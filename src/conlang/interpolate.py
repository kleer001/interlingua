"""Phase 1 of the Stage-6 cutover — natural-neighbor interpolation library.

Given a feature's position in Gemma-residual space, return a phonotactically
valid (C)V stem in the 16C/5V lexicon inventory by:

1. Finding the top-k nearest anchor positions in
   `data/processed/anchor-positions-v1.parquet` (146 (concept,attribute)
   points, each in 2304-d Gemma layer-12 residual space).
2. Computing inverse-distance weights over those k neighbors as a
   practical substitute for full d-dimensional Sibson weights (which are
   infeasible at d=2304).
3. Mixing each neighbor's *concept-level* modal projection per panphon
   segment position, weighted by Sibson weight. The modal projection
   comes from `data/processed/anchors-v1.parquet`.
4. Discretizing each blended panphon vector to the nearest 16C/5V
   inventory phoneme.
5. Running a phonotactic gate that enforces (C)V syllables ≥ 2 — when
   the first-nearest phoneme would break (C)V, fall back to the
   second-nearest that keeps the syllable legal.

`stem_for_position(query_vec)` is the top-level entry. The library is
free of any side effect on `lexicon.json` — that swap is Phase 2.

Companion: `semanticphonology.md` §"Phase 1 — Interpolation infrastructure".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import PROCESSED_DIR
from .anchors.phon_features import featurize_ipa
from .phonology import SINGLE_CONSONANTS, VOWELS, syllabify

DEFAULT_SUBSTRATE = PROCESSED_DIR / "substrate-v1-n2000.parquet"
DEFAULT_ANCHORS = PROCESSED_DIR / "anchors-v1.parquet"
DEFAULT_ANCHOR_POSITIONS = PROCESSED_DIR / "anchor-positions-v1.parquet"


# ── inventory features ───────────────────────────────────────────────────


# Orthography → IPA when panphon's tokenizer disagrees with the lexicon's
# Latin spellings. The lexicon writes /g/ as Latin 'g' (U+0067); panphon
# only recognizes 'ɡ' (U+0261). Same for /y/ palatal glide (Latin 'y' → IPA 'j').
_LEXICON_IPA_OVERRIDE: dict[str, str] = {
    "g": "ɡ",  # voiced velar stop
    "y": "j",  # palatal glide (English y as in "yes")
}


@lru_cache(maxsize=1)
def lexicon_phoneme_features() -> dict[str, list[int]]:
    """Return {phoneme: 24-d panphon feature vector} for the 16C/5V lexicon.

    Mirrors `anchors.inventory.phoneme_features` but uses the lexicon's
    16-consonant 5-vowel set from `phonology.py` (the inventory the
    cutover stems are written in). The output dict is keyed on the
    *orthographic* spelling used by `phonology.py`; panphon featurization
    routes through `_LEXICON_IPA_OVERRIDE` when needed.
    """
    out: dict[str, list[int]] = {}
    for p in SINGLE_CONSONANTS + VOWELS:
        ipa = _LEXICON_IPA_OVERRIDE.get(p, p)
        segs = featurize_ipa(ipa)
        if not segs:
            raise RuntimeError(f"panphon could not featurize lexicon phoneme {p!r} (ipa={ipa!r})")
        out[p] = segs[0]
    return out


def _squared_euclid(a: Iterable[float], b: Iterable[float]) -> float:
    return float(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))


def project_to_lexicon(seg: Iterable[float], *, only: tuple[str, ...] | None = None) -> str:
    """Nearest 16C/5V phoneme by squared-Euclidean panphon feature distance.

    `only` lets the phonotactic gate restrict the search to consonants or
    vowels (or, on a fallback retry, all but one specific phoneme).
    """
    feats = lexicon_phoneme_features()
    candidates = only if only is not None else tuple(feats)
    best: str | None = None
    best_d: float | None = None
    for p in candidates:
        d = _squared_euclid(seg, feats[p])
        if best_d is None or d < best_d:
            best_d = d
            best = p
    assert best is not None
    return best


def ranked_lexicon_candidates(seg: Iterable[float]) -> list[str]:
    """Return all 16C+5V phonemes sorted by feature distance to `seg` ascending."""
    feats = lexicon_phoneme_features()
    seg_list = list(seg)
    return sorted(feats, key=lambda p: _squared_euclid(seg_list, feats[p]))


# ── neighbor weights ─────────────────────────────────────────────────────


def neighbor_weights(
    query: np.ndarray,
    anchor_positions: np.ndarray,
    *,
    k: int = 5,
    epsilon: float = 1e-6,
) -> list[tuple[int, float]]:
    """Top-k anchors by cosine distance with inverse-distance weights.

    Returns (anchor_idx, weight) sorted by ascending distance (descending
    weight). `epsilon` guards against divide-by-zero when the query lands
    exactly on an anchor.
    """
    q = query.astype(np.float64)
    A = anchor_positions.astype(np.float64)
    qn = np.linalg.norm(q)
    An = np.linalg.norm(A, axis=1)
    denom = qn * An
    denom[denom == 0] = epsilon
    sims = (A @ q) / denom
    dists = np.clip(1.0 - sims, 0.0, 2.0)
    top_idx = np.argsort(dists)[:k]
    top_d = dists[top_idx]
    inv = 1.0 / (top_d + epsilon)
    w = inv / inv.sum()
    return [(int(i), float(wi)) for i, wi in zip(top_idx, w, strict=True)]


# ── signature mixer ──────────────────────────────────────────────────────


def _panphon_segs(text: str) -> list[list[int]]:
    return featurize_ipa(text) if text else []


def mix_signatures(
    weighted_anchors: list[tuple[int, float]],
    modal_projections: list[str],
) -> list[list[float]]:
    """Per-segment-position weighted blend of anchor modal projections.

    Target length = round(weighted mean of segment counts among the top-k
    anchors). Positions beyond a given anchor's segment list are filled
    with the zero vector (i.e., that anchor contributes nothing at that
    position).

    Returns a list of 24-d continuous panphon feature targets, one per
    segment slot.
    """
    if not weighted_anchors:
        return []
    seg_lists = [_panphon_segs(modal_projections[idx]) for idx, _ in weighted_anchors]
    if not any(seg_lists):
        return []
    dim = next(len(s[0]) for s in seg_lists if s)

    target_len = round(
        sum(w * len(segs) for (_, w), segs in zip(weighted_anchors, seg_lists, strict=True))
    )
    target_len = max(target_len, 2)  # always at least one syllable's worth

    out: list[list[float]] = []
    for p in range(target_len):
        blend = [0.0] * dim
        total_w = 0.0
        for (_, w), segs in zip(weighted_anchors, seg_lists, strict=True):
            if p < len(segs):
                for j in range(dim):
                    blend[j] += w * float(segs[p][j])
                total_w += w
        if total_w > 0:
            blend = [x / total_w for x in blend]
        out.append(blend)
    return out


# ── discretization + phonotactic gate ────────────────────────────────────


@dataclass
class StemBuild:
    """Diagnostic output of stem_for_position so callers can audit it."""

    stem: str
    raw_discretization: str
    fallback_segments: int  # how many positions used second-nearest


def _is_consonant(phoneme: str) -> bool:
    return phoneme in SINGLE_CONSONANTS


def _is_vowel(phoneme: str) -> bool:
    return phoneme in VOWELS


def discretize_with_phonotactics(target_segs: list[list[float]]) -> StemBuild:
    """Walk segments left-to-right, emitting a (C)V-valid string.

    Strategy:
    - Discretize each segment to its nearest 16C/5V phoneme.
    - If two consonants sit adjacent (CC), retry the second one against
      vowel candidates only — the phonotactic gate is "fall back to the
      second-nearest phoneme that keeps the syllable legal".
    - If two vowels sit adjacent (VV), retry the second against consonants.
    - Drop a trailing coda consonant (no codas in this inventory).
    - If the final string syllabifies with ≥ 2 syllables, return it.
      Otherwise pad with /a/.
    """
    raw_chars: list[str] = [project_to_lexicon(seg) for seg in target_segs]
    raw = "".join(raw_chars)

    fixed_chars: list[str] = []
    fallbacks = 0
    for i, seg in enumerate(target_segs):
        candidate = project_to_lexicon(seg)
        if i == 0:
            # Free choice on first slot — consonant or vowel both make valid syllable starts
            fixed_chars.append(candidate)
            continue
        prev = fixed_chars[-1]
        prev_is_c = _is_consonant(prev)
        cand_is_c = _is_consonant(candidate)
        if prev_is_c == cand_is_c:
            # CC or VV — pick from opposite class
            want = VOWELS if prev_is_c else SINGLE_CONSONANTS
            candidate = project_to_lexicon(seg, only=want)
            fallbacks += 1
        fixed_chars.append(candidate)

    if fixed_chars and _is_consonant(fixed_chars[-1]):
        # No codas allowed — drop trailing consonant rather than appending a vowel,
        # since dropping is closer to the spec's "second-nearest" fallback.
        fixed_chars.pop()
        fallbacks += 1

    stem = "".join(fixed_chars)
    if not stem or syllabify(stem) is None or len(syllabify(stem) or []) < 2:
        # Pad with 'a' until ≥ 2 syllables. 'a' is the maximally-neutral vowel.
        while syllabify(stem) is None or len(syllabify(stem) or []) < 2:
            stem += "a"
            fallbacks += 1
            if len(stem) > 12:
                break
    return StemBuild(stem=stem, raw_discretization=raw, fallback_segments=fallbacks)


# ── top-level entry ──────────────────────────────────────────────────────


@dataclass
class InterpolationContext:
    """Parquet contents loaded once; pass into stem_for_position repeatedly."""

    anchor_positions: np.ndarray  # (n_anchors, d_model)
    anchor_concepts: list[str]
    concept_modal_projections: dict[str, str]


def load_context(
    anchors_path: Path = DEFAULT_ANCHORS,
    anchor_positions_path: Path = DEFAULT_ANCHOR_POSITIONS,
) -> InterpolationContext:
    import pyarrow.parquet as pq

    pos_tbl = pq.read_table(anchor_positions_path)
    pos_df = pos_tbl.to_pandas()
    anchor_positions = np.array([list(v) for v in pos_df["position"]], dtype=np.float64)
    anchor_concepts = pos_df["concept"].tolist()

    anchors_tbl = pq.read_table(anchors_path)
    anchors_df = anchors_tbl.to_pandas()
    concept_modal_projections = dict(
        zip(anchors_df["concept"].tolist(), anchors_df["modal_projection"].tolist(), strict=True)
    )
    missing = [c for c in anchor_concepts if c not in concept_modal_projections]
    if missing:
        raise KeyError(f"anchor positions reference unknown concepts: {missing[:3]} ...")
    return InterpolationContext(
        anchor_positions=anchor_positions,
        anchor_concepts=anchor_concepts,
        concept_modal_projections=concept_modal_projections,
    )


def stem_for_position(
    query_vec: np.ndarray,
    ctx: InterpolationContext,
    *,
    k: int = 5,
) -> StemBuild:
    weighted = neighbor_weights(query_vec, ctx.anchor_positions, k=k)
    modal_list = [ctx.concept_modal_projections[ctx.anchor_concepts[idx]] for idx, _ in weighted]
    # Build a parallel weights-with-projections list aligned to modal_list indices
    weighted_for_mix = [(i, w) for i, (_, w) in enumerate(weighted)]
    target_segs = mix_signatures(weighted_for_mix, modal_list)
    if not target_segs:
        return StemBuild(stem="aa", raw_discretization="", fallback_segments=2)
    return discretize_with_phonotactics(target_segs)


def stem_for_feature_id(
    feature_id: int,
    *,
    substrate_path: Path = DEFAULT_SUBSTRATE,
    ctx: InterpolationContext | None = None,
    k: int = 5,
) -> StemBuild:
    import pyarrow.parquet as pq

    if ctx is None:
        ctx = load_context()
    tbl = pq.read_table(substrate_path, columns=["feature_id", "decoder_vec"])
    df = tbl.to_pandas()
    row = df[df["feature_id"] == feature_id]
    if row.empty:
        raise KeyError(f"feature_id {feature_id} not in substrate {substrate_path}")
    vec = np.array(list(row.iloc[0]["decoder_vec"]), dtype=np.float64)
    return stem_for_position(vec, ctx, k=k)
