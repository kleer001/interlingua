# ruff: noqa: E501
"""Phase 3 validation — Spearman ρ between decoder-space and stem-space distance.

Sample ≥10k random feature pairs from the N=2000 substrate. For each pair,
compute cosine distance in 2304-d decoder space and panphon-feature
phonological distance between the interpolated stems. Run Spearman ρ.

ρ ≳ 0.15 promotes the 41% round-trip from vibe-check to signed-off.
Spec: `semanticphonology.md` §"Phase 3 — Validation".
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from conlang import PROCESSED_DIR
from conlang.interpolate import (
    _LEXICON_IPA_OVERRIDE,
    load_context,
    stem_for_position,
)
from conlang.lab.project import phonological_distance

SUBSTRATE_PATH = PROCESSED_DIR / "substrate-v1-n2000.parquet"
REPORT_PATH = PROCESSED_DIR / "phase3_spearman_report.txt"
PAIRS_PATH = PROCESSED_DIR / "phase3_spearman_pairs.parquet"
N_PAIRS = 10_000
SEED = 0


def stem_to_ipa(stem: str) -> str:
    """Apply the lexicon→IPA override so panphon featurizes Latin g/y correctly."""
    return "".join(_LEXICON_IPA_OVERRIDE.get(c, c) for c in stem)


def build_stems(decoder_vecs: np.ndarray, ctx) -> list[str]:
    """One-pass: interpolate a stem for every substrate feature."""
    stems: list[str] = []
    for i, vec in enumerate(decoder_vecs):
        stems.append(stem_for_position(vec, ctx).stem)
        if (i + 1) % 500 == 0:
            print(f"  stems: {i + 1}/{len(decoder_vecs)}")
    return stems


def sample_pairs(n_features: int, n_pairs: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n_pairs:
        i = rng.randrange(n_features)
        j = rng.randrange(n_features)
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def main() -> None:
    print(f"loading substrate from {SUBSTRATE_PATH}")
    df = pd.read_parquet(SUBSTRATE_PATH)
    feature_ids = df["feature_id"].tolist()
    decoder_vecs = np.stack([np.asarray(v, dtype=np.float64) for v in df["decoder_vec"]])
    print(f"  {len(feature_ids)} features × {decoder_vecs.shape[1]}-d decoder")

    print("loading interpolation context (anchor positions + modal projections)")
    ctx = load_context()
    print(f"  {len(ctx.anchor_concepts)} anchors")

    print(f"interpolating stems for {len(feature_ids)} features")
    stems = build_stems(decoder_vecs, ctx)
    stem_ipa = [stem_to_ipa(s) for s in stems]

    print(f"sampling {N_PAIRS} pairs (seed={SEED})")
    pairs = sample_pairs(len(feature_ids), N_PAIRS, SEED)

    print("computing distances")
    cos_d = np.empty(len(pairs), dtype=np.float64)
    phon_d = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        cos_d[k] = float(cosine(decoder_vecs[i], decoder_vecs[j]))
        phon_d[k] = phonological_distance(stem_ipa[i], stem_ipa[j])
        if (k + 1) % 2000 == 0:
            print(f"  pairs: {k + 1}/{len(pairs)}")

    rho, p_value = spearmanr(cos_d, phon_d)

    interp = (
        "preserves" if rho > 0.15 else "does not preserve"
    )
    report = (
        f"Phase 3 Spearman validation\n"
        f"---------------------------\n"
        f"substrate:      {SUBSTRATE_PATH.name}\n"
        f"n_features:     {len(feature_ids)}\n"
        f"n_pairs:        {len(pairs)}\n"
        f"seed:           {SEED}\n"
        f"\n"
        f"Spearman rho:   {rho:.6f}\n"
        f"p-value:        {p_value:.3e}\n"
        f"\n"
        f"cosine_distance:        mean={cos_d.mean():.4f}  median={np.median(cos_d):.4f}  std={cos_d.std():.4f}\n"
        f"phonological_distance:  mean={phon_d.mean():.4f}  median={np.median(phon_d):.4f}  std={phon_d.std():.4f}\n"
        f"\n"
        f"interpretation: rho = {rho:.4f} -> interpolation {interp} semantic structure (threshold 0.15)\n"
    )

    print()
    print(report)
    REPORT_PATH.write_text(report)
    print(f"wrote report: {REPORT_PATH}")

    pair_df = pd.DataFrame(
        {
            "feature_id_a": [feature_ids[i] for i, _ in pairs],
            "feature_id_b": [feature_ids[j] for _, j in pairs],
            "cosine_distance": cos_d,
            "phonological_distance": phon_d,
        }
    )
    pair_df.to_parquet(PAIRS_PATH)
    print(f"wrote pairs:  {PAIRS_PATH}")


if __name__ == "__main__":
    main()
