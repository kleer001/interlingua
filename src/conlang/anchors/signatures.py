"""Per-concept phonological signatures from anchor-v1 entries.

For each concept we aggregate:
- the unique IPA forms (one per language, plus extras),
- the panphon feature vectors of those forms,
- mean / variance per feature across the pooled segments,
- a single-number *sharpness* score that compresses variance into one float
  so we can sort and visualize.

Sketch §"Cross-linguistic variation is data, not noise": cow=/m_u/ is sharp,
pig is fuzzy. Sharpness here is the inverse of mean per-feature variance,
normalized to [0, 1] (1 = identical features across all entries, 0 = max
variance for ternary features).

Output: `signatures-v1.jsonl`, one row per concept, with the
`concept_attribute_table` from Phase 6 staying independent.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .phon_features import feature_names, featurize_ipa, mean_var
from .schema import AnchorEntry


@dataclass
class ConceptSignature:
    concept: str
    n_entries: int
    n_with_ipa: int
    n_languages: int
    languages: list[str]
    feature_names: list[str]
    mean_features: list[float]
    var_features: list[float]
    sharpness: float
    # Up to ~5 representative examples (language + IPA + form) for hover/UI.
    examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# Max feature variance for a ternary feature with values in {-1, 0, 1}
# is 1.0 (a 50/50 split between -1 and +1). We normalize against that.
_MAX_VARIANCE = 1.0


def _sharpness_from_var(vars_: list[float], names: list[str]) -> float:
    """Lower variance = sharper. Average over information-bearing features,
    weighted by panphon's natural feature ordering. Tone features (`hitone`,
    `hireg`) are usually all-zero for our onomatopoeia, so excluding them
    keeps the score from being artificially close to 1.0."""
    skip = {"hitone", "hireg"}
    keep = [(j, v) for j, v in enumerate(vars_) if names[j] not in skip]
    if not keep:
        return 0.0
    avg_var = sum(v for _, v in keep) / len(keep)
    return max(0.0, 1.0 - avg_var / _MAX_VARIANCE)


def signature_for_concept(concept: str, entries: list[AnchorEntry]) -> ConceptSignature:
    rows = [e for e in entries if e.concept == concept]
    with_ipa = [e for e in rows if e.ipa]
    langs = sorted({e.language_code for e in rows if e.language_code})
    names = feature_names()

    # One bag-of-features vector per form: mean over segments. Mixing
    # consonant rows and vowel rows from the same IPA string into a single
    # pool destroys structure (syl flips +1/-1 per segment), so the
    # per-form summary is the natural unit for cross-linguistic variance.
    form_vectors: list[list[float]] = []
    examples: list[dict] = []
    seen_langs: set[str] = set()
    for e in with_ipa:
        segs = featurize_ipa(e.ipa or "")
        if not segs:
            continue
        per_form = [sum(seg[j] for seg in segs) / len(segs) for j in range(len(names))]
        form_vectors.append(per_form)
        if e.language_code not in seen_langs and len(examples) < 8:
            examples.append(
                {
                    "language": e.language,
                    "language_code": e.language_code,
                    "form": e.orthography,
                    "romanization": e.romanization,
                    "ipa": e.ipa,
                    "n_segments": len(segs),
                }
            )
            seen_langs.add(e.language_code or "")

    means, vars_ = mean_var(form_vectors)
    sharpness = _sharpness_from_var(vars_, names)
    return ConceptSignature(
        concept=concept,
        n_entries=len(rows),
        n_with_ipa=len(with_ipa),
        n_languages=len(langs),
        languages=langs,
        feature_names=names,
        mean_features=means,
        var_features=vars_,
        sharpness=sharpness,
        examples=examples,
    )


def build_all_signatures(entries: Iterable[AnchorEntry]) -> list[ConceptSignature]:
    entries_list = list(entries)
    concepts = sorted({e.concept for e in entries_list})
    return [signature_for_concept(c, entries_list) for c in concepts]


def write_signatures(sigs: Iterable[ConceptSignature], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for s in sigs:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n
