"""GLUE Path 1 audit: hunt grammatical-glue features in the existing lexicon.

Per `GLUE-TODO.md` §"Path 1 — Audit the existing lexicon", steps 1-3:

1. Build a vocabulary of grammatical-concept indicator regexes (the
   English-shaped baseline — 20 categories).
2. Tag matches over the Neuronpedia auto-interp labels. First match wins.
3. Compute co-activation promiscuity per candidate:
     - n_distinct_clusters_in_neighbors  (over top-k PMI neighbors)
     - entropy_over_neighbor_clusters    (Shannon, base e)
   Genuine glue should have high cluster-entropy and few same-cluster
   neighbors; content features (an "apple" feature) sit inside their own
   semantic field with low entropy.

Output: `data/processed/function_lexicon.json` with high/medium/low
audit_confidence per surviving candidate. Steps 4-5 (Neuronpedia
cross-validation + final cut) need network and a human eyeball and live
outside this module.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .. import INTERIM_DIR, PROCESSED_DIR, RAW_DIR

# Regex table, exact patterns from GLUE-TODO.md §"Path 1" step 1.
# Order matters: first match wins.
CATEGORIES: tuple[tuple[str, str], ...] = (
    (
        "negation / polarity",
        r"\b(negation|negat(ed|ing|ion)|not\b|denial|absence of|lack of|refus|contradiction)\b",
    ),
    (
        "number (plural)",
        r"\b(plural(ity)?|multiple instances|collective|several\b.*items|enumeration of)\b",
    ),
    ("tense / past", r"\b(past (tense|events?)|previously|completed actions?|historical)\b"),
    (
        "tense / future",
        r"\b(future (tense|events?|reference)|upcoming|anticipated|will\b.*(occur|happen))\b",
    ),
    ("aspect / progressive", r"\b(ongoing|in progress|continuing actions?|continuous)\b"),
    ("aspect / perfect", r"\b(completed|finished|having (done|been)|resultative)\b"),
    ("person / 1st", r"\b(first[- ]person|speaker self[- ]reference|narrator\b)\b"),
    ("person / 2nd", r"\b(second[- ]person|addressee|direct address)\b"),
    ("definiteness", r"\b(definite|specific (referent|entity)|previously mentioned|anaphoric)\b"),
    ("possession", r"\b(possession|ownership|belonging to|possessive)\b"),
    ("spatial deixis", r"\b(proximal|distal|here vs there|spatial reference|location indicator)\b"),
    ("temporal deixis", r"\b(now\b|then\b|temporal (reference|deixis|adverb))\b"),
    (
        "conjunction / additive",
        r"\b(addition(al)?|in addition|also\b|conjunctive|listing items|enumerating)\b",
    ),
    (
        "conjunction / contrast",
        r"\b(contrast|opposition|adversative|however|nevertheless|despite)\b",
    ),
    ("conjunction / causal", r"\b(caus(e|al|ation)|because|therefore|consequently|reason for)\b"),
    ("question / interrog.", r"\b(interrog(ative|ation)|question(s|ing)?|inquiry)\b"),
    ("comparative", r"\b(compar(ative|ison)|more than|greater (degree|extent)|gradient)\b"),
    ("modality / necessity", r"\b(necessity|must\b|obligation|requirement|deontic)\b"),
    ("modality / possibility", r"\b(possibility|may\b|might\b|epistemic|uncertain(ty)?)\b"),
    (
        "discourse marker",
        r"\b(discourse marker|hedging|topic shift|attention indicator|emphasis)\b",
    ),
)


def _compile_categories() -> list[tuple[str, re.Pattern]]:
    return [(name, re.compile(pat, flags=re.IGNORECASE)) for name, pat in CATEGORIES]


def categorize_label(label: str, compiled: list[tuple[str, re.Pattern]]) -> str | None:
    """First-match-wins category, or None if no regex hits."""
    for name, pat in compiled:
        if pat.search(label):
            return name
    return None


def neighbor_promiscuity(
    *,
    pmi_row: np.ndarray,
    hdbscan_labels: np.ndarray,
    self_idx: int,
    top_k: int,
    min_pmi: float,
) -> dict:
    """Top-k PMI neighbors, count distinct HDBSCAN clusters and Shannon entropy.

    Noise points (HDBSCAN label -1) are folded into the count as a single
    pseudo-cluster -1, since a glue feature firing alongside many noise
    points still demonstrates cross-domain firing.

    Returns:
        n_neighbors: int — how many neighbors made the cut
        n_distinct_clusters: int — count of unique hdbscan labels in neighbors
        entropy: float — Shannon entropy (nats) over the cluster distribution
        same_cluster_neighbors: int — neighbors sharing this node's hdbscan label
                                       (or 0 if this node is noise)
    """
    row = pmi_row.astype(np.float64, copy=True)
    row[self_idx] = -np.inf
    # Top-k by PMI, but require positive PMI
    order = np.argsort(-row)
    pick: list[int] = []
    for j in order:
        p = float(row[j])
        if not np.isfinite(p) or p <= min_pmi:
            break
        pick.append(int(j))
        if len(pick) >= top_k:
            break

    self_label = int(hdbscan_labels[self_idx])
    counts: Counter[int] = Counter(int(hdbscan_labels[j]) for j in pick)
    total = sum(counts.values())
    if total == 0:
        return {
            "n_neighbors": 0,
            "n_distinct_clusters": 0,
            "entropy": 0.0,
            "same_cluster_neighbors": 0,
        }
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
    same_cluster = counts.get(self_label, 0) if self_label != -1 else 0
    return {
        "n_neighbors": total,
        "n_distinct_clusters": len(counts),
        "entropy": entropy,
        "same_cluster_neighbors": same_cluster,
    }


def confidence_tier(
    *,
    entropy: float,
    n_distinct_clusters: int,
    n_neighbors: int,
    same_cluster_neighbors: int,
    entropy_high: float,
    entropy_medium: float,
) -> str:
    """Map (entropy, clusters, same-cluster count) to high/medium/low.

    Glue signature is "many small clusters, no single dominant one".
    Concrete features cluster tightly: low entropy, lots of same-cluster
    neighbors. We threshold on entropy and check that the candidate isn't
    living mostly inside one cluster.
    """
    if n_neighbors == 0:
        return "low"
    same_share = same_cluster_neighbors / n_neighbors
    if entropy >= entropy_high and n_distinct_clusters >= 3 and same_share <= 0.5:
        return "high"
    if entropy >= entropy_medium and n_distinct_clusters >= 2:
        return "medium"
    return "low"


def run_audit(
    *,
    features_path: Path,
    pmi_path: Path,
    labels_path: Path,
    top_k: int,
    min_pmi: float,
    entropy_high: float,
    entropy_medium: float,
) -> dict:
    features = [json.loads(line) for line in features_path.open() if line.strip()]
    pmi = np.load(pmi_path)
    labels = np.load(labels_path)
    if pmi.shape != (len(features), len(features)):
        raise ValueError(f"pmi shape {pmi.shape} disagrees with features count {len(features)}")
    if labels.shape != (len(features),):
        raise ValueError(
            f"labels shape {labels.shape} disagrees with features count {len(features)}"
        )

    compiled = _compile_categories()
    entries: list[dict] = []
    category_counts: Counter[str] = Counter()
    for i, f in enumerate(features):
        cat = categorize_label(f["label"], compiled)
        if cat is None:
            continue
        promiscuity = neighbor_promiscuity(
            pmi_row=pmi[i],
            hdbscan_labels=labels,
            self_idx=i,
            top_k=top_k,
            min_pmi=min_pmi,
        )
        tier = confidence_tier(
            entropy=promiscuity["entropy"],
            n_distinct_clusters=promiscuity["n_distinct_clusters"],
            n_neighbors=promiscuity["n_neighbors"],
            same_cluster_neighbors=promiscuity["same_cluster_neighbors"],
            entropy_high=entropy_high,
            entropy_medium=entropy_medium,
        )
        category_counts[cat] += 1
        entries.append(
            {
                "feature_id": int(f["feature_id"]),
                "slice_idx": i,
                "label": str(f["label"]),
                "grammatical_category": cat,
                "hdbscan_cluster": int(labels[i]),
                "n_pmi_neighbors": promiscuity["n_neighbors"],
                "n_distinct_clusters_in_neighbors": promiscuity["n_distinct_clusters"],
                "entropy_over_neighbor_clusters": promiscuity["entropy"],
                "same_cluster_neighbors": promiscuity["same_cluster_neighbors"],
                "audit_confidence": tier,
                "neuronpedia_url": _neuronpedia_url(int(f["feature_id"])),
            }
        )

    entries.sort(
        key=lambda e: (
            {"high": 0, "medium": 1, "low": 2}[e["audit_confidence"]],
            -e["entropy_over_neighbor_clusters"],
        )
    )

    tier_counts = Counter(e["audit_confidence"] for e in entries)
    return {
        "schema_version": 1,
        "source": "path-1-audit",
        "n_features_scanned": len(features),
        "n_entries": len(entries),
        "params": {
            "top_k_pmi_neighbors": top_k,
            "min_pmi": min_pmi,
            "entropy_high_threshold": entropy_high,
            "entropy_medium_threshold": entropy_medium,
        },
        "category_counts": dict(category_counts),
        "tier_counts": dict(tier_counts),
        "entries": entries,
    }


def _neuronpedia_url(feature_id: int) -> str:
    return f"https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/{feature_id}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=RAW_DIR / "features.jsonl")
    p.add_argument("--pmi", type=Path, default=INTERIM_DIR / "coactivation" / "pmi.npy")
    p.add_argument("--labels", type=Path, default=INTERIM_DIR / "hdbscan_labels.npy")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-pmi", type=float, default=0.0)
    p.add_argument("--entropy-high", type=float, default=1.5)
    p.add_argument("--entropy-medium", type=float, default=0.7)
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "function_lexicon.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"[1/2] Scanning {args.features.name} for grammatical-glue candidates ...", flush=True)
    out = run_audit(
        features_path=args.features,
        pmi_path=args.pmi,
        labels_path=args.labels,
        top_k=args.top_k,
        min_pmi=args.min_pmi,
        entropy_high=args.entropy_high,
        entropy_medium=args.entropy_medium,
    )
    print(
        f"      {out['n_entries']} candidates across {len(out['category_counts'])} categories",
        flush=True,
    )
    print(f"      tier counts: {out['tier_counts']}", flush=True)
    print(f"      category counts: {out['category_counts']}", flush=True)

    print(f"[2/2] Writing {args.out} ...", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    size_kb = args.out.stat().st_size / 1024
    print(f"      {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
