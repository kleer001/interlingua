"""Stage 6 lexicon: assign noun class + phonosemantic stem + apply affixes.

Three pieces:
1. `assign_class(label)` — keyword heuristic over the Neuronpedia label text
   maps each node to one of the 11 noun classes. Bias is explicit in
   KEYWORD_RULES; unmatched labels fall through to class 11 (abstract).
2. `build_stem(...)` — phonosemantic 3-syllable stem where:
     CV1 encodes the HDBSCAN cluster (siblings share CV1)
     CV2 = hash(parent_feature_id) (PMI-related nodes share CV2)
     CV3 = hash(self_feature_id) (uniqueness within bucket)
   Collisions get a CV4 appended; if still colliding, CV5; etc.
3. `build_lexicon()` orchestrates: load regularized.json + features.jsonl +
   hdbscan_labels.npy, assign classes, generate stems, apply class prefix +
   negation, and write `data/processed/lexicon.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from .phonology import (
    CLASS_PREFIXES,
    SINGLE_CONSONANTS,
    VOWELS,
    apply_class_prefix,
    is_valid_word,
    negate,
)


# Keyword → class_id. First match wins; unmatched labels fall through to
# class 11 (abstract). Five active singular classes (1, 5, 7, 9, 11) —
# Neuronpedia labels don't use plural markers in a way that's useful for
# class-assignment, so 2/4/6/8/10 stay empty.
#
# Patterns avoid the "terms/phrases/words/expressions" scaffolding that
# appears in nearly every label — that's about *how* the feature was
# described, not what it represents.
KEYWORD_RULES: list[tuple[str, int, str]] = [
    # Class 1: humans
    (r"\b(person|people|individuals?|men|women|child(ren)?|users?|speakers?|"
     r"authors?|writers?|figures?|name(s)?\s+of\s+(individuals|people|"
     r"public)|character(s)?|leaders?|workers?|teachers?|students?)\b",
     1, "human"),

    # Class 9: animals + languages
    (r"\b(animals?|cats?|dogs?|birds?|insects?|fish|mammals?|reptiles?|"
     r"creatures?|languages?|speech|dialects?|tongues?)\b",
     9, "animal/language"),

    # Class 5: physical natural objects (fruits, body parts, geography)
    (r"\b(fruits?|flowers?|crops?|leaves?|seeds?|trees?|plants?|"
     r"eyes?|ears?|hands?|feet|hearts?|brains?|organs?|bodies|"
     r"stones?|rocks?|hills?|mountains?|rivers?|lakes?|oceans?|"
     r"foods?|meals?|drinks?)\b",
     5, "natural object"),

    # Class 7: tools, systems, technical artifacts, processes, methods
    (r"\b(tools?|systems?|devices?|machines?|software|hardware|programs?|"
     r"algorithms?|methods?|formats?|protocols?|frameworks?|structures?|"
     r"processes?|procedures?|operations?|actions?|activities?|behaviors?|"
     r"movements?|techniques?|approaches?|technologies?|"
     r"equations?|formulas?|symbols?|notations?|measurements?|metrics?)\b",
     7, "tool/process"),
]

DEFAULT_CLASS_ID = 11  # abstract — catches everything else


def assign_class(label: str) -> int:
    """First-matching-keyword class assignment. Unmatched → class 11."""
    for pattern, class_id, _desc in KEYWORD_RULES:
        if re.search(pattern, label, re.IGNORECASE):
            return class_id
    return DEFAULT_CLASS_ID


# Fixed CV1 per HDBSCAN cluster. -1 = noise/singleton.
# Cluster IDs encountered in this slice: -1, 0..6 (7 real clusters).
# Use distinct CVs so siblings audibly share a first syllable.
CLUSTER_TO_CV1: dict[int, str] = {
    -1: "wa",  # noise/singleton — most nodes
    0:  "pa",
    1:  "te",
    2:  "ki",
    3:  "bo",
    4:  "mu",
    5:  "li",
    6:  "ne",
}
DEFAULT_CLUSTER_CV = "wa"


def _hash_to_cv(seed: str | int) -> str:
    """Deterministic single CV from a stable hash."""
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    c = SINGLE_CONSONANTS[digest[0] % len(SINGLE_CONSONANTS)]
    v = VOWELS[digest[1] % len(VOWELS)]
    return c + v


def _hash_to_cv_with_salt(seed: str | int, salt: int) -> str:
    return _hash_to_cv(f"{seed}::{salt}")


def build_stem(
    feature_id: int,
    cluster_id: int,
    parent_feature_id: int | None,
) -> str:
    """Phonosemantic 3-syllable stem: CV(cluster) + CV(parent) + CV(self)."""
    cv1 = CLUSTER_TO_CV1.get(cluster_id, DEFAULT_CLUSTER_CV)
    cv2 = _hash_to_cv(parent_feature_id) if parent_feature_id is not None else "ya"
    cv3 = _hash_to_cv(feature_id)
    return cv1 + cv2 + cv3


def _resolve_stem_collisions(entries: list[dict]) -> None:
    """Mutate entries in place: append CV4/CV5/... to dupe stems until unique."""
    by_stem: dict[str, list[int]] = {}
    for idx, e in enumerate(entries):
        by_stem.setdefault(e["stem"], []).append(idx)
    salt = 0
    while any(len(group) > 1 for group in by_stem.values()):
        salt += 1
        if salt > 50:
            # extremely unlikely with sha256; bail rather than spin forever
            raise RuntimeError("could not resolve stem collisions after 50 salt rounds")
        for stem, group in list(by_stem.items()):
            if len(group) <= 1:
                continue
            # First entry keeps current stem; others get a salted CV appended.
            for idx in group[1:]:
                e = entries[idx]
                extra = _hash_to_cv_with_salt(e["feature_id"], salt)
                e["stem"] = e["stem"] + extra
        # Rebuild map
        by_stem = {}
        for idx, e in enumerate(entries):
            by_stem.setdefault(e["stem"], []).append(idx)


def build_lexicon(
    features: list[dict],
    regularized_nodes: list[dict],
    hdbscan_labels: np.ndarray,
) -> list[dict]:
    """Compose the lexicon. One entry per node.

    Each entry has: feature_id, label, hdbscan_cluster, class_id, class_name,
    parent_feature_id, stem, surface, antonym.
    """
    assert len(features) == len(regularized_nodes) == len(hdbscan_labels), (
        f"length mismatch: {len(features)}, {len(regularized_nodes)}, "
        f"{len(hdbscan_labels)}"
    )

    # First pass: build entries with raw stems.
    entries: list[dict] = []
    for slice_idx, (feat, node) in enumerate(zip(features, regularized_nodes)):
        assert feat["feature_id"] == node["feature_id"], (
            f"feature_id mismatch at slice_idx {slice_idx}"
        )
        cluster = int(hdbscan_labels[slice_idx])
        parent = node["parent"]
        parent_feature_id = (
            features[parent["slice_idx"]]["feature_id"] if parent is not None else None
        )
        class_id = assign_class(feat["label"])
        stem = build_stem(
            feature_id=feat["feature_id"],
            cluster_id=cluster,
            parent_feature_id=parent_feature_id,
        )
        entries.append({
            "feature_id": feat["feature_id"],
            "slice_idx": slice_idx,
            "label": feat["label"],
            "hdbscan_cluster": cluster,
            "class_id": class_id,
            "class_name": CLASS_PREFIXES[class_id][1],
            "parent_feature_id": parent_feature_id,
            "stem": stem,
            # surface + antonym filled in after collision resolution
            "surface": None,
            "antonym": None,
        })

    # Second pass: resolve stem collisions (append CV4 until unique).
    _resolve_stem_collisions(entries)

    # Third pass: apply class prefix + negation now that stems are final.
    for e in entries:
        e["surface"] = apply_class_prefix(e["stem"], e["class_id"])
        e["antonym"] = negate(e["surface"])
        assert is_valid_word(e["surface"]), f"invalid surface form: {e['surface']!r}"
        assert is_valid_word(e["antonym"]), f"invalid antonym form: {e['antonym']!r}"

    return entries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=RAW_DIR / "features.jsonl")
    p.add_argument(
        "--regularized", type=Path, default=PROCESSED_DIR / "regularized.json"
    )
    p.add_argument(
        "--labels", type=Path, default=INTERIM_DIR / "hdbscan_labels.npy"
    )
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "lexicon.json")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    print("[1/3] Loading inputs ...", flush=True)
    with open(args.features) as f:
        features = [json.loads(line) for line in f]
    regularized = json.loads(args.regularized.read_text())
    nodes = regularized["nodes"]
    labels = np.load(args.labels)
    print(
        f"      {len(features)} features, {len(nodes)} regularized nodes, "
        f"{labels.shape} labels",
        flush=True,
    )

    print("[2/3] Building lexicon ...", flush=True)
    entries = build_lexicon(features, nodes, labels)

    by_class: dict[int, int] = {}
    for e in entries:
        by_class[e["class_id"]] = by_class.get(e["class_id"], 0) + 1
    print(
        "      class distribution: "
        + ", ".join(f"{cid}={by_class.get(cid, 0)}" for cid in sorted(CLASS_PREFIXES)),
        flush=True,
    )
    stem_lengths = [len(e["stem"]) for e in entries]
    print(
        f"      stems: min={min(stem_lengths)}, max={max(stem_lengths)}, "
        f"unique={len({e['stem'] for e in entries})}/{len(entries)}",
        flush=True,
    )

    out_doc = {
        "schema_version": 1,
        "n_entries": len(entries),
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2))
    print(f"[3/3] Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)", flush=True)
    return args.out


def main() -> None:
    args = parse_args(sys.argv[1:])
    run(args)


if __name__ == "__main__":
    main()
