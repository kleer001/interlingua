"""Stage 5 regularization: collapse the Stage 3 multigraph into a per-node schema.

Each node is assigned:
  parent   — highest-PMI co-activation neighbor (or None if no positive PMI pair)
  siblings — other nodes sharing the same non-noise HDBSCAN cluster
  near     — top-K cosine neighbors not already parent/sibling, above min_cosine

Per spec v0.2 §7 (Commitment 7) the schema deliberately omits a `transformation`
primitive. Antonymy/negation is handled compositionally via affix at Stage 6
phonology, not as a graph edge — Tegmark function-vector crystals failed to
bridge to Gemma Scope SAE decoders, so we don't promote a transformation
relation we can't ground.

Usage:
    python -m conlang.regularize
        [--min-pmi 0.0] [--near-top-k 5] [--min-cosine 0.30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import INTERIM_DIR, PROCESSED_DIR, RAW_DIR


def pick_parent(
    pmi_row: np.ndarray,
    self_idx: int,
    min_pmi: float,
) -> tuple[int, float] | None:
    """Highest-PMI neighbor strictly above min_pmi, or None."""
    row = pmi_row.astype(np.float64, copy=True)
    row[self_idx] = -np.inf
    j = int(np.argmax(row))
    p = float(row[j])
    if p <= min_pmi or not np.isfinite(p):
        return None
    return j, p


def siblings_in_cluster(labels: np.ndarray, self_idx: int) -> list[int]:
    """Other nodes sharing this node's HDBSCAN label. Noise (-1) returns []."""
    label = int(labels[self_idx])
    if label == -1:
        return []
    same = np.where(labels == label)[0]
    return [int(i) for i in same if int(i) != self_idx]


def near_neighbors(
    sim_row: np.ndarray,
    self_idx: int,
    exclude: set[int],
    top_k: int,
    min_cosine: float,
) -> list[tuple[int, float]]:
    """Top-K cosine neighbors above min_cosine, excluding self/parent/siblings."""
    row = sim_row.astype(np.float64, copy=True)
    row[self_idx] = -np.inf
    for j in exclude:
        row[j] = -np.inf
    order = np.argsort(-row)
    out: list[tuple[int, float]] = []
    for j in order:
        s = float(row[j])
        if not np.isfinite(s) or s < min_cosine:
            break
        out.append((int(j), s))
        if len(out) >= top_k:
            break
    return out


def regularize(
    features: list[dict],
    sim: np.ndarray,
    labels: np.ndarray,
    pmi: np.ndarray,
    min_pmi: float = 0.0,
    near_top_k: int = 5,
    min_cosine: float = 0.30,
) -> list[dict]:
    n = len(features)
    assert sim.shape == (n, n), f"sim shape {sim.shape} != ({n}, {n})"
    assert labels.shape == (n,), f"labels shape {labels.shape} != ({n},)"
    assert pmi.shape == (n, n), f"pmi shape {pmi.shape} != ({n}, {n})"

    nodes: list[dict] = []
    for i in range(n):
        parent = pick_parent(pmi[i], i, min_pmi)
        siblings = siblings_in_cluster(labels, i)
        exclude = set(siblings)
        if parent is not None:
            exclude.add(parent[0])
        near = near_neighbors(sim[i], i, exclude, near_top_k, min_cosine)
        feat = features[i]
        nodes.append(
            {
                "slice_idx": i,
                "feature_id": feat["feature_id"],
                "label": feat["label"],
                "parent": (
                    {"slice_idx": parent[0], "pmi": parent[1]}
                    if parent is not None
                    else None
                ),
                "siblings": siblings,
                "near": [{"slice_idx": j, "cosine": s} for j, s in near],
            }
        )
    return nodes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--min-pmi",
        type=float,
        default=0.0,
        help="Minimum PMI for a parent edge. Default 0 = strictly positive association.",
    )
    p.add_argument("--near-top-k", type=int, default=5)
    p.add_argument(
        "--min-cosine",
        type=float,
        default=0.30,
        help="Minimum cosine similarity for a `near` edge. Matches slice default.",
    )
    p.add_argument("--features", type=Path, default=RAW_DIR / "features.jsonl")
    p.add_argument("--sim", type=Path, default=INTERIM_DIR / "sim_matrix.npy")
    p.add_argument("--labels", type=Path, default=INTERIM_DIR / "hdbscan_labels.npy")
    p.add_argument(
        "--pmi", type=Path, default=INTERIM_DIR / "coactivation" / "pmi.npy"
    )
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "regularized.json")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    print("[1/3] Loading inputs ...", flush=True)
    with open(args.features) as f:
        features = [json.loads(line) for line in f]
    sim = np.load(args.sim)
    labels = np.load(args.labels)
    pmi = np.load(args.pmi)
    print(
        f"      {len(features)} features, sim {sim.shape}, "
        f"labels {labels.shape} ({int((labels==-1).sum())} noise), "
        f"pmi {pmi.shape}",
        flush=True,
    )

    print("[2/3] Regularizing ...", flush=True)
    nodes = regularize(
        features=features,
        sim=sim,
        labels=labels,
        pmi=pmi,
        min_pmi=args.min_pmi,
        near_top_k=args.near_top_k,
        min_cosine=args.min_cosine,
    )
    n_with_parent = sum(1 for nd in nodes if nd["parent"] is not None)
    n_with_siblings = sum(1 for nd in nodes if nd["siblings"])
    near_counts = [len(nd["near"]) for nd in nodes]
    print(f"      {n_with_parent}/{len(nodes)} nodes have a parent", flush=True)
    print(
        f"      {n_with_siblings}/{len(nodes)} nodes have ≥1 sibling "
        f"(non-noise HDBSCAN cluster members)",
        flush=True,
    )
    print(
        f"      near edges per node: median {int(np.median(near_counts))}, "
        f"max {max(near_counts)}, total {sum(near_counts)}",
        flush=True,
    )

    out = {
        "schema_version": 1,
        "params": {
            "min_pmi": args.min_pmi,
            "near_top_k": args.near_top_k,
            "min_cosine": args.min_cosine,
        },
        "n_nodes": len(nodes),
        "nodes": nodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    size_kb = args.out.stat().st_size / 1024
    print(f"[3/3] Wrote {args.out} ({size_kb:.0f} KB)", flush=True)
    return args.out


def main() -> None:
    args = parse_args(sys.argv[1:])
    run(args)


if __name__ == "__main__":
    main()
