"""Run the SAE↔crystal bridge end-to-end.

Loads:
  - Gemma Scope SAE decoder vectors (full 16384 features at layer 12 width 16k)
  - Per-relation hidden-state difference CSVs from data/interim/crystals/...
  - Neuronpedia bulk explanations for human-readable top-feature reports

Writes:
  - data/processed/crystal_bridge/alignment_raw.npy        (16384, 12)
  - data/processed/crystal_bridge/alignment_lda.npy        (16384, 12)
  - data/processed/crystal_bridge/coverage.json
  - data/processed/crystal_bridge/top_features_per_crystal.json
  - data/processed/crystal_bridge/best_crystal_per_feature.parquet (or csv)
"""

from __future__ import annotations

# HF_HUB_CACHE first — see hf-hub-cache-import-order memory.
import os

os.environ.setdefault("HF_HUB_CACHE", "/media/menser/fauna/interlingua/hf-cache")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import PROCESSED_DIR  # noqa: E402
from .edges.crystals import (  # noqa: E402
    DEFAULT_CRYSTAL_DIR,
    best_crystal_per_feature,
    build_bridge,
    coverage_stats,
    distinctive_top_features_per_crystal,
    distinctiveness_stats,
    load_relation_diffs,
    top_features_per_crystal,
)
from .ingest import decoder_vectors, load_bulk_explanations, load_sae  # noqa: E402

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
DEFAULT_MARGIN_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    p.add_argument("--neuronpedia-model", default="gemma-2-2b")
    p.add_argument("--neuronpedia-source", default="12-gemmascope-res-16k")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--normalize", type=int, default=0)
    p.add_argument("--n-lda", type=int, default=8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--crystal-root", type=Path, default=DEFAULT_CRYSTAL_DIR)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR / "crystal_bridge")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading SAE {args.sae_release}::{args.sae_id} ...", flush=True)
    sae, _cfg = load_sae(args.sae_release, args.sae_id)
    decoder = decoder_vectors(sae)
    n_features, d_model = decoder.shape
    print(f"      decoder: ({n_features}, {d_model})", flush=True)

    crystal_dir = args.crystal_root / f"gemma-fv-layer{args.layer}-norm{args.normalize}-diff"
    print(f"[2/5] Loading relation diffs from {crystal_dir} ...", flush=True)
    diffs = load_relation_diffs(crystal_dir)
    for name, arr in diffs.items():
        print(f"        {name:25s}: {arr.shape}", flush=True)

    print(f"[3/5] Building bridge (LDA n_components={args.n_lda}) ...", flush=True)
    bridge = build_bridge(decoder, diffs, n_lda_components=args.n_lda)

    np.save(args.out_dir / "alignment_raw.npy", bridge.alignment_raw)
    np.save(args.out_dir / "alignment_lda.npy", bridge.alignment_lda)
    np.save(args.out_dir / "lda_scalings.npy", bridge.lda_scalings)
    np.save(args.out_dir / "centroids_raw.npy", bridge.centroids)
    np.save(args.out_dir / "centroids_lda.npy", bridge.lda_centroids)

    print("[4/5] Coverage + distinctiveness stats ...", flush=True)
    cov_raw = coverage_stats(bridge.alignment_raw, DEFAULT_THRESHOLDS)
    cov_lda = coverage_stats(bridge.alignment_lda, DEFAULT_THRESHOLDS)
    distinct_raw = distinctiveness_stats(bridge.alignment_raw, DEFAULT_MARGIN_THRESHOLDS)
    distinct_lda = distinctiveness_stats(bridge.alignment_lda, DEFAULT_MARGIN_THRESHOLDS)
    coverage = {
        "raw": {str(k): v for k, v in cov_raw.items()},
        "lda": {str(k): v for k, v in cov_lda.items()},
        "distinct_raw": {str(k): v for k, v in distinct_raw.items()},
        "distinct_lda": {str(k): v for k, v in distinct_lda.items()},
        "n_features": n_features,
        "n_relations": len(bridge.relations),
        "relations": bridge.relations,
    }
    (args.out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print("      raw alignment ≥ T:", flush=True)
    for t, info in cov_raw.items():
        print(f"        T={t:.2f}  {info['n_features']:>5d} features  ({info['fraction']*100:5.2f}%)",
              flush=True)
    print("      LDA alignment ≥ T:", flush=True)
    for t, info in cov_lda.items():
        print(f"        T={t:.2f}  {info['n_features']:>5d} features  ({info['fraction']*100:5.2f}%)",
              flush=True)
    print("      LDA-space distinctiveness margin ≥ M (best - 2nd-best):", flush=True)
    for m, info in distinct_lda.items():
        print(f"        M={m:.2f}  {info['n_features']:>5d} features  ({info['fraction']*100:5.2f}%)",
              flush=True)

    print(f"[5/5] Top-{args.top_k} features per crystal (Neuronpedia labels) ...", flush=True)
    rows = load_bulk_explanations(args.neuronpedia_model, args.neuronpedia_source)
    label_by_idx: dict[int, str] = {}
    for r in rows:
        try:
            label_by_idx[int(r["index"])] = (r.get("description") or "").strip()
        except (KeyError, ValueError, TypeError):
            continue

    def _label(i: int) -> str:
        return label_by_idx.get(i, "<no label>")

    top_raw = top_features_per_crystal(bridge.alignment_raw, bridge.relations, k=args.top_k)
    top_lda = top_features_per_crystal(bridge.alignment_lda, bridge.relations, k=args.top_k)
    top_distinct = distinctive_top_features_per_crystal(
        bridge.alignment_lda, bridge.relations, k=args.top_k, min_margin=0.0
    )

    def _serialize(top: dict[str, list[tuple]]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for rel, pairs in top.items():
            rows = []
            for tup in pairs:
                if len(tup) == 2:
                    fid, score = tup
                    rows.append({"feature_id": fid, "score": score, "label": _label(fid)})
                else:
                    fid, score, margin = tup
                    rows.append({
                        "feature_id": fid, "score": score, "margin": margin, "label": _label(fid)
                    })
            out[rel] = rows
        return out

    (args.out_dir / "top_features_raw.json").write_text(
        json.dumps(_serialize(top_raw), indent=2)
    )
    (args.out_dir / "top_features_lda.json").write_text(
        json.dumps(_serialize(top_lda), indent=2)
    )
    (args.out_dir / "top_features_distinct.json").write_text(
        json.dumps(_serialize(top_distinct), indent=2)
    )

    # Print distinctive tops to stdout — that's the meaningful view for inspection.
    # (raw is dominated by distractor features, LDA scores saturate, distinctive
    # margin combines best of both: LDA's semantic ranking + a meaningful
    # threshold).
    print("\n--- Top distinctive features per crystal (LDA cosine, ranked by margin) ---",
          flush=True)
    for rel in bridge.relations:
        print(f"\n  {rel}:", flush=True)
        rows = top_distinct[rel][: min(args.top_k, 5)]
        if not rows:
            print("    (no features chose this crystal as their best match)", flush=True)
            continue
        for fid, score, margin in rows:
            print(
                f"    score={score:+.3f}  margin={margin:+.3f}  "
                f"feat#{fid:5d}  {_label(fid)[:75]!r}",
                flush=True,
            )

    # Per-feature best-crystal table
    best_idx, best_score = best_crystal_per_feature(bridge.alignment_lda, bridge.relations)
    table_path = args.out_dir / "best_crystal_per_feature.csv"
    with table_path.open("w") as f:
        f.write("feature_id,best_crystal,best_lda_score,label\n")
        for fid in range(n_features):
            f.write(
                f"{fid},{bridge.relations[best_idx[fid]]},{best_score[fid]:.6f},"
                f"{json.dumps(_label(fid))}\n"
            )
    print(f"\nwrote {table_path}", flush=True)
    print(f"all outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
