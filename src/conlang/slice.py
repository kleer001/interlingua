"""Vertical slice orchestrator (Milestone 1).

End-to-end run of Stages 1-2 + visualization on the first N filter-passing
Gemma Scope features.

Usage:
    python -m conlang.slice \\
        --sae-release gemma-scope-2b-pt-res-canonical \\
        --sae-id layer_12/width_16k/canonical \\
        --neuronpedia-model gemma-2-2b \\
        --neuronpedia-source 12-gemmascope-res-16k \\
        --top-n 200 \\
        --dedup-threshold 0.85 \\
        --edge-threshold 0.50

All intermediate artifacts are saved under fauna/interlingua/data/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from .dedupe import (
    cluster_by_threshold,
    cluster_hdbscan,
    cosine_matrix,
    pick_representatives,
    save_clusters,
    threshold_distribution_summary,
)
from .ingest import (
    decoder_vectors,
    first_n_passing_filter,
    load_bulk_explanations,
    load_sae,
    save_node_set,
)
from .viz import RENDERERS, build_graph_data, write_slice_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-release", required=True)
    p.add_argument("--sae-id", required=True)
    p.add_argument("--neuronpedia-model", required=True)
    p.add_argument("--neuronpedia-source", required=True)
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument(
        "--dedup-method",
        choices=["threshold", "hdbscan"],
        default="hdbscan",
        help="threshold = union-find on cosine ≥ --dedup-threshold (v0.1 default). "
             "hdbscan = density-based clustering per LessWrong prior art (recommended).",
    )
    p.add_argument("--dedup-threshold", type=float, default=0.30,
                   help="Cosine threshold for the 'threshold' dedup method. "
                        "Ignored when --dedup-method=hdbscan.")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=5)
    p.add_argument("--hdbscan-min-samples", type=int, default=None)
    p.add_argument("--edge-threshold", type=float, default=0.10,
                   help="Cosine similarity above which an edge is drawn in the viz.")
    p.add_argument("--pmi-threshold", type=float, default=5.0,
                   help="PMI above which a co-activation (orange) edge is drawn. "
                        "Requires data/interim/coactivation/pmi.npy on disk.")
    p.add_argument(
        "--pmi-path",
        type=Path,
        default=INTERIM_DIR / "coactivation" / "pmi.npy",
        help="Path to PMI matrix from run_coactivation. If missing, only cosine "
             "edges are drawn.",
    )
    p.add_argument(
        "--bridge-dir",
        type=Path,
        default=PROCESSED_DIR / "crystal_bridge",
        help="If this directory has alignment_lda.npy + coverage.json, "
             "the viz hover will include each node's best-matching crystal.",
    )
    p.add_argument(
        "--viz-backend",
        choices=sorted(RENDERERS),
        default="cytoscape",
        help="Renderer for the slice HTML. cytoscape scales past 1k nodes; "
             "pyvis is kept for parity but is slow at this size.",
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    print(f"[1/6] Loading SAE {args.sae_release}::{args.sae_id} ...", flush=True)
    sae, _cfg = load_sae(args.sae_release, args.sae_id)
    decoder = decoder_vectors(sae)
    print(f"      decoder: {decoder.shape}, dtype={decoder.dtype}", flush=True)

    print(f"[2/6] Loading bulk Neuronpedia explanations "
          f"({args.neuronpedia_model}/{args.neuronpedia_source}) ...", flush=True)
    rows = load_bulk_explanations(args.neuronpedia_model, args.neuronpedia_source)
    print(f"      {len(rows)} explanations on disk", flush=True)

    print(f"[3/6] Applying §6 filter and taking first {args.top_n} ...", flush=True)
    features = first_n_passing_filter(rows, decoder, n=args.top_n)
    print(f"      kept {len(features)} features", flush=True)
    meta_path, vecs_path = save_node_set(features, out_dir=RAW_DIR)
    print(f"      wrote {meta_path.name}, {vecs_path.name}", flush=True)

    print("[4/6] Cosine similarity + dedup ...", flush=True)
    vecs = np.stack([f.decoder_vec for f in features])
    sim = cosine_matrix(vecs)
    summary = threshold_distribution_summary(sim)
    print(f"      pair similarity: {summary}", flush=True)
    np.save(INTERIM_DIR / "sim_matrix.npy", sim)
    (INTERIM_DIR / "sim_summary.json").write_text(json.dumps(summary, indent=2))

    if args.dedup_method == "threshold":
        raw_clusters = cluster_by_threshold(sim, threshold=args.dedup_threshold)
        method_desc = f"threshold>={args.dedup_threshold}"
    else:
        raw_clusters, hdbscan_labels = cluster_hdbscan(
            sim,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
        )
        np.save(INTERIM_DIR / "hdbscan_labels.npy", hdbscan_labels)
        n_noise = int((hdbscan_labels == -1).sum())
        method_desc = (
            f"HDBSCAN min_cluster_size={args.hdbscan_min_cluster_size} "
            f"({n_noise} noise points)"
        )

    # No Neuronpedia confidence in the bulk dump; pick by lowest feature_id for
    # determinism. Refine when we have a better proxy for description quality.
    feature_ids = [f.feature_id for f in features]
    cluster_keys = [-fid for fid in feature_ids]
    clusters = pick_representatives(raw_clusters, cluster_keys)
    n_non_singleton = sum(1 for c in clusters if len(c.members) > 1)
    print(f"      {len(clusters)} clusters via {method_desc} "
          f"({n_non_singleton} non-singleton)", flush=True)
    save_clusters(clusters, out_dir=INTERIM_DIR)

    # Sample of strongest pairs above the viz threshold — useful for sanity-checking
    iu, ju = np.triu_indices(sim.shape[0], k=1)
    pair_sims = sim[iu, ju]
    mask = pair_sims >= args.edge_threshold
    top_pairs = sorted(
        zip(iu[mask].tolist(), ju[mask].tolist(), pair_sims[mask].tolist()),
        key=lambda t: -t[2],
    )[:10]
    print(f"      {int(mask.sum())} viz edges at threshold {args.edge_threshold}", flush=True)
    for a, b, s in top_pairs:
        print(f"        {s:.3f}  {features[a].label[:60]!r}  ↔  {features[b].label[:60]!r}",
              flush=True)

    print(f"[5/6] Rendering slice HTML with {args.viz_backend} ...", flush=True)
    features_meta = [
        {"feature_id": f.feature_id, "label": f.label} for f in features
    ]

    # Build hdbscan_labels indexed by slice position (0..len(features)-1) if available.
    hdbscan_labels_for_viz = None
    if args.dedup_method == "hdbscan":
        full_labels = np.load(INTERIM_DIR / "hdbscan_labels.npy")
        hdbscan_labels_for_viz = full_labels  # same length as `features` (= sim.shape[0])

    # Build crystal overlay if bridge data is on disk.
    crystal_overlay = None
    bridge_dir = args.bridge_dir
    if (bridge_dir / "alignment_lda.npy").exists() and (bridge_dir / "coverage.json").exists():
        alignment_lda = np.load(bridge_dir / "alignment_lda.npy")
        relations = json.loads((bridge_dir / "coverage.json").read_text())["relations"]
        # Map slice features (subset of 16384) → their alignment rows.
        rows = np.stack([alignment_lda[f.feature_id] for f in features])
        sorted_desc = np.sort(rows, axis=1)[:, ::-1]
        margin = sorted_desc[:, 0] - sorted_desc[:, 1]
        best_idx = rows.argmax(axis=1)
        best_score = rows.max(axis=1)
        crystal_overlay = {
            "relations": relations,
            "best_idx": best_idx,
            "best_score": best_score,
            "margin": margin,
        }
        print(f"      crystal overlay: {len(relations)} relations, "
              f"distinctiveness margin median={float(np.median(margin)):.4f}", flush=True)

    pmi_for_viz = None
    if args.pmi_path.is_file():
        pmi_for_viz = np.load(args.pmi_path)
        if pmi_for_viz.shape != sim.shape:
            print(
                f"      warning: PMI shape {pmi_for_viz.shape} != sim shape "
                f"{sim.shape}; skipping co-activation edges",
                flush=True,
            )
            pmi_for_viz = None
        else:
            print(f"      PMI overlay loaded from {args.pmi_path}", flush=True)

    graph_data = build_graph_data(
        features_meta=features_meta,
        sim=sim,
        clusters=clusters,
        edge_threshold=args.edge_threshold,
        hdbscan_labels=hdbscan_labels_for_viz,
        crystal_overlay=crystal_overlay,
        pmi=pmi_for_viz,
        pmi_threshold=args.pmi_threshold,
    )
    renderer = RENDERERS[args.viz_backend]
    html_path = renderer(graph_data, PROCESSED_DIR / "slice.html")
    print(f"      wrote {html_path}", flush=True)

    print("[6/6] Manifest ...", flush=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = write_slice_manifest(
        out_dir=PROCESSED_DIR,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        neuronpedia_model=args.neuronpedia_model,
        neuronpedia_source=args.neuronpedia_source,
        n_features_requested=args.top_n,
        n_after_filter=len(features),
        n_clusters=len(clusters),
        cosine_dedup_threshold=args.dedup_threshold,
        edge_viz_threshold=args.edge_threshold,
    )
    print(f"      wrote {manifest_path}", flush=True)
    print(f"\nDone. Open {html_path} in a browser.")
    return html_path


def main() -> None:
    args = parse_args(sys.argv[1:])
    run(args)


if __name__ == "__main__":
    main()
