"""Vertical-slice visualization: interactive HTML of nodes + cosine edges."""

from __future__ import annotations

import colorsys
import json
from pathlib import Path

import numpy as np

from . import PROCESSED_DIR
from .dedupe import Cluster


def _palette(n: int) -> list[str]:
    """Generate n visually distinct hex colors using evenly-spaced HSL hues."""
    if n <= 0:
        return []
    return [
        "#{:02x}{:02x}{:02x}".format(
            *(int(255 * c) for c in colorsys.hls_to_rgb((i / n) % 1.0, 0.55, 0.65))
        )
        for i in range(n)
    ]


def build_pyvis_graph(
    features_meta: list[dict],
    sim: np.ndarray,
    clusters: list[Cluster],
    edge_threshold: float,
    out_path: Path | None = None,
    hdbscan_labels: np.ndarray | None = None,
    crystal_overlay: dict | None = None,
):
    """Render an interactive HTML graph of deduped features.

    Nodes: cluster representatives.
    Edges: cosine similarities >= edge_threshold between representatives.
    Hover: label, cluster size, optional HDBSCAN cluster, optional best crystal.

    Optional overlays:
    - hdbscan_labels: array of length len(features_meta). Non-negative ids get a
      cluster color; -1 is rendered gray (noise).
    - crystal_overlay: {"relations": [...], "best_idx": np.ndarray, "best_score":
      np.ndarray, "margin": np.ndarray} — all length len(features_meta).
      Adds best-crystal info to the hover.
    """
    from pyvis.network import Network

    rep_indices = [c.representative for c in clusters]
    rep_to_node_id = {r: i for i, r in enumerate(rep_indices)}

    # Color map: one color per unique non-negative HDBSCAN label, gray for -1.
    color_for_label: dict[int, str] = {}
    if hdbscan_labels is not None:
        unique_real = sorted({int(x) for x in hdbscan_labels if x >= 0})
        palette = _palette(len(unique_real))
        color_for_label = dict(zip(unique_real, palette))
    NOISE_COLOR = "#555"
    DEFAULT_COLOR = "#88c"

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#111",
        font_color="white",
        notebook=False,
    )
    net.force_atlas_2based(spring_length=120)

    for rep, cluster in zip(rep_indices, clusters):
        meta = features_meta[rep]
        title_lines = [
            f"feature_id={meta['feature_id']}",
            f"cluster_size={len(cluster.members)}",
        ]
        color = DEFAULT_COLOR
        if hdbscan_labels is not None:
            lab = int(hdbscan_labels[rep])
            if lab >= 0:
                color = color_for_label.get(lab, DEFAULT_COLOR)
                title_lines.append(f"hdbscan_cluster={lab}")
            else:
                color = NOISE_COLOR
                title_lines.append("hdbscan_cluster=noise")
        if crystal_overlay is not None:
            relations = crystal_overlay["relations"]
            bi = int(crystal_overlay["best_idx"][rep])
            bs = float(crystal_overlay["best_score"][rep])
            mg = float(crystal_overlay["margin"][rep])
            title_lines.append(
                f"best_crystal={relations[bi]}  score={bs:+.3f}  margin={mg:+.3f}"
            )
        title_lines.append(f"label: {meta['label']}")
        net.add_node(
            rep_to_node_id[rep],
            label=meta["label"][:40],
            title="\n".join(title_lines),
            value=len(cluster.members),
            color=color,
        )

    for i, a in enumerate(rep_indices):
        for b in rep_indices[i + 1:]:
            s = float(sim[a, b])
            if s >= edge_threshold:
                net.add_edge(rep_to_node_id[a], rep_to_node_id[b], value=s, title=f"cos={s:.3f}")

    if out_path is None:
        out_path = PROCESSED_DIR / "slice.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path), notebook=False)
    return out_path


def write_slice_manifest(
    out_dir: Path,
    sae_release: str,
    sae_id: str,
    neuronpedia_model: str,
    neuronpedia_source: str,
    n_features_requested: int,
    n_after_filter: int,
    n_clusters: int,
    cosine_dedup_threshold: float,
    edge_viz_threshold: float,
) -> Path:
    manifest = {
        "sae_release": sae_release,
        "sae_id": sae_id,
        "neuronpedia_model": neuronpedia_model,
        "neuronpedia_source": neuronpedia_source,
        "n_features_requested": n_features_requested,
        "n_after_filter": n_after_filter,
        "n_clusters": n_clusters,
        "cosine_dedup_threshold": cosine_dedup_threshold,
        "edge_viz_threshold": edge_viz_threshold,
    }
    path = out_dir / "slice_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
