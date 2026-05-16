"""Stage 2: cosine deduplication.

Computes pairwise cosine similarity on decoder vectors, clusters near-duplicates,
picks a representative per cluster (highest Neuronpedia score), and emits a
deduped node set ready for Stage 3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import INTERIM_DIR


@dataclass
class Cluster:
    members: list[int]  # indices into the input feature list
    representative: int


def cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    """Return the (n, n) cosine similarity matrix for a (n, d) feature matrix."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / np.clip(norms, 1e-12, None)
    return normed @ normed.T


def cluster_by_threshold(sim: np.ndarray, threshold: float) -> list[list[int]]:
    """Union-find over edges (i, j) where sim[i, j] >= threshold and i != j.

    Returns clusters as lists of indices (singletons included).
    """
    n = sim.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # iterate strictly upper triangle
    iu, ju = np.triu_indices(n, k=1)
    mask = sim[iu, ju] >= threshold
    for i, j in zip(iu[mask].tolist(), ju[mask].tolist()):
        union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def cluster_hdbscan(
    sim: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
) -> tuple[list[list[int]], np.ndarray]:
    """HDBSCAN over cosine *distance* (1 - sim), clamped to [0, 2].

    Per the LessWrong post (Lim et al., Oct 2024), HDBSCAN on Gemma Scope decoder
    cosine distances tends to label >90% of features as noise but produces highly
    coherent clusters out of the rest. Each noise feature becomes its own
    singleton cluster in our output, so downstream code (representative picking,
    viz) doesn't have to special-case noise.

    Returns (clusters, labels) where labels[i] is the HDBSCAN label for feature
    i (-1 = noise, otherwise the cluster id).
    """
    from sklearn.cluster import HDBSCAN

    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    if min_samples is None:
        min_samples = min_cluster_size
    model = HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = model.fit_predict(dist.astype(np.float64))

    # Group by label, but split each noise (-1) into its own singleton.
    groups_by_label: dict[int, list[int]] = {}
    next_singleton_id = int(labels.max() if labels.size else -1) + 1
    for i, lab in enumerate(labels):
        lab = int(lab)
        if lab == -1:
            groups_by_label[next_singleton_id] = [i]
            next_singleton_id += 1
        else:
            groups_by_label.setdefault(lab, []).append(i)
    return list(groups_by_label.values()), labels


def pick_representatives(
    clusters: list[list[int]],
    scores: list[float],
) -> list[Cluster]:
    """Per cluster, choose the member with the highest auto-interp score."""
    out: list[Cluster] = []
    for members in clusters:
        rep = max(members, key=lambda i: scores[i])
        out.append(Cluster(members=members, representative=rep))
    return out


def save_clusters(clusters: list[Cluster], out_dir: Path = INTERIM_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "clusters.jsonl"
    with path.open("w") as f:
        for c in clusters:
            f.write(
                json.dumps({"representative": c.representative, "members": c.members}) + "\n"
            )
    return path


def threshold_distribution_summary(sim: np.ndarray) -> dict[str, float]:
    """Summarize off-diagonal cosine similarity for picking a threshold."""
    n = sim.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    off = sim[iu, ju]
    return {
        "min": float(off.min()),
        "p50": float(np.median(off)),
        "p90": float(np.quantile(off, 0.90)),
        "p99": float(np.quantile(off, 0.99)),
        "max": float(off.max()),
        "n_pairs": int(off.size),
    }
