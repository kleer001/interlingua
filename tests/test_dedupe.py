"""Tests for the cosine dedup helpers. No torch/HF needed."""

import numpy as np

from conlang.dedupe import (
    cluster_by_threshold,
    cluster_hdbscan,
    cosine_matrix,
    pick_representatives,
    threshold_distribution_summary,
)


def test_cosine_matrix_diagonal_is_one():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sim = cosine_matrix(vecs)
    assert np.allclose(np.diag(sim), 1.0)


def test_cosine_matrix_orthogonal_pair_is_zero():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = cosine_matrix(vecs)
    assert abs(sim[0, 1]) < 1e-9


def test_cluster_groups_high_similarity_pairs():
    # Two clusters: {0, 1} near-duplicates; {2} alone; {3, 4} also similar.
    vecs = np.array(
        [
            [1.0, 0.01],
            [1.0, 0.02],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-1.0, 0.01],
        ]
    )
    sim = cosine_matrix(vecs)
    clusters = cluster_by_threshold(sim, threshold=0.99)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2, 2]


def test_pick_representatives_takes_highest_score():
    clusters = [[0, 1, 2], [3]]
    scores = [0.1, 0.9, 0.5, 0.7]
    reps = pick_representatives(clusters, scores)
    assert reps[0].representative == 1
    assert reps[1].representative == 3


def test_threshold_distribution_summary_keys():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((10, 4))
    sim = cosine_matrix(vecs)
    summary = threshold_distribution_summary(sim)
    assert {"min", "p50", "p90", "p99", "max", "n_pairs"} <= set(summary)
    assert summary["n_pairs"] == 10 * 9 // 2


def test_cluster_hdbscan_finds_two_dense_clusters():
    """Two well-separated dense clusters in 8-dim space — HDBSCAN's natural use case.

    Identical points break HDBSCAN (mutual reachability distance = 0 → density
    undefined). And tiny clusters trigger over-splitting from micro-noise. So the
    test uses two 30-point Gaussian clusters in 8 dims, well-separated.
    """
    rng = np.random.default_rng(42)
    centers = np.array([
        [+1, +1, +1, +1, 0, 0, 0, 0],
        [-1, -1, -1, -1, 0, 0, 0, 0],
    ], dtype=np.float32)
    cluster_a = centers[0] + rng.normal(scale=0.05, size=(30, 8))
    cluster_b = centers[1] + rng.normal(scale=0.05, size=(30, 8))
    vecs = np.vstack([cluster_a, cluster_b])
    sim = cosine_matrix(vecs)
    clusters, labels = cluster_hdbscan(sim, min_cluster_size=10, min_samples=5)

    # Contract: every input appears in exactly one returned cluster.
    seen: set[int] = set()
    for c in clusters:
        seen.update(c)
    assert seen == set(range(60))

    # Each true cluster's points should mostly share one HDBSCAN label.
    a_labels = [int(labels[i]) for i in range(30)]
    b_labels = [int(labels[i]) for i in range(30, 60)]
    # The dominant label per group must be non-negative (i.e. a real cluster).
    a_dom = max(set(a_labels), key=a_labels.count)
    b_dom = max(set(b_labels), key=b_labels.count)
    assert a_dom >= 0
    assert b_dom >= 0
    # And the two groups must end up with DIFFERENT dominant labels.
    assert a_dom != b_dom
    # Most members of each group share the dominant label (allow a few noise).
    assert a_labels.count(a_dom) >= 25
    assert b_labels.count(b_dom) >= 25


def test_cluster_hdbscan_returns_label_per_input_point():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((20, 4))
    sim = cosine_matrix(vecs)
    clusters, labels = cluster_hdbscan(sim, min_cluster_size=3)
    assert labels.shape == (20,)
    # every input index appears in exactly one returned cluster
    seen = set()
    for c in clusters:
        seen.update(c)
    assert seen == set(range(20))
