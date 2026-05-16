"""Tests for the SAE↔crystal bridge math. No torch/HF/network."""

import numpy as np
import pytest

from conlang.edges.crystals import (
    best_crystal_per_feature,
    build_bridge,
    coverage_stats,
    distinctive_top_features_per_crystal,
    distinctiveness_stats,
    fit_lda_basis,
    relation_centroids,
    top_features_per_crystal,
)


def _toy_diffs() -> dict[str, np.ndarray]:
    """Two relations in 4-dim space, centered around different means.

    Relation A diffs cluster around [+3, 0, 0, 0]; relation B around [0, +3, 0, 0].
    Many samples per class so LDA has something to fit on.
    """
    rng = np.random.default_rng(0)
    a = np.array([3.0, 0, 0, 0]) + rng.normal(scale=0.1, size=(40, 4))
    b = np.array([0, 3.0, 0, 0]) + rng.normal(scale=0.1, size=(40, 4))
    return {"a-rel": a.astype(np.float32), "b-rel": b.astype(np.float32)}


def test_relation_centroids_recover_means():
    diffs = _toy_diffs()
    names, c = relation_centroids(diffs)
    assert names == ["a-rel", "b-rel"]
    assert c.shape == (2, 4)
    assert c[0, 0] > 2.5 and abs(c[0, 1]) < 0.5
    assert c[1, 1] > 2.5 and abs(c[1, 0]) < 0.5


def test_fit_lda_basis_returns_correct_shapes():
    diffs = _toy_diffs()
    scalings, mean = fit_lda_basis(diffs, n_components=1)
    assert scalings.shape == (4, 1)  # 2 classes → max 1 LDA dim
    assert mean.shape == (4,)


def test_build_bridge_aligns_decoders_with_their_centroid():
    diffs = _toy_diffs()
    # Decoder vectors deliberately parallel to each centroid + a noise vector.
    decoder = np.array(
        [
            [10.0, 0.0, 0.0, 0.0],   # parallel to a-rel centroid
            [0.0, 10.0, 0.0, 0.0],   # parallel to b-rel centroid
            [0.0, 0.0, 1.0, 1.0],    # orthogonal to both
        ],
        dtype=np.float32,
    )
    bridge = build_bridge(decoder, diffs, n_lda_components=1)
    assert bridge.alignment_raw.shape == (3, 2)
    # Feature 0 should align with relation 0 (a-rel) much more than relation 1.
    assert bridge.alignment_raw[0, 0] > 0.9
    assert bridge.alignment_raw[0, 1] < 0.1
    assert bridge.alignment_raw[1, 1] > 0.9
    assert bridge.alignment_raw[1, 0] < 0.1
    # Feature 2 should be ~zero against both.
    assert abs(bridge.alignment_raw[2, 0]) < 0.1
    assert abs(bridge.alignment_raw[2, 1]) < 0.1


def test_coverage_stats_counts_features_above_threshold():
    alignment = np.array(
        [
            [0.9, 0.1],
            [0.5, 0.6],
            [0.05, 0.04],
            [0.7, 0.2],
        ],
        dtype=np.float32,
    )
    stats = coverage_stats(alignment, [0.1, 0.5, 0.8])
    assert stats[0.1]["n_features"] == 3   # rows 0, 1, 3 have a max ≥ 0.1
    assert stats[0.5]["n_features"] == 3
    assert stats[0.8]["n_features"] == 1
    assert stats[0.5]["fraction"] == pytest.approx(0.75)


def test_top_features_per_crystal_returns_descending():
    alignment = np.array(
        [
            [0.1, 0.5],
            [0.9, 0.2],
            [0.3, 0.8],
            [0.2, 0.1],
        ],
        dtype=np.float32,
    )
    top = top_features_per_crystal(alignment, ["x", "y"], k=2)
    assert top["x"] == [(1, pytest.approx(0.9)), (2, pytest.approx(0.3))]
    assert top["y"] == [(2, pytest.approx(0.8)), (0, pytest.approx(0.5))]


def test_best_crystal_per_feature_returns_argmax_and_max():
    alignment = np.array(
        [
            [0.1, 0.5, 0.3],
            [0.9, 0.2, 0.4],
            [0.05, 0.04, 0.06],
        ],
        dtype=np.float32,
    )
    idx, score = best_crystal_per_feature(alignment, ["a", "b", "c"])
    np.testing.assert_array_equal(idx, [1, 0, 2])
    np.testing.assert_allclose(score, [0.5, 0.9, 0.06], atol=1e-6)


def test_distinctiveness_stats_counts_features_with_high_margin():
    # Feature 0: 0.9 vs 0.1 → margin 0.8 (very distinctive)
    # Feature 1: 0.6 vs 0.5 → margin 0.1 (distractor-ish)
    # Feature 2: 0.05 vs 0.04 → margin 0.01 (random)
    alignment = np.array(
        [
            [0.9, 0.1, 0.05],
            [0.6, 0.5, 0.4],
            [0.05, 0.04, 0.03],
        ],
        dtype=np.float32,
    )
    stats = distinctiveness_stats(alignment, [0.05, 0.2, 0.5])
    assert stats[0.05]["n_features"] == 2
    assert stats[0.2]["n_features"] == 1
    assert stats[0.5]["n_features"] == 1


def test_distinctive_top_features_filters_by_best_match():
    # Feature 0 best = relation 0, margin = 0.4 (0.5 - 0.1)
    # Feature 1 best = relation 1, margin = 0.3 (0.6 - 0.3)
    # Feature 2 best = relation 0, margin = 0.05 (0.95 - 0.9, distractor-y)
    alignment = np.array(
        [
            [0.5, 0.1, 0.0],
            [0.3, 0.6, 0.2],
            [0.95, 0.9, 0.85],
        ],
        dtype=np.float32,
    )
    top = distinctive_top_features_per_crystal(alignment, ["a", "b", "c"], k=10, min_margin=0.0)
    # Relation 'a' should have feat 0 (margin 0.4) ranked above feat 2 (margin 0.05).
    assert [t[0] for t in top["a"]] == [0, 2]
    assert top["a"][0][2] == pytest.approx(0.4, abs=1e-5)
    assert top["a"][1][2] == pytest.approx(0.05, abs=1e-5)
    # Relation 'b' has feat 1 only (best match = b).
    assert [t[0] for t in top["b"]] == [1]
    # Relation 'c' has no features whose best is c.
    assert top["c"] == []


def test_distinctive_top_features_min_margin_excludes_low():
    alignment = np.array(
        [
            [0.5, 0.1],
            [0.6, 0.55],   # best = a but margin only 0.05
        ],
        dtype=np.float32,
    )
    top = distinctive_top_features_per_crystal(alignment, ["a", "b"], k=10, min_margin=0.2)
    # feat 0 has margin 0.4, feat 1 has margin 0.05 → only feat 0 survives
    assert [t[0] for t in top["a"]] == [0]
