"""Tests for Stage 5 regularization. Pure numpy, no torch/HF."""

import numpy as np

from conlang.regularize import (
    near_neighbors,
    pick_parent,
    regularize,
    siblings_in_cluster,
)


def test_pick_parent_returns_highest_pmi_neighbor():
    pmi_row = np.array([0.0, 0.5, 2.0, 1.0])
    result = pick_parent(pmi_row, self_idx=0, min_pmi=0.0)
    assert result == (2, 2.0)


def test_pick_parent_excludes_self():
    # self_idx has the max value; the function must skip it.
    pmi_row = np.array([10.0, 1.0, 0.5])
    result = pick_parent(pmi_row, self_idx=0, min_pmi=0.0)
    assert result == (1, 1.0)


def test_pick_parent_returns_none_when_all_below_threshold():
    pmi_row = np.array([0.0, 0.0, 0.0, 0.0])
    assert pick_parent(pmi_row, self_idx=0, min_pmi=0.0) is None


def test_pick_parent_returns_none_when_exactly_at_threshold():
    # min_pmi is exclusive — independent pairs (PMI=0) are not parents.
    pmi_row = np.array([0.0, 0.0, 0.0])
    assert pick_parent(pmi_row, self_idx=0, min_pmi=0.0) is None


def test_siblings_returns_other_cluster_members():
    labels = np.array([0, 0, 1, 0, -1])
    assert sorted(siblings_in_cluster(labels, self_idx=0)) == [1, 3]


def test_siblings_returns_empty_for_noise_point():
    labels = np.array([-1, -1, 0, 0])
    assert siblings_in_cluster(labels, self_idx=0) == []


def test_siblings_singleton_cluster_returns_empty():
    labels = np.array([0, 1, 2])
    assert siblings_in_cluster(labels, self_idx=0) == []


def test_near_neighbors_respects_top_k():
    sim_row = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    near = near_neighbors(sim_row, self_idx=0, exclude=set(), top_k=2, min_cosine=0.0)
    assert [j for j, _ in near] == [1, 2]


def test_near_neighbors_excludes_parent_and_siblings():
    sim_row = np.array([1.0, 0.9, 0.8, 0.7])
    near = near_neighbors(
        sim_row, self_idx=0, exclude={1, 2}, top_k=5, min_cosine=0.0
    )
    assert [j for j, _ in near] == [3]


def test_near_neighbors_filters_below_min_cosine():
    sim_row = np.array([1.0, 0.5, 0.4, 0.2])
    near = near_neighbors(
        sim_row, self_idx=0, exclude=set(), top_k=5, min_cosine=0.45
    )
    assert [j for j, _ in near] == [1]


def test_regularize_end_to_end_shape():
    features = [
        {"feature_id": 10, "label": "a"},
        {"feature_id": 20, "label": "b"},
        {"feature_id": 30, "label": "c"},
        {"feature_id": 40, "label": "d"},
    ]
    # Node 0 most similar to node 1; nodes 0+1 in same HDBSCAN cluster; node 2 noise.
    sim = np.array(
        [
            [1.0, 0.9, 0.4, 0.1],
            [0.9, 1.0, 0.3, 0.2],
            [0.4, 0.3, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0],
        ]
    )
    labels = np.array([0, 0, -1, 1])
    # Strongest PMI for node 0 is with node 2; node 0's parent should be 2,
    # not its high-cosine HDBSCAN sibling 1.
    pmi = np.array(
        [
            [0.0, 0.5, 2.0, 0.0],
            [0.5, 0.0, 0.1, 0.4],
            [2.0, 0.1, 0.0, 0.3],
            [0.0, 0.4, 0.3, 0.0],
        ]
    )
    nodes = regularize(
        features, sim, labels, pmi,
        min_pmi=0.0, near_top_k=5, min_cosine=0.15,
    )
    assert len(nodes) == 4
    n0 = nodes[0]
    assert n0["feature_id"] == 10
    assert n0["parent"] == {"slice_idx": 2, "pmi": 2.0}
    assert n0["siblings"] == [1]
    # Near must exclude self (0), parent (2), siblings (1) → only 3 left,
    # but cos(0,3)=0.1 < min_cosine 0.15 → empty.
    assert n0["near"] == []


def test_regularize_noise_point_has_no_siblings():
    features = [{"feature_id": i, "label": str(i)} for i in range(3)]
    sim = np.eye(3) + 0.5 * (np.ones((3, 3)) - np.eye(3))
    labels = np.array([-1, -1, -1])
    pmi = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.2], [0.5, 0.2, 0.0]])
    nodes = regularize(
        features, sim, labels, pmi,
        min_pmi=0.0, near_top_k=5, min_cosine=0.0,
    )
    for nd in nodes:
        assert nd["siblings"] == []


def test_regularize_no_transformation_field():
    """Commitment 7: schema must not expose a transformation primitive."""
    features = [{"feature_id": 1, "label": "x"}, {"feature_id": 2, "label": "y"}]
    sim = np.array([[1.0, 0.5], [0.5, 1.0]])
    labels = np.array([-1, -1])
    pmi = np.array([[0.0, 1.0], [1.0, 0.0]])
    nodes = regularize(features, sim, labels, pmi)
    expected_keys = {"slice_idx", "feature_id", "label", "parent", "siblings", "near"}
    for nd in nodes:
        assert set(nd.keys()) == expected_keys
        assert "transformation" not in nd
        assert "opposite" not in nd
