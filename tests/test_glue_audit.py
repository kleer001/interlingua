"""GLUE Path 1 audit tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conlang.glue.audit import (
    CATEGORIES,
    _compile_categories,
    categorize_label,
    confidence_tier,
    neighbor_promiscuity,
    run_audit,
)


def test_compile_returns_all_categories():
    out = _compile_categories()
    assert len(out) == len(CATEGORIES)
    names = [n for n, _ in out]
    assert "negation / polarity" in names
    assert "discourse marker" in names


@pytest.mark.parametrize(
    "label,expected",
    [
        ("phrases expressing negation of a claim", "negation / polarity"),
        ("words denoting absence of light", "negation / polarity"),
        ("plurality markers in noun phrases", "number (plural)"),
        ("references to past events in narrative", "tense / past"),
        ("ongoing actions in progress", "aspect / progressive"),
        ("first-person speaker self-reference", "person / 1st"),
        ("definite specific entity reference", "definiteness"),
        ("contrast and adversative connectors", "conjunction / contrast"),
        ("causal because-clauses linking events", "conjunction / causal"),
        ("comparative more than constructions", "comparative"),
        ("epistemic possibility might-clauses", "modality / possibility"),
    ],
)
def test_categorize_label_matches_baseline(label, expected):
    compiled = _compile_categories()
    assert categorize_label(label, compiled) == expected


def test_categorize_returns_none_for_concrete_features():
    compiled = _compile_categories()
    assert categorize_label("recipes for cooking pasta", compiled) is None
    assert categorize_label("an apple tree in autumn", compiled) is None


def test_categorize_first_match_wins():
    """Label containing two patterns should bucket to the first table entry."""
    compiled = _compile_categories()
    # 'negation' matches first, 'plurality' would have matched 'number (plural)'
    label = "negation markers in plural contexts"
    assert categorize_label(label, compiled) == "negation / polarity"


def test_neighbor_promiscuity_high_entropy_glue_signature():
    # Self at 0, four neighbors split across 4 distinct clusters
    pmi = np.array(
        [
            [-np.inf, 5.0, 4.0, 3.0, 2.0],
            [5.0, -np.inf, 0, 0, 0],
            [4.0, 0, -np.inf, 0, 0],
            [3.0, 0, 0, -np.inf, 0],
            [2.0, 0, 0, 0, -np.inf],
        ]
    )
    labels = np.array([7, 1, 2, 3, 4])  # self in cluster 7, neighbors in 4 distinct
    out = neighbor_promiscuity(
        pmi_row=pmi[0], hdbscan_labels=labels, self_idx=0, top_k=4, min_pmi=0.0
    )
    assert out["n_neighbors"] == 4
    assert out["n_distinct_clusters"] == 4
    assert out["entropy"] == pytest.approx(np.log(4))
    assert out["same_cluster_neighbors"] == 0


def test_neighbor_promiscuity_concentrated_content_feature():
    # 5 neighbors all in same cluster — low entropy, glue-suspect signature absent
    labels = np.array([3, 3, 3, 3, 3, 3])
    pmi_full = np.zeros((6, 6))
    pmi_full[0, 0] = -np.inf
    pmi_full[0, 1:] = 5.0
    out = neighbor_promiscuity(
        pmi_row=pmi_full[0], hdbscan_labels=labels, self_idx=0, top_k=5, min_pmi=0.0
    )
    assert out["n_neighbors"] == 5
    assert out["n_distinct_clusters"] == 1
    assert out["entropy"] == pytest.approx(0.0)
    assert out["same_cluster_neighbors"] == 5


def test_neighbor_promiscuity_stops_at_nonpositive_pmi():
    # Top 5 are positive, rest are zero — should pick exactly 2
    n = 10
    pmi_row = np.array([-np.inf, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    labels = np.zeros(n, dtype=int)
    out = neighbor_promiscuity(
        pmi_row=pmi_row, hdbscan_labels=labels, self_idx=0, top_k=5, min_pmi=0.0
    )
    assert out["n_neighbors"] == 2


def test_confidence_tier_high_for_diverse_high_entropy():
    tier = confidence_tier(
        entropy=2.0,
        n_distinct_clusters=5,
        n_neighbors=10,
        same_cluster_neighbors=0,
        entropy_high=1.5,
        entropy_medium=0.7,
    )
    assert tier == "high"


def test_confidence_tier_medium_for_moderate():
    tier = confidence_tier(
        entropy=1.0,
        n_distinct_clusters=3,
        n_neighbors=5,
        same_cluster_neighbors=1,
        entropy_high=1.5,
        entropy_medium=0.7,
    )
    assert tier == "medium"


def test_confidence_tier_low_for_dominant_cluster():
    # High entropy by chance but most neighbors in same cluster — likely content
    tier = confidence_tier(
        entropy=1.6,
        n_distinct_clusters=3,
        n_neighbors=10,
        same_cluster_neighbors=8,
        entropy_high=1.5,
        entropy_medium=0.7,
    )
    assert tier == "medium"  # demoted from high by same_share check


def test_run_audit_end_to_end_synthetic(tmp_path):
    features = [
        {"feature_id": 1, "label": "phrases of negation and denial"},
        {"feature_id": 2, "label": "ongoing in progress actions"},
        {"feature_id": 3, "label": "an apple in autumn"},
        {"feature_id": 4, "label": "another apple in autumn"},
    ]
    fpath = tmp_path / "features.jsonl"
    with fpath.open("w") as f:
        for r in features:
            f.write(json.dumps(r) + "\n")

    pmi = np.array(
        [
            [-1e9, 2.0, 2.0, 2.0],  # neg fires with everyone (diverse)
            [2.0, -1e9, 2.0, 2.0],
            [2.0, 2.0, -1e9, 5.0],  # apples cluster tightly together
            [2.0, 2.0, 5.0, -1e9],
        ]
    )
    pmi_path = tmp_path / "pmi.npy"
    np.save(pmi_path, pmi)

    labels = np.array([0, 1, 2, 2])
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)

    out = run_audit(
        features_path=fpath,
        pmi_path=pmi_path,
        labels_path=labels_path,
        top_k=3,
        min_pmi=0.0,
        entropy_high=0.5,
        entropy_medium=0.2,
    )
    assert out["n_entries"] == 2  # apples didn't categorize
    cats = {e["grammatical_category"] for e in out["entries"]}
    assert cats == {"negation / polarity", "aspect / progressive"}


def test_run_audit_rejects_shape_mismatch(tmp_path):
    fpath = tmp_path / "f.jsonl"
    fpath.write_text(json.dumps({"feature_id": 1, "label": "negation marker"}) + "\n")
    pmi_path = tmp_path / "pmi.npy"
    np.save(pmi_path, np.zeros((3, 3)))
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, np.zeros(1))
    with pytest.raises(ValueError, match="pmi shape"):
        run_audit(
            features_path=fpath,
            pmi_path=pmi_path,
            labels_path=labels_path,
            top_k=3,
            min_pmi=0.0,
            entropy_high=1.5,
            entropy_medium=0.7,
        )
