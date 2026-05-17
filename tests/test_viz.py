"""Tests for the renderer-agnostic viz layer.

Covers `build_graph_data` (data prep) plus a thin smoke test of
`render_cytoscape` (HTML output structure). The pyvis renderer is not
unit-tested here — it delegates to the upstream `pyvis.network.Network`,
which we trust.
"""

from pathlib import Path

import numpy as np

from conlang.dedupe import Cluster
from conlang.viz import (
    COSINE_EDGE_COLOR,
    DEFAULT_COLOR,
    NOISE_COLOR,
    PMI_EDGE_COLOR,
    RENDERERS,
    build_graph_data,
    render_cytoscape,
)


def _clusters_for(rep_indices: list[int]) -> list[Cluster]:
    """Each rep is its own cluster of size 1 (matches HDBSCAN noise behavior)."""
    return [Cluster(representative=r, members=(r,)) for r in rep_indices]


def _features(n: int) -> list[dict]:
    return [{"feature_id": 100 + i, "label": f"feat {i}"} for i in range(n)]


def test_build_graph_data_one_node_per_representative():
    features = _features(4)
    sim = np.eye(4, dtype=np.float32)
    clusters = _clusters_for([0, 1, 2, 3])
    data = build_graph_data(features, sim, clusters, edge_threshold=0.5)
    assert len(data.nodes) == 4
    assert {n.id for n in data.nodes} == {0, 1, 2, 3}


def test_build_graph_data_cosine_edges_use_threshold():
    features = _features(3)
    sim = np.array([
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.6],
        [0.2, 0.6, 1.0],
    ], dtype=np.float32)
    clusters = _clusters_for([0, 1, 2])
    data = build_graph_data(features, sim, clusters, edge_threshold=0.5)
    cosine = [e for e in data.edges if e.kind == "cosine"]
    # Edges above 0.5: (0,1)=0.8 and (1,2)=0.6 — both above; (0,2)=0.2 — below.
    assert sorted((e.src, e.dst) for e in cosine) == [(0, 1), (1, 2)]
    assert all(e.color == COSINE_EDGE_COLOR for e in cosine)


def test_build_graph_data_pmi_edges_separate_kind():
    features = _features(3)
    sim = np.zeros((3, 3), dtype=np.float32)
    pmi = np.array([
        [0.0, 6.0, 1.0],
        [6.0, 0.0, 2.0],
        [1.0, 2.0, 0.0],
    ], dtype=np.float32)
    clusters = _clusters_for([0, 1, 2])
    data = build_graph_data(
        features, sim, clusters, edge_threshold=0.5, pmi=pmi, pmi_threshold=5.0
    )
    assert data.edge_count("cosine") == 0
    pmi_edges = [e for e in data.edges if e.kind == "pmi"]
    assert len(pmi_edges) == 1
    assert (pmi_edges[0].src, pmi_edges[0].dst) == (0, 1)
    assert pmi_edges[0].color == PMI_EDGE_COLOR


def test_build_graph_data_node_colors_from_hdbscan():
    features = _features(3)
    sim = np.eye(3, dtype=np.float32)
    clusters = _clusters_for([0, 1, 2])
    labels = np.array([0, 0, -1])
    data = build_graph_data(features, sim, clusters, edge_threshold=2.0,
                            hdbscan_labels=labels)
    nodes_by_id = {n.id: n for n in data.nodes}
    # Cluster 0 members get the same color; noise gets NOISE_COLOR.
    assert nodes_by_id[0].color == nodes_by_id[1].color
    assert nodes_by_id[0].color != NOISE_COLOR
    assert nodes_by_id[2].color == NOISE_COLOR


def test_build_graph_data_default_color_without_hdbscan():
    features = _features(2)
    sim = np.eye(2, dtype=np.float32)
    clusters = _clusters_for([0, 1])
    data = build_graph_data(features, sim, clusters, edge_threshold=2.0)
    for n in data.nodes:
        assert n.color == DEFAULT_COLOR


def test_build_graph_data_node_title_includes_label_and_feature_id():
    features = [{"feature_id": 42, "label": "the meaning of life"}]
    sim = np.eye(1, dtype=np.float32)
    clusters = _clusters_for([0])
    data = build_graph_data(features, sim, clusters, edge_threshold=2.0)
    assert "feature_id=42" in data.nodes[0].title
    assert "the meaning of life" in data.nodes[0].title


def test_build_graph_data_edge_count_helper():
    features = _features(3)
    sim = np.array([[1, .9, 0], [.9, 1, .8], [0, .8, 1]], dtype=np.float32)
    pmi = np.array([[0, 6, 0], [6, 0, 0], [0, 0, 0]], dtype=np.float32)
    clusters = _clusters_for([0, 1, 2])
    data = build_graph_data(features, sim, clusters, edge_threshold=0.5,
                            pmi=pmi, pmi_threshold=5.0)
    assert data.edge_count("cosine") == 2  # (0,1) and (1,2)
    assert data.edge_count("pmi") == 1


def test_renderers_registry_has_both_backends():
    assert set(RENDERERS) == {"pyvis", "cytoscape"}


def test_render_cytoscape_writes_self_contained_html(tmp_path: Path):
    features = _features(3)
    sim = np.array([[1, .9, 0], [.9, 1, .8], [0, .8, 1]], dtype=np.float32)
    clusters = _clusters_for([0, 1, 2])
    data = build_graph_data(features, sim, clusters, edge_threshold=0.5)
    out = tmp_path / "viz.html"
    render_cytoscape(data, out)
    body = out.read_text()
    assert "<!doctype html>" in body
    assert "cytoscape" in body.lower()
    assert "elements" in body
    # Both nodes (1 < 3) and edges present in the JSON payload.
    assert '"group": "nodes"' in body or '"group":"nodes"' in body
    assert '"group": "edges"' in body or '"group":"edges"' in body
    # Brace-escapes resolved correctly (no literal '{{').
    assert "{{" not in body
