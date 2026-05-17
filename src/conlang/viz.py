"""Vertical-slice visualization: renderer-agnostic graph data + two backends.

`build_graph_data` does all the data preparation (color, hover text, edge
filtering) and returns a plain dataclass. `render_pyvis` and
`render_cytoscape` are thin presenters that consume the same data.
"""

from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import PROCESSED_DIR
from .dedupe import Cluster


NOISE_COLOR = "#555"
DEFAULT_COLOR = "#88c"
COSINE_EDGE_COLOR = "#cccccc"
PMI_EDGE_COLOR = "orange"


def _palette(n: int) -> list[str]:
    """n visually distinct hex colors via evenly-spaced HSL hues."""
    if n <= 0:
        return []
    return [
        "#{:02x}{:02x}{:02x}".format(
            *(int(255 * c) for c in colorsys.hls_to_rgb((i / n) % 1.0, 0.55, 0.65))
        )
        for i in range(n)
    ]


@dataclass
class GraphNode:
    id: int           # 0..n_reps-1, stable across renderers
    feature_id: int
    label: str        # short, for inline display
    title: str        # multi-line hover text
    color: str        # hex
    value: float      # for node-size weighting


@dataclass
class GraphEdge:
    src: int          # GraphNode.id
    dst: int          # GraphNode.id
    weight: float
    kind: str         # "cosine" | "pmi"
    title: str
    color: str


@dataclass
class GraphData:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def edge_count(self, kind: str) -> int:
        return sum(1 for e in self.edges if e.kind == kind)


def _node_color(
    rep: int,
    hdbscan_labels: np.ndarray | None,
    color_for_label: dict[int, str],
) -> str:
    if hdbscan_labels is None:
        return DEFAULT_COLOR
    lab = int(hdbscan_labels[rep])
    if lab < 0:
        return NOISE_COLOR
    return color_for_label.get(lab, DEFAULT_COLOR)


def _node_title_lines(
    rep: int,
    meta: dict,
    cluster_size: int,
    hdbscan_labels: np.ndarray | None,
    crystal_overlay: dict | None,
) -> list[str]:
    lines = [
        f"feature_id={meta['feature_id']}",
        f"cluster_size={cluster_size}",
    ]
    if hdbscan_labels is not None:
        lab = int(hdbscan_labels[rep])
        lines.append("hdbscan_cluster=" + ("noise" if lab < 0 else str(lab)))
    if crystal_overlay is not None:
        relations = crystal_overlay["relations"]
        bi = int(crystal_overlay["best_idx"][rep])
        bs = float(crystal_overlay["best_score"][rep])
        mg = float(crystal_overlay["margin"][rep])
        lines.append(
            f"best_crystal={relations[bi]}  score={bs:+.3f}  margin={mg:+.3f}"
        )
    lines.append(f"label: {meta['label']}")
    return lines


def _pair_edges(
    matrix: np.ndarray,
    rep_indices: list[int],
    rep_to_node_id: dict[int, int],
    threshold: float,
    kind: str,
    color: str,
    title_fmt: str,
) -> list[GraphEdge]:
    """Iterate the upper triangle over `rep_indices`; emit one edge per pair
    where matrix[a, b] >= threshold."""
    out: list[GraphEdge] = []
    for i, a in enumerate(rep_indices):
        for b in rep_indices[i + 1:]:
            v = float(matrix[a, b])
            if v >= threshold:
                out.append(
                    GraphEdge(
                        src=rep_to_node_id[a],
                        dst=rep_to_node_id[b],
                        weight=v,
                        kind=kind,
                        title=title_fmt.format(v),
                        color=color,
                    )
                )
    return out


def build_graph_data(
    features_meta: list[dict],
    sim: np.ndarray,
    clusters: list[Cluster],
    edge_threshold: float,
    hdbscan_labels: np.ndarray | None = None,
    crystal_overlay: dict | None = None,
    pmi: np.ndarray | None = None,
    pmi_threshold: float = 5.0,
) -> GraphData:
    """Renderer-agnostic. Produces nodes + edges from the slice artifacts."""
    rep_indices = [c.representative for c in clusters]
    rep_to_node_id = {r: i for i, r in enumerate(rep_indices)}

    color_for_label: dict[int, str] = {}
    if hdbscan_labels is not None:
        unique_real = sorted({int(x) for x in hdbscan_labels if x >= 0})
        color_for_label = dict(zip(unique_real, _palette(len(unique_real))))

    nodes: list[GraphNode] = []
    for rep, cluster in zip(rep_indices, clusters):
        meta = features_meta[rep]
        title_lines = _node_title_lines(
            rep, meta, len(cluster.members), hdbscan_labels, crystal_overlay
        )
        nodes.append(
            GraphNode(
                id=rep_to_node_id[rep],
                feature_id=meta["feature_id"],
                label=meta["label"][:40],
                title="\n".join(title_lines),
                color=_node_color(rep, hdbscan_labels, color_for_label),
                value=float(len(cluster.members)),
            )
        )

    edges: list[GraphEdge] = _pair_edges(
        sim, rep_indices, rep_to_node_id,
        edge_threshold, "cosine", COSINE_EDGE_COLOR, "cos={:.3f}",
    )
    if pmi is not None:
        edges.extend(_pair_edges(
            pmi, rep_indices, rep_to_node_id,
            pmi_threshold, "pmi", PMI_EDGE_COLOR, "pmi={:+.2f}",
        ))

    print(
        f"      viz data: {len(nodes)} nodes, "
        f"{sum(1 for e in edges if e.kind == 'cosine')} cosine "
        f"(>= {edge_threshold:.2f}), "
        f"{sum(1 for e in edges if e.kind == 'pmi')} co-activation "
        f"(PMI >= {pmi_threshold:.2f})",
        flush=True,
    )
    return GraphData(nodes=nodes, edges=edges)


def render_pyvis(data: GraphData, out_path: Path) -> Path:
    """Thin pyvis presenter. Slow at > ~1k nodes; kept for legacy parity."""
    from pyvis.network import Network

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#111",
        font_color="white",
        notebook=False,
    )
    net.force_atlas_2based(spring_length=120)
    for n in data.nodes:
        net.add_node(n.id, label=n.label, title=n.title, value=n.value, color=n.color)
    for e in data.edges:
        net.add_edge(e.src, e.dst, value=e.weight, title=e.title, color=e.color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path), notebook=False)
    return out_path


_CYTOSCAPE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>interlingua slice — cytoscape viz</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #111;
                color: #ddd; font-family: ui-sans-serif, system-ui, sans-serif; }}
  #cy {{ position: absolute; top: 0; left: 0; right: 380px; bottom: 0; }}
  #panel {{ position: absolute; top: 0; right: 0; width: 380px; bottom: 0;
            background: #1a1a1a; border-left: 1px solid #333;
            padding: 1em; overflow-y: auto; box-sizing: border-box; }}
  #panel h2 {{ color: #f80; margin-top: 0; font-size: 1em; }}
  #panel pre {{ white-space: pre-wrap; word-break: break-word; font-size: .85em;
                color: #ddd; }}
  #legend {{ font-size: .8em; color: #888; margin-top: 1em; }}
  #legend .swatch {{ display: inline-block; width: .8em; height: .8em;
                     vertical-align: middle; margin-right: .3em; }}
  #search {{ width: 100%; box-sizing: border-box; background: #111; color: #ddd;
             border: 1px solid #333; padding: .4em; margin-bottom: .5em; }}
  .count {{ color: #888; font-size: .8em; }}
</style>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
</head>
<body>
<div id="cy"></div>
<div id="panel">
  <h2>interlingua slice</h2>
  <input id="search" placeholder="filter by label…">
  <div class="count" id="count"></div>
  <div id="info">
    <p style="color:#888">Click a node for details.</p>
  </div>
  <div id="legend">
    <div><span class="swatch" style="background:{COSINE_EDGE_COLOR}"></span> cosine edge</div>
    <div><span class="swatch" style="background:{PMI_EDGE_COLOR}"></span> co-activation (PMI) edge</div>
    <div><span class="swatch" style="background:{NOISE_COLOR}"></span> HDBSCAN noise singleton</div>
    <div><span class="swatch" style="background:{DEFAULT_COLOR}"></span> uncategorized</div>
    <p class="count" id="totals"></p>
  </div>
</div>
<script>
  const DATA = {DATA_JSON};

  const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements: DATA.elements,
    style: [
      {{ selector: 'node',
         style: {{
           'background-color': 'data(color)',
           'label': 'data(label)',
           'color': '#ddd',
           'font-size': 8,
           'text-valign': 'bottom',
           'text-margin-y': 4,
           'text-wrap': 'wrap',
           'text-max-width': 90,
           'width': 'mapData(value, 1, 50, 8, 28)',
           'height': 'mapData(value, 1, 50, 8, 28)',
           'border-width': 0
         }} }},
      {{ selector: 'edge.cosine',
         style: {{
           'line-color': '{COSINE_EDGE_COLOR}',
           'opacity': 0.35,
           'width': 'mapData(weight, 0, 1, 0.4, 2.5)',
           'curve-style': 'haystack'
         }} }},
      {{ selector: 'edge.pmi',
         style: {{
           'line-color': '{PMI_EDGE_COLOR}',
           'opacity': 0.7,
           'width': 'mapData(weight, 0, 10, 0.8, 3)',
           'curve-style': 'haystack'
         }} }},
      {{ selector: 'node:selected',
         style: {{
           'border-width': 3,
           'border-color': '#f80'
         }} }},
      {{ selector: '.dim',
         style: {{
           'opacity': 0.1
         }} }}
    ],
    layout: {{
      name: 'cose',
      animate: false,
      randomize: true,
      idealEdgeLength: 80,
      nodeRepulsion: 6000,
      gravity: 0.25,
      numIter: 1200,
      fit: true,
      padding: 30
    }}
  }});

  document.getElementById('totals').textContent =
    DATA.n_nodes + ' nodes, ' +
    DATA.n_cosine_edges + ' cosine edges, ' +
    DATA.n_pmi_edges + ' PMI edges';

  cy.on('tap', 'node', evt => {{
    const n = evt.target;
    const info = document.getElementById('info');
    info.replaceChildren();
    const pre = document.createElement('pre');
    pre.textContent = n.data('hoverText');
    info.appendChild(pre);
  }});
  cy.on('tap', evt => {{ if (evt.target === cy) {{
    const info = document.getElementById('info');
    info.replaceChildren();
    const p = document.createElement('p');
    p.style.color = '#888';
    p.textContent = 'Click a node for details.';
    info.appendChild(p);
  }} }});

  const search = document.getElementById('search');
  const countEl = document.getElementById('count');
  function applyFilter() {{
    const q = (search.value || '').trim().toLowerCase();
    let shown = 0;
    cy.batch(() => {{
      cy.nodes().forEach(n => {{
        const txt = (n.data('label') + ' ' + n.data('hoverText')).toLowerCase();
        const match = !q || txt.includes(q);
        n.toggleClass('dim', !match);
        if (match) shown += 1;
      }});
      cy.edges().forEach(e => {{
        const dim = e.source().hasClass('dim') || e.target().hasClass('dim');
        e.toggleClass('dim', dim);
      }});
    }});
    countEl.textContent = shown + ' / ' + cy.nodes().length + ' nodes';
  }}
  search.addEventListener('input', applyFilter);
  applyFilter();
</script>
</body>
</html>
"""


def render_cytoscape(data: GraphData, out_path: Path) -> Path:
    """Cytoscape.js presenter. Handles dense graphs much better than pyvis."""
    elements = []
    for n in data.nodes:
        elements.append({
            "group": "nodes",
            "data": {
                "id": f"n{n.id}",
                "label": n.label,
                "color": n.color,
                "value": n.value,
                "hoverText": n.title,
            },
        })
    for i, e in enumerate(data.edges):
        elements.append({
            "group": "edges",
            "data": {
                "id": f"e{i}",
                "source": f"n{e.src}",
                "target": f"n{e.dst}",
                "weight": e.weight,
                "color": e.color,
            },
            "classes": e.kind,
        })
    payload = {
        "elements": elements,
        "n_nodes": len(data.nodes),
        "n_cosine_edges": data.edge_count("cosine"),
        "n_pmi_edges": data.edge_count("pmi"),
    }
    html_doc = _CYTOSCAPE_HTML.format(
        DATA_JSON=json.dumps(payload),
        COSINE_EDGE_COLOR=COSINE_EDGE_COLOR,
        PMI_EDGE_COLOR=PMI_EDGE_COLOR,
        NOISE_COLOR=NOISE_COLOR,
        DEFAULT_COLOR=DEFAULT_COLOR,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)
    return out_path


RENDERERS = {
    "pyvis": render_pyvis,
    "cytoscape": render_cytoscape,
}


def build_pyvis_graph(
    features_meta: list[dict],
    sim: np.ndarray,
    clusters: list[Cluster],
    edge_threshold: float,
    out_path: Path | None = None,
    hdbscan_labels: np.ndarray | None = None,
    crystal_overlay: dict | None = None,
    pmi: np.ndarray | None = None,
    pmi_threshold: float = 5.0,
):
    """Legacy thin wrapper: build data + render with pyvis.

    Kept for any external caller; new code should use `build_graph_data` +
    a renderer of choice from `RENDERERS`.
    """
    data = build_graph_data(
        features_meta=features_meta,
        sim=sim,
        clusters=clusters,
        edge_threshold=edge_threshold,
        hdbscan_labels=hdbscan_labels,
        crystal_overlay=crystal_overlay,
        pmi=pmi,
        pmi_threshold=pmi_threshold,
    )
    if out_path is None:
        out_path = PROCESSED_DIR / "slice.html"
    return render_pyvis(data, out_path)


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
