"""Static HTML browser for the anchor pool.

One self-contained HTML page (no external deps) with:
- a sortable summary table: concept | category | n_languages | n_ipa | sharpness
- a per-concept section beneath that lists all entries with their language,
  form, romanization, IPA, source.

Sortable via a tiny inline JS click-sort. Survives offline because all data
is embedded.
"""

from __future__ import annotations

import argparse
import html
from collections import defaultdict
from pathlib import Path

from . import ANCHOR_PROCESSED
from .concepts import BY_SLUG
from .schema import AnchorEntry, read_jsonl
from .signatures import ConceptSignature, build_all_signatures


def _esc(s: str | None) -> str:
    return html.escape(s, quote=True) if s else ""


def _summary_row(sig: ConceptSignature) -> str:
    concept = sig.concept
    inv = BY_SLUG.get(concept)
    category = inv.category if inv else ""
    cls = f"sharp-{int(sig.sharpness * 10)}"
    return (
        f'<tr class="row {cls}" data-concept="{_esc(concept)}">'
        f'<td><a href="#c-{_esc(concept)}">{_esc(concept)}</a></td>'
        f"<td>{_esc(category)}</td>"
        f"<td>{sig.n_entries}</td>"
        f"<td>{sig.n_with_ipa}</td>"
        f"<td>{sig.n_languages}</td>"
        f'<td class="proj">{_esc(sig.modal_projection)}</td>'
        f"<td>{sig.sharpness:.3f}</td>"
        f"</tr>"
    )


def _concept_block(sig: ConceptSignature, entries: list[AnchorEntry]) -> str:
    rows: list[str] = []
    for e in entries:
        ipa = _esc(e.ipa or "")
        proj = _esc(e.projected_form or "")
        roman = _esc(e.romanization or "")
        ortho = _esc(e.orthography or "")
        rows.append(
            f"<tr>"
            f"<td>{_esc(e.language)}</td>"
            f"<td>{_esc(e.language_code or '')}</td>"
            f'<td class="form">{ortho}</td>'
            f'<td class="roman">{roman}</td>'
            f'<td class="ipa">{ipa}</td>'
            f'<td class="proj">{proj}</td>'
            f"<td>{_esc(e.source or '')}</td>"
            f"</tr>"
        )
    examples_html = ""
    if sig.examples:
        items = []
        for ex in sig.examples:
            items.append(
                f"<li><b>{_esc(ex['language'])}</b> "
                f"<span class='form'>{_esc(ex.get('form') or '')}</span> "
                f"<span class='ipa'>{_esc(ex.get('ipa') or '')}</span> "
                f"<span class='proj'>&rarr; {_esc(ex.get('projected_form') or '')}</span></li>"
            )
        examples_html = "<ul class='examples'>" + "".join(items) + "</ul>"

    proj_hist_html = ""
    if sig.projection_histogram:
        items = []
        for form, count in sig.projection_histogram:
            items.append(
                f'<span class="proj-bin">{_esc(form)} <span class="count">x{count}</span></span>'
            )
        modal = _esc(sig.modal_projection)
        proj_hist_html = (
            f'<div class="proj-summary">'
            f'<b>modal projection:</b> <span class="proj big">{modal}</span> '
            f'&middot; <span class="muted">{sig.n_distinct_projections} distinct</span>'
            f'<div class="proj-hist">{" ".join(items)}</div></div>'
        )

    feat_pairs = list(zip(sig.feature_names, sig.mean_features, sig.var_features, strict=True))
    feat_pairs.sort(key=lambda t: t[2])  # by variance ascending = stable first
    feat_html = "".join(
        f"<tr><td>{_esc(n)}</td><td>{m:+.2f}</td><td>{v:.3f}</td></tr>"
        for n, m, v in feat_pairs[:8]
    )

    return f"""
<section id="c-{_esc(sig.concept)}" class="concept">
  <h2>{_esc(sig.concept)}</h2>
  <div class="meta">
    sharpness {sig.sharpness:.3f} ·
    n_entries {sig.n_entries} ·
    n_with_ipa {sig.n_with_ipa} ·
    n_languages {sig.n_languages}
  </div>
  {proj_hist_html}
  {examples_html}
  <details>
    <summary>top-stable features (mean ± lowest variance)</summary>
    <table class="feats"><thead>
    <tr><th>feature</th><th>mean</th><th>var</th></tr>
    </thead><tbody>{feat_html}</tbody></table>
  </details>
  <details open>
    <summary>{len(entries)} entries</summary>
    <table class="entries"><thead>
    <tr><th>language</th><th>code</th><th>form</th>
        <th>romanization</th><th>ipa</th><th>projected</th><th>source</th></tr>
    </thead><tbody>
    {"".join(rows)}
    </tbody></table>
  </details>
</section>
"""


_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Anchor pool browser</title>
<style>
body {
  font: 14px/1.5 system-ui, -apple-system, sans-serif;
  max-width: 1100px; margin: 1em auto; padding: 0 1em; color: #222;
}
h1 { font-size: 1.4em; }
h2 {
  font-size: 1.1em; margin-top: 2em;
  border-bottom: 1px solid #ccc; padding-bottom: 0.2em;
}
.meta { color: #666; font-size: 12px; margin-bottom: 0.5em; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0; }
th, td {
  padding: 3px 8px; border-bottom: 1px solid #eee;
  text-align: left; vertical-align: top;
}
th {
  background: #f4f4f4; cursor: pointer; user-select: none;
  position: sticky; top: 0;
}
th:hover { background: #ebebeb; }
.summary tbody tr:nth-child(odd) { background: #fafafa; }
.form { font-weight: 600; }
.ipa {
  font-family: "Charis SIL", "Doulos SIL", "Lucida Sans Unicode",
    "Segoe UI", monospace;
  color: #06539e;
}
.roman { color: #555; font-style: italic; }
.proj {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  color: #5a2691;
  font-weight: 600;
}
.proj.big { font-size: 1.2em; padding: 0 4px; background: #f3eafe; }
.proj-summary { margin: 0.3em 0; }
.proj-hist {
  margin: 0.3em 0; font-size: 12px;
  display: flex; flex-wrap: wrap; gap: 6px;
}
.proj-bin { padding: 1px 6px; background: #f3eafe; border-radius: 3px; color: #5a2691; }
.proj-bin .count { color: #999; font-size: 11px; }
.muted { color: #888; font-size: 12px; }
.examples { margin: 0.4em 0 0.7em 1em; }
.examples li { margin-bottom: 0.2em; }
details summary { cursor: pointer; user-select: none; padding: 0.3em 0; color: #444; }
details[open] summary { color: #222; }
.feats td:nth-child(2) { font-family: monospace; }
.feats td:nth-child(3) { font-family: monospace; color: #888; }
.sharp-6 { background: #e6f6e6; }
.sharp-7 { background: #d5f0d5; }
.sharp-4 { background: #fdf3df; }
.sharp-3 { background: #fbe7c2; }
nav {
  position: sticky; top: 0; background: #fff;
  padding: 0.5em 0; border-bottom: 1px solid #ccc; z-index: 10;
}
</style>
<script>
function sortBy(idx, numeric) {
  const tbody = document.querySelector('.summary tbody');
  const rows = Array.from(tbody.rows);
  const cur = tbody.dataset.sort || '';
  const next = (cur === idx + ':asc') ? idx + ':desc' : idx + ':asc';
  rows.sort((a, b) => {
    let av = a.cells[idx].innerText, bv = b.cells[idx].innerText;
    if (numeric) { av = parseFloat(av); bv = parseFloat(bv); }
    if (av < bv) return next.endsWith('asc') ? -1 : 1;
    if (av > bv) return next.endsWith('asc') ? 1 : -1;
    return 0;
  });
  tbody.dataset.sort = next;
  rows.forEach(r => tbody.appendChild(r));
}
</script>
<h1>Anchor pool browser</h1>
<nav>summary &middot; sortable; click a concept name to jump to its entries</nav>
"""


def render_html(
    entries: list[AnchorEntry],
    sigs: list[ConceptSignature],
) -> str:
    by_concept: dict[str, list[AnchorEntry]] = defaultdict(list)
    for e in entries:
        by_concept[e.concept].append(e)
    sigs_sorted = sorted(sigs, key=lambda s: -s.sharpness)

    summary_rows = "".join(_summary_row(s) for s in sigs_sorted)
    summary_html = f"""
<table class="summary">
  <thead>
    <tr>
      <th onclick="sortBy(0,false)">concept</th>
      <th onclick="sortBy(1,false)">category</th>
      <th onclick="sortBy(2,true)">n_entries</th>
      <th onclick="sortBy(3,true)">n_with_ipa</th>
      <th onclick="sortBy(4,true)">n_languages</th>
      <th onclick="sortBy(5,false)">modal projection</th>
      <th onclick="sortBy(6,true)">sharpness</th>
    </tr>
  </thead>
  <tbody>{summary_rows}</tbody>
</table>
"""

    concept_blocks = "\n".join(
        _concept_block(s, by_concept.get(s.concept, [])) for s in sigs_sorted
    )

    return _HEAD + summary_html + concept_blocks


def write_browser(
    entries: list[AnchorEntry],
    sigs: list[ConceptSignature],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(entries, sigs), encoding="utf-8")
    return path


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", type=Path, default=ANCHOR_PROCESSED / "anchors-v1.jsonl")
    ap.add_argument("--out", type=Path, default=ANCHOR_PROCESSED / "anchors-v1.html")
    ap.add_argument(
        "--sigs",
        type=Path,
        default=None,
        help="If set, write the signatures JSONL here too",
    )
    args = ap.parse_args()
    entries = read_jsonl(args.anchors)
    sigs = build_all_signatures(entries)
    if args.sigs:
        from .signatures import write_signatures

        write_signatures(sigs, args.sigs)
    p = write_browser(entries, sigs, args.out)
    print(
        f"[viz] {len(entries)} entries, {len(sigs)} concepts -> {p}",
        flush=True,
    )
    print(f"[viz] file size: {p.stat().st_size / 1024:.1f} KiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "render_html",
    "write_browser",
]
