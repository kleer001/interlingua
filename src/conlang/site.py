"""Stage 6 single-page deliverable: grammar sketch + 1000-row lexicon table.

Phase A target audience: anyone who wants a one-file demo. Phase B builds out
the multi-page site + reproducible notebook for other audiences.

Inputs:
  data/processed/lexicon.json    (from `python -m conlang.lexicon`)

Output:
  data/processed/lexicon.html    single self-contained file, ~600 KB
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import sys
from pathlib import Path

from . import PROCESSED_DIR

_REPO_DOCS_STATIC = Path(__file__).resolve().parents[2] / "docs" / "static"
from .phonology import (
    CLASS_PREFIXES,
    NASAL_DIGRAPHS,
    NEGATION_PREFIX,
    PRENASALIZED_ONSETS,
    SINGLE_CONSONANTS,
    VOWELS,
    apply_class_prefix,
    negate,
)


_DEMO_STEM = "paka"  # used for the "all classes on one stem" demo table


_HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>interlingua — bottom-up SAE-derived conlang</title>
<style>
  :root {
    --bg: #111; --fg: #ddd; --accent: #f80; --dim: #888; --row: #1a1a1a;
  }
  body { background: var(--bg); color: var(--fg); font-family: ui-sans-serif, system-ui, sans-serif;
         max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
  h1, h2, h3 { color: var(--accent); }
  h1 { border-bottom: 1px solid var(--accent); padding-bottom: .25em; }
  code { background: var(--row); padding: 0 .25em; border-radius: 3px; }
  table { width: 100%; border-collapse: collapse; margin: .5em 0; font-size: .9em; }
  th, td { padding: .35em .6em; border-bottom: 1px solid #333; text-align: left; vertical-align: top; }
  th { background: var(--row); color: var(--accent); position: sticky; top: 0; cursor: pointer; user-select: none; }
  th:hover { background: #222; }
  tr:nth-child(even) td { background: var(--row); }
  .stem { font-family: ui-monospace, monospace; color: var(--accent); }
  .neg { font-family: ui-monospace, monospace; color: var(--dim); }
  .meta { color: var(--dim); font-size: .85em; }
  .search { margin: 1em 0; }
  .search input { background: var(--row); color: var(--fg); border: 1px solid #333;
                  padding: .4em .8em; width: 60%; font-size: 1em; border-radius: 4px; }
  .count { margin-left: 1em; color: var(--dim); }
  details { background: var(--row); padding: .5em 1em; border-radius: 4px; margin: .5em 0; }
  summary { cursor: pointer; color: var(--accent); }
</style>
</head>
<body>
"""

_HTML_TAIL = """
<script>
  // Sortable + filterable lexicon table. Minimal vanilla JS.
  (function() {
    const table = document.getElementById("lexicon");
    if (!table) return;
    const tbody = table.tBodies[0];
    const allRows = Array.from(tbody.rows);
    const input = document.getElementById("search");
    const count = document.getElementById("count");

    function applyFilter() {
      const q = (input.value || "").trim().toLowerCase();
      let shown = 0;
      for (const r of allRows) {
        const match = !q || r.textContent.toLowerCase().includes(q);
        r.style.display = match ? "" : "none";
        if (match) shown += 1;
      }
      count.textContent = shown + " / " + allRows.length + " entries";
    }
    input.addEventListener("input", applyFilter);

    // Click-to-sort by column. Numeric columns get numeric sort.
    Array.from(table.tHead.rows[0].cells).forEach((th, idx) => {
      let asc = true;
      const numeric = th.dataset.numeric === "true";
      th.addEventListener("click", () => {
        const rows = Array.from(tbody.rows);
        rows.sort((a, b) => {
          const av = a.cells[idx].textContent.trim();
          const bv = b.cells[idx].textContent.trim();
          if (numeric) return (asc ? 1 : -1) * (Number(av) - Number(bv));
          return (asc ? 1 : -1) * av.localeCompare(bv);
        });
        asc = !asc;
        for (const r of rows) tbody.appendChild(r);
      });
    });

    applyFilter();
  })();
</script>
</body>
</html>
"""


def _e(s: str) -> str:
    return html.escape(str(s))


def _render_phonology_section() -> str:
    rows = []
    rows.append("<h2>Phonology</h2>")
    rows.append(
        f"<p><strong>Vowels ({len(VOWELS)}):</strong> "
        f"<code>{' '.join(VOWELS)}</code></p>"
    )
    rows.append(
        f"<p><strong>Single consonants ({len(SINGLE_CONSONANTS)}):</strong> "
        f"<code>{' '.join(SINGLE_CONSONANTS)}</code></p>"
    )
    rows.append(
        f"<p><strong>Nasal digraphs:</strong> "
        f"<code>{' '.join(NASAL_DIGRAPHS)}</code> "
        f"(treated as single phonemes)</p>"
    )
    rows.append(
        f"<p><strong>Prenasalized onsets:</strong> "
        f"<code>{' '.join(PRENASALIZED_ONSETS)}</code> "
        f"(arise from class-9 N + stop)</p>"
    )
    rows.append(
        "<p><strong>Syllable template:</strong> <code>(C)V</code>. "
        "Words are ≥ 2 syllables. No codas, no glide insertion, "
        "no tone.</p>"
    )
    return "\n".join(rows)


def _render_morphology_section() -> str:
    rows = ["<h2>Morphology</h2>",
            "<p>Eleven noun classes (six sg/pl pairs + class 11). "
            "Composition order: <code>[negation?] + [class prefix] + [stem]</code>. "
            f"Productive negation: prefix <code>{NEGATION_PREFIX}-</code> "
            "(per spec v0.2 §7 Commitment 7).</p>",
            "<table>",
            "<thead><tr><th>Class</th><th>Prefix</th><th>Domain</th>"
            f"<th>{_DEMO_STEM} + class</th>"
            f"<th>negation</th></tr></thead>",
            "<tbody>"]
    for cid in sorted(CLASS_PREFIXES):
        pfx, desc = CLASS_PREFIXES[cid]
        surface = apply_class_prefix(_DEMO_STEM, cid)
        ant = negate(surface)
        rows.append(
            f"<tr><td>{cid}</td><td><code>{_e(pfx)}-</code></td>"
            f"<td>{_e(desc)}</td>"
            f"<td class='stem'>{_e(surface)}</td>"
            f"<td class='neg'>{_e(ant)}</td></tr>"
        )
    rows.append("</tbody></table>")
    rows.append(
        "<details><summary>Class 9 sandhi note</summary>"
        "<p>Class 9 takes one of two allomorphs depending on the stem onset:</p>"
        "<ul>"
        "<li>Prenasalized stop (<code>mp/mb/nt/nd</code>) before "
        "<code>p, b, t, d</code> initials.</li>"
        "<li><code>yi-</code> (or <code>y-</code> before a vowel) for "
        "every other onset. Pure homorganic sandhi (<code>ng+k</code>, "
        "<code>n+s</code>, etc.) would produce phonotactically illegal "
        "clusters in this inventory.</li>"
        "</ul></details>"
    )
    return "\n".join(rows)


def _render_lexicon_section(entries: list[dict]) -> str:
    rows = ["<h2>Lexicon</h2>",
            f"<p class='meta'>{len(entries)} entries derived from the "
            "Gemma Scope SAE 1000-feature slice. "
            "Stem structure is phonosemantic: CV1 encodes HDBSCAN cluster "
            "(siblings share CV1); CV2 encodes the highest-PMI co-activation "
            "parent (PMI-related nodes share CV2); CV3 is uniqueness. "
            "Collisions get a CV4 appended.</p>",
            "<div class='search'>",
            "<input id='search' placeholder='filter by stem, label, class…'>",
            "<span class='count' id='count'></span>",
            "</div>",
            "<table id='lexicon'>",
            "<thead><tr>"
            "<th>Stem</th>"
            "<th>Surface</th>"
            "<th>Negation</th>"
            "<th data-numeric='true'>Class</th>"
            "<th>Class domain</th>"
            "<th>Label (Neuronpedia)</th>"
            "<th data-numeric='true'>Feature</th>"
            "</tr></thead>",
            "<tbody>"]
    for e in entries:
        rows.append(
            "<tr>"
            f"<td class='stem'>{_e(e['stem'])}</td>"
            f"<td class='stem'>{_e(e['surface'])}</td>"
            f"<td class='neg'>{_e(e['antonym'])}</td>"
            f"<td>{e['class_id']}</td>"
            f"<td>{_e(e['class_name'])}</td>"
            f"<td>{_e(e['label'])}</td>"
            f"<td>{e['feature_id']}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _render_header() -> str:
    return (
        "<h1>interlingua</h1>"
        "<p class='meta'>A bottom-up conlang whose lexicon is derived from "
        "sparse-autoencoder features of Gemma 2 (2B). 1000 nodes, "
        "Bantu-shaped phonology, productive compositional negation. "
        "See <code>spec.md</code> v0.2 and <code>prior-work.md</code> v0.3 "
        "for the full design.</p>"
    )


def _render_methodology_section() -> str:
    return (
        "<h2>Methodology (one-paragraph)</h2>"
        "<p>For each of 1000 features from the Gemma Scope SAE "
        "(<code>gemma-scope-2b-pt-res-canonical layer_12/width_16k/canonical</code>), "
        "we collect a Neuronpedia description, cluster the decoder vectors with "
        "HDBSCAN, and accumulate pairwise co-activation PMI on ~190k FLORES-200 "
        "tokens. The resulting graph (cosine + co-activation edges) is regularized "
        "into parent/sibling/near structure per spec v0.2 §7. We then assign a "
        "Bantu-shaped noun class (keyword heuristic over the description) and "
        "generate a phonosemantic stem where shared cluster and shared parent are "
        "audible in the first two syllables. Negation per Commitment 7 is a single "
        "productive prefix; no <code>transformation</code> primitive exists in "
        "the graph because the function-vector crystal bridge to SAE decoders "
        "failed the v0.2 §7 distinctiveness gate (0% margin ≥ 0.05).</p>"
    )


def render_html(entries: list[dict]) -> str:
    return (
        _HTML_HEAD
        + _render_header()
        + _render_phonology_section()
        + _render_morphology_section()
        + _render_methodology_section()
        + _render_lexicon_section(entries)
        + f"<p class='meta'>Generated {dt.datetime.now(dt.UTC).isoformat()}.</p>"
        + _HTML_TAIL
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lexicon",
        type=Path,
        default=PROCESSED_DIR / "lexicon.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_DOCS_STATIC / "lexicon.html",
        help="Default: docs/static/lexicon.html. MkDocs serves this as a "
             "static file alongside the doc pages.",
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    print(f"[1/2] Loading {args.lexicon} ...", flush=True)
    doc = json.loads(args.lexicon.read_text())
    entries = doc["entries"]
    print(f"      {len(entries)} entries", flush=True)
    print("[2/2] Rendering HTML ...", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(entries))
    size_kb = args.out.stat().st_size / 1024
    print(f"      wrote {args.out} ({size_kb:.0f} KB)", flush=True)
    return args.out


def main() -> None:
    args = parse_args(sys.argv[1:])
    run(args)


if __name__ == "__main__":
    main()
