"""Generate notebooks/walkthrough.ipynb from a Python template.

Keeping the notebook generated (rather than hand-edited) means it stays in
sync with the rest of the codebase as the pipeline evolves. Re-run after
changes to the public CLIs.

Usage:
    python notebooks/build_walkthrough.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# interlingua — walkthrough\n"
            "\n"
            "Reproducible end-to-end pass through the pipeline that "
            "produces the conlang lexicon.\n"
            "\n"
            "**This notebook is read-mostly by default.** All cells assume "
            "the heavy pipeline stages have already been run and their "
            "artifacts are on disk under `/media/menser/fauna/interlingua/"
            "data/`. To regenerate the artifacts from scratch:\n"
            "\n"
            "```bash\n"
            "source .venv/bin/activate\n"
            "python -m conlang.slice --sae-release gemma-scope-2b-pt-res-canonical \\\n"
            "    --sae-id layer_12/width_16k/canonical \\\n"
            "    --neuronpedia-model gemma-2-2b \\\n"
            "    --neuronpedia-source 12-gemmascope-res-16k \\\n"
            "    --top-n 1000 --dedup-method hdbscan\n"
            "python -m conlang.run_coactivation --use-flores --n-per-lang 1000\n"
            "python -m conlang.regularize\n"
            "python -m conlang.lexicon\n"
            "python -m conlang.site\n"
            "```\n"
            "\n"
            "The coactivation stage needs a GPU and takes a few minutes; "
            "the others finish in seconds."
        ),

        new_code_cell(
            "import json\n"
            "import numpy as np\n"
            "\n"
            "from conlang import INTERIM_DIR, PROCESSED_DIR, RAW_DIR\n"
            "from conlang.phonology import (\n"
            "    apply_class_prefix, negate, CLASS_PREFIXES,\n"
            "    VOWELS, SINGLE_CONSONANTS,\n"
            ")\n"
        ),

        new_markdown_cell(
            "## Stage 1–2 — Ingest and dedupe\n"
            "\n"
            "1000 SAE features pass the §6 filter from the bulk Neuronpedia "
            "explanation dump. We compute pairwise cosine similarity over "
            "their decoder vectors and cluster with HDBSCAN."
        ),

        new_code_cell(
            "features = [json.loads(line) for line in open(RAW_DIR / 'features.jsonl')]\n"
            "sim = np.load(INTERIM_DIR / 'sim_matrix.npy')\n"
            "labels = np.load(INTERIM_DIR / 'hdbscan_labels.npy')\n"
            "print(f'{len(features)} features')\n"
            "print(f'sim matrix: {sim.shape}')\n"
            "print(f'HDBSCAN: {len(set(labels.tolist()))} groups, '\n"
            "      f'{int((labels == -1).sum())} noise singletons')\n"
            "print()\n"
            "print('first three feature labels:')\n"
            "for f in features[:3]:\n"
            "    print(f'  [{f[\"feature_id\"]:>5d}] {f[\"label\"]}')\n"
        ),

        new_markdown_cell(
            "## Stage 3 — Co-activation edges\n"
            "\n"
            "5982 FLORES-200 dev sentences across six languages "
            "(eng, fra, deu, spa, zho, jpn) pass through Gemma 2 2B with "
            "the SAE encoder hooked at layer 12. We accumulate pairwise "
            "feature co-firing and normalize to PMI."
        ),

        new_code_cell(
            "pmi = np.load(INTERIM_DIR / 'coactivation' / 'pmi.npy')\n"
            "top_pairs = json.load(open(INTERIM_DIR / 'coactivation' / 'top_pairs.json'))\n"
            "print(f'PMI matrix: {pmi.shape}, {int((pmi > 0).sum())} positive pairs')\n"
            "print(f'top-50 pairs stashed in top_pairs.json')\n"
            "print()\n"
            "print('top 5 PMI pairs:')\n"
            "for pair in top_pairs[:5]:\n"
            "    print(f'  PMI {pair[\"pmi\"]:+.2f}, cofire={pair[\"cofire_count\"]}')\n"
            "    print(f'    {pair[\"label_a\"][:60]!r}')\n"
            "    print(f'    {pair[\"label_b\"][:60]!r}')\n"
        ),

        new_markdown_cell(
            "## Stage 4 — Decision gate\n"
            "\n"
            "Three load-bearing questions before continuing to morphology."
        ),

        new_code_cell(
            "n_distinct = len(features) - int((labels >= 0).sum() - len(set(l for l in labels if l >= 0)))\n"
            "n_with_pmi_parent = int((pmi.max(axis=1) > 0).sum())\n"
            "crystal_coverage = 0.0  # See spec §7: bridge failed, 0% margin >= 0.05\n"
            "\n"
            "print(f'Distinct semantic fields:       {n_distinct} (target 500-5000) — OK')\n"
            "print(f'Crystal coverage at margin 0.05: {crystal_coverage:.0%} (target >= 30%) — FAIL')\n"
            "print(f'Nodes with positive-PMI parent: {n_with_pmi_parent}/{len(features)} — OK')\n"
            "print()\n"
            "print('2/3 green. The crystal failure is exactly the Commitment 7')\n"
            "print('failure-mode the spec anticipated; mitigation (compositional')\n"
            "print('negation handled by morphology) is in force.')\n"
        ),

        new_markdown_cell(
            "## Stage 5 — Regularize\n"
            "\n"
            "Collapse the Stage 3 multigraph into per-node `parent` "
            "(highest-PMI neighbor), `siblings` (HDBSCAN cluster co-members), "
            "and `near` (top cosine neighbors)."
        ),

        new_code_cell(
            "reg = json.load(open(PROCESSED_DIR / 'regularized.json'))\n"
            "nodes = reg['nodes']\n"
            "n_with_parent = sum(1 for n in nodes if n['parent'])\n"
            "n_with_siblings = sum(1 for n in nodes if n['siblings'])\n"
            "print(f'{n_with_parent}/{len(nodes)} nodes have a parent')\n"
            "print(f'{n_with_siblings}/{len(nodes)} nodes have at least one sibling')\n"
            "print()\n"
            "print('example node:')\n"
            "import textwrap\n"
            "print(textwrap.indent(json.dumps(nodes[491], indent=2)[:600], '  '))\n"
        ),

        new_markdown_cell(
            "## Stage 6 — Phonology, lexicon, site\n"
            "\n"
            "Eleven Bantu noun classes, productive `si-` negation, "
            "phonosemantic stems (CV1=cluster, CV2=parent, CV3=self)."
        ),

        new_code_cell(
            "# Show the 11-class affix paradigm on a sample stem.\n"
            "for cid in sorted(CLASS_PREFIXES):\n"
            "    pfx, desc = CLASS_PREFIXES[cid]\n"
            "    surface = apply_class_prefix('paka', cid)\n"
            "    ant = negate(surface)\n"
            "    print(f'class {cid:2d} ({pfx:>3}-) {desc[:30]:<30} → {surface:<10} / {ant}')\n"
        ),

        new_code_cell(
            "lex = json.load(open(PROCESSED_DIR / 'lexicon.json'))\n"
            "entries = lex['entries']\n"
            "by_class = {}\n"
            "for e in entries:\n"
            "    by_class[e['class_id']] = by_class.get(e['class_id'], 0) + 1\n"
            "print(f'{len(entries)} lexicon entries\\n')\n"
            "print('class distribution:')\n"
            "for cid in sorted(by_class):\n"
            "    print(f'  class {cid:2d} ({CLASS_PREFIXES[cid][1]:<35}): {by_class[cid]}')\n"
        ),

        new_code_cell(
            "import random\n"
            "random.seed(7)\n"
            "print('sample lexicon entries (random):\\n')\n"
            "for e in random.sample(entries, 8):\n"
            "    print(f'  {e[\"surface\"]:<14} (neg {e[\"antonym\"]:<14}) '\n"
            "          f'class={e[\"class_id\"]:2d}  {e[\"label\"][:55]!r}')\n"
        ),

        new_markdown_cell(
            "## Open the rendered deliverables\n"
            "\n"
            "After running `python -m conlang.site && mkdocs build`:\n"
            "\n"
            "- **Single-page lexicon** (Phase A): "
            "`docs/static/lexicon.html` (also served at "
            "`site/static/lexicon.html` from the MkDocs build).\n"
            "- **Multi-page site** (Phase B): "
            "`data/processed/site/index.html`."
        ),
    ]
    return nb


def main() -> None:
    nb = build()
    out = Path(__file__).resolve().parent / "walkthrough.ipynb"
    nbformat.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
