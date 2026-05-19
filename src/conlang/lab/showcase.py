# ruff: noqa: E501
"""Single-page informal showcase of the anchor pool.

A notebook-style HTML page (no JS dependencies except a small inline filter
on the browse table). Tells the story of what's in `anchors-v1.jsonl`:
- 5,338 onomatopoeic words across 84+ languages
- projected to a fixed 10C/5V CV(n) inventory
- "sharp vs fuzzy" cross-linguistic anchors

Sections:
  1. headline + stats
  2. the inventory
  3. featured concepts (curated)
  4. sharp vs fuzzy contrast
  5. browse-everything table (with inline filter)
  6. method notes
"""

from __future__ import annotations

import argparse
import html
from collections import defaultdict
from pathlib import Path

from conlang.anchors import ANCHOR_PROCESSED

from .concepts import BY_SLUG
from .inventory import CONSONANTS, VOWELS
from .schema import AnchorEntry, read_jsonl
from .signatures import ConceptSignature, build_all_signatures


def _esc(s: str | None) -> str:
    return html.escape(s, quote=True) if s else ""


# Curated featured set — the concepts most people will recognize and want to
# see laid out cross-linguistically. Roughly: domestic mammals, common birds,
# core human, weather + insect.
FEATURED_SLUGS: tuple[str, ...] = (
    "cat_meowing",
    "cow_mooing",
    "dog_barking",
    "rooster_crowing",
    "pig_grunting",
    "sheep_bleating",
    "frog_croaking",
    "duck_quacking",
    "owl_hooting",
    "snake_hissing",
    "bee_buzzing",
    "horse_whinnying",
    "donkey_braying",
    "crow_calling",
    "lion_roaring",
    "sneezing",
    "laughter",
    "kiss",
    "yawning",
    "wow",
)

# Roughly which concept to spotlight as the "sharp anchor" example and the
# "fuzzy anchor" example. We pick by hand so the contrast is meaningful (small
# sample sizes inflate sharpness scores in the auto-rank, e.g. flap/tick).
SHARP_FEATURE = "cow_mooing"
FUZZY_FEATURE = "dog_barking"


def _stat_box(label: str, value: str) -> str:
    return (
        f'<div class="stat"><div class="stat-v">{value}</div>'
        f'<div class="stat-l">{label}</div></div>'
    )


def _example_row(e: AnchorEntry) -> str:
    return (
        "<tr>"
        f'<td class="lang">{_esc(e.language)}</td>'
        f'<td class="form">{_esc(e.orthography)}</td>'
        f'<td class="roman">{_esc(e.romanization or "")}</td>'
        f'<td class="ipa">{_esc(e.ipa or "")}</td>'
        f'<td class="proj">{_esc(e.projected_form or "")}</td>'
        "</tr>"
    )


def _featured_block(
    slug: str,
    sig: ConceptSignature,
    entries: list[AnchorEntry],
    max_rows: int = 10,
) -> str:
    """Per-concept featured block. Shows modal projection prominently +
    a few language examples."""
    inv = BY_SLUG.get(slug)
    description = inv.description if inv else slug.replace("_", " ")
    # Include ALL entries with an orthography (English compound forms like
    # "cock-a-doodle-doo" don't always have IPA — projection columns just
    # stay empty for them).
    candidates = [e for e in entries if e.orthography]
    # Pick one example per language, preferring high-traffic languages first
    # AND preferring entries that have IPA when alternatives exist.
    priority = [
        "en",
        "ja",
        "es",
        "fr",
        "de",
        "ru",
        "it",
        "cmn",
        "ko",
        "ar",
        "hi",
        "vi",
        "pt",
        "tr",
        "fi",
        "pl",
        "sv",
        "nl",
        "el",
        "he",
    ]
    by_lang: dict[str, AnchorEntry] = {}
    for e in candidates:
        code = e.language_code or ""
        if not code:
            continue
        existing = by_lang.get(code)
        if existing is None or (not existing.ipa and e.ipa):
            by_lang[code] = e
    ordered = [by_lang[c] for c in priority if c in by_lang]
    other = [e for c, e in by_lang.items() if c not in priority]
    other.sort(key=lambda e: e.language or "")
    examples = (ordered + other)[:max_rows]
    rows = "".join(_example_row(e) for e in examples)

    hist_chips = []
    for form, count in sig.projection_histogram[:8]:
        hist_chips.append(
            f'<span class="chip"><span class="proj">{_esc(form)}</span>'
            f'<span class="chip-c">×{count}</span></span>'
        )
    hist_html = " ".join(hist_chips)

    meta_text = (
        f"{sig.n_with_ipa} forms &middot; {sig.n_languages} langs "
        f"&middot; {sig.n_distinct_projections} distinct projections "
        f"&middot; sharpness {sig.sharpness:.2f}"
    )
    return f"""
<article class="concept" id="featured-{_esc(slug)}">
  <header>
    <h3>{_esc(description.lower())}</h3>
    <div class="cmeta">
      <span class="modal-pill">&rarr;
        <span class="proj big">{_esc(sig.modal_projection)}</span></span>
      <span class="muted">{meta_text}</span>
    </div>
  </header>
  <table class="examples">
    <thead><tr>
      <th>language</th><th>form</th><th>roman</th>
      <th>IPA</th><th>→ projection</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="proj-hist">{hist_html}</div>
</article>
"""


def _inventory_section() -> str:
    cs = " ".join(f'<span class="phon">{c}</span>' for c in CONSONANTS)
    vs = " ".join(f'<span class="phon">{v}</span>' for v in VOWELS)
    return f"""
<section class="inventory">
  <h2>the inventory</h2>
  <p>
    Every form gets projected onto this fixed phoneme set —
    10 consonants, 5 vowels, optional /n/ coda — by mapping each
    IPA segment to its nearest neighbor by articulatory features.
    All cross-linguistic variation collapses into one alphabet.
  </p>
  <div class="phon-row"><b>C&times;10:</b> {cs}</div>
  <div class="phon-row"><b>V&times;5:</b> {vs}</div>
</section>
"""


def _sharp_vs_fuzzy(
    sigs_by_slug: dict[str, ConceptSignature],
    entries_by_slug: dict[str, list[AnchorEntry]],
) -> str:
    def panel(slug: str, role: str) -> str:
        sig = sigs_by_slug[slug]
        entries = entries_by_slug[slug]
        # Show ALL distinct projections as chips, biggest first
        hist = " ".join(
            f'<span class="chip"><span class="proj">{_esc(f)}</span>'
            f'<span class="chip-c">×{c}</span></span>'
            for f, c in sig.projection_histogram[:14]
        )
        # Examples (top-N langs)
        by_lang: dict[str, AnchorEntry] = {}
        for e in entries:
            if e.projected_form and (e.language_code or "") and e.language_code not in by_lang:
                by_lang[e.language_code] = e
        sample = list(by_lang.values())[:8]
        lines = "".join(
            f"<li><b>{_esc(e.language)}</b>: "
            f'<span class="form">{_esc(e.orthography)}</span> '
            f'<span class="ipa">{_esc(e.ipa or "")}</span> '
            f'<span class="proj">→ {_esc(e.projected_form or "")}</span></li>'
            for e in sample
        )
        inv = BY_SLUG.get(slug)
        title = (inv.description if inv else slug).replace("_", " ").lower()
        return f"""
<div class="contrast-panel {role}">
  <div class="role-tag">{role.upper()}</div>
  <h3>{_esc(title)}</h3>
  <div class="muted">
    modal: <span class="proj big">{_esc(sig.modal_projection)}</span>
    &middot; {sig.n_distinct_projections} distinct
    &middot; sharpness {sig.sharpness:.2f}
  </div>
  <div class="proj-hist">{hist}</div>
  <ul class="examples">{lines}</ul>
</div>
"""

    return f"""
<section class="contrast">
  <h2>sharp vs fuzzy anchors</h2>
  <p>
    Some sounds are physically distinctive enough that human ears converge on the
    same imitation — those are <em>sharp anchors</em>. Others have so much room
    for interpretation that cultural convention dominates over iconicity —
    those are <em>fuzzy anchors</em>. Cross-linguistic variation isn't noise;
    it's a measurable property of each concept.
  </p>
  <div class="contrast-grid">
    {panel(SHARP_FEATURE, "sharp")}
    {panel(FUZZY_FEATURE, "fuzzy")}
  </div>
</section>
"""


def _browse_table(
    entries: list[AnchorEntry],
    sigs_by_slug: dict[str, ConceptSignature],
) -> str:
    rows: list[str] = []
    for e in entries:
        if not e.projected_form and not e.ipa:
            continue
        rows.append(
            '<tr class="brow" data-search="'
            + _esc(
                " ".join(
                    [
                        e.concept,
                        e.language or "",
                        e.language_code or "",
                        e.orthography or "",
                        e.romanization or "",
                        e.ipa or "",
                        e.projected_form or "",
                    ]
                ).lower()
            )
            + '">'
            f"<td>{_esc(e.concept)}</td>"
            f'<td class="lang">{_esc(e.language)}</td>'
            f'<td class="form">{_esc(e.orthography)}</td>'
            f'<td class="roman">{_esc(e.romanization or "")}</td>'
            f'<td class="ipa">{_esc(e.ipa or "")}</td>'
            f'<td class="proj">{_esc(e.projected_form or "")}</td>'
            "</tr>"
        )
    return f"""
<section id="browse">
  <h2>browse everything</h2>
  <p class="muted">
    Every row with IPA or projection. Type to filter — concept, language, form,
    IPA, projection are all searchable in one box.
  </p>
  <input id="q" type="search" placeholder="type to filter — concept, language, form, IPA, projection…" />
  <div class="count-line">
    <span id="count"></span> of {len(rows)} rows.
  </div>
  <table class="browse">
    <thead><tr>
      <th>concept</th><th>language</th><th>form</th>
      <th>roman</th><th>IPA</th><th>projection</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <script>
    (function () {{
      const q = document.getElementById('q');
      const rows = document.querySelectorAll('tr.brow');
      const count = document.getElementById('count');
      function apply() {{
        const t = q.value.toLowerCase().trim();
        let n = 0;
        rows.forEach(r => {{
          const hit = !t || r.dataset.search.includes(t);
          r.style.display = hit ? '' : 'none';
          if (hit) n++;
        }});
        count.textContent = n;
      }}
      q.addEventListener('input', apply);
      apply();
    }})();
  </script>
</section>
"""


def _stats_banner(entries: list[AnchorEntry]) -> str:
    n_total = len(entries)
    n_with_ipa = sum(1 for e in entries if e.ipa)
    n_projected = sum(1 for e in entries if e.projected_form)
    langs = {e.language_code for e in entries if e.language_code}
    concepts = {e.concept for e in entries}
    return f"""
<section class="stats">
  {_stat_box("entries", f"{n_total:,}")}
  {_stat_box("with IPA", f"{n_with_ipa:,}")}
  {_stat_box("projected", f"{n_projected:,}")}
  {_stat_box("languages", str(len(langs)))}
  {_stat_box("concepts", str(len(concepts)))}
</section>
"""


_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Anchor pool — cross-linguistic onomatopoeia</title>
<style>
:root {
  --ink: #1f1f1f; --paper: #fffdfa;
  --muted: #6b6b6b; --rule: #d8d2c4;
  --pop: #7a2e8e; --pop-bg: #f3eafe;
  --ipa: #06539e; --sharp: #2b7a3b; --fuzzy: #a04020;
}
body {
  font: 17px/1.55 "Iowan Old Style", "Cambria", "Georgia", serif;
  background: var(--paper); color: var(--ink);
  max-width: 780px; margin: 1.5em auto 4em; padding: 0 1.2em;
}
h1 { font: 700 32px/1.1 "Iowan Old Style", Georgia, serif; margin: 0.2em 0 0; letter-spacing: -0.5px; }
h2 { font: 600 22px/1.2 system-ui, sans-serif; margin: 2em 0 0.4em; }
h3 { font: 600 17px/1.2 system-ui, sans-serif; margin: 1em 0 0.3em; }
p { margin: 0.5em 0 1em; }
.muted { color: var(--muted); font-size: 14px; }
em { font-style: italic; }
b, strong { font-weight: 600; }

.tagline { color: var(--muted); margin: 0.2em 0 1.5em; font-style: italic; }
.stats {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;
  margin: 1.5em 0 2.5em;
}
.stat { padding: 12px; background: #f7f2e8; border-radius: 4px; text-align: center; }
.stat-v { font: 700 22px/1 system-ui, sans-serif; }
.stat-l { color: var(--muted); font-size: 13px; margin-top: 3px; }

.inventory { background: #fff8e8; padding: 1em 1.2em; border-left: 3px solid #e0c060; }
.phon-row { font-family: ui-monospace, monospace; margin: 0.4em 0; }
.phon { display: inline-block; padding: 1px 7px; background: white; border: 1px solid var(--rule); border-radius: 3px; margin: 0 2px; font-weight: 600; }

.concept { margin: 2em 0; }
.concept header { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
.cmeta { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.modal-pill .proj.big {
  font: 700 20px/1 ui-monospace, monospace; padding: 2px 10px;
  background: var(--pop-bg); color: var(--pop); border-radius: 3px;
}

.ipa { font-family: "Charis SIL", "Doulos SIL", "Lucida Sans Unicode", monospace; color: var(--ipa); }
.proj { font-family: ui-monospace, "SF Mono", monospace; color: var(--pop); font-weight: 600; }
.proj.big { font-size: 1.05em; }
.form { font-weight: 600; }
.roman { color: #555; font-style: italic; font-size: 14px; }
.lang { color: #444; }

table.examples, table.browse {
  width: 100%; border-collapse: collapse; margin: 0.6em 0;
  font-size: 14.5px;
}
table.examples td, table.examples th, table.browse td, table.browse th {
  padding: 4px 8px; border-bottom: 1px solid var(--rule); text-align: left;
  vertical-align: top;
}
table.examples th, table.browse th { background: #f7f2e8; font-weight: 600; }

.proj-hist {
  display: flex; flex-wrap: wrap; gap: 4px; margin: 0.4em 0 0.6em;
}
.chip {
  padding: 1px 6px; background: var(--pop-bg); border-radius: 3px;
  font-size: 13px; display: inline-flex; gap: 4px;
}
.chip-c { color: #9686a8; font-size: 12px; }

.contrast-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  margin: 1em 0;
}
@media (max-width: 720px) {
  .contrast-grid { grid-template-columns: 1fr; }
  .stats { grid-template-columns: repeat(2, 1fr); }
}
.contrast-panel {
  padding: 14px; border: 1px solid var(--rule); border-radius: 4px;
}
.contrast-panel.sharp { border-color: var(--sharp); background: #f1faf2; }
.contrast-panel.fuzzy { border-color: var(--fuzzy); background: #fbf2ee; }
.role-tag {
  font: 700 11px/1 system-ui, sans-serif;
  padding: 2px 6px; border-radius: 2px;
  display: inline-block; margin-bottom: 6px;
}
.sharp .role-tag { background: var(--sharp); color: white; }
.fuzzy .role-tag { background: var(--fuzzy); color: white; }
.contrast-panel ul.examples { padding-left: 1em; }
.contrast-panel ul.examples li { margin-bottom: 0.2em; font-size: 14px; }

#q {
  width: 100%; padding: 8px 12px; font-size: 15px; box-sizing: border-box;
  border: 1px solid var(--rule); border-radius: 4px; margin: 0.4em 0 0.3em;
}
.count-line { color: var(--muted); font-size: 13px; margin-bottom: 0.5em; }
table.browse { font-size: 13.5px; }
table.browse tbody tr:nth-child(odd) { background: #fbfaf6; }

footer {
  margin-top: 3em; padding-top: 1em; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: 13.5px;
}
footer code { background: #f7f2e8; padding: 1px 4px; border-radius: 2px; }

nav.toc {
  margin: 1em 0 1.5em; font-size: 13px; color: var(--muted);
}
nav.toc a { color: var(--muted); text-decoration: none; margin-right: 12px; }
nav.toc a:hover { color: var(--pop); text-decoration: underline; }
</style>
"""


def render_showcase(entries: list[AnchorEntry], sigs: list[ConceptSignature]) -> str:
    by_slug_sig = {s.concept: s for s in sigs}
    by_slug_entries: dict[str, list[AnchorEntry]] = defaultdict(list)
    for e in entries:
        by_slug_entries[e.concept].append(e)

    featured_blocks = "\n".join(
        _featured_block(s, by_slug_sig[s], by_slug_entries[s])
        for s in FEATURED_SLUGS
        if s in by_slug_sig
    )

    return (
        _HEAD
        + f"""
<header>
  <h1>anchor pool</h1>
  <p class="tagline">cross-linguistic onomatopoeia, projected onto one alphabet</p>
</header>

<nav class="toc">
  <a href="#what">what</a>
  <a href="#inventory">inventory</a>
  <a href="#featured">featured</a>
  <a href="#contrast">sharp / fuzzy</a>
  <a href="#browse">browse</a>
  <a href="#how">how</a>
</nav>

<section id="what">
  <h2>what's this?</h2>
  <p>
    Words that imitate sounds — onomatopoeia — across as many languages as
    we could find. A bee buzzes <em>bzzz</em> in English, <em>brum</em> in
    French, <em>zoem</em> in Dutch, <em>ブンブン</em> (bunbun) in Japanese.
    Each language paraphrases the same physical noise through its own
    sound system, and those paraphrases turn out to be a really good way
    to see what each phonological inventory is capable of.
  </p>
  <p>
    We pulled the data from Wikipedia's cross-linguistic-onomatopoeias
    page and individual Wiktionary entries, ran the spelled forms
    through Epitran to get IPA where possible (and Wiktionary's own
    pronunciation pages where Epitran can't — Hebrew, Greek, Bulgarian,
    Danish, Icelandic, English…), then projected every IPA segment onto
    a fixed 15-phoneme inventory so we can read the matrix straight
    across.
  </p>
</section>

{_stats_banner(entries)}

<section id="inventory">
{_inventory_section()}
</section>

<section id="featured">
  <h2>featured</h2>
  <p>
    Twenty concepts you'll recognize. Each shows the cross-linguistic
    distribution of projected forms in the inventory — <span class="proj">→ projection</span>.
    The boxed <span class="modal-pill"><span class="proj big">modal</span></span>
    is the single most-common projection across languages.
  </p>
  {featured_blocks}
</section>

<section id="contrast">
{_sharp_vs_fuzzy(by_slug_sig, by_slug_entries)}
</section>

{_browse_table(entries, by_slug_sig)}

<section id="how">
  <h2>how</h2>
  <ul>
    <li><b>sources.</b> Wikipedia <em>cross-linguistic onomatopoeias</em> (the wide table of language × concept) and individual Wiktionary entries for ~120 English seed words (the translations sections).</li>
    <li><b>IPA.</b> <code>Epitran</code> (David Mortensen / CMU) for ~54 languages with shippable g2p rules; Wiktionary form-page IPA scrapes for the rest.</li>
    <li><b>features.</b> Each IPA segment becomes a 24-dim ternary vector via <code>panphon</code> (PHOIBLE-aligned).</li>
    <li><b>projection.</b> Nearest-neighbor mapping by squared-Euclidean distance from each input segment to the fixed 10C/5V inventory. Voicing, length, palatal place, tone all collapse uniformly.</li>
    <li><b>code &amp; data.</b> <code>conlang.anchors.*</code> in the repo; raw + processed artifacts on local NVMe (not in git).</li>
  </ul>
</section>

<footer>
  Built with <code>conlang.anchors</code>. {len(entries):,} rows, {sum(1 for e in entries if e.ipa):,} with IPA, {len({e.language_code for e in entries if e.language_code})} language codes.
</footer>
"""
    )


def write_showcase(entries: list[AnchorEntry], sigs: list[ConceptSignature], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_showcase(entries, sigs), encoding="utf-8")
    return path


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", type=Path, default=ANCHOR_PROCESSED / "anchors-v1.jsonl")
    ap.add_argument("--out", type=Path, default=ANCHOR_PROCESSED / "anchors-showcase.html")
    args = ap.parse_args()
    entries = read_jsonl(args.anchors)
    sigs = build_all_signatures(entries)
    p = write_showcase(entries, sigs, args.out)
    print(f"[showcase] {len(entries):,} entries -> {p}", flush=True)
    print(f"[showcase] file size: {p.stat().st_size / 1024:.1f} KiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
