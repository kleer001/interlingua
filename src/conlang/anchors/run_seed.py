"""Phase-1 seed pipeline: fetch + parse Wikipedia → canonical anchors file.

Output:
    {ANCHOR_PROCESSED}/anchors-seed.jsonl
    {ANCHOR_PROCESSED}/anchors-seed.csv

Both files are deduped on (concept, language, orthography, romanization, ipa).

Re-running is safe — fetches are idempotent and the parser is pure on cached
files. Pass `--refetch` to force a new MediaWiki pull (picks up new revids).

CLI:
    python -m conlang.anchors.run_seed
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import ANCHOR_PROCESSED, ANCHOR_RAW
from .fetch_wikipedia import SEED_PAGES, fetch_page
from .parse_wikipedia import parse_file
from .schema import AnchorEntry, write_jsonl

CSV_COLUMNS: tuple[str, ...] = (
    "concept",
    "category",
    "language",
    "language_code",
    "orthography",
    "romanization",
    "ipa",
    "projected_form",
    "source",
    "source_url",
    "source_revid",
    "captured_at",
    "notes",
    "concept_label",
    "section_path",
)


def _latest_cached(page: str, raw_dir: Path) -> tuple[Path, Path] | None:
    """Find the highest-revid cached (html, meta) pair for `page`, or None."""
    slug = page.replace(" ", "_").replace("/", "__")
    htmls = sorted(raw_dir.glob(f"{slug}-rev*.html"))
    if not htmls:
        return None
    latest = htmls[-1]
    meta = latest.with_suffix(".meta.json")
    if not meta.exists():
        return None
    return latest, meta


def _dedupe(entries: list[AnchorEntry]) -> list[AnchorEntry]:
    seen: set[tuple] = set()
    out: list[AnchorEntry] = []
    for e in entries:
        k = (e.concept, e.language, e.orthography, e.romanization, e.ipa)
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def _write_csv(entries: list[AnchorEntry], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for e in entries:
            d = asdict(e)
            extra = d.pop("extra") or {}
            row = {k: d.get(k) for k in CSV_COLUMNS}
            row["concept_label"] = extra.get("concept_label")
            row["section_path"] = extra.get("section_path")
            w.writerow(row)
            n += 1
    return n


def _print_summary(entries: list[AnchorEntry]) -> None:
    concepts = Counter(e.concept for e in entries)
    langs = Counter(e.language for e in entries)
    cats = Counter(e.category for e in entries)
    print(f"[summary] entries={len(entries)}", file=sys.stderr)
    print(f"[summary] concepts={len(concepts)}", file=sys.stderr)
    print(f"[summary] languages={len(langs)}", file=sys.stderr)
    print(f"[summary] categories={len(cats)}", file=sys.stderr)
    print(f"[summary] entries_with_ipa={sum(1 for e in entries if e.ipa)}", file=sys.stderr)
    print(
        f"[summary] entries_with_romanization={sum(1 for e in entries if e.romanization)}",
        file=sys.stderr,
    )
    print(f"[summary] entries_with_notes={sum(1 for e in entries if e.notes)}", file=sys.stderr)
    no_code = [e for e in entries if not e.language_code]
    if no_code:
        bad_langs = sorted(set(e.language for e in no_code))
        print(
            f"[warn] {len(no_code)} entries missing language_code (languages: {bad_langs})",
            file=sys.stderr,
        )
    print(file=sys.stderr)
    print("Top 10 concepts by count:", file=sys.stderr)
    for c, n in concepts.most_common(10):
        print(f"  {c:32}  {n}", file=sys.stderr)
    print(file=sys.stderr)
    print("Top 10 languages by count:", file=sys.stderr)
    for lang, n in langs.most_common(10):
        print(f"  {lang:32}  {n}", file=sys.stderr)


def run(refetch: bool = False, dest: Path = ANCHOR_PROCESSED) -> tuple[Path, Path]:
    raw_dir = ANCHOR_RAW / "wikipedia"
    all_entries: list[AnchorEntry] = []
    for page in SEED_PAGES:
        if refetch or _latest_cached(page, raw_dir) is None:
            fr = fetch_page(page, raw_dir)
            html_path, meta_path = fr.html_path, fr.meta_path
        else:
            cached = _latest_cached(page, raw_dir)
            assert cached is not None
            html_path, meta_path = cached
        print(f"[parse] {page} <- {html_path.name}", file=sys.stderr)
        page_entries = parse_file(html_path, meta_path)
        all_entries.extend(page_entries)

    deduped = _dedupe(all_entries)
    dest.mkdir(parents=True, exist_ok=True)
    jsonl_path = dest / "anchors-seed.jsonl"
    csv_path = dest / "anchors-seed.csv"
    n_jsonl = write_jsonl(deduped, jsonl_path)
    n_csv = _write_csv(deduped, csv_path)
    print(
        f"[write] {n_jsonl} rows -> {jsonl_path}\n[write] {n_csv} rows -> {csv_path}",
        file=sys.stderr,
    )
    _print_summary(deduped)
    return jsonl_path, csv_path


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refetch", action="store_true", help="Force re-fetch of source pages")
    ap.add_argument(
        "--dest",
        type=Path,
        default=ANCHOR_PROCESSED,
        help="Output dir (default: fauna anchoring/processed)",
    )
    args = ap.parse_args()
    run(refetch=args.refetch, dest=args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
