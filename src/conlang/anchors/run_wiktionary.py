"""Orchestrator for the Wiktionary seed pass (Phase 3 — concept expansion).

Fetches every unique english_seed across the concept inventory, throttles to
one request per second, parses each cached page, and writes:

    {ANCHOR_INTERIM}/wiktionary-rows.jsonl
    {ANCHOR_INTERIM}/wiktionary-rows.csv

CLI:
    python -m conlang.anchors.run_wiktionary           # fetch + parse
    python -m conlang.anchors.run_wiktionary --parse-only   # use cache
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from conlang.lab.schema import AnchorEntry, write_jsonl

from . import ANCHOR_INTERIM, ANCHOR_RAW
from .fetch_wiktionary import collect_unique_seeds, fetch_all
from .parse_wiktionary import parse_seed_file
from .run_seed import CSV_COLUMNS


def _write_csv(entries: list[AnchorEntry], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(CSV_COLUMNS) + ["seed", "gloss"]
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in entries:
            d = asdict(e)
            extra = d.pop("extra") or {}
            row = {k: d.get(k) for k in CSV_COLUMNS}
            row["seed"] = extra.get("seed")
            row["gloss"] = extra.get("gloss")
            w.writerow(row)
            n += 1
    return n


def run(parse_only: bool = False, delay: float = 6.0) -> tuple[Path, Path]:
    raw_dir = ANCHOR_RAW / "wiktionary"
    seeds = collect_unique_seeds()
    print(f"[run] {len(seeds)} unique seeds", file=sys.stderr)
    if not parse_only:
        fetch_all(seeds=seeds, dest_dir=raw_dir, min_delay_s=delay)

    # Parse every meta-html pair on disk (covers both fresh and cached).
    all_entries: list[AnchorEntry] = []
    n_pages = 0
    n_missing = 0
    for meta_path in sorted(raw_dir.glob("*.meta.json")):
        if "missing" in meta_path.name:
            n_missing += 1
            continue
        # find sibling html
        html_path = meta_path.with_suffix(".html")
        if not html_path.exists():
            # naming uses .meta.json suffix; strip .meta to get base
            stem = meta_path.name.replace(".meta.json", "")
            html_path = meta_path.with_name(f"{stem}.html")
        if not html_path.exists():
            continue
        n_pages += 1
        try:
            all_entries.extend(parse_seed_file(html_path, meta_path))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] parse failed for {html_path.name}: {e!r}", file=sys.stderr)

    # Dedup on (concept, language_code or language, orthography, romanization).
    seen: set[tuple] = set()
    deduped: list[AnchorEntry] = []
    for e in all_entries:
        k = (
            e.concept,
            e.language_code or e.language,
            e.orthography,
            e.romanization,
        )
        if k in seen:
            continue
        seen.add(k)
        deduped.append(e)

    out_dir = ANCHOR_INTERIM
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "wiktionary-rows.jsonl"
    csv_path = out_dir / "wiktionary-rows.csv"
    n_jsonl = write_jsonl(deduped, jsonl_path)
    n_csv = _write_csv(deduped, csv_path)
    print(
        f"[parse] pages={n_pages}  missing={n_missing}  "
        f"raw={len(all_entries)}  dedup={len(deduped)}",
        file=sys.stderr,
    )
    print(
        f"[write] {n_jsonl} rows -> {jsonl_path}\n[write] {n_csv} rows -> {csv_path}",
        file=sys.stderr,
    )
    concepts = Counter(e.concept for e in deduped)
    langs = Counter(e.language_code for e in deduped)
    print(f"[summary] distinct concepts covered: {len(concepts)}", file=sys.stderr)
    print(f"[summary] distinct language codes:   {len(langs)}", file=sys.stderr)
    print("Top 10 concepts by entry count:", file=sys.stderr)
    for c, n in concepts.most_common(10):
        print(f"  {c:32}  {n}", file=sys.stderr)
    return jsonl_path, csv_path


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--delay", type=float, default=6.0)
    args = ap.parse_args()
    run(parse_only=args.parse_only, delay=args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
