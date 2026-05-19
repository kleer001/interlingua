"""Merge per-source AnchorEntry JSONLs into a single anchors-v1.jsonl.

Inputs (defaults to the canonical paths on fauna):
    {ANCHOR_PROCESSED}/anchors-seed.jsonl       (Wikipedia, Phase 1)
    {ANCHOR_INTERIM}/wiktionary-rows.jsonl       (Wiktionary, Phase 3)

Output:
    {ANCHOR_PROCESSED}/anchors-merged.jsonl     (combined, no IPA enrichment)
    {ANCHOR_PROCESSED}/anchors-v1.jsonl         (after Epitran enrichment)
    {ANCHOR_PROCESSED}/anchors-v1.csv

During merge we:
  - Canonicalize each row's `concept` via the inventory (Wikipedia's
    `dog_or_wolf_howling` -> `dog_howling`, etc.).
  - Dedup on (concept, language_code or language, orthography, romanization).
  - Preserve provenance: the same canonical row may come from multiple sources;
    we keep the first occurrence and record the others as `extra.also_in`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from conlang.lab.concepts import canonical_slug
from conlang.lab.schema import AnchorEntry, read_jsonl, write_jsonl

from . import ANCHOR_INTERIM, ANCHOR_PROCESSED, ANCHOR_RAW
from .enrich_epitran import run as run_enrich
from .parse_wiktionary_ipa import apply_lookup, build_ipa_lookup
from .run_seed import CSV_COLUMNS


def _canonicalize(entry: AnchorEntry) -> AnchorEntry | None:
    """Map the entry's concept to its canonical inventory slug.

    Returns None if the concept isn't in the inventory (drop these — they
    indicate the source mentioned a concept we haven't decided to track).
    """
    new = canonical_slug(entry.concept)
    if new is None:
        return None
    if new == entry.concept:
        return entry
    return AnchorEntry(
        concept=new,
        category=entry.category,
        language=entry.language,
        language_code=entry.language_code,
        orthography=entry.orthography,
        romanization=entry.romanization,
        ipa=entry.ipa,
        source=entry.source,
        source_url=entry.source_url,
        source_revid=entry.source_revid,
        captured_at=entry.captured_at,
        notes=entry.notes,
        extra={**(entry.extra or {}), "source_concept": entry.concept},
    )


def merge_streams(
    streams: dict[str, Iterable[AnchorEntry]],
) -> tuple[list[AnchorEntry], dict[str, int]]:
    """`streams` is {source_name: iterable_of_entries}. Returns merged list."""
    seen: dict[tuple, AnchorEntry] = {}
    stats: dict[str, int] = {f"in_{n}": 0 for n in streams}
    stats["dropped_unknown_concept"] = 0
    stats["dedup_merged_provenance"] = 0
    for name, stream in streams.items():
        for e in stream:
            stats[f"in_{name}"] += 1
            canon = _canonicalize(e)
            if canon is None:
                stats["dropped_unknown_concept"] += 1
                continue
            # Dedup on (concept, language, orthography). Romanization and IPA
            # are derivations of the form, not distinguishing features — a row
            # with a romanization should subsume one without if the form matches.
            key = (
                canon.concept,
                canon.language_code or canon.language,
                canon.orthography,
            )
            if key in seen:
                stats["dedup_merged_provenance"] += 1
                existing = seen[key]
                also = existing.extra.setdefault("also_in", [])
                also.append({"source": canon.source, "source_url": canon.source_url})
                # Backfill IPA if we have it now but didn't before.
                if not existing.ipa and canon.ipa:
                    existing.ipa = canon.ipa
                # Pick up a romanization or notes if missing.
                if not existing.romanization and canon.romanization:
                    existing.romanization = canon.romanization
                if not existing.notes and canon.notes:
                    existing.notes = canon.notes
                continue
            seen[key] = canon
    stats["out_merged"] = len(seen)
    return list(seen.values()), stats


def synthesize_english_anchors(
    entries: list[AnchorEntry],
) -> tuple[list[AnchorEntry], int]:
    """For each canonical concept that lacks an English row, synthesize one
    from the inventory's first ``english_seed``. Wikipedia's wide table only
    covers ~50 concepts, so most of the 114-concept inventory (water sounds,
    weather, impacts, mechanical, exclamations) had no English entry until
    this step. Provenance: ``source = "inventory-english-seed"``.

    IPA and projection are filled later by the Wiktionary-form-page lookup
    and the projection step.
    """
    from conlang.lab.concepts import CONCEPTS

    have_en = {e.concept for e in entries if e.language_code == "en" and e.orthography}
    today = entries[0].captured_at if entries else "2026-05-16"
    added = 0
    for c in CONCEPTS:
        if c.slug in have_en:
            continue
        if not c.english_seeds:
            continue
        for idx, seed in enumerate(c.english_seeds):
            entries.append(
                AnchorEntry(
                    concept=c.slug,
                    category=c.category,
                    language="English",
                    language_code="en",
                    orthography=seed,
                    romanization=None,
                    ipa=None,
                    source="inventory-english-seed",
                    source_url="",
                    source_revid=None,
                    captured_at=today,
                    notes=None,
                    extra={"seed_origin": f"concepts.english_seeds[{idx}]"},
                )
            )
            added += 1
    return entries, added


def _write_csv(entries: list[AnchorEntry], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(CSV_COLUMNS) + ["source_concept", "seed", "ipa_source"]
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in entries:
            d = asdict(e)
            extra = d.pop("extra") or {}
            row = {k: d.get(k) for k in CSV_COLUMNS}
            row["source_concept"] = extra.get("source_concept")
            row["seed"] = extra.get("seed")
            row["ipa_source"] = extra.get("ipa_source")
            w.writerow(row)
            n += 1
    return n


def run(
    wiki_jsonl: Path = ANCHOR_PROCESSED / "anchors-seed.jsonl",
    wikt_jsonl: Path = ANCHOR_INTERIM / "wiktionary-rows.jsonl",
    merged_jsonl: Path = ANCHOR_PROCESSED / "anchors-merged.jsonl",
    v1_jsonl: Path = ANCHOR_PROCESSED / "anchors-v1.jsonl",
    v1_csv: Path = ANCHOR_PROCESSED / "anchors-v1.csv",
    wiktionary_ipa_cache: Path = ANCHOR_RAW / "wiktionary_ipa",
    skip_enrich: bool = False,
    skip_wiktionary_ipa: bool = False,
) -> dict[str, int]:
    streams: dict[str, Iterable[AnchorEntry]] = {}
    if wiki_jsonl.exists():
        streams["wikipedia"] = read_jsonl(wiki_jsonl)
    else:
        print(f"[warn] missing {wiki_jsonl}", file=sys.stderr)
    if wikt_jsonl.exists():
        streams["wiktionary"] = read_jsonl(wikt_jsonl)
    else:
        print(f"[warn] missing {wikt_jsonl}", file=sys.stderr)
    merged, stats = merge_streams(streams)

    # Synthesize an English row from the inventory's first english_seed for
    # any canonical concept lacking one. Wikipedia's wide table doesn't
    # cover every concept we now track, so without this step the matrix
    # has English-shaped holes for water sounds, weather, impacts, etc.
    merged, n_synth = synthesize_english_anchors(merged)
    stats["synthesized_english"] = n_synth

    n_merged = write_jsonl(merged, merged_jsonl)
    print(f"[merge] {n_merged} -> {merged_jsonl}", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k:32} {v}", file=sys.stderr)

    if skip_enrich:
        return stats

    enrich_stats = run_enrich(merged_jsonl, v1_jsonl, overwrite=False)
    stats.update({f"enrich_{k}": v for k, v in enrich_stats.items()})

    # Wiktionary IPA fallback for anything Epitran couldn't handle.
    if not skip_wiktionary_ipa and wiktionary_ipa_cache.exists():
        lookup = build_ipa_lookup(wiktionary_ipa_cache)
        if lookup:
            entries = read_jsonl(v1_jsonl)
            filled, ipa_stats = apply_lookup(entries, lookup)
            write_jsonl(filled, v1_jsonl)
            print(
                f"[wiktionary-ipa] cache pairs={len(lookup)} "
                f"filled={ipa_stats['filled']} kept={ipa_stats['kept']} "
                f"no_match={ipa_stats['no_match']}",
                file=sys.stderr,
            )
            stats.update({f"wiktipa_{k}": v for k, v in ipa_stats.items()})

    # Project every IPA row into the 10C/5V inventory. Self-contained
    # representation that downstream Stage-6 reads as the matrix.
    enriched = read_jsonl(v1_jsonl)
    from conlang.lab.project import project_ipa

    n_proj = 0
    for e in enriched:
        if e.ipa:
            p = project_ipa(e.ipa)
            if p:
                e.projected_form = p
                n_proj += 1
    write_jsonl(enriched, v1_jsonl)
    print(f"[project] {n_proj} rows projected to inventory", file=sys.stderr)
    stats["projected"] = n_proj

    n_csv = _write_csv(enriched, v1_csv)
    print(f"[csv]   {n_csv} -> {v1_csv}", file=sys.stderr)
    return stats


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki", type=Path, default=ANCHOR_PROCESSED / "anchors-seed.jsonl")
    ap.add_argument("--wikt", type=Path, default=ANCHOR_INTERIM / "wiktionary-rows.jsonl")
    ap.add_argument("--merged", type=Path, default=ANCHOR_PROCESSED / "anchors-merged.jsonl")
    ap.add_argument("--v1", type=Path, default=ANCHOR_PROCESSED / "anchors-v1.jsonl")
    ap.add_argument("--v1-csv", type=Path, default=ANCHOR_PROCESSED / "anchors-v1.csv")
    ap.add_argument("--skip-enrich", action="store_true", help="Skip the Epitran step")
    ap.add_argument(
        "--skip-wiktionary-ipa",
        action="store_true",
        help="Skip the Wiktionary-form-page IPA fallback",
    )
    ap.add_argument(
        "--wiktionary-ipa-cache",
        type=Path,
        default=ANCHOR_RAW / "wiktionary_ipa",
    )
    args = ap.parse_args()
    run(
        wiki_jsonl=args.wiki,
        wikt_jsonl=args.wikt,
        merged_jsonl=args.merged,
        v1_jsonl=args.v1,
        v1_csv=args.v1_csv,
        wiktionary_ipa_cache=args.wiktionary_ipa_cache,
        skip_enrich=args.skip_enrich,
        skip_wiktionary_ipa=args.skip_wiktionary_ipa,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
