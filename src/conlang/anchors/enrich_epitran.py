"""Phase-2 (redirected): fill the `ipa` field on AnchorEntry rows via Epitran.

Per anchor-data-plan.md §Tooling, the planned IPA pipeline is
    orthography  --Epitran-->  IPA  --PHOIBLE-->  articulatory features.

This module runs the first arrow on the merged anchor JSONL. Coverage is
limited by Epitran's bundled mapping files (~158 lang-script pairs as of
v1.35.1) — see EPITRAN_CODE_MAP below.

Notable gaps:
  - English (eng-Latn requires CMU Flite `lex_lookup` and isn't shipped).
  - Greek, Hebrew, Danish, Icelandic, Bulgarian, Macedonian.

For Mandarin / Cantonese (cmn-Latn, yue-Latn) Epitran expects *pinyin*
input, not Hanzi. We pass `romanization` instead of `orthography` for those.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from conlang.lab.schema import AnchorEntry, read_jsonl, write_jsonl

# BCP-47 short code -> Epitran (ISO 639-3 + ISO 15924) code.
# Built from the intersection of Epitran v1.35.1's bundled mappings and the
# language codes seen in our anchor data.
EPITRAN_CODE_MAP: dict[str, str] = {
    "af": "afr-Latn",
    "sq": "sqi-Latn",
    "ar": "ara-Arab",
    "az": "aze-Cyrl",
    "bn": "ben-Beng",
    "ca": "cat-Latn",
    "ceb": "ceb-Latn",
    "cmn": "cmn-Latn",  # consumes pinyin via romanization field
    "cs": "ces-Latn",
    "cy": "cym-Latn",
    "de": "deu-Latn",
    "es": "spa-Latn",
    "et": "est-Latn",
    "fa": "fas-Arab",
    "fi": "fin-Latn",
    "fil": "tgl-Latn",
    "fr": "fra-Latn",
    "ga": "gle-Latn",
    "gl": "glg-Latn",
    "hi": "hin-Deva",
    "hr": "hrv-Latn",
    "hu": "hun-Latn",
    "id": "ind-Latn",
    "it": "ita-Latn",
    "ja": "jpn-Kana",  # most Japanese onomatopoeia is in Katakana
    "ka": "kat-Geor",
    "kk": "kaz-Cyrl",
    "kn": "kan-Knda",
    "ko": "kor-Hang",
    "ky": "kir-Arab",
    "lt": "lit-Latn",
    "lv": "lav-Latn",
    "mar": "mar-Deva",
    "mr": "mar-Deva",
    "ml": "mal-Mlym",
    "ms": "msa-Latn",
    "nl": "nld-Latn",
    "pl": "pol-Latn",
    "pt": "por-Latn",
    "ro": "ron-Latn",
    "ru": "rus-Cyrl",
    "si": "sin-Sinh",
    "sl": "slv-Latn",
    "so": "som-Latn",
    "sr": "srp-Cyrl",
    "sv": "swe-Latn",
    "ta": "tam-Taml",
    "te": "tel-Telu",
    "th": "tha-Thai",
    "tl": "tgl-Latn",
    "tr": "tur-Latn",
    "uk": "ukr-Cyrl",
    "ur": "urd-Arab",
    "vi": "vie-Latn",
    "yue": "yue-Latn",  # consumes Jyutping-like romanization
}

# Languages where Epitran expects romanized input rather than the native
# script (e.g., pinyin/jyutping rather than Hanzi).
_USE_ROMANIZATION_INSTEAD = frozenset({"cmn", "yue"})


def _make_engine(epitran_code: str):
    """Lazy-construct an Epitran engine; let exceptions propagate to caller."""
    import epitran

    return epitran.Epitran(epitran_code)


def enrich(
    entries: Iterable[AnchorEntry],
    *,
    code_map: dict[str, str] = EPITRAN_CODE_MAP,
    overwrite: bool = False,
) -> tuple[list[AnchorEntry], dict[str, int]]:
    """Return (enriched_entries, stats) where stats counts hits/misses by code.

    `overwrite=False` keeps any IPA already on the row (Wikipedia gives us 48).
    """
    cache: dict[str, object] = {}
    stats: dict[str, int] = {
        "filled": 0,
        "kept": 0,
        "skipped_no_code": 0,
        "skipped_no_text": 0,
        "failed": 0,
    }
    out: list[AnchorEntry] = []
    for e in entries:
        if e.ipa and not overwrite:
            stats["kept"] += 1
            out.append(e)
            continue
        code = e.language_code
        ep_code = code_map.get(code) if code else None
        if not ep_code:
            stats["skipped_no_code"] += 1
            out.append(e)
            continue
        text = e.romanization or "" if code in _USE_ROMANIZATION_INSTEAD else (e.orthography or "")
        if not text:
            stats["skipped_no_text"] += 1
            out.append(e)
            continue
        try:
            if ep_code not in cache:
                cache[ep_code] = _make_engine(ep_code)
            engine = cache[ep_code]
            ipa = engine.transliterate(text)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            stats["failed"] += 1
            out.append(e)
            continue
        if not ipa or ipa == text:
            stats["failed"] += 1
            out.append(e)
            continue
        stats["filled"] += 1
        out.append(
            AnchorEntry(
                concept=e.concept,
                category=e.category,
                language=e.language,
                language_code=e.language_code,
                orthography=e.orthography,
                romanization=e.romanization,
                ipa=f"/{ipa}/",
                source=e.source,
                source_url=e.source_url,
                source_revid=e.source_revid,
                captured_at=e.captured_at,
                notes=e.notes,
                extra={**(e.extra or {}), "ipa_source": f"epitran:{ep_code}"},
            )
        )
    return out, stats


def run(in_path: Path, out_path: Path, overwrite: bool = False) -> dict[str, int]:
    entries = read_jsonl(in_path)
    print(f"[enrich] input rows: {len(entries)}  ({in_path})", file=sys.stderr)
    enriched, stats = enrich(entries, overwrite=overwrite)
    n = write_jsonl(enriched, out_path)
    print(f"[enrich] output rows: {n}  ({out_path})", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k:24} {v}", file=sys.stderr)
    return stats


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true", help="Replace existing IPA")
    args = ap.parse_args()
    run(args.in_path, args.out_path, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
