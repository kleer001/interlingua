"""Phase 2+ fallback: pull IPA from each form's own Wiktionary page.

Epitran covers ~54 of our 84 languages; the rest (English, Greek, Hebrew,
Bulgarian, Macedonian, Danish, Icelandic, Norwegian, ...) leave 2,146 rows
without IPA. Many of those forms have their own en.wiktionary.org/wiki/{form}
page with a per-language Pronunciation section carrying an IPA span — we
harvest those here.

We fetch one cache file per unique orthography (regardless of how many
AnchorEntry rows it covers) to keep the network load manageable. To keep the
run finite we cap fetches at `max_forms` per invocation and prefer the most
frequent missing forms first.

Per global policy: 6 s minimum delay between requests.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import ANCHOR_PROCESSED, ANCHOR_RAW
from .enrich_epitran import EPITRAN_CODE_MAP
from .schema import read_jsonl

WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"
USER_AGENT = (
    "interlingua-anchors/0.1 "
    "(https://github.com/kleer001/interlingua; kleer001code@gmail.com) "
    "httpx"
)


@dataclass
class FormFetchResult:
    form: str
    revid: int | None
    html_path: Path
    meta_path: Path
    missing: bool


def _form_slug(form: str) -> str:
    return form.replace("/", "__").replace(" ", "_")


def fetch_form(form: str, dest_dir: Path = ANCHOR_RAW / "wiktionary_ipa") -> FormFetchResult:
    dest_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "action": "parse",
        "page": form,
        "format": "json",
        "formatversion": "2",
        "prop": "text|revid|displaytitle",
        "redirects": "1",
    }
    # Retry transient network failures (e.g., flaky DNS in the sandbox).
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(
                timeout=60.0, headers={"User-Agent": USER_AGENT}
            ) as cli:
                resp = cli.get(WIKTIONARY_API, params=params)
                resp.raise_for_status()
                body = resp.json()
            break
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_err = e
            time.sleep(2 ** attempt * 2.0)  # 2s, 4s, 8s
    else:
        raise RuntimeError(f"network failure after retries fetching {form!r}") from last_err

    slug = _form_slug(form)
    if "error" in body:
        meta_path = dest_dir / f"{slug}-missing.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "form": form,
                    "missing": True,
                    "api_error": body["error"],
                    "captured_at": datetime.now(timezone.utc).date().isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return FormFetchResult(
            form=form,
            revid=None,
            html_path=Path("/dev/null"),
            meta_path=meta_path,
            missing=True,
        )

    parse = body["parse"]
    revid = int(parse["revid"])
    html = parse["text"]
    html_path = dest_dir / f"{slug}-rev{revid}.html"
    meta_path = dest_dir / f"{slug}-rev{revid}.meta.json"
    if not html_path.exists():
        html_path.write_text(html, encoding="utf-8")
    meta = {
        "form": form,
        "page": parse.get("title", form),
        "revid": revid,
        "url": f"https://en.wiktionary.org/wiki/{form}",
        "api_endpoint": WIKTIONARY_API,
        "captured_at": datetime.now(timezone.utc).date().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return FormFetchResult(
        form=form, revid=revid, html_path=html_path, meta_path=meta_path, missing=False
    )


def _existing_cache(form: str, dest_dir: Path) -> tuple[Path, Path] | None:
    slug = _form_slug(form)
    if (dest_dir / f"{slug}-missing.meta.json").exists():
        return None  # known missing; don't retry
    htmls = sorted(dest_dir.glob(f"{slug}-rev*.html"))
    if not htmls:
        return None
    h = htmls[-1]
    m = h.with_suffix(".meta.json")
    return (h, m) if m.exists() else None


def select_target_forms(
    anchors_jsonl: Path,
    *,
    max_forms: int | None,
    min_form_length: int = 2,
) -> list[tuple[str, int]]:
    """Pick orthographies to fetch IPA for: rows missing IPA whose language
    isn't covered by Epitran, ranked by how many AnchorEntry rows would
    benefit from the lookup.
    """
    entries = read_jsonl(anchors_jsonl)
    candidates = Counter()
    for e in entries:
        if e.ipa:
            continue
        if not e.orthography:
            continue
        if len(e.orthography) < min_form_length:
            continue
        # Prioritize forms whose language Epitran can't transliterate.
        if e.language_code in EPITRAN_CODE_MAP:
            continue
        candidates[e.orthography] += 1
    ranked = candidates.most_common()
    if max_forms is not None:
        ranked = ranked[:max_forms]
    return ranked


def fetch_targets(
    targets: list[tuple[str, int]],
    *,
    dest_dir: Path = ANCHOR_RAW / "wiktionary_ipa",
    min_delay_s: float = 6.0,
    refetch: bool = False,
) -> list[FormFetchResult]:
    results: list[FormFetchResult] = []
    last_fetch = 0.0
    for form, count in targets:
        cached = None if refetch else _existing_cache(form, dest_dir)
        if cached is not None:
            html_path, meta_path = cached
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            results.append(
                FormFetchResult(
                    form=form,
                    revid=int(meta["revid"]),
                    html_path=html_path,
                    meta_path=meta_path,
                    missing=False,
                )
            )
            print(f"[cache] {form!r} ({count} rows)", flush=True)
            continue
        wait = max(0.0, min_delay_s - (time.monotonic() - last_fetch))
        if wait > 0:
            time.sleep(wait)
        r = fetch_form(form, dest_dir)
        last_fetch = time.monotonic()
        if r.missing:
            print(f"[miss]  {form!r} ({count} rows)", flush=True)
        else:
            print(f"[ok]    {form!r} rev{r.revid} ({count} rows)", flush=True)
        results.append(r)
    return results


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--anchors",
        type=Path,
        default=ANCHOR_PROCESSED / "anchors-v1.jsonl",
    )
    ap.add_argument("--dest", type=Path, default=ANCHOR_RAW / "wiktionary_ipa")
    ap.add_argument(
        "--max-forms",
        type=int,
        default=300,
        help="Cap on unique forms fetched per run (None = no cap)",
    )
    ap.add_argument("--delay", type=float, default=6.0)
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    targets = select_target_forms(args.anchors, max_forms=args.max_forms)
    print(
        f"[plan] {len(targets)} unique forms to fetch "
        f"(top by row-coverage); ~{len(targets) * args.delay / 60:.1f} min at {args.delay}s/req",
        flush=True,
    )
    fetch_targets(targets, dest_dir=args.dest, min_delay_s=args.delay, refetch=args.refetch)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
