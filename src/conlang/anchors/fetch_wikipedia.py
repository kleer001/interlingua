"""Phase-1 seed source: MediaWiki action=parse for a page → raw HTML + metadata.

Idempotent: if a file for the current revid already exists, the fetch is a
no-op apart from refreshing the meta file. Each fetch records the revid so
downstream parsers can prove which snapshot they ran against.

CLI:
    python -m conlang.anchors.fetch_wikipedia
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import ANCHOR_RAW

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "interlingua-anchors/0.1 "
    "(https://github.com/kleer001/interlingua; kleer001code@gmail.com) "
    "httpx"
)

# Phase-1 seed pages. Cross-linguistic_onomatopoeias is the main one; the
# others are kept on the list because the plan calls for filling concept gaps
# from related pages once the main scrape is in.
SEED_PAGES: tuple[str, ...] = ("Cross-linguistic_onomatopoeias",)


@dataclass
class FetchResult:
    page: str
    revid: int
    html_path: Path
    meta_path: Path
    bytes_html: int


def fetch_page(page: str, dest_dir: Path = ANCHOR_RAW / "wikipedia") -> FetchResult:
    """Pull `page` via MediaWiki action=parse. Writes `<slug>-rev<revid>.html`
    and `<slug>-rev<revid>.meta.json` to dest_dir.

    Headers include a contact User-Agent per the Wikimedia policy.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "action": "parse",
        "page": page,
        "format": "json",
        "formatversion": "2",
        "prop": "text|revid|displaytitle",
        "redirects": "1",
    }
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as cli:
        resp = cli.get(WIKIPEDIA_API, params=params)
        resp.raise_for_status()
        body = resp.json()

    if "error" in body:
        raise RuntimeError(f"MediaWiki API error fetching {page!r}: {body['error']}")
    parse = body["parse"]
    revid = int(parse["revid"])
    html = parse["text"]
    display_title = parse.get("displaytitle", page)

    slug = page.replace(" ", "_").replace("/", "__")
    html_path = dest_dir / f"{slug}-rev{revid}.html"
    meta_path = dest_dir / f"{slug}-rev{revid}.meta.json"

    if not html_path.exists():
        html_path.write_text(html, encoding="utf-8")
    meta = {
        "page": page,
        "display_title": display_title,
        "revid": revid,
        "url": f"https://en.wikipedia.org/wiki/{page}",
        "permalink": f"https://en.wikipedia.org/w/index.php?title={page}&oldid={revid}",
        "api_endpoint": WIKIPEDIA_API,
        "captured_at": datetime.now(timezone.utc).date().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return FetchResult(
        page=page,
        revid=revid,
        html_path=html_path,
        meta_path=meta_path,
        bytes_html=len(html.encode("utf-8")),
    )


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--page",
        action="append",
        help="Page title (may be passed multiple times). Default: SEED_PAGES.",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=ANCHOR_RAW / "wikipedia",
        help="Output directory (default: fauna anchoring/raw/wikipedia)",
    )
    args = ap.parse_args()
    pages = tuple(args.page) if args.page else SEED_PAGES
    for p in pages:
        r = fetch_page(p, args.dest)
        print(f"[ok] {r.page} rev{r.revid}  {r.bytes_html / 1024:.1f} KiB  → {r.html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
