"""Phase-3: fetch English-Wiktionary pages for onomatopoeic seed words.

For each seed (e.g. "woof", "moo", "meow") we hit MediaWiki action=parse on
en.wiktionary.org and cache the rendered HTML on fauna for offline parsing.
Per Wikimedia etiquette we throttle to one request every `min_delay_s`
seconds and send a contact User-Agent.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import ANCHOR_RAW
from .concepts import CONCEPTS

WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"
USER_AGENT = (
    "interlingua-anchors/0.1 "
    "(https://github.com/kleer001/interlingua; kleer001code@gmail.com) "
    "httpx"
)


@dataclass
class WiktFetchResult:
    seed: str
    revid: int | None
    html_path: Path
    meta_path: Path
    missing: bool  # True if page not found (404 / missingtitle)


def collect_unique_seeds() -> list[str]:
    """All distinct english_seeds across the concept inventory."""
    seen: set[str] = set()
    out: list[str] = []
    for c in CONCEPTS:
        for s in c.english_seeds:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _slug(seed: str) -> str:
    return seed.replace(" ", "_").replace("/", "__")


def fetch_seed(seed: str, dest_dir: Path = ANCHOR_RAW / "wiktionary") -> WiktFetchResult:
    """Hit MediaWiki action=parse for a single seed; cache HTML + meta.

    Returns missing=True if the page doesn't exist (some seeds like
    "pitter-patter" may not have their own Wiktionary entry — we record
    the gap and move on).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "action": "parse",
        "page": seed,
        "format": "json",
        "formatversion": "2",
        "prop": "text|revid|displaytitle",
        "redirects": "1",
    }
    with httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT}) as cli:
        resp = cli.get(WIKTIONARY_API, params=params)
        resp.raise_for_status()
        body = resp.json()

    slug = _slug(seed)
    if "error" in body:
        # missingtitle is the common case for non-existent pages
        code = body["error"].get("code", "")
        missing = code == "missingtitle"
        meta_path = dest_dir / f"{slug}-missing.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "missing": True,
                    "api_error": body["error"],
                    "captured_at": datetime.now(timezone.utc).date().isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return WiktFetchResult(
            seed=seed,
            revid=None,
            html_path=Path("/dev/null"),
            meta_path=meta_path,
            missing=missing or False,
        )

    parse = body["parse"]
    revid = int(parse["revid"])
    html = parse["text"]
    html_path = dest_dir / f"{slug}-rev{revid}.html"
    meta_path = dest_dir / f"{slug}-rev{revid}.meta.json"

    if not html_path.exists():
        html_path.write_text(html, encoding="utf-8")
    meta = {
        "seed": seed,
        "page": parse.get("title", seed),
        "display_title": parse.get("displaytitle", seed),
        "revid": revid,
        "url": f"https://en.wiktionary.org/wiki/{seed}",
        "permalink": f"https://en.wiktionary.org/w/index.php?title={seed}&oldid={revid}",
        "api_endpoint": WIKTIONARY_API,
        "captured_at": datetime.now(timezone.utc).date().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return WiktFetchResult(
        seed=seed, revid=revid, html_path=html_path, meta_path=meta_path, missing=False
    )


def _existing_cache_for(seed: str, dest_dir: Path) -> tuple[Path, Path] | None:
    slug = _slug(seed)
    htmls = sorted(dest_dir.glob(f"{slug}-rev*.html"))
    if not htmls:
        return None
    h = htmls[-1]
    m = h.with_suffix(".meta.json")
    return (h, m) if m.exists() else None


def fetch_all(
    seeds: list[str] | None = None,
    dest_dir: Path = ANCHOR_RAW / "wiktionary",
    min_delay_s: float = 6.0,
    refetch: bool = False,
) -> list[WiktFetchResult]:
    """Fetch every seed (or `seeds`), throttling between network calls.

    Cached seeds are skipped unless `refetch=True`. Returns one result per
    seed (whether it was network-fetched or cache-confirmed).
    """
    seeds = seeds or collect_unique_seeds()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[WiktFetchResult] = []
    last_fetch = 0.0
    for seed in seeds:
        cached = None if refetch else _existing_cache_for(seed, dest_dir)
        if cached is not None:
            html_path, meta_path = cached
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out.append(
                WiktFetchResult(
                    seed=seed,
                    revid=int(meta["revid"]),
                    html_path=html_path,
                    meta_path=meta_path,
                    missing=False,
                )
            )
            print(f"[cache] {seed} -> {html_path.name}", flush=True)
            continue
        wait = max(0.0, min_delay_s - (time.monotonic() - last_fetch))
        if wait > 0:
            time.sleep(wait)
        r = fetch_seed(seed, dest_dir)
        last_fetch = time.monotonic()
        if r.missing:
            print(f"[miss] {seed}", flush=True)
        else:
            print(f"[ok]   {seed} rev{r.revid}", flush=True)
        out.append(r)
    return out


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", action="append", help="Specific seed to fetch (repeatable)")
    ap.add_argument(
        "--dest",
        type=Path,
        default=ANCHOR_RAW / "wiktionary",
    )
    ap.add_argument("--delay", type=float, default=6.0, help="Min seconds between fetches")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    fetch_all(seeds=args.seed, dest_dir=args.dest, min_delay_s=args.delay, refetch=args.refetch)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
