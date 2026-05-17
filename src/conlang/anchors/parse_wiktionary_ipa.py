"""Pull per-language IPA strings out of a cached Wiktionary form page.

The page is structured as <h2>Language Name</h2> -> [...sections...] ->
<span class="IPA">/xxx/</span>. IPA spans lack a `lang=` attribute on this
site, so we attribute them to the most-recent preceding <h2>.

We skip:
- the "Translingual" pseudo-section (no language to bind to)
- rhyme-suffix spans (start with "-", e.g. <span class="IPA">-iː</span>)
- IPA-shaped strings that are clearly nicknames or part of an enPR/IPAchar
  span (we identify the canonical Pronunciation IPA by requiring the
  surrounding chars to be / or [).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from .lang_codes import code_for


def _is_canonical_ipa(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if s.startswith("-"):
        return False
    return (s.startswith("/") and "/" in s[1:]) or (s.startswith("[") and "]" in s)


def parse_form_html(html: str) -> dict[str, list[str]]:
    """Return `{language_code: [ipa, ...]}` for one cached Wiktionary form page.

    Multiple IPA strings per language are preserved (regional variants).
    Languages without an ISO code mapping are dropped silently.
    """
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, list[str]] = {}
    current_lang: str | None = None
    for el in soup.find_all(["h2", "span"]):
        if not isinstance(el, Tag):
            continue
        if el.name == "h2":
            text = el.get_text(" ", strip=True)
            # Strip [edit] suffix
            text = text.split("[")[0].strip()
            if text in ("Translingual", "Contents", "References", ""):
                current_lang = None
            else:
                current_lang = text
            continue
        if el.name == "span":
            classes = el.get("class") or []
            if "IPA" not in classes:
                continue
            if current_lang is None:
                continue
            ipa = el.get_text(strip=True)
            if not _is_canonical_ipa(ipa):
                continue
            code = code_for(current_lang)
            if not code:
                continue
            out.setdefault(code, []).append(ipa)
    return out


def parse_form_file(html_path: Path) -> dict[str, list[str]]:
    return parse_form_html(html_path.read_text(encoding="utf-8"))


def build_ipa_lookup(
    cache_dir: Path,
) -> dict[tuple[str, str], str]:
    """Walk the cache, return `{(form, lang_code): first_ipa}` mapping.

    `form` is the un-slugified orthography (taken from the meta file's
    `form` field). When multiple IPAs are listed for a language we keep the
    first canonical one.
    """
    lookup: dict[tuple[str, str], str] = {}
    for meta_path in sorted(cache_dir.glob("*.meta.json")):
        if "missing" in meta_path.name:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        form = meta.get("form")
        if not form:
            continue
        html_path = meta_path.with_suffix(".html")
        if not html_path.exists():
            stem = meta_path.name.replace(".meta.json", "")
            html_path = meta_path.with_name(f"{stem}.html")
        if not html_path.exists():
            continue
        try:
            per_lang = parse_form_file(html_path)
        except Exception:  # noqa: BLE001
            continue
        for code, ipas in per_lang.items():
            if not ipas:
                continue
            key = (form, code)
            if key not in lookup:
                lookup[key] = ipas[0]
    return lookup


def _lookup_with_variants(
    form: str, lang_code: str, lookup: dict[tuple[str, str], str]
) -> str | None:
    """Try the exact (form, lang) key first, then hyphenated/concatenated
    variants for multi-word forms (so "cock a doodle doo" matches the
    cock-a-doodle-doo page)."""
    direct = lookup.get((form, lang_code))
    if direct:
        return direct
    variants: list[str] = []
    if " " in form:
        variants.append(form.replace(" ", "-"))
        words = form.split()
        if len(words) >= 2 and len(set(words)) == 1:
            variants.append(words[0])
        variants.append(form.replace(" ", ""))
    if "-" in form and " " not in form:
        variants.append(form.replace("-", " "))
        variants.append(form.replace("-", ""))
    if " " not in form and "-" not in form and len(form) >= 4 and len(form) % 2 == 0:
        half = len(form) // 2
        if form[:half] == form[half:]:
            variants.append(f"{form[:half]}-{form[:half]}")
            variants.append(form[:half])
    for v in variants:
        hit = lookup.get((v, lang_code))
        if hit:
            return hit
    return None


def apply_lookup(
    entries: Iterable,
    lookup: dict[tuple[str, str], str],
    *,
    overwrite: bool = False,
) -> tuple[list, dict[str, int]]:
    """Backfill `ipa` on AnchorEntry rows using the lookup.

    Returns (entries_out, stats).
    """
    from .schema import AnchorEntry

    stats = {"filled": 0, "kept": 0, "no_match": 0, "skipped_no_form": 0}
    out: list[AnchorEntry] = []
    for e in entries:
        if e.ipa and not overwrite:
            stats["kept"] += 1
            out.append(e)
            continue
        if not e.orthography or not e.language_code:
            stats["skipped_no_form"] += 1
            out.append(e)
            continue
        ipa = _lookup_with_variants(e.orthography, e.language_code, lookup)
        if not ipa:
            stats["no_match"] += 1
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
                ipa=ipa,
                source=e.source,
                source_url=e.source_url,
                source_revid=e.source_revid,
                captured_at=e.captured_at,
                notes=e.notes,
                extra={**(e.extra or {}), "ipa_source": "wiktionary-form-page"},
            )
        )
    return out, stats
