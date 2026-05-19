"""Parse a cached Wikipedia HTML page into AnchorEntry rows.

The page uses `<table class="wikitable">` with languages as rows and concept
columns as `<th>`. Cells use a rich, machine-friendly markup:

    <span title="Japanese-language text"><i lang="ja">native</i></span>
    (<span title="Japanese-language romanization"><i lang="ja-Latn">roman</i></span>)
    <span class="IPA" lang="ja-Latn-fonipa"><a>...</a></span>

So we extract:
- orthography     <- native-text span (or bare <i> for Latin-script langs)
- romanization    <- romanization span (or bare <i lang="xx-Latn">)
- ipa             <- IPA span text
- language_code   <- `lang` attr (preferred); else our name->code dict

Entries within a cell are separated by top-level commas (commas inside
parentheses that wrap a romanization span are NOT entry separators). When a
new native-text or romanization span appears while the current entry already
has that field, we flush the current entry first (handles multi-form cells
that use spaces instead of commas as separators, as in some Arabic rows).

Heuristic, not exhaustive. Cells that do not fit the patterns above are still
emitted with a best-effort `orthography` and the residual text as `notes`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from conlang.lab.schema import AnchorEntry

from .lang_codes import code_for, normalize_language_name

CONCEPT_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_concept(label: str) -> str:
    s = label.strip().lower()
    s = CONCEPT_SLUG_RE.sub("_", s)
    return s.strip("_")


def _headings_above(table: Tag) -> list[str]:
    """Return section headings above `table`, outermost-first (h2 -> h3 -> h4)."""
    found: dict[str, str] = {}
    node = table
    while True:
        node = node.find_previous(["h2", "h3", "h4"])
        if node is None:
            break
        tag = node.name
        if tag in found:
            continue
        text = node.get_text(" ", strip=True)
        text = re.sub(r"\s*\[\s*edit\s*\]\s*$", "", text, flags=re.IGNORECASE)
        if text in ("Contents",):
            continue
        found[tag] = text
        if "h2" in found and "h3" in found and "h4" in found:
            break
    return [found[t] for t in ("h2", "h3", "h4") if t in found]


@dataclass
class _Pending:
    orthography: str | None = None
    romanization: str | None = None
    ipa: str | None = None
    language_code: str | None = None
    literal_text: str = ""

    def is_empty(self) -> bool:
        return not (self.orthography or self.romanization or self.ipa)

    def finalize_notes(self) -> str | None:
        t = self.literal_text
        for _ in range(3):
            t = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        t = t.strip("()[]{}.;:&-, \t\n")
        return t or None


def _classify_tag(tag: Tag) -> dict:
    """Return a dict of field contributions from one tag.

    Keys may include: orthography, romanization, ipa, language_code, literal_text.
    """
    name = (tag.name or "").lower()
    out: dict = {}
    if name == "span":
        title = (tag.get("title") or "").strip()
        classes = tag.get("class") or []
        if "IPA" in classes:
            out["ipa"] = tag.get_text(strip=True)
            return out
        if title.endswith("-language text"):
            inner = tag.find(["i", "span"]) or tag
            out["orthography"] = inner.get_text(strip=True)
            inner_lang = inner.get("lang") if isinstance(inner, Tag) else None
            lang = inner_lang or tag.get("lang")
            if lang:
                out["language_code"] = lang
            return out
        if title.endswith("-language romanization"):
            inner = tag.find(["i", "span"]) or tag
            out["romanization"] = inner.get_text(strip=True)
            inner_lang = inner.get("lang") if isinstance(inner, Tag) else None
            lang = (inner_lang or tag.get("lang") or "").strip()
            if lang:
                out["language_code"] = lang.split("-")[0]
            return out
        out["literal_text"] = " " + tag.get_text(" ", strip=True)
        return out
    if name == "i":
        lang = (tag.get("lang") or "").strip()
        text = tag.get_text(strip=True)
        if lang.endswith("-Latn"):
            out["romanization"] = text
            out["language_code"] = lang.split("-")[0]
        else:
            out["orthography"] = text
            if lang:
                out["language_code"] = lang
        return out
    if name in ("a", "b", "small", "br", "wbr"):
        out["literal_text"] = " " + tag.get_text(" ", strip=True)
        return out
    out["literal_text"] = " " + tag.get_text(" ", strip=True)
    return out


def _would_overwrite(cur: _Pending, contrib: dict) -> bool:
    for k in ("orthography", "romanization", "ipa"):
        if k in contrib and getattr(cur, k):
            return True
    return False


def _apply_contrib(cur: _Pending, contrib: dict) -> None:
    if "literal_text" in contrib:
        cur.literal_text += contrib["literal_text"]
    if "language_code" in contrib and not cur.language_code:
        cur.language_code = contrib["language_code"]
    for k in ("orthography", "romanization", "ipa"):
        if k in contrib:
            setattr(cur, k, contrib[k])


def _split_inline_alternatives(p: _Pending) -> list[_Pending]:
    """If `p.orthography` or `p.romanization` contains a top-level comma,
    fan out into multiple _Pending entries sharing the same language_code.
    Top-level comma = comma outside any parens.
    """

    def split_on_top_commas(s: str) -> list[str]:
        out: list[str] = []
        depth = 0
        buf: list[str] = []
        for c in s:
            if c in "([{":
                depth += 1
                buf.append(c)
            elif c in ")]}":
                depth = max(0, depth - 1)
                buf.append(c)
            elif c == "," and depth == 0:
                out.append("".join(buf).strip())
                buf = []
            else:
                buf.append(c)
        tail = "".join(buf).strip()
        if tail:
            out.append(tail)
        return [x for x in out if x]

    ortho_parts = split_on_top_commas(p.orthography) if p.orthography else [None]
    roman_parts = split_on_top_commas(p.romanization) if p.romanization else [None]

    if len([x for x in ortho_parts if x]) > 1 and (
        p.romanization is None or len(roman_parts) == len(ortho_parts)
    ):
        out = []
        for j, op in enumerate(ortho_parts):
            out.append(
                _Pending(
                    orthography=op,
                    romanization=(roman_parts[j] if p.romanization else None),
                    ipa=p.ipa if j == 0 else None,
                    language_code=p.language_code,
                    literal_text=p.literal_text if j == 0 else "",
                )
            )
        return out
    if p.orthography is None and len([x for x in roman_parts if x]) > 1:
        out = []
        for j, rp in enumerate(roman_parts):
            out.append(
                _Pending(
                    orthography=None,
                    romanization=rp,
                    ipa=p.ipa if j == 0 else None,
                    language_code=p.language_code,
                    literal_text=p.literal_text if j == 0 else "",
                )
            )
        return out
    return [p]


def parse_cell(cell: Tag, fallback_language_code: str | None = None) -> list[_Pending]:
    """Walk a `<td>` and emit one _Pending per alternative form."""
    for sup in cell.find_all("sup"):
        sup.extract()
    for style in cell.find_all(["style", "script"]):
        style.extract()

    out: list[_Pending] = []
    cur = _Pending()
    paren_depth = 0
    # Snapshot of cur.literal_text at the moment paren_depth went 0->1, so we
    # can recover orthography from cells written as: NATIVE_SCRIPT (<i>roman</i>)
    # with no <span title="X-language text"> wrapper around the native part.
    pre_paren_text: str | None = None

    def flush() -> None:
        nonlocal cur, pre_paren_text
        if not cur.is_empty():
            if not cur.language_code:
                cur.language_code = fallback_language_code
            out.extend(_split_inline_alternatives(cur))
        cur = _Pending()
        pre_paren_text = None

    for child in cell.children:
        if isinstance(child, NavigableString):
            text = str(child)
            for c in text:
                if c in "([":
                    if paren_depth == 0:
                        pre_paren_text = cur.literal_text
                    paren_depth += 1
                    cur.literal_text += c
                elif c in ")]":
                    paren_depth = max(0, paren_depth - 1)
                    if paren_depth == 0:
                        pre_paren_text = None
                    cur.literal_text += c
                elif (c == "," or c == ";") and paren_depth == 0:
                    flush()
                else:
                    cur.literal_text += c
        elif isinstance(child, Tag):
            contrib = _classify_tag(child)
            # Case-2 recovery: bare <i> inside parens is a romanization, not
            # a new orthography. Pair it with either the already-set
            # orthography (1a) or the text we accumulated before the paren (1b).
            if "orthography" in contrib and "romanization" not in contrib and paren_depth > 0:
                if cur.orthography:
                    contrib["romanization"] = contrib.pop("orthography")
                elif pre_paren_text is not None:
                    pp = pre_paren_text.strip().strip("()[]{}.;:&-, \t\n")
                    if pp:
                        cur.orthography = pp
                        if cur.literal_text.startswith(pre_paren_text):
                            cur.literal_text = cur.literal_text[len(pre_paren_text) :]
                        contrib["romanization"] = contrib.pop("orthography")
            if _would_overwrite(cur, contrib):
                flush()
            _apply_contrib(cur, contrib)

    flush()
    return [p for p in out if not p.is_empty()]


def parse_html(
    html: str,
    *,
    source: str,
    source_url: str,
    source_revid: str | None,
    captured_at: str,
) -> list[AnchorEntry]:
    soup = BeautifulSoup(html, "lxml")
    entries: list[AnchorEntry] = []
    for table in soup.find_all("table", class_="wikitable"):
        headings = _headings_above(table)
        category = headings[-1] if headings else ""
        rows = table.find_all("tr")
        if not rows:
            continue
        header_cells = rows[0].find_all(["th", "td"])
        col_labels = [c.get_text(" ", strip=True) for c in header_cells]
        if len(col_labels) < 2 or col_labels[0].lower() != "language":
            continue
        concept_labels = col_labels[1:]
        concept_slugs = [slugify_concept(c) for c in concept_labels]

        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            raw_lang = cells[0].get_text(" ", strip=True)
            lang_name = normalize_language_name(raw_lang)
            if not lang_name:
                continue
            lang_code = code_for(lang_name)
            for j, c in enumerate(cells[1:]):
                if j >= len(concept_slugs):
                    break
                parsed = parse_cell(c, fallback_language_code=lang_code)
                for p in parsed:
                    entries.append(
                        AnchorEntry(
                            concept=concept_slugs[j],
                            category=category,
                            language=lang_name,
                            language_code=(p.language_code or lang_code or None),
                            orthography=(p.orthography or p.romanization or ""),
                            romanization=(
                                p.romanization if (p.orthography and p.romanization) else None
                            ),
                            ipa=p.ipa,
                            source=source,
                            source_url=source_url,
                            source_revid=source_revid,
                            captured_at=captured_at,
                            notes=p.finalize_notes(),
                            extra={
                                "concept_label": concept_labels[j],
                                "section_path": " / ".join(headings),
                            },
                        )
                    )
    return entries


def parse_file(html_path: Path, meta_path: Path) -> list[AnchorEntry]:
    """Parse an html file plus its sidecar meta.json (from fetch_wikipedia)."""
    html = html_path.read_text(encoding="utf-8")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return parse_html(
        html,
        source=f"wikipedia:{meta['page']}",
        source_url=meta["url"],
        source_revid=str(meta["revid"]),
        captured_at=meta["captured_at"],
    )
