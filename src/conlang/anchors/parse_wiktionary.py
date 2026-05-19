"""Parse a cached Wiktionary HTML page into AnchorEntry rows.

Wiktionary uses well-structured markup:
- Each translation block is `<div class="NavFrame">` with a `<div class="NavHead">`
  carrying the sense gloss ("the sound of a dog barking").
- Inside, `<table class="translations">` contains a `<ul>` of `<li>` items,
  one per language.
- Each `<li>` text looks like
      "Arabic: <span class="Arab" lang="ar">native</span>
              (<span class="tr Latn" lang="ar-Latn">roman</span>)"
- Some `<li>`s wrap multiple sub-language entries in nested `<dl><dd>` (e.g.,
  Chinese splits into Mandarin / Cantonese).

We bind each translation table to one or more inventory concepts by matching
the NavHead gloss against per-concept keywords (slug words, description
words, alias words — minus the seed itself and stop-words).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from conlang.lab.concepts import CONCEPTS, ConceptDef
from conlang.lab.schema import AnchorEntry

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "in",
        "on",
        "to",
        "for",
        "with",
        "as",
        "is",
        "by",
        "at",
        "from",
        "that",
        "this",
        "these",
        "those",
        "be",
        "being",
        "been",
        "made",
        "make",
        "makes",
        "sound",
        "noise",
        "etc",
        "etc.",
    }
)

_KEYWORD_RE = re.compile(r"[a-z][a-z\-]+")


def _concept_keywords(c: ConceptDef) -> frozenset[str]:
    """Derive matching keywords for a concept.

    Avoids using the english_seeds themselves so that the "(noun) loud
    rumor" sense of *buzz* doesn't grab the "bee_buzzing" concept just
    because the lemma equals the seed.
    """
    words: set[str] = set()
    for source in (
        c.slug,
        c.description,
        *(a for a in c.aliases),
    ):
        for w in _KEYWORD_RE.findall(source.lower()):
            words.add(w)
    seeds_lower = {s.lower() for s in c.english_seeds}
    return frozenset(w for w in words if w not in _STOP_WORDS and w not in seeds_lower)


_WORD_RE_GLOSS = re.compile(r"\b[a-z][a-z\-]+\b")
# Common English inflectional suffixes. We match a keyword against a gloss
# word if they share a stem of length >= 3 and BOTH residual suffixes appear
# in this set. That distinguishes ``sneeze`` ~ ``sneezing`` (e + ing, valid)
# from ``cat`` ~ ``cattle`` (∅ + tle, invalid).
_INFLECTIONS: frozenset[str] = frozenset(
    {"", "s", "es", "ed", "d", "e", "ing", "er", "ly", "ied", "ies", "y"}
)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _gloss_matches(gloss: str, keywords: frozenset[str]) -> bool:
    """Match if any keyword equals a gloss word, or shares a stem of len>=3
    with one and the two residual suffixes are both common inflections."""
    g_words = set(_WORD_RE_GLOSS.findall(gloss.lower()))
    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in g_words:
            return True
        for word in g_words:
            stem_len = _common_prefix_len(kw_l, word)
            if stem_len < 3:
                continue
            if kw_l[stem_len:] in _INFLECTIONS and word[stem_len:] in _INFLECTIONS:
                return True
    return False


def _navhead_gloss(table: Tag) -> str:
    nav = table.find_parent("div", class_="NavFrame")
    if not nav:
        return ""
    head = nav.find("div", class_="NavHead")
    if not head:
        return ""
    return head.get_text(" ", strip=True)


# Only translation tables whose preceding part-of-speech header is one of
# these are kept. Verb tables include lexical translations like "to bark"
# (Russian *láyat'*) which are NOT onomatopoeic — they would dilute the
# cross-linguistic phonological signature we want.
_KEEP_POS = frozenset({"Interjection", "Noun", "Phrase", "Particle"})


# Headings that name a part of speech. Different Wiktionary pages place POS
# at different heading levels (h3 for single-etymology pages, h4 for
# multi-etymology pages), and put intermediate headings like "Translations"
# or "Synonyms" between the POS and the translation table. We walk back
# through all heading levels and return the first one whose text is a
# recognized POS word.
_POS_WORDS = frozenset(
    {
        "Interjection",
        "Noun",
        "Verb",
        "Adjective",
        "Adverb",
        "Preposition",
        "Conjunction",
        "Pronoun",
        "Article",
        "Numeral",
        "Particle",
        "Determiner",
        "Phrase",
        "Proper noun",
        "Letter",
    }
)


def _preceding_pos(table: Tag) -> str | None:
    node = table
    while True:
        node = node.find_previous(["h2", "h3", "h4", "h5", "h6"])
        if node is None:
            return None
        text = re.sub(
            r"\s*\[\s*edit\s*\]\s*$",
            "",
            node.get_text(" ", strip=True),
            flags=re.IGNORECASE,
        ).strip()
        if text in _POS_WORDS:
            return text
        # Stop if we cross a language boundary (h2 is the language section).
        if node.name == "h2":
            return None


def _container_label(container: Tag) -> str:
    """Pull the language label: text before the first colon at this level."""
    parts: list[str] = []
    for child in container.children:
        if hasattr(child, "name") and child.name in ("dl", "ul", "ol"):
            break
        s = child.get_text(" ", strip=True) if hasattr(child, "name") else str(child)
        parts.append(s)
        if ":" in s:
            break
    raw = " ".join(p for p in parts if p)
    return raw.split(":")[0].strip()


def _extract_forms_local(container: Tag) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (natives, romans) for spans inside `container` but not inside
    any nested `<dd>` (those are handled by recursion into sub-languages).

    Classification by attributes, not class whitelist:
      - `<span class="tr">` (with or without script suffix)              → romanization
      - `<span lang="xx-Latn">` (BCP-47 -Latn suffix)                    → romanization
      - `<span lang="xx">` with neither of those                         → native form
    """
    natives: list[tuple[str, str]] = []
    romans: list[tuple[str, str]] = []
    seen: set[int] = set()
    for span in container.find_all("span"):
        if id(span) in seen:
            continue
        seen.add(id(span))
        nested_dd = span.find_parent("dd")
        if nested_dd is not None and nested_dd is not container:
            continue
        cls = span.get("class") or []
        lang = (span.get("lang") or "").strip()
        # Skip annotation parens and Wiktionary cross-links.
        if any(c.startswith("mention-") for c in cls) or "tpos" in cls:
            continue
        is_roman = ("tr" in cls) or lang.endswith("-Latn")
        if is_roman:
            text = span.get_text(strip=True)
            if text:
                romans.append((text, lang.split("-")[0] if lang else ""))
            continue
        # Native = has lang attr and isn't romanization.
        if not lang:
            continue
        text = span.get_text(strip=True)
        if text:
            natives.append((text, lang))
    return natives, romans


def _pair_forms(
    natives: list[tuple[str, str]],
    romans: list[tuple[str, str]],
    lang_label: str,
    parent_label: str | None,
) -> list[dict]:
    out: list[dict] = []
    if natives:
        if len(romans) == len(natives):
            for (nf, ncode), (rf, _rcode) in zip(natives, romans, strict=True):
                out.append(
                    {
                        "language_label": lang_label,
                        "language_code": ncode or None,
                        "orthography": nf,
                        "romanization": rf,
                        "parent_label": parent_label,
                    }
                )
        else:
            for idx, (nf, ncode) in enumerate(natives):
                rf = romans[0][0] if (romans and idx == 0) else None
                out.append(
                    {
                        "language_label": lang_label,
                        "language_code": ncode or None,
                        "orthography": nf,
                        "romanization": rf,
                        "parent_label": parent_label,
                    }
                )
    elif romans:
        # Latin-script-only language entry — emit the romanization as the form.
        for rf, rcode in romans:
            out.append(
                {
                    "language_label": lang_label,
                    "language_code": rcode or None,
                    "orthography": rf,
                    "romanization": None,
                    "parent_label": parent_label,
                }
            )
    # Filter boilerplate
    return [
        x
        for x in out
        if x["orthography"] and "please add this translation" not in x["orthography"].lower()
    ]


def _parse_li_entry(li: Tag, parent_label: str | None = None) -> list[dict]:
    """Walk a single translation `<li>` and emit one dict per form found."""
    out: list[dict] = []
    # Sub-language rows: <li>Chinese: <dl><dd>Mandarin: ...</dd></dl></li>
    sub_dds = [
        dd
        for dd in li.find_all("dd")
        if dd.find_parent("dd") is None  # only first-level <dd>
    ]
    if sub_dds:
        outer_label = _container_label(li)
        for dd in sub_dds:
            inner = _parse_li_entry(dd, parent_label=outer_label or parent_label)
            out.extend(inner)

    # Top-level forms (spans not inside any nested <dd>).
    natives, romans = _extract_forms_local(li)
    lang_label = _container_label(li)
    out.extend(_pair_forms(natives, romans, lang_label, parent_label))
    return out


def parse_seed_html(
    html: str,
    seed: str,
    *,
    source_url: str,
    source_revid: str,
    captured_at: str,
    concepts: Iterable[ConceptDef] = CONCEPTS,
) -> list[AnchorEntry]:
    """Parse a Wiktionary page; emit AnchorEntry rows for matching concepts."""
    soup = BeautifulSoup(html, "lxml")
    concepts_list = list(concepts)
    # Pre-compute matching keywords per concept.
    keywords_per_concept = {c.slug: _concept_keywords(c) for c in concepts_list}
    # Filter to concepts that use this seed.
    seed_lower = seed.lower()
    relevant = [c for c in concepts_list if seed_lower in {s.lower() for s in c.english_seeds}]
    if not relevant:
        return []

    entries: list[AnchorEntry] = []
    for table in soup.find_all("table", class_="translations"):
        gloss = _navhead_gloss(table)
        if not gloss:
            continue
        pos = _preceding_pos(table)
        if pos and pos not in _KEEP_POS:
            # Verb / adjective / etc. translations are lexical, not
            # onomatopoeic — skip.
            continue
        matched_concepts = [
            c for c in relevant if _gloss_matches(gloss, keywords_per_concept[c.slug])
        ]
        if not matched_concepts:
            continue
        for li in table.find_all("li"):
            for item in _parse_li_entry(li):
                ortho = item["orthography"]
                roman = item.get("romanization")
                lang_label = item.get("language_label") or ""
                lang_code = item.get("language_code")
                parent_label = item.get("parent_label")
                full_lang_label = f"{parent_label}, {lang_label}" if parent_label else lang_label
                for c in matched_concepts:
                    entries.append(
                        AnchorEntry(
                            concept=c.slug,
                            category=c.category,
                            language=full_lang_label,
                            language_code=lang_code or None,
                            orthography=ortho,
                            romanization=roman,
                            ipa=None,
                            source=f"wiktionary:{seed}",
                            source_url=source_url,
                            source_revid=source_revid,
                            captured_at=captured_at,
                            notes=None,
                            extra={
                                "seed": seed,
                                "gloss": gloss,
                            },
                        )
                    )
    return entries


def parse_seed_file(html_path: Path, meta_path: Path) -> list[AnchorEntry]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("missing"):
        return []
    return parse_seed_html(
        html_path.read_text(encoding="utf-8"),
        seed=meta["seed"],
        source_url=meta["url"],
        source_revid=str(meta["revid"]),
        captured_at=meta["captured_at"],
    )
