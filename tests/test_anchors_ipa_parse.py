"""Smoke tests for the static HTML viz + the IPA-page parser."""

from pathlib import Path

import pytest

from conlang.anchors.parse_wiktionary_ipa import (
    apply_lookup,
    build_ipa_lookup,
    parse_form_html,
)
from conlang.anchors.schema import AnchorEntry

pytest.importorskip("panphon")


def _entry(
    concept="cow_mooing", lang_code="en", ipa=None, ortho="moo", language="English"
) -> AnchorEntry:
    return AnchorEntry(
        concept=concept,
        category="mammal",
        language=language,
        language_code=lang_code,
        orthography=ortho,
        romanization=None,
        ipa=ipa,
        source="test",
        source_url="https://example/",
        source_revid="0",
        captured_at="2026-05-16",
        notes=None,
        extra={},
    )


def test_parse_form_html_handles_canonical_layout():
    html = """
    <h2>English</h2>
    <p>some intro</p>
    <span class="IPA">/muː/</span>
    <span class="IPA">-uː</span>
    <h2>Japanese</h2>
    <span class="IPA">/moː/</span>
    <h2>Translingual</h2>
    <span class="IPA">/skip/</span>
    """
    out = parse_form_html(html)
    assert out["en"] == ["/muː/"]
    assert out["ja"] == ["/moː/"]
    assert "tl" not in out  # Translingual is skipped (no real language)


def test_parse_form_html_drops_rhyme_suffixes():
    html = (
        "<h2>English</h2>"
        '<span class="IPA">-uː</span>'  # rhyme suffix, skip
        '<span class="IPA">/muː/</span>'  # canonical, keep
    )
    out = parse_form_html(html)
    assert out["en"] == ["/muː/"]


def test_parse_form_html_drops_unmapped_language():
    html = '<h2>Made Up Language</h2><span class="IPA">/xxx/</span>'
    out = parse_form_html(html)
    assert out == {}


def test_apply_lookup_fills_missing_ipa_and_keeps_existing():
    rows = [
        _entry(lang_code="en", ortho="moo", ipa=None),
        _entry(lang_code="en", ortho="moo", ipa="/already/"),  # kept
        _entry(lang_code="he", ortho="מו", ipa=None),
    ]
    lookup = {("moo", "en"): "/muː/", ("מו", "he"): "/mu/"}
    out, stats = apply_lookup(rows, lookup)
    assert stats["filled"] == 2
    assert stats["kept"] == 1
    assert out[0].ipa == "/muː/"
    assert out[0].extra["ipa_source"] == "wiktionary-form-page"
    assert out[1].ipa == "/already/"
    assert out[2].ipa == "/mu/"


def test_apply_lookup_no_match_records_stat():
    rows = [_entry(lang_code="en", ortho="zorp", ipa=None)]
    out, stats = apply_lookup(rows, lookup={})
    assert stats["no_match"] == 1
    assert out[0].ipa is None


def test_build_ipa_lookup_walks_cache(tmp_path: Path):
    """Drop in two cached pages and check the lookup builder."""
    cache = tmp_path / "wiktionary_ipa"
    cache.mkdir()
    # form 'moo'
    (cache / "moo-rev1.html").write_text(
        '<h2>English</h2><span class="IPA">/muː/</span>'
        '<h2>Japanese</h2><span class="IPA">/moː/</span>',
        encoding="utf-8",
    )
    (cache / "moo-rev1.meta.json").write_text('{"form": "moo", "revid": 1}', encoding="utf-8")
    # missing-form record (should be skipped)
    (cache / "zorp-missing.meta.json").write_text(
        '{"form": "zorp", "missing": true}', encoding="utf-8"
    )
    lookup = build_ipa_lookup(cache)
    assert lookup[("moo", "en")] == "/muː/"
    assert lookup[("moo", "ja")] == "/moː/"
    assert ("zorp", "en") not in lookup
