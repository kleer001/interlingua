"""Smoke test for the static HTML viz."""

from pathlib import Path

import pytest

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


def test_viz_renders_without_error(tmp_path: Path):
    from conlang.anchors.signatures import build_all_signatures
    from conlang.anchors.viz import write_browser

    rows = [
        _entry(concept="cow_mooing", lang_code="en", ipa="/muː/"),
        _entry(concept="cow_mooing", lang_code="ja", ipa="/moː/", language="Japanese"),
        _entry(concept="dog_barking", lang_code="en", ipa="/wuːf/", ortho="woof"),
    ]
    sigs = build_all_signatures(rows)
    out = tmp_path / "anchors.html"
    write_browser(rows, sigs, out)
    txt = out.read_text(encoding="utf-8")
    assert '<table class="summary">' in txt
    assert "cow_mooing" in txt
    assert "dog_barking" in txt
    # IPA characters survive the HTML-escape round trip
    assert "muː" in txt
