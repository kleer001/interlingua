"""Epitran enrichment tests with a mock engine — no real models needed."""

from conlang.anchors import enrich_epitran
from conlang.lab.schema import AnchorEntry


class _FakeEngine:
    """Returns a predictable IPA string so we can assert routing without
    needing Epitran's actual data files installed."""

    def __init__(self, code: str):
        self.code = code

    def transliterate(self, text: str) -> str:
        return f"<{self.code}|{text}>"


def _entry(**over) -> AnchorEntry:
    d = dict(
        concept="dog_barking",
        category="mammal",
        language="English",
        language_code="en",
        orthography="woof",
        romanization=None,
        ipa=None,
        source="wikipedia",
        source_url="https://example/",
        source_revid="0",
        captured_at="2026-05-16",
        notes=None,
        extra={},
    )
    d.update(over)
    return AnchorEntry(**d)


def test_skip_when_no_language_code(monkeypatch):
    monkeypatch.setattr(enrich_epitran, "_make_engine", _FakeEngine)
    out, stats = enrich_epitran.enrich([_entry(language_code=None)])
    assert stats["skipped_no_code"] == 1
    assert out[0].ipa is None


def test_skip_when_no_orthography(monkeypatch):
    monkeypatch.setattr(enrich_epitran, "_make_engine", _FakeEngine)
    out, stats = enrich_epitran.enrich([_entry(language_code="fr", orthography="")])
    assert stats["skipped_no_text"] == 1
    assert out[0].ipa is None


def test_skip_when_language_unmapped(monkeypatch):
    """A real BCP-47 code that isn't in EPITRAN_CODE_MAP (e.g. Hebrew)."""
    monkeypatch.setattr(enrich_epitran, "_make_engine", _FakeEngine)
    out, stats = enrich_epitran.enrich([_entry(language_code="he", orthography="כלב")])
    assert stats["skipped_no_code"] == 1


def test_keeps_existing_ipa_by_default(monkeypatch):
    monkeypatch.setattr(enrich_epitran, "_make_engine", _FakeEngine)
    out, stats = enrich_epitran.enrich([_entry(language_code="fr", ipa="/already/")])
    assert stats["kept"] == 1
    assert out[0].ipa == "/already/"


def test_overwrites_when_requested(monkeypatch):
    monkeypatch.setattr(enrich_epitran, "_make_engine", _FakeEngine)
    out, stats = enrich_epitran.enrich(
        [_entry(language_code="fr", ipa="/old/", orthography="woof")], overwrite=True
    )
    assert stats["filled"] == 1
    assert out[0].ipa == "/<fra-Latn|woof>/"


def test_mandarin_uses_romanization_not_hanzi(monkeypatch):
    monkeypatch.setattr(enrich_epitran, "_make_engine", _FakeEngine)
    e = _entry(language_code="cmn", orthography="汪", romanization="wāng", language="Mandarin")
    out, stats = enrich_epitran.enrich([e])
    assert stats["filled"] == 1
    # The fake engine echoes whatever it gets, so we can read which text was passed
    assert "wāng" in out[0].ipa
    assert "汪" not in out[0].ipa
    assert out[0].extra["ipa_source"] == "epitran:cmn-Latn"


def test_filled_count_matches_routing(monkeypatch):
    monkeypatch.setattr(enrich_epitran, "_make_engine", _FakeEngine)
    rows = [
        _entry(language_code="fr", orthography="woof"),  # filled
        _entry(language_code="ja", orthography="ワン"),  # filled
        _entry(language_code="he", orthography="כלב"),  # no_code
        _entry(language_code="en", orthography="woof"),  # no_code (English unsupported)
        _entry(language_code="ru", orthography=""),  # no_text
        _entry(language_code="ru", orthography="гав", ipa="/x/"),  # kept
    ]
    out, stats = enrich_epitran.enrich(rows)
    assert stats["filled"] == 2
    assert stats["skipped_no_code"] == 2
    assert stats["skipped_no_text"] == 1
    assert stats["kept"] == 1
