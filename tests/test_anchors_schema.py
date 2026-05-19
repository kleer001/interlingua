"""Schema roundtrip and IO tests for anchor entries."""

import json
from pathlib import Path

from conlang.lab.schema import AnchorEntry, read_jsonl, write_jsonl


def _sample(**over) -> AnchorEntry:
    d = dict(
        concept="dog_barking",
        category="Cats and dogs",
        language="Japanese",
        language_code="ja",
        orthography="ワンワン",
        romanization="wan wan",
        ipa=None,
        source="wikipedia:Cross-linguistic_onomatopoeias",
        source_url="https://en.wikipedia.org/wiki/Cross-linguistic_onomatopoeias",
        source_revid="1234567",
        captured_at="2026-05-16",
        notes=None,
        extra={"concept_label": "Dog barking"},
    )
    d.update(over)
    return AnchorEntry(**d)


def test_to_jsonl_is_valid_json_with_unicode():
    e = _sample()
    line = e.to_jsonl()
    d = json.loads(line)
    assert d["orthography"] == "ワンワン"  # not \uXXXX-escaped
    assert d["language_code"] == "ja"
    assert d["extra"]["concept_label"] == "Dog barking"


def test_write_and_read_jsonl_roundtrip(tmp_path: Path):
    es = [
        _sample(),
        _sample(language="English", language_code="en", orthography="woof", romanization=None),
        _sample(orthography="", ipa="[ʔ]", romanization=None),
    ]
    p = tmp_path / "out.jsonl"
    n = write_jsonl(es, p)
    assert n == 3
    back = read_jsonl(p)
    assert len(back) == 3
    assert back[0].orthography == "ワンワン"
    assert back[1].language == "English"
    assert back[2].ipa == "[ʔ]"
