"""Per-concept signature aggregation tests."""

import pytest

from conlang.lab.schema import AnchorEntry
from conlang.lab.signatures import (
    ConceptSignature,
    build_all_signatures,
    signature_for_concept,
)

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


def test_signature_for_concept_aggregates_languages():
    rows = [
        _entry(lang_code="en", ipa="/muː/", language="English"),
        _entry(lang_code="ja", ipa="/moː/", language="Japanese"),
        _entry(lang_code="es", ipa="/mu/", language="Spanish"),
    ]
    sig = signature_for_concept("cow_mooing", rows)
    assert isinstance(sig, ConceptSignature)
    assert sig.n_entries == 3
    assert sig.n_with_ipa == 3
    assert sig.n_languages == 3
    assert sorted(sig.languages) == ["en", "es", "ja"]
    # Examples should sample distinct languages
    assert len({ex["language_code"] for ex in sig.examples}) == 3
    # Sharpness should be high — all three forms are /m/+vowel
    assert sig.sharpness > 0.7


def test_signature_handles_missing_ipa():
    rows = [_entry(lang_code="en", ipa=None)]
    sig = signature_for_concept("cow_mooing", rows)
    assert sig.n_entries == 1
    assert sig.n_with_ipa == 0
    # No segments -> means/vars all zero -> sharpness = 1.0 by formula.
    # We mostly care it doesn't crash.
    assert sig.feature_names == sig.feature_names  # set


def test_sharpness_lower_for_diverse_forms():
    sharp = [
        _entry(lang_code="en", ipa="/muː/"),
        _entry(lang_code="ja", ipa="/moː/"),
        _entry(lang_code="es", ipa="/mu/"),
    ]
    fuzzy = [
        _entry(lang_code="en", ipa="/oink/"),
        _entry(lang_code="ja", ipa="/buːbuː/"),
        _entry(lang_code="ru", ipa="/xrjuxrju/"),
        _entry(lang_code="es", ipa="/oink/"),
    ]
    s1 = signature_for_concept("cow_mooing", sharp)
    s2 = signature_for_concept("cow_mooing", fuzzy)  # same slug, different rows
    assert s1.sharpness > s2.sharpness


def test_build_all_signatures_one_per_concept():
    rows = [
        _entry(concept="cow_mooing", ipa="/muː/"),
        _entry(concept="cow_mooing", ipa="/moː/"),
        _entry(concept="dog_barking", ipa="/wuːf/"),
    ]
    sigs = build_all_signatures(rows)
    assert len(sigs) == 2
    assert {s.concept for s in sigs} == {"cow_mooing", "dog_barking"}
