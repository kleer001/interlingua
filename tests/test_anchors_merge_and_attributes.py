"""Merge canonicalization + Phase-6 attribute-bundle behaviour."""

from pathlib import Path

from conlang.anchors.attributes import (
    ATTRIBUTE_REGISTRY,
    build_attribute_anchor_table,
    signature_for,
)
from conlang.anchors.merge import _canonicalize, merge_streams, synthesize_english_anchors
from conlang.anchors.schema import AnchorEntry


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


def test_canonicalize_maps_wikipedia_alias_to_inventory_slug():
    e = _entry(concept="dog_or_wolf_howling")
    canon = _canonicalize(e)
    assert canon is not None
    assert canon.concept == "dog_howling"
    assert canon.extra["source_concept"] == "dog_or_wolf_howling"


def test_canonicalize_returns_none_for_unknown_concept():
    e = _entry(concept="not_a_real_slug")
    assert _canonicalize(e) is None


def test_merge_dedupes_within_and_across_sources():
    wiki = [
        _entry(orthography="woof"),
        _entry(orthography="bark"),
        # dup of the first — should collapse
        _entry(orthography="woof"),
    ]
    wikt = [
        _entry(orthography="woof", source="wiktionary:woof", romanization="wuf"),
        _entry(
            language_code="ja",
            orthography="ワンワン",
            romanization="wan wan",
            source="wiktionary:woof",
        ),
    ]
    merged, stats = merge_streams({"wiki": wiki, "wikt": wikt})
    keys = {(e.concept, e.language_code, e.orthography) for e in merged}
    assert ("dog_barking", "en", "woof") in keys
    assert ("dog_barking", "en", "bark") in keys
    assert ("dog_barking", "ja", "ワンワン") in keys
    assert stats["dedup_merged_provenance"] >= 2


def test_merge_backfills_missing_fields_from_later_source():
    wiki = [_entry(orthography="woof", romanization=None, ipa=None)]
    wikt = [_entry(orthography="woof", romanization="wuf", ipa="/wuf/", source="wiktionary:woof")]
    merged, _ = merge_streams({"wiki": wiki, "wikt": wikt})
    e = next(x for x in merged if x.orthography == "woof")
    assert e.romanization == "wuf"
    assert e.ipa == "/wuf/"


def test_synthesize_english_loops_all_seeds_for_lacking_concept():
    """Concepts lacking English should receive one entry per english_seed.

    Picks 'dog_howling' (seeds: 'awoo', 'howl') — a multi-seed concept — and
    starts from an empty entries list so the concept is guaranteed to lack
    English. Confirms both seeds land with distinct seed_origin provenance.
    """
    entries: list[AnchorEntry] = []
    out, added = synthesize_english_anchors(entries)
    howling = [e for e in out if e.concept == "dog_howling" and e.language_code == "en"]
    orthographies = {e.orthography for e in howling}
    assert "awoo" in orthographies
    assert "howl" in orthographies
    origins = {e.extra.get("seed_origin") for e in howling}
    assert origins == {"concepts.english_seeds[0]", "concepts.english_seeds[1]"}
    assert added >= 2


def test_synthesize_english_skips_concept_already_covered():
    """If a concept already has an English entry, no seeds should be added."""
    seed = _entry(concept="dog_barking", language_code="en", orthography="woof")
    entries = [seed]
    out, added = synthesize_english_anchors(entries)
    dog_bark_en = [e for e in out if e.concept == "dog_barking" and e.language_code == "en"]
    # dog_barking has 4 seeds but should be untouched because English already exists.
    assert len(dog_bark_en) == 1
    assert dog_bark_en[0].source == "wikipedia"


def test_attribute_registry_concepts_exist_in_inventory():
    from conlang.anchors.concepts import BY_SLUG

    for slug, bundle in ATTRIBUTE_REGISTRY.items():
        # The bundle's concept field must be in the inventory.
        assert bundle.concept in BY_SLUG, (
            f"AttributeBundle for {slug!r} targets unknown concept {bundle.concept!r}"
        )


def test_signature_for_aggregates_languages():
    entries = [
        _entry(language_code="en", orthography="woof"),
        _entry(language_code="ja", orthography="ワンワン", ipa="/wanwan/"),
        _entry(language_code="ru", orthography="гав"),
    ]
    sig = signature_for("dog_barking", entries)
    assert sig.n_languages == 3
    assert sig.n_entries == 3
    assert "ja" in sig.languages
    assert "/wanwan/" in sig.ipas


def test_build_attribute_anchor_table_yields_one_row_per_attribute(tmp_path: Path):
    entries = [_entry()]
    table = build_attribute_anchor_table(entries)
    # Each row should reference a registered concept and an attribute string.
    bundle_attr_count = sum(
        len(b.attributes) + len(b.cultural_attributes) for b in ATTRIBUTE_REGISTRY.values()
    )
    assert len(table) == bundle_attr_count
    # Cultural flag is set correctly.
    snake = [r for r in table if r.concept == "snake_hissing"]
    cultural = [r for r in snake if r.cultural]
    universal = [r for r in snake if not r.cultural]
    assert len(cultural) >= 1
    assert len(universal) >= 1


def test_minimum_attribute_count_per_bundle():
    """Every bundle must carry ≥12 attributes (perceptual + affect combined).

    Floor protects ρ density: a stub bundle with only 4-5 attrs would land
    nearly all probe mass at one residual-space point, undoing the
    attribute-level density gain that motivates the cutover.
    """
    for slug, bundle in ATTRIBUTE_REGISTRY.items():
        assert len(bundle.attributes) >= 12, (
            f"Bundle {slug!r} has only {len(bundle.attributes)} attributes; "
            f"floor is 12 (perceptual + affect combined)."
        )


def test_each_bundle_has_cultural_attributes():
    """Every bundle must carry ≥3 cultural attributes.

    Cross-cultural / contradictory valences are the philosophical lean
    that makes anchors richer than literal sound-feature tags.
    """
    for slug, bundle in ATTRIBUTE_REGISTRY.items():
        assert len(bundle.cultural_attributes) >= 3, (
            f"Bundle {slug!r} has only {len(bundle.cultural_attributes)} "
            f"cultural attributes; floor is 3."
        )


def test_registry_covers_at_least_40_signed_concepts():
    """Coverage gate for the Phase 2 cutover precondition.

    The #11 hypothesis requires ≥40 of the 63 signed concepts to be
    bundled before flipping `embed_positions.py` to attribute-level.
    Active gate as of Batch B3 (registry first reached 40 bundles).
    """
    assert len(ATTRIBUTE_REGISTRY) >= 40, (
        f"Registry has {len(ATTRIBUTE_REGISTRY)} bundles; "
        f"≥40 required to flip embed_positions to attribute-level."
    )
