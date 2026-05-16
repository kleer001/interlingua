"""Sanity tests for the canonical concept inventory."""

from collections import Counter

from conlang.anchors.concepts import (
    ALIAS_TO_SLUG,
    BY_SLUG,
    CONCEPTS,
    canonical_slug,
    concepts_by_category,
)


def test_slugs_are_unique():
    slugs = [c.slug for c in CONCEPTS]
    dups = [s for s, n in Counter(slugs).items() if n > 1]
    assert not dups, f"duplicate slugs: {dups}"


def test_every_alias_resolves_to_a_real_slug():
    for alias, slug in ALIAS_TO_SLUG.items():
        assert slug in BY_SLUG, f"alias {alias!r} -> unknown slug {slug!r}"


def test_canonical_slug_handles_known_wikipedia_slugs():
    # These come from Wikipedia's cross-linguistic onomatopoeias columns.
    assert canonical_slug("dog_or_wolf_howling") == "dog_howling"
    assert canonical_slug("lion_tiger_roaring") == "lion_roaring"
    assert canonical_slug("chicken_clucking") == "hen_clucking"
    assert canonical_slug("duck_calling") == "duck_quacking"
    assert canonical_slug("bird_singing") == "songbird_singing"
    assert canonical_slug("baby_crying") == "crying"
    # Identity slugs work too
    assert canonical_slug("dog_barking") == "dog_barking"


def test_canonical_slug_returns_none_for_unknown():
    assert canonical_slug("triceratops_bellowing") is None


def test_each_concept_has_at_least_one_seed():
    for c in CONCEPTS:
        assert len(c.english_seeds) >= 1, f"{c.slug} has no english_seeds"


def test_categories_match_data_plan_taxonomy():
    expected = {
        "mammal",
        "bird",
        "reptile",
        "insect",
        "marine",
        "water_sound",
        "fire_wind_weather",
        "hard_impact",
        "soft_impact",
        "resonant",
        "mechanical",
        "human_nonverbal",
        "movement",
        "texture_eating",
        "exclamation_affect",
    }
    got = set(c.category for c in CONCEPTS)
    assert got <= expected, f"unexpected categories: {got - expected}"


def test_concepts_by_category_partitions_inventory():
    by_cat = concepts_by_category()
    total = sum(len(v) for v in by_cat.values())
    assert total == len(CONCEPTS)
