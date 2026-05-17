"""Tests for Stage 6 lexicon. No torch/HF."""

import numpy as np

from conlang.lexicon import (
    DEFAULT_CLASS_ID,
    _resolve_stem_collisions,
    assign_class,
    build_lexicon,
    build_stem,
)
from conlang.phonology import is_valid_word


# --- class assignment -----------------------------------------------------


def test_assign_class_picks_human():
    assert assign_class("references to individuals and their life events") == 1
    assert assign_class("phrases about people who work in tech") == 1


def test_assign_class_picks_animal_language():
    assert assign_class("names of animals and birds") == 9
    assert assign_class("words from various languages") == 9


def test_assign_class_picks_tool_thing():
    assert assign_class("software tools and programs for analysis") == 7
    assert assign_class("processes related to manufacturing") == 7


def test_assign_class_picks_abstract():
    assert assign_class("concepts related to time and causality") == 11


def test_assign_class_falls_through_to_default():
    # No keyword matches → default class.
    assert assign_class("xyzzy plugh foo bar") == DEFAULT_CLASS_ID


def test_assign_class_first_match_wins():
    # If a label mentions both "person" and "system", the human rule fires
    # first because it appears earlier in KEYWORD_RULES.
    assert assign_class("a person who operates the system") == 1


# --- stem generation ------------------------------------------------------


def test_build_stem_is_deterministic():
    a = build_stem(feature_id=42, cluster_id=-1, parent_feature_id=7)
    b = build_stem(feature_id=42, cluster_id=-1, parent_feature_id=7)
    assert a == b


def test_build_stem_shares_cv1_for_same_cluster():
    a = build_stem(feature_id=1, cluster_id=3, parent_feature_id=10)
    b = build_stem(feature_id=2, cluster_id=3, parent_feature_id=20)
    assert a[:2] == b[:2], f"siblings should share CV1: {a} vs {b}"


def test_build_stem_differs_cv1_across_clusters():
    a = build_stem(feature_id=1, cluster_id=0, parent_feature_id=10)
    b = build_stem(feature_id=1, cluster_id=4, parent_feature_id=10)
    assert a[:2] != b[:2]


def test_build_stem_shares_cv2_for_same_parent():
    a = build_stem(feature_id=1, cluster_id=-1, parent_feature_id=99)
    b = build_stem(feature_id=2, cluster_id=-1, parent_feature_id=99)
    assert a[2:4] == b[2:4], f"same-parent should share CV2: {a} vs {b}"


def test_build_stem_no_parent_uses_default_cv2():
    a = build_stem(feature_id=1, cluster_id=-1, parent_feature_id=None)
    b = build_stem(feature_id=2, cluster_id=-1, parent_feature_id=None)
    assert a[2:4] == b[2:4] == "ya"


def test_build_stem_is_valid_phonology():
    # 3-syllable stem from valid inventory must syllabify as a valid word.
    stem = build_stem(feature_id=123, cluster_id=2, parent_feature_id=456)
    assert is_valid_word(stem)


# --- collision resolution -------------------------------------------------


def test_resolve_collisions_breaks_duplicates():
    entries = [
        {"feature_id": 1, "stem": "papapa"},
        {"feature_id": 2, "stem": "papapa"},
        {"feature_id": 3, "stem": "papapa"},
        {"feature_id": 4, "stem": "tetete"},
    ]
    _resolve_stem_collisions(entries)
    stems = [e["stem"] for e in entries]
    assert len(stems) == len(set(stems)), f"still colliding: {stems}"
    # First entry keeps its original stem.
    assert entries[0]["stem"] == "papapa"
    # Others got CV4 appended.
    assert all(e["stem"].startswith("papapa") for e in entries[:3])
    assert all(len(e["stem"]) > 6 for e in entries[1:3])
    assert entries[3]["stem"] == "tetete"


def test_resolve_collisions_noop_when_unique():
    entries = [
        {"feature_id": 1, "stem": "papapa"},
        {"feature_id": 2, "stem": "tetete"},
    ]
    _resolve_stem_collisions(entries)
    assert entries[0]["stem"] == "papapa"
    assert entries[1]["stem"] == "tetete"


# --- end-to-end build_lexicon ---------------------------------------------


def test_build_lexicon_assigns_one_entry_per_node():
    features = [
        {"feature_id": 10, "label": "a person who teaches mathematics"},
        {"feature_id": 20, "label": "names of programming languages"},
        {"feature_id": 30, "label": "abstract theory of computation"},
    ]
    regularized_nodes = [
        {"feature_id": 10, "parent": {"slice_idx": 1, "pmi": 2.0}, "siblings": [], "near": []},
        {"feature_id": 20, "parent": None, "siblings": [], "near": []},
        {"feature_id": 30, "parent": {"slice_idx": 0, "pmi": 1.5}, "siblings": [], "near": []},
    ]
    labels = np.array([0, 0, -1])
    entries = build_lexicon(features, regularized_nodes, labels)
    assert len(entries) == 3
    # Classes per the keyword rules:
    assert entries[0]["class_id"] == 1   # person → human
    assert entries[1]["class_id"] == 9   # languages → animal/language
    assert entries[2]["class_id"] == 11  # theory → abstract
    # Surfaces are valid Bantu words.
    for e in entries:
        assert is_valid_word(e["surface"])
        assert is_valid_word(e["antonym"])
        assert e["antonym"].startswith("s")
    # Same cluster → entries 0 and 1 share CV1.
    assert entries[0]["stem"][:2] == entries[1]["stem"][:2]


def test_build_lexicon_stems_are_globally_unique():
    # Deliberately force a collision: same cluster, same parent, then check
    # that the resolver produces distinct surfaces.
    features = [
        {"feature_id": 100, "label": "test"},
        {"feature_id": 200, "label": "test"},
    ]
    regularized_nodes = [
        {"feature_id": 100, "parent": {"slice_idx": 1, "pmi": 1.0}, "siblings": [], "near": []},
        {"feature_id": 200, "parent": {"slice_idx": 0, "pmi": 1.0}, "siblings": [], "near": []},
    ]
    labels = np.array([0, 0])
    entries = build_lexicon(features, regularized_nodes, labels)
    stems = [e["stem"] for e in entries]
    assert len(set(stems)) == len(stems)
