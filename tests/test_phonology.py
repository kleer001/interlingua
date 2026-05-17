"""Tests for Stage 6 phonology. Pure data + composition, no torch/HF."""

import pytest

from conlang.phonology import (
    CLASS_PREFIXES,
    NEGATION_PREFIX,
    apply_class_prefix,
    is_valid_syllable,
    is_valid_word,
    negate,
    syllabify,
)


# --- syllabifier ----------------------------------------------------------


def test_syllabify_simple_cv():
    assert syllabify("paka") == ["pa", "ka"]


def test_syllabify_v_only_syllable():
    assert syllabify("aki") == ["a", "ki"]


def test_syllabify_nasal_digraph():
    assert syllabify("nyumba") == ["nyu", "mba"]


def test_syllabify_velar_nasal_digraph():
    assert syllabify("ngoma") == ["ngo", "ma"]


def test_syllabify_prenasalized_onset():
    assert syllabify("mbuzi") == ["mbu", "zi"]


def test_syllabify_prenasalized_with_valid_consonants():
    # mp + a + nd + u  → "mpa" + "ndu"
    assert syllabify("mpandu") == ["mpa", "ndu"]


def test_syllabify_rejects_bare_consonant():
    assert syllabify("m") is None
    assert syllabify("pk") is None


def test_syllabify_rejects_consonant_at_end():
    assert syllabify("pak") is None


def test_syllabify_rejects_unknown_consonant():
    # r is not in our inventory
    assert syllabify("ra") is None
    assert syllabify("hapa") is None


def test_syllabify_empty_returns_none():
    assert syllabify("") is None


def test_is_valid_syllable_accepts_single_cv():
    assert is_valid_syllable("pa")
    assert is_valid_syllable("a")
    assert is_valid_syllable("nyu")


def test_is_valid_syllable_rejects_multi():
    assert not is_valid_syllable("paka")


def test_is_valid_word_enforces_min_syllables():
    assert is_valid_word("paka", min_syllables=2)
    assert not is_valid_word("pa", min_syllables=2)
    assert is_valid_word("pa", min_syllables=1)


# --- class prefixes -------------------------------------------------------


def test_apply_class_prefix_simple_concatenation():
    assert apply_class_prefix("paka", 7) == "kipaka"   # class 7
    assert apply_class_prefix("toto", 1) == "mutoto"   # class 1
    assert apply_class_prefix("toto", 2) == "batoto"   # class 2


def test_apply_class_prefix_elides_at_vowel_hiatus():
    # mu + ana → m + ana = "mana" (simple elision, no glide)
    assert apply_class_prefix("ana", 1) == "mana"
    # ki + ana → k + ana = "kana"
    assert apply_class_prefix("ana", 7) == "kana"


def test_apply_class_prefix_class_9_homorganic_with_labial():
    # N- + paka → mpaka (homorganic with p)
    assert apply_class_prefix("paka", 9) == "mpaka"
    assert apply_class_prefix("buku", 9) == "mbuku"


def test_apply_class_prefix_class_9_yi_for_velar():
    # Class 9 + k/g uses the yi- allomorph (homorganic ng+k would be invalid).
    assert apply_class_prefix("kuku", 9) == "yikuku"
    assert apply_class_prefix("guni", 9) == "yiguni"


def test_apply_class_prefix_class_9_alveolar_prenasalizes():
    # t/d still prenasalize: nt-, nd-.
    assert apply_class_prefix("toto", 9) == "ntoto"
    assert apply_class_prefix("dama", 9) == "ndama"


def test_apply_class_prefix_class_9_yi_for_other_consonants():
    # s/w/l/etc. take the yi- allomorph.
    assert apply_class_prefix("soko", 9) == "yisoko"
    assert apply_class_prefix("wamafe", 9) == "yiwamafe"
    assert apply_class_prefix("lupa", 9) == "yilupa"


def test_apply_class_prefix_class_9_y_before_vowel():
    # Vowel-initial stems get y- (yi elides before vowel).
    assert apply_class_prefix("andu", 9) == "yandu"


def test_apply_class_prefix_class_10_simple():
    # zi-paka = "zipaka"
    assert apply_class_prefix("paka", 10) == "zipaka"


def test_apply_class_prefix_unknown_class_raises():
    with pytest.raises(KeyError):
        apply_class_prefix("paka", 99)


def test_apply_class_prefix_output_is_valid_word_when_input_is():
    # Sanity: every prefixed stem of a 2-syllable valid stem is a valid word.
    stem = "paka"
    for class_id in CLASS_PREFIXES:
        out = apply_class_prefix(stem, class_id)
        assert is_valid_word(out), f"class {class_id} produced invalid word: {out!r}"


# --- negation -------------------------------------------------------------


def test_negate_prefixes_si():
    assert negate("kipaka") == "sikipaka"
    assert negate("batoto") == "sibatoto"


def test_negate_elides_at_vowel_hiatus():
    # si + ana → s + ana = "sana"
    assert negate("ana") == "sana"
    # si + utu → s + utu = "sutu"
    assert negate("utu") == "sutu"


def test_negate_composes_outside_class_prefix():
    # Stage 6 composition order: [neg] + [class] + [stem]
    classed = apply_class_prefix("paka", 7)
    assert classed == "kipaka"
    assert negate(classed) == "sikipaka"


def test_negate_is_productive_across_all_classes():
    # Per Commitment 7: negation works on every class-prefixed form.
    for class_id in CLASS_PREFIXES:
        classed = apply_class_prefix("paka", class_id)
        negated = negate(classed)
        assert negated.startswith(NEGATION_PREFIX) or negated.startswith("s")
        assert is_valid_word(negated)


def test_negate_double_application_stacks():
    # Conlang choice: double negation = double prefix (emphatic, not cancelling).
    once = negate("kipaka")
    twice = negate(once)
    assert twice == "sisikipaka"
    assert twice != "kipaka"


# --- inventory shape ------------------------------------------------------


def test_class_prefixes_cover_expected_ids():
    assert set(CLASS_PREFIXES) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}


def test_negation_prefix_is_si():
    assert NEGATION_PREFIX == "si"
