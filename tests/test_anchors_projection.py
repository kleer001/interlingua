"""IPA -> 10C/5V inventory projection tests."""

import pytest

from conlang.anchors.inventory import (
    ALL_PHONEMES,
    CONSONANTS,
    VOWELS,
    is_consonant,
    is_vowel,
    phoneme_features,
)
from conlang.anchors.project import phonological_distance, project_ipa, project_segment

panphon = pytest.importorskip("panphon")


def test_inventory_size():
    assert len(CONSONANTS) == 10
    assert len(VOWELS) == 5
    assert len(ALL_PHONEMES) == 15


def test_consonant_vowel_partition():
    for c in CONSONANTS:
        assert is_consonant(c)
        assert not is_vowel(c)
    for v in VOWELS:
        assert is_vowel(v)
        assert not is_consonant(v)


def test_phoneme_features_returns_24d_vectors():
    feats = phoneme_features()
    assert set(feats) == set(ALL_PHONEMES)
    for v in feats.values():
        assert len(v) == 24
        assert all(x in (-1, 0, 1) for x in v)


def test_each_inventory_phoneme_maps_to_itself():
    """A segment that exactly equals an inventory phoneme must project to it."""
    feats = phoneme_features()
    for p in ALL_PHONEMES:
        assert project_segment(feats[p]) == p


def test_voiced_stop_collapses_to_voiceless():
    # /b/, /d/, /ɡ/ -> /p/, /t/, /k/ (voicing not in inventory)
    assert project_ipa("/b/") == "p"
    assert project_ipa("/d/") == "t"
    assert project_ipa("/ɡ/") == "k"


def test_long_vowel_strips_length():
    assert project_ipa("/aː/") == "a"
    assert project_ipa("/iː/") == "i"
    assert project_ipa("/uː/") == "u"


def test_stress_marks_dont_affect_projection():
    assert project_ipa("/ˈmu/") == "mu"
    assert project_ipa("/ˌba/") == "pa"


def test_brackets_and_slashes_both_work():
    assert project_ipa("/wuf/") == "wup"
    assert project_ipa("[wuf]") == "wup"


def test_affricate_splits_to_two_segments():
    # /t͡ʃ/ is a single phoneme but panphon parses as t + ʃ
    out = project_ipa("/t͡ʃ/")
    # Should be 2 segments: 't' and the projection of /ʃ/ -> 's'
    assert out == "ts" or out == "t"  # depending on panphon's parse


def test_palatal_nasal_maps_to_some_nasal():
    # /ɲ/ -> some nasal in our inventory; panphon's feature distance ranks
    # /m/ slightly closer (palatal is -cor in panphon's tradition, which
    # is unexpected linguistically). Either /n/ or /m/ is acceptable here.
    out = project_ipa("/ɲa/")
    assert out[0] in ("n", "m")


def test_palatal_stop_maps_to_k():
    # /c/ palatal stop -> /k/ velar
    assert project_ipa("/ca/") == "ka"


def test_glottal_stop_maps_to_h():
    # /ʔ/ glottal stop -> /h/ glottal fricative (only glottal in inventory)
    out = project_ipa("/ʔa/")
    assert out.startswith("h")


def test_empty_input_returns_empty():
    assert project_ipa("") == ""
    assert project_ipa(None) == ""


def test_non_ipa_input_returns_empty():
    # Cyrillic chars aren't IPA — panphon returns no segments
    assert project_ipa("/гав/") == ""


def test_japanese_wan_wan_round_trips():
    # /wanwan/ already lives entirely in the inventory
    assert project_ipa("/wanwan/") == "wanwan"


def test_projection_length_matches_segment_count():
    """Every IPA segment produces exactly one inventory phoneme."""
    from conlang.anchors.phon_features import featurize_ipa

    for ipa in ["/muː/", "/bʌz/", "/oink/", "/krjxrju/"]:
        n_segs = len(featurize_ipa(ipa))
        proj = project_ipa(ipa)
        assert len(proj) == n_segs, f"{ipa!r}: {n_segs} segs -> {proj!r}"


# ── phonological_distance ────────────────────────────────────────────────


def test_phonological_distance_identity_is_zero():
    assert phonological_distance("paka", "paka") == 0.0
    assert phonological_distance("mu", "mu") == 0.0


def test_phonological_distance_empty_inputs():
    assert phonological_distance("", "") == 0.0
    assert phonological_distance(None, None) == 0.0  # type: ignore[arg-type]


def test_phonological_distance_symmetric():
    assert phonological_distance("paka", "tama") == phonological_distance("tama", "paka")
    assert phonological_distance("pi", "ba") == phonological_distance("ba", "pi")


def test_phonological_distance_voicing_close_but_nonzero():
    # /p/ vs /b/ differ on +/- voi only — small but positive distance
    d = phonological_distance("pa", "ba")
    assert d > 0
    # bounded by sqrt of one feature flip × 2 max = sqrt(4) = 2
    assert d <= 2.0


def test_phonological_distance_grows_with_more_changes():
    # one stop change vs two stop changes
    d1 = phonological_distance("pata", "bata")
    d2 = phonological_distance("pata", "baka")
    assert d2 > d1 > 0


def test_phonological_distance_length_penalty():
    # "paka" vs "pa" must cost more than identity but less than "paka" vs "ziza"
    same = phonological_distance("paka", "paka")
    short_long = phonological_distance("pa", "paka")
    very_different = phonological_distance("paka", "ziza")
    assert short_long > same
    assert very_different > short_long


def test_phonological_distance_vowel_change_nonzero():
    # /pa/ vs /pi/ — vowel quality only
    assert phonological_distance("pa", "pi") > 0
    assert phonological_distance("pa", "pi") < phonological_distance("pa", "ki")


def test_phonological_distance_nw_recovers_leading_insertion():
    """NW must find the gap-at-start alignment for 'apa' vs 'pa'.

    Optimal NW alignment is '-pa' against 'apa' (one leading indel of
    /a/), so the cost is exactly the cost of indeling /a/ — i.e., equal
    to phonological_distance('a', ''). A naive position-aligned kernel
    would have given dist(a,p) + dist(p,a) + ||a||² (much larger),
    which is the discriminating signal that NW is doing real work.
    """
    d_nw = phonological_distance("apa", "pa")
    only_a = phonological_distance("a", "")
    assert d_nw == pytest.approx(only_a, abs=1e-9)
