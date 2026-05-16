"""panphon feature extraction tests."""

import pytest

from conlang.anchors.phon_features import (
    feature_names,
    featurize_form,
    featurize_ipa,
    mean_var,
    normalize_ipa,
)

# Skip the suite if panphon isn't installed (it's an optional `anchors` dep).
panphon = pytest.importorskip("panphon")


def test_normalize_strips_slash_wrappers():
    assert normalize_ipa("/wuːf/") == "wuːf"
    assert normalize_ipa("[hap]") == "hap"
    assert normalize_ipa("  /m oː/  ") == "moː"


def test_normalize_strips_stress_marks():
    assert "ˈ" not in normalize_ipa("/ˈbiː/")
    assert "ˌ" not in normalize_ipa("/baˌba/")


def test_feature_names_count():
    # panphon's 24-feature schema (PHOIBLE-aligned)
    assert len(feature_names()) == 24
    assert "voi" in feature_names()
    assert "cons" in feature_names()


def test_featurize_ipa_returns_one_vector_per_segment():
    segs = featurize_ipa("muː")
    assert len(segs) == 2  # m + uː
    assert all(len(s) == 24 for s in segs)
    # 'm' is sonorant + nasal, 'uː' is syllabic
    assert segs[0][feature_names().index("nas")] == 1
    assert segs[1][feature_names().index("syl")] == 1


def test_featurize_ipa_handles_empty_and_non_ipa():
    assert featurize_ipa("") == []
    # Cyrillic isn't IPA — panphon should return an empty list
    assert featurize_ipa("/гав/") == []


def test_featurize_form_prefers_ipa_when_available():
    assert featurize_form(ipa="/muː/", orthography="moo") == featurize_ipa("/muː/")
    # Without IPA we return nothing (we don't trust orthography as IPA)
    assert featurize_form(ipa=None, orthography="moo") == []


def test_mean_var_constant_input_has_zero_variance():
    rows = [[1, 1, 0, -1] for _ in range(5)]
    means, vars_ = mean_var(rows)
    assert means == [1.0, 1.0, 0.0, -1.0]
    assert vars_ == [0.0, 0.0, 0.0, 0.0]


def test_mean_var_balanced_input_has_unit_variance():
    rows = [[1], [-1], [1], [-1]]
    means, vars_ = mean_var(rows)
    assert means == [0.0]
    assert vars_ == [1.0]


def test_mean_var_empty_input_returns_zero_vectors():
    means, vars_ = mean_var([])
    assert means == [0.0] * 24
    assert vars_ == [0.0] * 24
