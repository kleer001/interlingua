"""Gloss-keyword matching: inflection-aware stem matching."""

from conlang.anchors.parse_wiktionary import _gloss_matches


def test_exact_word_match():
    assert _gloss_matches("the sound of a dog barking", frozenset({"bark"}))
    assert _gloss_matches("the sound of a dog barking", frozenset({"dog"}))


def test_inflection_pair_matches_sneeze_sneezing():
    assert _gloss_matches("the sound of a sneeze", frozenset({"sneezing"}))


def test_inflection_pair_matches_snake_snakes():
    assert _gloss_matches("the hissing of snakes", frozenset({"snake"}))


def test_inflection_pair_matches_buzz_buzzing():
    assert _gloss_matches("buzzing of a bee", frozenset({"buzz"}))


def test_does_not_match_cat_cattle():
    """``cat`` must NOT match ``cattle`` even though it's a prefix."""
    assert not _gloss_matches("domesticated cattle", frozenset({"cat"}))


def test_no_match_for_unrelated_words():
    assert not _gloss_matches("a yarn used in weaving", frozenset({"dog", "bark"}))


def test_short_keywords_only_exact():
    """Three-letter keywords need exact match — no prefix expansion."""
    assert _gloss_matches("a small bee", frozenset({"bee"}))
    # 'ant' must not match 'antelope'
    assert not _gloss_matches("an antelope grazing", frozenset({"ant"}))
