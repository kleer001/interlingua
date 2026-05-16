"""Tests for the pure-Python parts of ingest (no torch/HF/network)."""

import numpy as np

from conlang.ingest import (
    _parse_embedding,
    first_n_passing_filter,
    passes_filter_rubric,
)


def test_filter_rubric_keeps_abstract_concepts():
    keep = [
        "expressions of gratitude and thanks",
        "concepts related to fairness and justice",
        "descriptions of weather and atmospheric conditions",
        "instructions related to cooking and food preparation",
    ]
    for d in keep:
        assert passes_filter_rubric(d), f"should keep: {d!r}"


def test_filter_rubric_drops_code_and_formatting():
    drop = [
        "terms related to programming languages",
        "javascript variable declarations",
        "tokenizer positional features",
        "markdown formatting tokens",
        "url and email address fragments",
        "git commit messages",
    ]
    for d in drop:
        assert not passes_filter_rubric(d), f"should drop: {d!r}"


def test_filter_rubric_drops_too_short():
    assert not passes_filter_rubric("ok")
    assert not passes_filter_rubric("a b c")
    assert not passes_filter_rubric("")


def test_filter_rubric_drops_vague_labels():
    assert not passes_filter_rubric("various words")
    assert not passes_filter_rubric("miscellaneous tokens")


def test_filter_rubric_drops_token_locked_descriptions():
    """Descriptions that pin to a specific quoted word are token features, not concepts."""
    drop = [
        'the word "several" indicating quantity or frequency references',
        'instances of the conjunction "and" in various contexts',
        'expressions related to "ex" or "excessive" and its variants',
        'instances of the word "express" and related forms',
        'the preposition "from" in spatial contexts',
    ]
    for d in drop:
        assert not passes_filter_rubric(d), f"should drop: {d!r}"


def test_filter_rubric_drops_numerical_and_statistical_codes():
    drop = [
        "numerical codes in scientific contexts",
        "specific numerical codes or values, often in a scientific context",
        "statistical references in academic writing",
        "numerical data and statistical references",
        "specific identifiers used in datasets",
    ]
    for d in drop:
        assert not passes_filter_rubric(d), f"should drop: {d!r}"


def test_filter_rubric_drops_meta_textual_labels():
    drop = [
        "references to labeled data or metadata in documents",
        "metadata in text",
        "labeled data references",
    ]
    for d in drop:
        assert not passes_filter_rubric(d), f"should drop: {d!r}"


def test_filter_rubric_drops_vague_reference_phrases():
    assert not passes_filter_rubric("references to specific categories")
    assert not passes_filter_rubric("references to specific identifiers in text")


def test_first_n_passing_filter_orders_by_index_and_attaches_decoder():
    decoder = np.arange(40).reshape(10, 4).astype(np.float32)
    rows = [
        {"index": "3", "description": "code syntax tokens", "embedding": None},  # drop
        {"index": "1", "description": "concepts of love and affection", "embedding": None},  # keep
        {"index": "0", "description": "various tokens", "embedding": None},  # drop
        {"index": "2", "description": "descriptions of natural landscapes", "embedding": None},  # keep
        {"index": "4", "description": "expressions of grief and loss", "embedding": None},  # keep
    ]
    feats = first_n_passing_filter(rows, decoder, n=2)
    assert [f.feature_id for f in feats] == [1, 2]
    assert np.array_equal(feats[0].decoder_vec, decoder[1])
    assert np.array_equal(feats[1].decoder_vec, decoder[2])


def test_first_n_passing_filter_skips_out_of_range_indices():
    decoder = np.zeros((3, 2), dtype=np.float32)
    rows = [
        {"index": "5", "description": "expressions of curiosity", "embedding": None},
        {"index": "0", "description": "descriptions of family relationships", "embedding": None},
    ]
    feats = first_n_passing_filter(rows, decoder, n=5)
    assert [f.feature_id for f in feats] == [0]


def test_parse_embedding_handles_none_and_bad_json():
    assert _parse_embedding(None) is None
    assert _parse_embedding("not json") is None
    e = _parse_embedding("[0.1, 0.2, 0.3]")
    assert isinstance(e, np.ndarray)
    assert e.shape == (3,)
    assert np.allclose(e, [0.1, 0.2, 0.3])
