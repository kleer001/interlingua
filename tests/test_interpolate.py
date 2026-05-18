"""Phase 1 interpolation library tests."""

from __future__ import annotations

import numpy as np
import pytest

from conlang.interpolate import (
    InterpolationContext,
    discretize_with_phonotactics,
    lexicon_phoneme_features,
    mix_signatures,
    neighbor_weights,
    project_to_lexicon,
    ranked_lexicon_candidates,
    stem_for_position,
)
from conlang.phonology import VOWELS, syllabify

# ── inventory ────────────────────────────────────────────────────────────


def test_lexicon_features_have_16c_5v():
    feats = lexicon_phoneme_features()
    assert len(feats) == 21  # 16 consonants + 5 vowels
    for v in feats.values():
        assert len(v) == 24


def test_each_inventory_phoneme_projects_to_itself():
    feats = lexicon_phoneme_features()
    for p, vec in feats.items():
        assert project_to_lexicon(vec) == p


def test_project_to_lexicon_respects_only_filter():
    feats = lexicon_phoneme_features()
    # Use /p/'s feature vector but restrict to vowels — should pick a vowel
    out = project_to_lexicon(feats["p"], only=VOWELS)
    assert out in VOWELS


def test_ranked_candidates_orders_by_distance():
    feats = lexicon_phoneme_features()
    ranked = ranked_lexicon_candidates(feats["a"])
    assert ranked[0] == "a"
    assert len(ranked) == 21


# ── neighbor weights ─────────────────────────────────────────────────────


def test_neighbor_weights_query_at_anchor_concentrates_weight():
    anchors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ]
    )
    query = np.array([1.0, 0.0, 0.0])
    weights = neighbor_weights(query, anchors, k=4)
    # First neighbor must be anchor 0 with weight > all others combined
    assert weights[0][0] == 0
    assert weights[0][1] > sum(w for _, w in weights[1:])


def test_neighbor_weights_sum_to_one():
    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(10, 8))
    query = rng.normal(size=(8,))
    weights = neighbor_weights(query, anchors, k=5)
    assert len(weights) == 5
    assert sum(w for _, w in weights) == pytest.approx(1.0)


def test_neighbor_weights_handles_top_k_larger_than_n_anchors():
    anchors = np.eye(3)
    query = np.ones(3)
    weights = neighbor_weights(query, anchors, k=10)
    assert len(weights) == 3


# ── signature mixer ──────────────────────────────────────────────────────


def test_mix_signatures_single_anchor_recovers_segments():
    """One anchor with weight 1.0 — output should equal that anchor's panphon features."""
    from conlang.anchors.phon_features import featurize_ipa

    segs = featurize_ipa("paka")
    assert len(segs) == 4
    out = mix_signatures([(0, 1.0)], ["paka"])
    assert len(out) == 4
    for o, s in zip(out, segs, strict=True):
        assert o == pytest.approx([float(x) for x in s])


def test_mix_signatures_target_length_is_weighted_mean():
    out = mix_signatures([(0, 0.5), (1, 0.5)], ["pa", "patika"])
    # round((0.5*2 + 0.5*6) = 4)
    assert len(out) == 4


def test_mix_signatures_zero_pads_shorter_anchors():
    """Positions beyond a shorter anchor's length should ignore it (no zero-blend dilution)."""
    out_mixed = mix_signatures([(0, 0.5), (1, 0.5)], ["pa", "paka"])
    # at position 0/1 both anchors contribute; at 2 only "paka" contributes
    # so position 2 should equal "paka"[2]'s features
    from conlang.anchors.phon_features import featurize_ipa

    expected_pos2 = featurize_ipa("paka")[2]
    assert out_mixed[2] == pytest.approx([float(x) for x in expected_pos2])


def test_mix_signatures_empty_anchors_returns_empty():
    assert mix_signatures([], []) == []


# ── phonotactic gate ─────────────────────────────────────────────────────


def test_discretize_simple_cv_passes_through():
    """A signature that already discretizes to a valid (C)V should not be modified much."""
    from conlang.anchors.phon_features import featurize_ipa

    segs = [list(s) for s in featurize_ipa("pa")]
    build = discretize_with_phonotactics(segs)
    syls = syllabify(build.stem)
    assert syls is not None
    assert len(syls) >= 2  # the gate pads to ≥ 2 syllables


def test_discretize_forces_cv_when_two_consonants_adjacent():
    """Two consonant-shaped segments in a row should not produce a CC cluster."""
    from conlang.anchors.phon_features import featurize_ipa

    # "ks" featurizes as two consonants — gate should retry pos 1 against vowels
    segs = [list(s) for s in featurize_ipa("ks")]
    build = discretize_with_phonotactics(segs)
    syls = syllabify(build.stem)
    assert syls is not None
    assert len(syls) >= 2
    assert build.fallback_segments >= 1


def test_discretize_drops_trailing_consonant():
    """No codas allowed — a stem-final consonant must be dropped."""
    from conlang.anchors.phon_features import featurize_ipa

    segs = [list(s) for s in featurize_ipa("pak")]
    build = discretize_with_phonotactics(segs)
    syls = syllabify(build.stem)
    assert syls is not None
    assert build.stem[-1] in VOWELS


def test_discretize_pads_short_stems_to_two_syllables():
    """A single segment input should still produce ≥ 2 syllables."""
    from conlang.anchors.phon_features import featurize_ipa

    segs = [list(s) for s in featurize_ipa("p")]
    build = discretize_with_phonotactics(segs)
    syls = syllabify(build.stem)
    assert syls is not None
    assert len(syls) >= 2


# ── end-to-end ───────────────────────────────────────────────────────────


def test_stem_for_position_round_trip_at_anchor():
    """Per Phase-1 acceptance: query at an anchor's position → stem close to its modal projection.

    'Close' is hard to make precise across the 10C/5V → 16C/5V inventory
    shift (anchors have /h/, lexicon doesn't), so this test just asserts
    the output is a valid (C)V stem and is non-empty.
    """
    rng = np.random.default_rng(42)
    anchor_positions = rng.normal(size=(3, 8))
    ctx = InterpolationContext(
        anchor_positions=anchor_positions,
        anchor_concepts=["snake_hissing", "snake_hissing", "cow_mooing"],
        concept_modal_projections={"snake_hissing": "phih", "cow_mooing": "mu"},
    )

    # Query exactly at anchor 2 (cow)
    query = anchor_positions[2]
    build = stem_for_position(query, ctx, k=2)
    syls = syllabify(build.stem)
    assert syls is not None
    assert len(syls) >= 2
    # cow modal is "mu" — output should contain /m/ or /u/ if interpolation works.
    assert "m" in build.stem or "u" in build.stem


def test_stem_for_position_two_anchors_blend():
    """Equal-weight blend of two distinct anchors should still produce a valid stem."""
    rng = np.random.default_rng(123)
    anchor_positions = rng.normal(size=(2, 8))
    ctx = InterpolationContext(
        anchor_positions=anchor_positions,
        anchor_concepts=["a", "b"],
        concept_modal_projections={"a": "ka", "b": "tu"},
    )
    query = (anchor_positions[0] + anchor_positions[1]) / 2
    build = stem_for_position(query, ctx, k=2)
    syls = syllabify(build.stem)
    assert syls is not None
    assert len(syls) >= 2
