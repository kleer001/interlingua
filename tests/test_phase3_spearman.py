"""Smoke test for scripts/phase3_spearman.py.

Imports the module and exercises its helper functions on a tiny toy input.
Does NOT run the full computation against the 2000-feature substrate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "conlang"
    / "lab"
    / "validate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("conlang.lab.validate", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_imports():
    mod = _load_module()
    assert hasattr(mod, "main")
    assert hasattr(mod, "sample_pairs")
    assert hasattr(mod, "stem_to_ipa")


def test_stem_to_ipa_applies_override():
    mod = _load_module()
    # Latin 'g' -> IPA 'ɡ'; Latin 'y' -> IPA 'j'; others pass through.
    assert mod.stem_to_ipa("aga") == "aɡa"
    assert mod.stem_to_ipa("yala") == "jala"
    assert mod.stem_to_ipa("tara") == "tara"


def test_sample_pairs_deterministic_and_unique():
    mod = _load_module()
    pairs = mod.sample_pairs(n_features=20, n_pairs=5, seed=0)
    assert len(pairs) == 5
    assert len(set(pairs)) == 5
    for i, j in pairs:
        assert 0 <= i < 20 and 0 <= j < 20
        assert i < j  # canonical ordering
    # Reproducible
    again = mod.sample_pairs(n_features=20, n_pairs=5, seed=0)
    assert pairs == again


def test_toy_pair_distance_pipeline():
    """End-to-end on 5 hand-picked pairs without the substrate parquet."""
    mod = _load_module()
    from scipy.spatial.distance import cosine
    from scipy.stats import spearmanr

    from conlang.lab.project import phonological_distance

    # Five toy "stems" of varying similarity
    stems = ["tara", "tana", "kapa", "miru", "saba"]
    stem_ipa = [mod.stem_to_ipa(s) for s in stems]

    # Toy decoder vectors so we can compute cosine
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((5, 8))

    pairs = mod.sample_pairs(n_features=5, n_pairs=5, seed=0)
    cos_d = np.array([float(cosine(vecs[i], vecs[j])) for i, j in pairs])
    phon_d = np.array(
        [phonological_distance(stem_ipa[i], stem_ipa[j]) for i, j in pairs]
    )
    assert cos_d.shape == phon_d.shape == (5,)
    assert (phon_d >= 0).all()
    # Self-distance sanity: identity returns 0
    assert phonological_distance(stem_ipa[0], stem_ipa[0]) == pytest.approx(0.0)
    # spearmanr accepts the arrays without error
    rho, _ = spearmanr(cos_d, phon_d)
    assert -1.0 <= rho <= 1.0
