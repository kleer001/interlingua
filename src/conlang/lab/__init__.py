"""Phonology experiment toolkit.

This subpackage holds the reusable apparatus from the Phase 3 cutover
experiment: a Needleman-Wunsch phonological-distance kernel, an
attribute-bundle anchor pool, a dual-mode Gemma embedding probe, the
Spearman validator, and the small inventory + signature helpers that
hang off them. The Phase 3 result narrative lives alongside the tools
in `results/`.

Designed to be forked or `pip install -e`'d from a sibling-repo conlang
project. Not on PyPI. For the full project context see
`docs/phase3-onomatopoeia.md`.

The heavy result artifacts (substrate parquets, anchor-position parquets)
are not bundled. They are expected at ``$CONLANG_DATA_ROOT`` or, by
default, ``/media/menser/fauna/interlingua/data/processed/`` — generated
by ``embed_positions.py`` and consumed by ``validate.py``. Override with
``CONLANG_DATA_ROOT=/your/path`` if running outside the original host.
"""

from __future__ import annotations

from .attributes import ATTRIBUTE_REGISTRY, AttributeBundle, build_attribute_anchor_table
from .concepts import CONCEPTS, ConceptDef, canonical_slug
from .embed_positions import (
    embed_texts,
    load_attribute_rows,
    load_concept_rows,
)
from .inventory import ALL_PHONEMES, CONSONANTS, VOWELS, phoneme_features
from .phon_features import (
    feature_names,
    featurize_ipa,
    mean_var,
    normalize_ipa,
)
from .project import phonological_distance, project_ipa, project_segment
from .schema import AnchorEntry, read_jsonl, write_jsonl
from .signatures import (
    ConceptSignature,
    build_all_signatures,
    signature_for_concept,
)

__all__ = [
    "ALL_PHONEMES",
    "ATTRIBUTE_REGISTRY",
    "AnchorEntry",
    "AttributeBundle",
    "CONCEPTS",
    "CONSONANTS",
    "ConceptDef",
    "ConceptSignature",
    "VOWELS",
    "build_all_signatures",
    "build_attribute_anchor_table",
    "canonical_slug",
    "embed_texts",
    "feature_names",
    "featurize_ipa",
    "load_attribute_rows",
    "load_concept_rows",
    "mean_var",
    "normalize_ipa",
    "phoneme_features",
    "phonological_distance",
    "project_ipa",
    "project_segment",
    "read_jsonl",
    "signature_for_concept",
    "write_jsonl",
]
