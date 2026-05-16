"""Phase-6 skeleton: bind concepts to semantic-attribute bundles.

In the anchor-pool design (anchor-pool-sketch.md §"Attribute bucket"), each
animal concept is not a single point in concept space but a *cloud* of
attribute points. The attribute list for snake is its real anchor —
{close-to-ground, elongated, sudden-strike, ...} — and the snake-shaped
cloud is what the LLM-extracted concepts get interpolated against.

The registry below is the *starter* hand-curated hypothesis (per the plan's
Phase 6 step). The real attribute set will come from the SAE feature
clustering once Stage 4 stabilizes; this file is the wiring that lets
Phase 6 emit `(concept, attribute, phonological_signature)` rows the moment
that embedding is available.

Cultural attributes are flagged separately because they're not universal
and may need per-language weighting in the final Voronoi tessellation.

This module does NOT do the embedding step itself. The function
`build_attribute_anchor_table()` returns rows with a stub
`attribute_embedding=None`; Stage 5 will fill that field by passing the
attribute text through Gemma 2 2B + the slice's SAE encoder and reading
back the active-feature vector.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schema import AnchorEntry, read_jsonl


@dataclass(frozen=True)
class AttributeBundle:
    concept: str
    attributes: tuple[str, ...]
    cultural_attributes: tuple[str, ...] = field(default_factory=tuple)


# fmt: off
ATTRIBUTE_REGISTRY: dict[str, AttributeBundle] = {
    "snake_hissing": AttributeBundle(
        concept="snake_hissing",
        attributes=(
            "close-to-ground", "elongated", "smooth-scaled",
            "flexible", "sinuous", "sudden-strike", "concealed",
            "cold-blooded", "venomous", "silent-movement",
        ),
        cultural_attributes=("cunning", "wisdom", "fertility-renewal"),
    ),
    "dog_barking": AttributeBundle(
        concept="dog_barking",
        attributes=(
            "domestic", "loyal", "alert", "protective", "playful",
            "loud", "fast-runner", "carnivore", "four-legged",
        ),
        cultural_attributes=("faithful-friend", "guardian"),
    ),
    "cat_meowing": AttributeBundle(
        concept="cat_meowing",
        attributes=(
            "domestic", "independent", "agile", "stealthy", "small-predator",
            "nocturnal", "feline", "four-legged",
        ),
        cultural_attributes=("aloof", "lucky-or-unlucky"),
    ),
    "cow_mooing": AttributeBundle(
        concept="cow_mooing",
        attributes=(
            "large", "domesticated-livestock", "herbivore", "ruminant",
            "slow-moving", "milk-producing", "four-legged", "horned",
        ),
        cultural_attributes=("sacred", "abundance"),
    ),
    "pig_grunting": AttributeBundle(
        concept="pig_grunting",
        attributes=(
            "domestic-livestock", "omnivore", "intelligent", "muddy",
            "stout", "pink-or-spotted", "four-legged", "rooting",
        ),
        cultural_attributes=("unclean", "greedy", "lucky-piggy-bank"),
    ),
    "horse_whinnying": AttributeBundle(
        concept="horse_whinnying",
        attributes=(
            "large", "fast-runner", "herbivore", "domesticated",
            "noble", "four-legged", "hooved", "strong",
        ),
        cultural_attributes=("freedom", "warrior", "elegance"),
    ),
    "rooster_crowing": AttributeBundle(
        concept="rooster_crowing",
        attributes=(
            "diurnal", "domestic-bird", "dawn-call", "loud",
            "territorial", "two-legged", "feathered",
        ),
        cultural_attributes=("vigilance", "courage", "pride"),
    ),
    "owl_hooting": AttributeBundle(
        concept="owl_hooting",
        attributes=(
            "nocturnal", "predator", "silent-flight", "large-eyes",
            "two-legged", "feathered", "perched",
        ),
        cultural_attributes=("wisdom", "ill-omen", "watchfulness"),
    ),
    "bee_buzzing": AttributeBundle(
        concept="bee_buzzing",
        attributes=(
            "small", "flying", "stinging", "colony", "busy",
            "honey-producing", "six-legged", "winged",
        ),
        cultural_attributes=("industriousness", "sweetness"),
    ),
    "frog_croaking": AttributeBundle(
        concept="frog_croaking",
        attributes=(
            "amphibian", "wet", "jumping", "small", "cold-blooded",
            "rain-associated", "four-legged",
        ),
        cultural_attributes=("transformation", "luck", "fertility"),
    ),
    "lion_roaring": AttributeBundle(
        concept="lion_roaring",
        attributes=(
            "large-predator", "mane", "loud", "regal",
            "savannah", "four-legged", "feline", "carnivore",
        ),
        cultural_attributes=("royalty", "courage", "ferocity"),
    ),
    "mouse_squeaking": AttributeBundle(
        concept="mouse_squeaking",
        attributes=(
            "tiny", "fast", "fearful", "nibbling",
            "four-legged", "tailed", "rodent",
        ),
        cultural_attributes=("timid", "household-pest"),
    ),
    "crow_calling": AttributeBundle(
        concept="crow_calling",
        attributes=(
            "black", "intelligent", "scavenger", "loud", "perched",
            "two-legged", "feathered",
        ),
        cultural_attributes=("ill-omen", "mystery", "cleverness"),
    ),
    "wolf_howling": AttributeBundle(
        concept="dog_howling",  # canonical slug includes wolf
        attributes=(
            "wild", "predator", "pack", "loud", "long-howl",
            "four-legged", "carnivore",
        ),
        cultural_attributes=("wilderness", "loneliness", "lunar"),
    ),
}
# fmt: on


@dataclass
class ConceptSignature:
    """Cross-linguistic phonological signature aggregated over one concept.

    This is the *fuzzy* version: list of forms keyed by language. Sharpness
    (how much variance across languages) is computable from it but not
    pre-computed here — that's a downstream measurement step.
    """

    concept: str
    n_languages: int
    n_entries: int
    languages: list[str]
    orthographies: list[str]
    ipas: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def signature_for(slug: str, entries: Iterable[AnchorEntry]) -> ConceptSignature:
    rows = [e for e in entries if e.concept == slug]
    langs = sorted({e.language_code for e in rows if e.language_code})
    return ConceptSignature(
        concept=slug,
        n_languages=len(langs),
        n_entries=len(rows),
        languages=langs,
        orthographies=[e.orthography for e in rows if e.orthography],
        ipas=[e.ipa for e in rows if e.ipa],
    )


@dataclass
class AttributeAnchor:
    concept: str
    attribute: str
    cultural: bool
    signature: ConceptSignature
    # Filled in Stage 5: vector position in SAE feature space.
    attribute_embedding: list[float] | None = None
    # Filled in Stage 6: aggregated articulatory-feature vector.
    phonological_features: list[float] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def build_attribute_anchor_table(
    entries: Iterable[AnchorEntry],
    registry: dict[str, AttributeBundle] = ATTRIBUTE_REGISTRY,
) -> list[AttributeAnchor]:
    entries_list = list(entries)
    out: list[AttributeAnchor] = []
    for bundle in registry.values():
        sig = signature_for(bundle.concept, entries_list)
        for attr in bundle.attributes:
            out.append(
                AttributeAnchor(
                    concept=bundle.concept, attribute=attr, cultural=False, signature=sig
                )
            )
        for attr in bundle.cultural_attributes:
            out.append(
                AttributeAnchor(
                    concept=bundle.concept, attribute=attr, cultural=True, signature=sig
                )
            )
    return out


def write_attribute_anchors(table: Iterable[AttributeAnchor], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in table:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def run_from_jsonl(anchors_jsonl: Path, dest_jsonl: Path) -> Path:
    """Convenience: read anchors-v1.jsonl, write attribute-anchors.jsonl."""
    entries = read_jsonl(anchors_jsonl)
    table = build_attribute_anchor_table(entries)
    write_attribute_anchors(table, dest_jsonl)
    return dest_jsonl
