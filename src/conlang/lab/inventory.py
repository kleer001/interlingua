"""The project's canonical 10C/5V CV(n) phoneme inventory.

Sketch §"The premise" and §"Aggregation mechanics" both reference a fixed
10-consonant / 5-vowel inventory with optional /n/ coda. Every cross-
linguistic IPA form is projected onto this inventory so the per-concept
phonological signature lives in a uniform representation regardless of
where its segments came from (Epitran transliteration, Wikipedia native
IPA, Wiktionary form-page IPA, manual transcription, ...).

Chosen for typological accessibility — every segment here is in the vast
majority of the world's languages. /p t k/ are the three places of
voiceless stop; /s/ the only fricative; /m n/ the two universal nasals;
/l r/ the two universal liquids; /w/ the most common glide; /h/ the
glottal anchor. Vowels are the standard 5-vowel "a e i o u" system that
half the world's languages use.

The "(n)" in CV(n) means an optional /n/ coda; the syllable shape isn't
enforced at the segment-projection layer (see project.py), but the
inventory deliberately omits other potential codas to keep the door open
for that constraint downstream.
"""

from __future__ import annotations

from functools import lru_cache

# Order matters: it's the canonical order used everywhere downstream.
CONSONANTS: tuple[str, ...] = ("p", "t", "k", "s", "h", "m", "n", "l", "r", "w")
VOWELS: tuple[str, ...] = ("a", "e", "i", "o", "u")

ALL_PHONEMES: tuple[str, ...] = CONSONANTS + VOWELS

# Optional coda. CV(n) means a syllable may end in /n/.
CODA: str = "n"


@lru_cache(maxsize=1)
def phoneme_features() -> dict[str, list[int]]:
    """Return `{phoneme: 24-dim panphon feature vector}` for every inventory
    phoneme. Computed once and cached.
    """
    from .phon_features import featurize_ipa

    out: dict[str, list[int]] = {}
    for p in ALL_PHONEMES:
        segs = featurize_ipa(p)
        if not segs:
            raise RuntimeError(f"panphon could not featurize inventory phoneme {p!r}")
        out[p] = segs[0]
    return out


def is_consonant(p: str) -> bool:
    return p in CONSONANTS


def is_vowel(p: str) -> bool:
    return p in VOWELS
