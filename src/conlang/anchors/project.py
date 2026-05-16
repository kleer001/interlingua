"""Project an IPA string onto the 10C/5V CV(n) inventory.

For each segment in the input IPA:
- get its panphon 24-dimensional feature vector
- compute squared-Euclidean distance to every inventory phoneme's vector
- emit the nearest inventory phoneme

The result is a string in the project's canonical inventory — uniform
across every row in the matrix regardless of which source supplied the
original IPA. This is the layer the sketch calls "self-contained": once
a form is projected, all the heterogeneity of the source IPA (slashes
vs brackets, stress marks vs no, fine vs broad transcription) is washed
out.

Syllable structure (CV vs CVN) is NOT enforced at the segment-projection
layer — that's a downstream cleanup that may want to insert epenthetic
vowels or drop illegal codas. Projecting first, syllabifying after.
"""

from __future__ import annotations

from .inventory import ALL_PHONEMES, phoneme_features
from .phon_features import featurize_ipa


def _distance(a: list[int], b: list[int]) -> int:
    """Squared-Euclidean. Both inputs are 24-d ternary vectors so this is
    a small fixed-cost loop; no need for numpy.
    """
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=True))


def project_segment(seg: list[int]) -> str:
    """Return the inventory phoneme closest to `seg` by feature distance.

    Ties broken by inventory order (CONSONANTS first, then VOWELS), which
    biases toward consonants when input is ambiguous — desirable since the
    matrix should preserve consonantal place/manner cues over vowel
    quality cues that wash out cross-linguistically anyway.
    """
    feats = phoneme_features()
    best = None
    best_d = None
    for p in ALL_PHONEMES:
        d = _distance(seg, feats[p])
        if best_d is None or d < best_d:
            best_d = d
            best = p
    return best  # type: ignore[return-value]


def project_ipa(ipa: str | None) -> str:
    """Project an IPA string -> inventory string. Empty input -> empty string."""
    if not ipa:
        return ""
    segs = featurize_ipa(ipa)
    if not segs:
        return ""
    return "".join(project_segment(s) for s in segs)


def project_entries(entries, *, field: str = "projected"):
    """Yield (entry, projection) for each entry that has IPA. Caller decides
    whether to mutate the entry or build a sidecar table.
    """
    for e in entries:
        p = project_ipa(e.ipa) if e.ipa else ""
        yield e, p
