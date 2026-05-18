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

`phonological_distance` lives here too — same feature space, same
squared-Euclidean kernel — because Phase 3 of the Stage-6 cutover
(`semanticphonology.md`) needs to score how well interpolation preserves
substrate geometry against the metric used for projection.
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


def phonological_distance(stem_a: str, stem_b: str) -> float:
    """Position-aligned panphon-feature Euclidean distance between two stems.

    Both inputs are featurized segment-by-segment via panphon. The two
    segment-vector lists are aligned by position; positions beyond the
    shorter list contribute the squared L2 norm of the longer's vector at
    that index (i.e., the missing-segment penalty is whatever feature
    weight the present segment carries). The returned distance is the
    sqrt of the summed squared deviations, i.e., a proper L2 over the
    concatenated 24-d-per-segment feature space with zero-padding.

    This is the metric Phase 3 of the Stage-6 cutover tunes Spearman ρ
    against (`semanticphonology.md` §3 Phase 3). Identity returns 0.0;
    swapping arguments preserves the value (symmetric).

    Position alignment is deliberate: it matches the existing
    `_distance` kernel and keeps the metric cheap. If Phase 3 reveals
    that position alignment is too brittle for variable-length stems,
    the natural upgrade is Needleman-Wunsch over the same per-segment
    squared-Euclidean cost — same metric semantics, different alignment.
    """
    fa = featurize_ipa(stem_a) if stem_a else []
    fb = featurize_ipa(stem_b) if stem_b else []
    if not fa and not fb:
        return 0.0
    n = max(len(fa), len(fb))
    dim = len(fa[0]) if fa else len(fb[0])
    zero = [0] * dim
    sq = 0
    for i in range(n):
        va = fa[i] if i < len(fa) else zero
        vb = fb[i] if i < len(fb) else zero
        sq += _distance(va, vb)
    return sq**0.5
