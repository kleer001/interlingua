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
    """Needleman-Wunsch alignment over per-segment squared-Euclidean cost.

    Both inputs are featurized segment-by-segment via panphon. NW DP finds
    the alignment minimizing total cost over three operations:
      - substitution: squared-Euclidean distance between segment vectors
      - indel (gap):  squared L2 norm of the segment being indeled (its
        distance to the zero vector — same penalty the prior
        position-aligned kernel charged for trailing unmatched segments)
    The returned distance is sqrt of the optimal alignment cost.

    NW supersedes the position-aligned kernel (introduced in 7343d35)
    after Phase 3 Spearman ρ landed at 0.03 against the N=2000 substrate:
    variable-length stems and shifted insertions were forced into an
    arbitrary positional zip that washed out fine structure. Same panphon
    kernel; different alignment. Identity returns 0.0; symmetric under
    argument swap.
    """
    fa = featurize_ipa(stem_a) if stem_a else []
    fb = featurize_ipa(stem_b) if stem_b else []
    if not fa and not fb:
        return 0.0
    n_a, n_b = len(fa), len(fb)
    dim = len(fa[0]) if fa else len(fb[0])
    zero = [0] * dim
    inf = float("inf")
    dp = [[inf] * (n_b + 1) for _ in range(n_a + 1)]
    dp[0][0] = 0
    for i in range(1, n_a + 1):
        dp[i][0] = dp[i - 1][0] + _distance(fa[i - 1], zero)
    for j in range(1, n_b + 1):
        dp[0][j] = dp[0][j - 1] + _distance(fb[j - 1], zero)
    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + _distance(fa[i - 1], fb[j - 1]),
                dp[i - 1][j] + _distance(fa[i - 1], zero),
                dp[i][j - 1] + _distance(fb[j - 1], zero),
            )
    return dp[n_a][n_b] ** 0.5
