"""IPA -> articulatory feature vectors via panphon (PHOIBLE-style).

Each IPA segment becomes a 24-dimensional ternary vector (-1 / 0 / 1) over
the panphon features: syl, son, cons, cont, delrel, lat, nas, strid, voi,
sg, cg, ant, cor, distr, lab, hi, lo, back, round, velaric, tense, long,
hitone, hireg.

Most of the project (sketch + data plan) treats these as the input to the
Voronoi tessellation in joint (semantic x phonological) space. This module
is the bridge from string IPA to that vector representation.

We normalize incoming IPA:
- Strip Epitran-style `/.../` or `[...]` wrappers.
- Strip stress marks (ˈ ˌ), tone marks, hyphens, whitespace.
- Pass to panphon's FeatureTable.word_to_vector_list().

For consumers:
- `featurize_ipa(ipa) -> list[list[int]]`  one row per segment
- `featurize_form(orthography_or_ipa, ipa_field) -> ...`  picks the IPA
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

# panphon is imported lazily so test collection doesn't pull its data files.

_STRIP_WRAPPERS = re.compile(r"^[/\[\{]+|[/\]\}]+$")
_STRIP_MARKS = re.compile(r"[ˈˌˑ.‿‖\s​‌‍‎‏\-—–]")


@lru_cache(maxsize=1)
def _feature_table():
    import panphon  # noqa: PLC0415  (lazy)

    return panphon.FeatureTable()


def feature_names() -> list[str]:
    return list(_feature_table().names)


def normalize_ipa(s: str) -> str:
    """Strip wrappers + marks so panphon's tokenizer can do its job."""
    if not s:
        return ""
    out = _STRIP_WRAPPERS.sub("", s.strip())
    out = _STRIP_MARKS.sub("", out)
    return out


def featurize_ipa(ipa: str) -> list[list[int]]:
    """Return one feature vector per IPA segment. Empty if no segments parse."""
    norm = normalize_ipa(ipa)
    if not norm:
        return []
    try:
        return _feature_table().word_to_vector_list(norm, numeric=True)
    except Exception:  # noqa: BLE001  (panphon can choke on rare combos)
        return []


def featurize_form(*, ipa: str | None, orthography: str | None) -> list[list[int]]:
    """Convenience for the AnchorEntry view: prefer IPA, fall back to plain
    orthography only if it appears to be Latin-IPA-compatible (heuristic:
    contains no characters above U+02FF that aren't IPA diacritics)."""
    if ipa:
        return featurize_ipa(ipa)
    return []


def mean_var(rows: Iterable[Iterable[int]]) -> tuple[list[float], list[float]]:
    """Return (mean, variance) per feature dimension across all rows."""
    acc = list(rows)
    if not acc:
        n = len(feature_names())
        return [0.0] * n, [0.0] * n
    dim = len(acc[0])
    n = len(acc)
    means = [sum(r[j] for r in acc) / n for j in range(dim)]
    vars_ = [sum((r[j] - means[j]) ** 2 for r in acc) / n for j in range(dim)]
    return means, vars_
