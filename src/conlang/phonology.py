"""Stage 6 phonology: minimal Bantu-shaped phoneme inventory + affix table.

Scope: enough machinery to assign valid forms to the 1000 nodes in
data/processed/regularized.json (that work happens in Stage 6 form assignment,
not here). This module is pure data + composition.

Design choices:
- 5 vowels, 15 single consonants, 2 nasal digraphs (ny, ng) treated as single
  phonemes.
- Syllable template: (C)V. Onsets allow single consonants, the nasal digraphs,
  and four prenasalized clusters (mp, mb, nt, nd) that fall out of class-9
  N-prefix sandhi.
- 11 noun classes (six sg/pl pairs + class 11) with Swahili-shaped surface
  prefixes. Class 9 uses a homorganic nasal that surfaces as m-/n-/ng-
  depending on the following stop.
- Negation (per spec v0.2 §7 Commitment 7) is a single productive prefix
  `si-` that attaches outside any class prefix:
      [negation?] + [class] + [stem]
- Vowel hiatus across prefix + stem boundary is resolved by simple elision
  of the prefix-final vowel. No glide insertion (keeps the affix table flat).
"""

from __future__ import annotations

VOWELS: tuple[str, ...] = ("a", "e", "i", "o", "u")

SINGLE_CONSONANTS: tuple[str, ...] = (
    "p", "t", "k", "b", "d", "g",
    "m", "n",
    "s", "z", "f", "v",
    "l", "w", "y",
)

NASAL_DIGRAPHS: tuple[str, ...] = ("ny", "ng")
PRENASALIZED_ONSETS: tuple[str, ...] = ("mp", "mb", "nt", "nd")

_VOWEL_SET = frozenset(VOWELS)
_SINGLE_C_SET = frozenset(SINGLE_CONSONANTS)
_TWO_CHAR_ONSETS = frozenset(NASAL_DIGRAPHS + PRENASALIZED_ONSETS)


# class_id -> (surface prefix, description). Class 9 prefix is the abstract
# nasal "n" that gets homorganic in `apply_class_prefix`.
CLASS_PREFIXES: dict[int, tuple[str, str]] = {
    1:  ("mu", "human, singular"),
    2:  ("ba", "human, plural"),
    3:  ("u",  "plant/object, singular"),
    4:  ("mi", "plant/object, plural"),
    5:  ("li", "fruit/paired thing, singular"),
    6:  ("ma", "fruit/paired thing, plural"),
    7:  ("ki", "tool/thing, singular"),
    8:  ("vi", "tool/thing, plural"),
    9:  ("n",  "animal/language, singular (homorganic stop or yi-)"),
    10: ("zi", "animal/language, plural"),
    11: ("lu", "long thing / mass / abstract"),
}

NEGATION_PREFIX: str = "si"


def syllabify(word: str) -> list[str] | None:
    """Parse `word` into a list of (C)V syllables. Return None if invalid.

    Onset preference (greedy, longest match first):
      mp/mb/nt/nd/ny/ng + V  →  3 chars
      single C + V           →  2 chars
      V                      →  1 char
    """
    if not word:
        return None
    syllables: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        if i + 2 < n and word[i:i + 2] in _TWO_CHAR_ONSETS and word[i + 2] in _VOWEL_SET:
            syllables.append(word[i:i + 3])
            i += 3
            continue
        if i + 1 < n and word[i] in _SINGLE_C_SET and word[i + 1] in _VOWEL_SET:
            syllables.append(word[i:i + 2])
            i += 2
            continue
        if word[i] in _VOWEL_SET:
            syllables.append(word[i])
            i += 1
            continue
        return None
    return syllables


def is_valid_syllable(s: str) -> bool:
    syls = syllabify(s)
    return syls is not None and len(syls) == 1


def is_valid_word(word: str, min_syllables: int = 2) -> bool:
    """Bantu word-minimum: ≥ 2 syllables, strict (C)V phonotactics."""
    syls = syllabify(word)
    return syls is not None and len(syls) >= min_syllables


def _class_9_form(stem: str) -> str:
    """Class-9 surface form. Two allomorphs:

    - Prenasalized stop (mp, mb, nt, nd) for stems starting with p, b, t, d.
    - `yi-` (or `y-` before a vowel) for every other onset. Keeps the class-9
      marker phonotactically valid for the full inventory; pure homorganic
      sandhi can't because the conlang doesn't permit `nw`, `ns`, `nk` etc.
      as onsets.
    """
    if not stem:
        return "yi"
    c = stem[0]
    if c in ("p", "b"):
        return "m" + stem
    if c in ("t", "d"):
        return "n" + stem
    if c in _VOWEL_SET:
        return "y" + stem
    return "yi" + stem


def apply_class_prefix(stem: str, class_id: int) -> str:
    """Compose a class prefix with a stem.

    Class 9 uses the two-allomorph rule in `_class_9_form`. All other classes
    are simple concatenation with prefix-final-V eliding before a stem-initial V.
    """
    if class_id not in CLASS_PREFIXES:
        raise KeyError(f"unknown class_id={class_id}; valid: {sorted(CLASS_PREFIXES)}")
    if class_id == 9:
        return _class_9_form(stem)
    prefix, _desc = CLASS_PREFIXES[class_id]
    if prefix and stem and prefix[-1] in _VOWEL_SET and stem[0] in _VOWEL_SET:
        return prefix[:-1] + stem
    return prefix + stem


def negate(form: str) -> str:
    """Productive negation per Commitment 7. Prefixes `si-` outside class prefix.

    Hiatus: if `form` starts with a vowel, drop the prefix-final `i`.
    """
    if not form:
        return NEGATION_PREFIX
    if form[0] in _VOWEL_SET:
        return NEGATION_PREFIX[:-1] + form
    return NEGATION_PREFIX + form
