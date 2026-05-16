"""Canonical anchor record schema.

One row per `(concept, language, source)`. Sources may disagree; we keep them
separate at this layer and only merge/dedupe downstream when computing the
cross-linguistic phonological signature per concept.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AnchorEntry:
    concept: str  # e.g. "dog_bark" — slug from our inventory
    category: str  # e.g. "mammal" — see anchor-data-plan.md table
    language: str  # canonical English name, e.g. "Japanese"
    language_code: str | None  # ISO 639-3 if confidently mappable, else None
    orthography: str  # native-script form, e.g. "ワンワン"
    romanization: str | None  # latin transliteration if given, else None
    ipa: str | None  # IPA if given by source, else None (filled later)
    source: str  # e.g. "wikipedia:cross-linguistic-onomatopoeias"
    source_url: str
    source_revid: str | None  # for reproducibility (Wikipedia revision id, etc.)
    captured_at: str  # ISO date of capture
    notes: str | None = None  # free text — disambiguators, dialect tags
    extra: dict = field(default_factory=dict)  # source-specific structured extras

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_jsonl(entries: Iterable[AnchorEntry], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.to_jsonl())
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[AnchorEntry]:
    out: list[AnchorEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(AnchorEntry(**d))
    return out
