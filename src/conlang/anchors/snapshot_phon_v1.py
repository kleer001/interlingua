"""Freeze the phon-side anchor pool into a single parquet snapshot.

Per `semanticphonology.md` Phase 0, the cutover needs a frozen anchor
table that Phase 1 (interpolation) reads from. This module joins the
three live anchor outputs

- `anchoring/processed/signatures-v1.jsonl`     — per-concept modal projection,
                                                  mean / var panphon features
- `anchoring/processed/attribute-anchors.jsonl` — per (concept, attribute) rows
- `anchoring/processed/anchors-v1.jsonl`        — per (concept, language) rows;
                                                  here only the per-concept counts
                                                  are kept for provenance

into one parquet at `data/processed/anchors-v1.parquet`, one row per
concept-with-signature (63 concepts as of 2026-05-16).

SAE-feature-space embeddings of concepts are deliberately NOT in this
snapshot — they get computed inside Phase 1's interpolation step. See
`semanticphonology.md` §"Phase 0".
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .. import PROCESSED_DIR

DEFAULT_ANCHORS_ROOT = Path("/media/menser/fauna/interlingua/anchoring/processed")
DEFAULT_OUTPUT = PROCESSED_DIR / "anchors-v1.parquet"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_rows(anchors_root: Path) -> list[dict]:
    sigs = _load_jsonl(anchors_root / "signatures-v1.jsonl")
    attrs = _load_jsonl(anchors_root / "attribute-anchors.jsonl")

    attrs_by_concept: dict[str, list[dict]] = defaultdict(list)
    for a in attrs:
        attrs_by_concept[a["concept"]].append(
            {"attribute": a["attribute"], "cultural": bool(a.get("cultural", False))}
        )

    rows: list[dict] = []
    for s in sigs:
        concept = s["concept"]
        attr_list = sorted(attrs_by_concept.get(concept, []), key=lambda r: r["attribute"])
        rows.append(
            {
                "concept": concept,
                "n_entries": int(s["n_entries"]),
                "n_with_ipa": int(s["n_with_ipa"]),
                "n_languages": int(s["n_languages"]),
                "languages": list(s["languages"]),
                "sharpness": float(s["sharpness"]),
                "modal_projection": s["modal_projection"],
                "n_distinct_projections": int(s["n_distinct_projections"]),
                "projection_histogram": [
                    {"form": form, "count": int(count)} for form, count in s["projection_histogram"]
                ],
                "mean_features": [float(x) for x in s["mean_features"]],
                "var_features": [float(x) for x in s["var_features"]],
                "examples": list(s.get("examples", [])),
                "attributes": [r["attribute"] for r in attr_list],
                "cultural_attributes": [r["attribute"] for r in attr_list if r["cultural"]],
            }
        )
    rows.sort(key=lambda r: r["concept"])
    return rows


def write_parquet(rows: list[dict], output: Path, feature_names: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    table = table.replace_schema_metadata(
        {
            b"schema_version": b"1",
            b"feature_names": json.dumps(feature_names).encode(),
            b"source": b"signatures-v1.jsonl + attribute-anchors.jsonl",
            b"note": (
                b"Phon-side only. SAE-feature-space concept embeddings are "
                b"computed downstream in conlang.interpolate (Phase 1)."
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchors-root",
        type=Path,
        default=DEFAULT_ANCHORS_ROOT,
        help="Directory containing signatures-v1.jsonl and attribute-anchors.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path",
    )
    args = parser.parse_args()

    sigs_path = args.anchors_root / "signatures-v1.jsonl"
    sigs_first = _load_jsonl(sigs_path)[0]
    feature_names = list(sigs_first["feature_names"])

    rows = build_rows(args.anchors_root)
    write_parquet(rows, args.output, feature_names)

    print(f"wrote {len(rows)} concept rows -> {args.output}")


if __name__ == "__main__":
    main()
