"""Freeze the SAE substrate for the Stage 6 cutover.

Per `semanticphonology.md` Phase 0:

  > Snapshot SAE feature vectors and the HDBSCAN cluster outputs into
  > `data/processed/substrate-v1.parquet`. Every later phase reads from this;
  > nothing reads from a live extraction. Filename embeds N
  > (`substrate-v1-n2000.parquet`) so a future scale-up doesn't silently
  > clobber.

This module reads the live Stage 1 + Stage 2 artifacts

  - `data/raw/features.jsonl`        — feature_id, label (per slice index)
  - `data/raw/decoder_vecs.npy`      — (N, decoder_dim) float32 SAE decoder vectors
  - `data/interim/hdbscan_labels.npy` — (N,) int64 HDBSCAN cluster labels
  - `data/processed/slice_manifest.json` — SAE / Neuronpedia provenance

and emits one row per feature with the decoder vector inlined as a list and
the cluster label attached. Output is parameterized by N: the default path
is `data/processed/substrate-v1-n{N}.parquet`.

PMI (the N×N co-activation matrix) lives as a sidecar `pmi.npy` already;
the regularized parent/sibling graph lives as `regularized.json`. Neither
is bundled into the substrate parquet — the substrate is the raw materials
the per-feature snapshot, not the derived graph.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from . import INTERIM_DIR, PROCESSED_DIR, RAW_DIR


def load_substrate(
    *,
    features_path: Path = RAW_DIR / "features.jsonl",
    decoder_path: Path = RAW_DIR / "decoder_vecs.npy",
    labels_path: Path = INTERIM_DIR / "hdbscan_labels.npy",
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    features = [json.loads(line) for line in features_path.open() if line.strip()]
    decoder = np.load(decoder_path)
    labels = np.load(labels_path)
    if len(features) != decoder.shape[0]:
        raise ValueError(
            f"features ({len(features)}) and decoder rows ({decoder.shape[0]}) disagree on N"
        )
    if len(features) != labels.shape[0]:
        raise ValueError(
            f"features ({len(features)}) and HDBSCAN labels ({labels.shape[0]}) disagree on N"
        )
    return features, decoder, labels


def build_rows(features: list[dict], decoder: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for i, f in enumerate(features):
        rows.append(
            {
                "slice_idx": i,
                "feature_id": int(f["feature_id"]),
                "label": str(f["label"]),
                "hdbscan_cluster": int(labels[i]),
                "decoder_vec": [float(x) for x in decoder[i]],
            }
        )
    return rows


def write_parquet(
    rows: list[dict],
    output: Path,
    *,
    n: int,
    decoder_dim: int,
    manifest: dict | None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    md = {
        b"schema_version": b"1",
        b"n_features": str(n).encode(),
        b"decoder_dim": str(decoder_dim).encode(),
        b"created_at": _dt.datetime.now(_dt.UTC).isoformat().encode(),
        b"note": (
            b"Phase 0 substrate snapshot for the Stage 6 cutover. "
            b"SAE feature vectors + HDBSCAN clusters, frozen. "
            b"Co-activation PMI and regularized parent/sibling graph live as separate files."
        ),
    }
    if manifest:
        for k in ("sae_release", "sae_id", "neuronpedia_model", "neuronpedia_source"):
            if k in manifest:
                md[k.encode()] = str(manifest[k]).encode()
    table = table.replace_schema_metadata(md)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=RAW_DIR / "features.jsonl",
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=RAW_DIR / "decoder_vecs.npy",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=INTERIM_DIR / "hdbscan_labels.npy",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROCESSED_DIR / "slice_manifest.json",
        help="Optional. Provenance copied into parquet metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to data/processed/substrate-v1-n{N}.parquet",
    )
    args = parser.parse_args()

    features, decoder, labels = load_substrate(
        features_path=args.features,
        decoder_path=args.decoder,
        labels_path=args.labels,
    )
    n = len(features)
    decoder_dim = int(decoder.shape[1])

    manifest = None
    if args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text())

    rows = build_rows(features, decoder, labels)

    output = args.output or PROCESSED_DIR / f"substrate-v1-n{n}.parquet"
    write_parquet(rows, output, n=n, decoder_dim=decoder_dim, manifest=manifest)

    print(f"wrote {n} feature rows × {decoder_dim}-d decoder -> {output}")


if __name__ == "__main__":
    main()
