"""substrate-v1-n{N}.parquet builder tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conlang.snapshot_substrate import build_rows, load_substrate, write_parquet


def _write_substrate_fixture(tmp_path: Path, n: int = 3, dim: int = 4) -> dict[str, Path]:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    interim.mkdir()

    features = [{"feature_id": 100 + i, "label": f"label-{i}"} for i in range(n)]
    fpath = raw / "features.jsonl"
    with fpath.open("w") as f:
        for r in features:
            f.write(json.dumps(r) + "\n")

    decoder = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    dpath = raw / "decoder_vecs.npy"
    np.save(dpath, decoder)

    labels = np.array([-1, 0, 0], dtype=np.int64)[:n]
    lpath = interim / "hdbscan_labels.npy"
    np.save(lpath, labels)

    return {"features": fpath, "decoder": dpath, "labels": lpath}


def test_load_substrate_aligns_features_decoder_labels(tmp_path):
    paths = _write_substrate_fixture(tmp_path, n=3, dim=4)
    features, decoder, labels = load_substrate(
        features_path=paths["features"],
        decoder_path=paths["decoder"],
        labels_path=paths["labels"],
    )
    assert len(features) == 3
    assert decoder.shape == (3, 4)
    assert labels.shape == (3,)
    assert features[0]["feature_id"] == 100


def test_load_substrate_rejects_size_mismatch(tmp_path):
    paths = _write_substrate_fixture(tmp_path, n=3, dim=4)
    # truncate features.jsonl so it has only 2 rows
    lines = paths["features"].read_text().splitlines()
    paths["features"].write_text("\n".join(lines[:2]) + "\n")
    with pytest.raises(ValueError, match="disagree on N"):
        load_substrate(
            features_path=paths["features"],
            decoder_path=paths["decoder"],
            labels_path=paths["labels"],
        )


def test_build_rows_inlines_decoder_and_keeps_provenance(tmp_path):
    paths = _write_substrate_fixture(tmp_path, n=2, dim=3)
    features, decoder, labels = load_substrate(
        features_path=paths["features"],
        decoder_path=paths["decoder"],
        labels_path=paths["labels"],
    )
    rows = build_rows(features, decoder, labels)
    assert len(rows) == 2
    assert rows[0] == {
        "slice_idx": 0,
        "feature_id": 100,
        "label": "label-0",
        "hdbscan_cluster": -1,
        "decoder_vec": [0.0, 1.0, 2.0],
    }
    assert rows[1]["decoder_vec"] == [3.0, 4.0, 5.0]


def test_write_parquet_roundtrips(tmp_path):
    paths = _write_substrate_fixture(tmp_path, n=2, dim=3)
    features, decoder, labels = load_substrate(
        features_path=paths["features"],
        decoder_path=paths["decoder"],
        labels_path=paths["labels"],
    )
    rows = build_rows(features, decoder, labels)

    out = tmp_path / "substrate-v1-n2.parquet"
    write_parquet(
        rows,
        out,
        n=2,
        decoder_dim=3,
        manifest={
            "sae_release": "test-release",
            "sae_id": "test-id",
            "neuronpedia_model": "test-model",
            "neuronpedia_source": "test-src",
        },
    )

    import pyarrow.parquet as pq

    table = pq.read_table(out)
    assert table.num_rows == 2
    md = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    assert md["schema_version"] == "1"
    assert md["n_features"] == "2"
    assert md["decoder_dim"] == "3"
    assert md["sae_release"] == "test-release"
    assert md["sae_id"] == "test-id"
    df = table.to_pandas()
    assert list(df["feature_id"]) == [100, 101]
    assert list(df.iloc[0]["decoder_vec"]) == [0.0, 1.0, 2.0]
