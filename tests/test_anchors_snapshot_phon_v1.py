"""Phon-side anchor snapshot builder tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conlang.lab.snapshot import build_rows, write_parquet


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _stub_signature(concept: str, *, modal: str = "pa", n_entries: int = 5) -> dict:
    return {
        "concept": concept,
        "n_entries": n_entries,
        "n_with_ipa": n_entries,
        "n_languages": n_entries,
        "languages": [f"l{i}" for i in range(n_entries)],
        "feature_names": ["syl", "son", "cons"],
        "mean_features": [0.0, 0.5, -0.5],
        "var_features": [0.1, 0.2, 0.3],
        "sharpness": 0.42,
        "modal_projection": modal,
        "projection_histogram": [[modal, 3], ["pi", 1]],
        "n_distinct_projections": 2,
        "examples": [{"language": "Test", "language_code": "tt", "form": "wuf", "ipa": "/wuf/"}],
    }


def test_build_rows_joins_signatures_and_attributes(tmp_path):
    sigs = [
        _stub_signature("snake_hissing"),
        _stub_signature("bee_buzzing"),
    ]
    attrs = [
        {"concept": "snake_hissing", "attribute": "close-to-ground", "cultural": False},
        {"concept": "snake_hissing", "attribute": "cunning", "cultural": True},
        {"concept": "bee_buzzing", "attribute": "small", "cultural": False},
    ]
    _write_jsonl(tmp_path / "signatures-v1.jsonl", sigs)
    _write_jsonl(tmp_path / "attribute-anchors.jsonl", attrs)

    rows = build_rows(tmp_path)
    assert [r["concept"] for r in rows] == ["bee_buzzing", "snake_hissing"]
    snake = next(r for r in rows if r["concept"] == "snake_hissing")
    assert snake["attributes"] == ["close-to-ground", "cunning"]
    assert snake["cultural_attributes"] == ["cunning"]
    assert snake["modal_projection"] == "pa"
    assert snake["sharpness"] == pytest.approx(0.42)
    assert len(snake["mean_features"]) == 3
    assert snake["projection_histogram"][0] == {"form": "pa", "count": 3}


def test_build_rows_handles_concept_with_no_attributes(tmp_path):
    sigs = [_stub_signature("solo_concept")]
    _write_jsonl(tmp_path / "signatures-v1.jsonl", sigs)
    _write_jsonl(tmp_path / "attribute-anchors.jsonl", [])

    rows = build_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["attributes"] == []
    assert rows[0]["cultural_attributes"] == []


def test_write_parquet_roundtrips(tmp_path):
    sigs = [_stub_signature("dog_barking")]
    attrs = [{"concept": "dog_barking", "attribute": "loud", "cultural": False}]
    _write_jsonl(tmp_path / "signatures-v1.jsonl", sigs)
    _write_jsonl(tmp_path / "attribute-anchors.jsonl", attrs)

    rows = build_rows(tmp_path)
    out = tmp_path / "anchors-v1.parquet"
    write_parquet(rows, out, feature_names=["syl", "son", "cons"])
    assert out.exists()

    import pyarrow.parquet as pq

    table = pq.read_table(out)
    assert table.num_rows == 1
    assert table.column_names[:3] == ["concept", "n_entries", "n_with_ipa"]
    meta = dict(table.schema.metadata)
    assert json.loads(meta[b"feature_names"]) == ["syl", "son", "cons"]
    assert meta[b"schema_version"] == b"1"
