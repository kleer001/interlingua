"""Stage 1: ingest SAE features + Neuronpedia explanations.

Loads decoder vectors from a Gemma Scope SAE checkpoint and auto-interp
descriptions from Neuronpedia's bulk S3 dump
(`v1/{model}/{source}/explanations/batch-*.jsonl.gz`), then applies the
spec §6 filter rubric and takes the first N survivors.

The bulk dump has no confidence score (that's only on per-feature API calls), so
filtering is by description content, not score. The rubric is intentionally
crude on first pass; refine against the slice's output, not in advance.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

from . import RAW_DIR

NP_S3_BASE = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1"


@dataclass
class Feature:
    feature_id: int
    label: str
    decoder_vec: np.ndarray  # shape (d_model,)
    np_embedding: np.ndarray | None = None  # auto-interp embedding from Neuronpedia

    def to_meta(self) -> dict:
        d = asdict(self)
        d.pop("decoder_vec")
        d.pop("np_embedding")
        return d


def load_sae(release: str, sae_id: str):
    """Load a Gemma Scope SAE via sae_lens. Imported lazily so test collection
    doesn't pull torch."""
    from sae_lens import SAE

    sae, cfg_dict, _sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release, sae_id=sae_id
    )
    return sae, cfg_dict


def decoder_vectors(sae) -> np.ndarray:
    """Return the SAE decoder matrix as (n_features, d_model) float32."""
    import torch

    with torch.no_grad():
        return sae.W_dec.detach().cpu().to(torch.float32).numpy()


def download_bulk_explanations(
    model: str,
    source: str,
    dest_dir: Path,
    n_batches: int = 17,
) -> Path:
    """Download Neuronpedia's bulk explanation batches for one SAE to dest_dir.

    Idempotent: skips files already present and non-empty. Returns dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120.0, follow_redirects=True) as cli:
        for i in range(n_batches):
            key = f"explanations/batch-{i}.jsonl.gz"
            out = dest_dir / key.replace("/", "__")
            if out.exists() and out.stat().st_size > 0:
                continue
            r = cli.get(f"{NP_S3_BASE}/{model}/{source}/{key}")
            r.raise_for_status()
            out.write_bytes(r.content)
    return dest_dir


def load_bulk_explanations(
    model: str,
    source: str,
    raw_dir: Path = RAW_DIR,
) -> list[dict]:
    """Read every explanation row from the cached S3 batches for one SAE.

    Returns a list of dicts with keys: id, index (str), description, embedding (str),
    umap_x, umap_y, umap_cluster, umap_log_feature_sparsity, etc.
    """
    dest = raw_dir / "np" / model / source
    if not dest.exists():
        download_bulk_explanations(model, source, dest)
    rows: list[dict] = []
    for path in sorted(dest.glob("explanations__batch-*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


# --- Spec §6 filter rubric ----------------------------------------------------

_EXCLUDE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Programming / code
        r"\b(programming|code|coding|source code|html|css|javascript|python|java(script)?|"
        r"function call|api|json|xml|sql|database|syntax|variable|compiler|regex|"
        r"git|commit|repository|github)\b",
        # URLs / emails / formatting / markup
        r"\b(url|hyperlink|email address|markdown|markup|formatting|whitespace|"
        r"punctuation|capitalization|line break|newline|tab character)\b",
        # Tokenizer / positional / dead-feature signals
        r"\b(token|tokenizer|positional|position-related|sequence position|"
        r"start of (sentence|sequence|document)|end of (sentence|sequence|document))\b",
        # Vague labels
        r"^\s*(various|miscellaneous|unclear|noise|random|mixed)\s+(words|tokens|text|content)\s*$",
        # Brands / products / proper-noun-heavy hints (light)
        r"\b(brand name|product name|company|corporation|app name|software name)\b",
        # Token-locked features: descriptions that pin to a specific quoted word or phrase
        # ("the word \"several\" indicating ...", "instances of the conjunction \"and\" ...")
        r"\bthe (word|phrase|term|conjunction|preposition)\s+\"[^\"]+\"",
        r"\binstances of (the (word|phrase|term|conjunction|preposition)\s+)?\"[^\"]+\"",
        r"\"[^\"]+\"\s+(and (its )?(variants|related forms|similar (words|phrases|terms)))",
        # Numerical / statistical codes / identifiers (not the concept of number)
        r"\b(numerical|statistical) (codes?|references?|data|values?|identifiers?)\b",
        r"\b(specific|particular) (numerical|number)\b",
        r"\bspecific (codes?|identifiers?|values?)\b",
        # Meta-textual / dataset labels
        r"\b(labelled?|labeled) data\b",
        r"\bmetadata( in (documents|text))?\b",
        r"\b(data or metadata|metadata or data)\b",
        r"\breferences to (labeled|labelled) data\b",
        # Hedge-vague: "references to specific X" with no semantic anchor
        r"^references to specific (categories|items|terms|identifiers|values|codes)\b",
    ]
]


def passes_filter_rubric(description: str) -> bool:
    """True if description survives the spec §6 exclude rules.

    The rubric is intentionally rough on first pass. Inspect what survives,
    then tighten.
    """
    desc = description.strip()
    if len(desc) < 12:
        return False
    if len(desc.split()) < 3:
        return False
    for pat in _EXCLUDE_PATTERNS:
        if pat.search(desc):
            return False
    return True


def _parse_embedding(s: str | None) -> np.ndarray | None:
    if s is None:
        return None
    try:
        return np.asarray(json.loads(s), dtype=np.float32)
    except (json.JSONDecodeError, TypeError):
        return None


def first_n_passing_filter(
    rows: list[dict],
    decoder: np.ndarray,
    n: int,
) -> list[Feature]:
    """Take the first N explanations (by feature index) whose description
    passes the §6 rubric. Attach the SAE decoder vector and (optional) NP
    embedding."""
    # Order by integer feature index — bulk dump is unordered.
    rows_sorted = sorted(rows, key=lambda r: int(r["index"]))
    seen: set[int] = set()
    out: list[Feature] = []
    for r in rows_sorted:
        idx = int(r["index"])
        if idx in seen:
            continue
        seen.add(idx)
        desc = r.get("description") or ""
        if not passes_filter_rubric(desc):
            continue
        if idx >= decoder.shape[0]:
            continue
        out.append(
            Feature(
                feature_id=idx,
                label=desc.strip(),
                decoder_vec=decoder[idx],
                np_embedding=_parse_embedding(r.get("embedding")),
            )
        )
        if len(out) >= n:
            break
    return out


def save_node_set(features: list[Feature], out_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    """Persist features as a metadata JSONL + a stacked decoder matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "features.jsonl"
    vecs_path = out_dir / "decoder_vecs.npy"
    with meta_path.open("w") as f:
        for feat in features:
            f.write(json.dumps(feat.to_meta()) + "\n")
    np.save(vecs_path, np.stack([f.decoder_vec for f in features]))
    return meta_path, vecs_path
