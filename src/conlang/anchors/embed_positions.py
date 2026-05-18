"""Embed each anchor concept into SAE residual space.

Per `semanticphonology.md` Phase 1, the natural-neighbor lookup needs
anchors and features to live in the same space. Substrate features
already carry their SAE decoder vector (2304-d residual direction);
anchors don't have a position yet — `signatures-v1.jsonl` only has the
phon-side modal projection, not a semantic-space embedding.

This module fills the gap. For each of the 63 concepts with a phon
signature, we look up the concept's primary English seed in
`concepts.py` (e.g. snake_hissing → "hiss"), run that text through
Gemma 2 2B, and mean-pool the layer-12 residual hidden state over its
non-pad tokens. The result is a 2304-d vector in the same coordinate
system as each feature's decoder vector, so cosine distance between an
anchor and a feature is well-defined and meaningful.

We anchor by *concept*, not by (concept, attribute), because only 14 of
63 signed concepts have curated attribute bundles — using
attribute-level rows would leave the substrate's other 49 concept
regions without nearby anchors, which is what
`semanticphonology.md`'s anchor-density floor is meant to prevent.
Attribute-level interpolation is a follow-up enhancement once the
attribute roster covers the rest of the concept inventory.

Output: `data/processed/anchor-positions-v1.parquet`. Filename does NOT
embed N because the anchor positions are independent of slice size —
they live in residual space, not in slice-feature space.
"""

from __future__ import annotations

# HF_HUB_CACHE before any transformers/sae_lens import.
import os

os.environ.setdefault("HF_HUB_CACHE", "/media/menser/fauna/interlingua/hf-cache")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from .. import PROCESSED_DIR  # noqa: E402
from .concepts import CONCEPTS  # noqa: E402

DEFAULT_INPUT = Path("/media/menser/fauna/interlingua/anchoring/processed/signatures-v1.jsonl")
DEFAULT_OUTPUT = PROCESSED_DIR / "anchor-positions-v1.parquet"

LAYER_INDEX = 13  # hidden_states[13] = output of transformer block 12 (matches run_coactivation.py)


def _concept_seed_lookup() -> dict[str, str]:
    """Map concept slug → primary English seed from concepts.py."""
    out: dict[str, str] = {}
    for c in CONCEPTS:
        if c.english_seeds:
            out[c.slug] = c.english_seeds[0]
        else:
            out[c.slug] = c.slug.replace("_", " ")
    return out


def load_concept_rows(path: Path) -> list[dict]:
    """Read signatures-v1.jsonl and join the concept slug with its English seed."""
    seeds = _concept_seed_lookup()
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sig = json.loads(line)
            concept = sig["concept"]
            seed = seeds.get(concept, concept.replace("_", " "))
            rows.append(
                {
                    "concept": concept,
                    "seed": seed,
                    "n_entries": sig.get("n_entries", 0),
                    "n_languages": sig.get("n_languages", 0),
                }
            )
    return rows


@torch.no_grad()
def embed_texts(
    texts: list[str],
    model,
    tokenizer,
    *,
    layer_index: int = LAYER_INDEX,
    batch_size: int = 32,
) -> np.ndarray:
    """Mean-pool layer-`layer_index` residual over non-padding tokens.

    Returns float32 array of shape (len(texts), d_model).
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = next(model.parameters()).device

    outs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=32)
        enc = {k: v.to(device) for k, v in enc.items()}
        result = model(**enc, output_hidden_states=True, use_cache=False)
        hidden = result.hidden_states[layer_index]  # (B, T, D)
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        # mean over non-pad tokens
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = (summed / denom).to(torch.float32).cpu().numpy()
        outs.append(pooled)
    return np.concatenate(outs, axis=0)


def write_parquet(rows: list[dict], output: Path, *, d_model: int, layer_index: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    md = {
        b"schema_version": b"1",
        b"d_model": str(d_model).encode(),
        b"layer_index": str(layer_index).encode(),
        b"pool": b"mean-over-non-pad-tokens",
        b"model": b"google/gemma-2-2b",
        b"note": (
            b"Anchor positions in Gemma 2 2B layer-12 residual stream. "
            b"Mean-pooled over the attribute text's non-pad tokens. "
            b"Same coordinate system as the SAE decoder vectors in "
            b"substrate-v1-n{N}.parquet."
        ),
    }
    table = table.replace_schema_metadata(md)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    args = parser.parse_args()

    print("[1/3] Loading Gemma 2 2B (bf16) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b", torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    d_model = model.config.hidden_size
    print(f"      d_model={d_model}, layers={model.config.num_hidden_layers}", flush=True)

    print(f"[2/3] Loading concept rows from {args.input.name} ...", flush=True)
    rows = load_concept_rows(args.input)
    texts = [r["seed"] for r in rows]
    print(f"      {len(rows)} concepts; texts e.g. {texts[:3]}", flush=True)

    print(f"[3/3] Forward + mean-pool at layer {args.layer_index} ...", flush=True)
    embeddings = embed_texts(
        texts, model, tokenizer, layer_index=args.layer_index, batch_size=args.batch_size
    )
    print(f"      embeddings shape: {embeddings.shape}", flush=True)

    out_rows = []
    for r, vec in zip(rows, embeddings, strict=True):
        out_rows.append(
            {
                "concept": r["concept"],
                "seed": r["seed"],
                "n_entries": int(r["n_entries"]),
                "n_languages": int(r["n_languages"]),
                "position": [float(x) for x in vec],
            }
        )
    write_parquet(out_rows, args.output, d_model=d_model, layer_index=args.layer_index)
    print(f"      wrote {len(out_rows)} anchor positions -> {args.output}")


if __name__ == "__main__":
    main()
