"""Embed each anchor (concept or attribute) into SAE residual space.

Per `semanticphonology.md` Phase 1, the natural-neighbor lookup needs
anchors and features to live in the same space. Substrate features
already carry their SAE decoder vector (2304-d residual direction);
anchors don't have a position yet — `signatures-v1.jsonl` only has the
phon-side modal projection, not a semantic-space embedding.

This module fills the gap. Two modes:

**Attribute-level (default).** Iterate `ATTRIBUTE_REGISTRY`; for each
(concept, attribute) pair whose concept has a phon signature, build the
embed text `f"{slug} ({seed}) :: {attribute}"` (e.g.
"snake_hissing (hiss) :: evil-bringer-abrahamic"), run through Gemma
2 2B, and mean-pool layer-12 residual over non-pad tokens. Slug-prefix
disambiguates shared attribute words across concepts (cat_hissing vs.
snake_hissing); seed-in-parens preserves the iconic phonological hook
that pure slug-prefix loses. Yields ~1600 anchors at full inventory
coverage, supplying the residual-space density that 63 concept-level
anchors cannot.

**Concept-level (legacy).** One row per signed concept; embed text is
`english_seeds[0]` ("hiss", "woof", ...). Kept under `--anchor-level
concept` for A/B comparison against the attribute regime.

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
from .attributes import ATTRIBUTE_REGISTRY  # noqa: E402
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


def _read_signed_concepts(path: Path) -> list[dict]:
    """Load each signed concept's metadata from signatures-v1.jsonl."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sig = json.loads(line)
            rows.append(
                {
                    "concept": sig["concept"],
                    "n_entries": sig.get("n_entries", 0),
                    "n_languages": sig.get("n_languages", 0),
                }
            )
    return rows


def load_concept_rows(path: Path) -> list[dict]:
    """Concept-level: one row per signed concept, text = english_seeds[0]."""
    seeds = _concept_seed_lookup()
    rows: list[dict] = []
    for sig in _read_signed_concepts(path):
        concept = sig["concept"]
        seed = seeds.get(concept, concept.replace("_", " "))
        rows.append(
            {
                "concept": concept,
                "seed": seed,
                "attribute": None,
                "cultural": None,
                "text": seed,
                "n_entries": sig["n_entries"],
                "n_languages": sig["n_languages"],
            }
        )
    return rows


def load_attribute_rows(path: Path) -> list[dict]:
    """Attribute-level: one row per (signed concept, attribute) bundle entry.

    Embed text is `f"{slug} ({seed}) :: {attribute}"` — slug-prefix
    disambiguates concepts that share attribute words (e.g.
    `cat_hissing` and `snake_hissing` both have `snake-mimic`), seed
    in parentheses restores the iconic phonological hook that pure
    slug-prefix loses (`flatulence :: ...` vs the iconic `fart`).

    Prior format `f"{slug} :: {attribute}"` measured ρ = 0.0178 on
    2026-05-18 — *halved* baseline concept-level NW (ρ = 0.0365),
    diagnosing the seed-iconicity loss. See semanticphonology.md §3
    measurements log.

    Concepts with a phon signature but no bundle are skipped — but as
    of B7 the registry covers all 63 signed concepts, so this should be
    empty in practice.
    """
    seeds = _concept_seed_lookup()
    signed = {s["concept"]: s for s in _read_signed_concepts(path)}
    rows: list[dict] = []
    for bundle in ATTRIBUTE_REGISTRY.values():
        concept = bundle.concept
        if concept not in signed:
            continue
        sig = signed[concept]
        seed = seeds.get(concept, concept.replace("_", " "))
        for attr in bundle.attributes:
            rows.append(
                {
                    "concept": concept,
                    "seed": seed,
                    "attribute": attr,
                    "cultural": False,
                    "text": f"{concept} ({seed}) :: {attr}",
                    "n_entries": sig["n_entries"],
                    "n_languages": sig["n_languages"],
                }
            )
        for attr in bundle.cultural_attributes:
            rows.append(
                {
                    "concept": concept,
                    "seed": seed,
                    "attribute": attr,
                    "cultural": True,
                    "text": f"{concept} ({seed}) :: {attr}",
                    "n_entries": sig["n_entries"],
                    "n_languages": sig["n_languages"],
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


def write_parquet(
    rows: list[dict], output: Path, *, d_model: int, layer_index: int, anchor_level: str
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    note = (
        f"Anchor positions in Gemma 2 2B layer-{layer_index - 1} residual stream. "
        f"Mean-pooled over the embed text's non-pad tokens. Same coordinate system "
        f"as the SAE decoder vectors in substrate-v1-n{{N}}.parquet. "
        f"anchor_level={anchor_level}; text format: "
        + (
            "'<slug> (<seed>) :: <attribute>'"
            if anchor_level == "attribute"
            else "english_seeds[0]"
        )
        + "."
    )
    md = {
        b"schema_version": b"2",
        b"d_model": str(d_model).encode(),
        b"layer_index": str(layer_index).encode(),
        b"pool": b"mean-over-non-pad-tokens",
        b"model": b"google/gemma-2-2b",
        b"anchor_level": anchor_level.encode(),
        b"note": note.encode(),
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
    parser.add_argument(
        "--anchor-level",
        choices=("concept", "attribute"),
        default="attribute",
        help="concept: one embed per signed concept (english_seeds[0]). "
        "attribute: one embed per (concept, attribute) using "
        "f'{slug} ({seed}) :: {attribute}' (default).",
    )
    args = parser.parse_args()

    print("[1/3] Loading Gemma 2 2B (bf16) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b", torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    d_model = model.config.hidden_size
    print(f"      d_model={d_model}, layers={model.config.num_hidden_layers}", flush=True)

    print(
        f"[2/3] Loading {args.anchor_level}-level rows from {args.input.name} ...",
        flush=True,
    )
    if args.anchor_level == "attribute":
        rows = load_attribute_rows(args.input)
    else:
        rows = load_concept_rows(args.input)
    texts = [r["text"] for r in rows]
    print(f"      {len(rows)} rows; texts e.g. {texts[:3]}", flush=True)

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
                "attribute": r["attribute"],
                "cultural": r["cultural"],
                "n_entries": int(r["n_entries"]),
                "n_languages": int(r["n_languages"]),
                "position": [float(x) for x in vec],
            }
        )
    write_parquet(
        out_rows,
        args.output,
        d_model=d_model,
        layer_index=args.layer_index,
        anchor_level=args.anchor_level,
    )
    print(f"      wrote {len(out_rows)} anchor positions -> {args.output}")


if __name__ == "__main__":
    main()
