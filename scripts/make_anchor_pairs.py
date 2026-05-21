"""Compute pairwise distances among the 63 concept-level anchors.

Embeds each anchor's `english_seeds[0]` through Gemma 2 (2B) at layer 13
(matching the headline Phase 3 experiment). Computes pairwise cosine
distance (semantic) and Needleman-Wunsch phonological distance (panphon
features). Writes the resulting (1953 pairs × 6 cols) parquet so the
plot script can read it without re-running the model.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_CACHE", "/media/menser/fauna/interlingua/hf-cache")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.distance import cosine  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from conlang.lab.concepts import CONCEPTS  # noqa: E402
from conlang.lab.embed_positions import embed_texts  # noqa: E402
from conlang.lab.project import phonological_distance  # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[1] / "src" / "conlang" / "lab" / "results" / "anchor_pairs.parquet"


def main() -> None:
    rows = [(c.slug, c.english_seeds[0]) for c in CONCEPTS if c.english_seeds]
    print(f"loading Gemma 2 2B")
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    seeds = [r[1] for r in rows]
    slugs = [r[0] for r in rows]
    print(f"embedding {len(seeds)} concept seeds")
    with torch.no_grad():
        embs = embed_texts(seeds, model, tok, layer_index=13, batch_size=32)
    print(f"embeddings shape: {embs.shape}")

    pairs = []
    for i, sa in enumerate(slugs):
        for j in range(i + 1, len(slugs)):
            sb = slugs[j]
            phon = phonological_distance(seeds[i], seeds[j])
            cos = float(cosine(embs[i], embs[j]))
            pairs.append(
                {"a": sa, "b": sb, "seed_a": seeds[i], "seed_b": seeds[j], "phon": phon, "cos": cos}
            )
    df = pd.DataFrame(pairs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"wrote {len(df)} pairs → {OUT_PATH}")

    from scipy.stats import spearmanr

    rho, p = spearmanr(df["phon"], df["cos"])
    print(f"anchor-pair Spearman ρ = {rho:.4f}, p={p:.2e}")


if __name__ == "__main__":
    main()
