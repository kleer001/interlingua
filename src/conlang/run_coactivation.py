"""Run co-activation edges on the slice's 1000 features.

Loads:
  - Gemma 2 2B + Gemma Scope SAE (layer 12, width 16k, canonical)
  - A small multilingual corpus (FLORES dev subset — fallback to a curated
    inline list if the dataset isn't cached)
  - The slice's feature_ids from data/raw/features.jsonl

Writes:
  - data/interim/coactivation/cofire.npy        (n_slice, n_slice) int64
  - data/interim/coactivation/fires.npy         (n_slice,)         int64
  - data/interim/coactivation/pmi.npy           (n_slice, n_slice) float32
  - data/interim/coactivation/top_pairs.json    top-50 by PMI w/ labels
  - data/interim/coactivation/summary.json      counts and threshold info
"""

from __future__ import annotations

# HF_HUB_CACHE first
import os

os.environ.setdefault("HF_HUB_CACHE", "/media/menser/fauna/interlingua/hf-cache")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from . import INTERIM_DIR, RAW_DIR  # noqa: E402
from .edges.coactivation import compute_coactivation, pmi_normalize, top_cofiring_pairs  # noqa: E402
from .ingest import load_sae  # noqa: E402


# Small fallback corpus — curated multilingual sample.
# Concept-diverse so co-activation across topics is exercised.
FALLBACK_CORPUS = [
    # English
    "The cat sat on the mat and watched the rain fall.",
    "Justice requires that all people be treated equally under the law.",
    "She wrote a beautiful poem about love and loss.",
    "Mathematical proofs build on axioms and previous theorems.",
    "The volcano erupted, sending lava flowing down the mountain.",
    "Children laughed and played in the garden after school.",
    "Economic growth depends on innovation and productivity.",
    "Music has the power to evoke deep emotions in listeners.",
    # French
    "Le chat est assis sur le tapis et regarde la pluie tomber.",
    "La justice exige que toutes les personnes soient traitées également.",
    "Elle a écrit un beau poème sur l'amour et la perte.",
    "Les preuves mathématiques s'appuient sur des axiomes et des théorèmes antérieurs.",
    # German
    "Die Katze sitzt auf der Matte und schaut zu, wie der Regen fällt.",
    "Gerechtigkeit erfordert, dass alle Menschen vor dem Gesetz gleich behandelt werden.",
    "Sie schrieb ein schönes Gedicht über Liebe und Verlust.",
    # Spanish
    "El gato se sentó en la alfombra y miró caer la lluvia.",
    "La justicia exige que todas las personas sean tratadas por igual ante la ley.",
    "Ella escribió un hermoso poema sobre el amor y la pérdida.",
    # Concept-heavy abstract
    "Truth and falsehood depend on careful definition.",
    "Past, present, and future flow into one another.",
    "Good and evil are not always easy to tell apart.",
    "Big and small, fast and slow, hot and cold — opposites everywhere.",
    "Hope persists even in the darkest hours of human history.",
    "Knowledge accumulates through observation, experiment, and dialogue.",
    "Family bonds shape who we become as adults.",
    "The forest at dawn is alive with bird calls and rustling leaves.",
    "Water flows from mountains to the sea, carving valleys as it goes.",
    "Memory and forgetting are equal partners in shaping identity.",
    "Conflict and cooperation define the long arc of human society.",
    "A simple meal shared among friends is its own kind of celebration.",
]


FLORES_LANGUAGES = ["eng_Latn", "fra_Latn", "deu_Latn", "spa_Latn", "zho_Hans", "jpn_Jpan"]
FLORES_ROOT = RAW_DIR / "flores200_dataset"


def load_corpus(use_flores: bool, n_per_lang: int) -> list[str]:
    """Load corpus. `--use-flores` reads the FAIR FLORES-200 tarball from RAW_DIR.

    Without --use-flores: returns the curated 30-sentence FALLBACK_CORPUS.
    With --use-flores: requires the tarball to be extracted at
    data/raw/flores200_dataset/ (download from
    https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz). Fails loudly
    if missing — no silent fallback, since the whole point of the flag is to
    scale beyond the curated corpus.
    """
    if not use_flores:
        return FALLBACK_CORPUS
    dev_dir = FLORES_ROOT / "dev"
    if not dev_dir.is_dir():
        raise FileNotFoundError(
            f"FLORES-200 not found at {dev_dir}. Run:\n"
            f"  curl -sL -o {RAW_DIR}/flores200_dataset.tar.gz "
            f"https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz\n"
            f"  tar -xzf {RAW_DIR}/flores200_dataset.tar.gz -C {RAW_DIR}/"
        )
    out: list[str] = []
    for lang in FLORES_LANGUAGES:
        path = dev_dir / f"{lang}.dev"
        if not path.is_file():
            raise FileNotFoundError(f"FLORES file missing: {path}")
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        texts = [line for line in lines if line][:n_per_lang]
        out.extend(texts)
    return out


def load_slice_feature_ids() -> list[int]:
    rows = [json.loads(l) for l in (RAW_DIR / "features.jsonl").open()]
    return [int(r["feature_id"]) for r in rows]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    p.add_argument("--layer-index", type=int, default=13,
                   help="HuggingFace hidden_states index. For gemma-scope-2b-pt-res "
                        "layer_12 SAEs (hook_resid_post block 12), use 13 "
                        "(hidden_states[0] is embedding, [1..N] are layer outputs).")
    p.add_argument("--activation-threshold", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--use-flores", action="store_true",
                   help="Try to load FLORES-200 dev subset (~50 sentences × 6 langs).")
    p.add_argument("--n-per-lang", type=int, default=50)
    p.add_argument("--out-dir", type=Path, default=INTERIM_DIR / "coactivation")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading SAE + Gemma 2 2B (bf16) ...", flush=True)
    sae, sae_cfg = load_sae(args.sae_release, args.sae_id)
    sae = sae.to("cuda:0").eval()
    print(f"      SAE hook: {sae_cfg.get('hook_name', '<unknown>')}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b", torch_dtype=torch.bfloat16, device_map="cuda:0",
    ).eval()
    print(f"      model layers: {model.config.num_hidden_layers}", flush=True)

    print(f"[2/5] Loading corpus ...", flush=True)
    corpus = load_corpus(args.use_flores, args.n_per_lang)
    print(f"      {len(corpus)} sentences", flush=True)

    print("[3/5] Loading slice feature IDs ...", flush=True)
    slice_ids = load_slice_feature_ids()
    print(f"      {len(slice_ids)} slice features", flush=True)

    print("[4/5] Forward + accumulate co-activation ...", flush=True)
    result = compute_coactivation(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sentences=corpus,
        slice_feature_ids=slice_ids,
        layer_index=args.layer_index,
        activation_threshold=args.activation_threshold,
        batch_size=args.batch_size,
    )
    print(f"      processed {result.total_tokens} tokens", flush=True)
    n_active = int((result.fires > 0).sum())
    print(f"      {n_active} of {len(slice_ids)} slice features fired ≥ once "
          f"(threshold={args.activation_threshold})", flush=True)

    print("[5/5] PMI + top pairs ...", flush=True)
    pmi = pmi_normalize(result.cofire, result.fires, result.total_tokens).astype(np.float32)
    np.save(args.out_dir / "cofire.npy", result.cofire)
    np.save(args.out_dir / "fires.npy", result.fires)
    np.save(args.out_dir / "pmi.npy", pmi)

    top = top_cofiring_pairs(pmi, result.cofire, k=50, min_cofire_count=3)
    features = [json.loads(l) for l in (RAW_DIR / "features.jsonl").open()]
    top_serializable = [
        {
            "slice_a": a, "slice_b": b,
            "pmi": p, "cofire_count": c,
            "label_a": features[a]["label"], "label_b": features[b]["label"],
            "feature_id_a": features[a]["feature_id"], "feature_id_b": features[b]["feature_id"],
        }
        for a, b, p, c in top
    ]
    (args.out_dir / "top_pairs.json").write_text(json.dumps(top_serializable, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps({
        "n_sentences": len(corpus),
        "n_tokens": result.total_tokens,
        "n_slice_features": len(slice_ids),
        "n_active_features": n_active,
        "activation_threshold": args.activation_threshold,
        "layer_index": args.layer_index,
        "n_top_pairs": len(top),
    }, indent=2))

    print(f"\n--- Top 15 co-firing pairs (PMI, min cofire=3) ---", flush=True)
    for row in top_serializable[:15]:
        print(
            f"  pmi={row['pmi']:+.2f} cofire={row['cofire_count']:>3d}  "
            f"{row['label_a'][:60]!r}\n"
            f"                   ↔  {row['label_b'][:60]!r}",
            flush=True,
        )
    print(f"\noutputs in {args.out_dir}")


if __name__ == "__main__":
    main()
