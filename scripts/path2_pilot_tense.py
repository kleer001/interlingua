"""GLUE Path 2 pilot on past tense.

Runs Stage A (template pairs) -> Stage B (Gemma 2 2B layer-12 residual
cache, both mean-pool and last-token in one forward pass) -> Stage C
(difference-in-means + 80/20 held-out accuracy) -> Stage D (16k SAE
projection) -> Stage E (verdict) on a single Tier 1 category.

The pilot also sweeps N ∈ {200, 400, 800} to show that sample size has
near-zero effect on the recovered direction once it exceeds ~100 pairs.

Run:
  HF_HUB_CACHE=/media/menser/fauna/interlingua/hf-cache \
  python scripts/path2_pilot_tense.py
"""

from __future__ import annotations

import os
import random
import time

os.environ.setdefault("HF_HUB_CACHE", "/media/menser/fauna/interlingua/hf-cache")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:0"
MODEL_ID = "google/gemma-2-2b"
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_FILE = "layer_12/width_16k/average_l0_82/params.npz"
LAYER_INDEX = 13   # hidden_states[13] = output of transformer block 12

random.seed(42)
rng = np.random.RandomState(42)

VERBS = [
    ("walk","walks","walked"),("talk","talks","talked"),("ask","asks","asked"),
    ("work","works","worked"),("learn","learns","learned"),("play","plays","played"),
    ("look","looks","looked"),("listen","listens","listened"),
    ("watch","watches","watched"),("finish","finishes","finished"),
    ("start","starts","started"),("jump","jumps","jumped"),
    ("paint","paints","painted"),("cook","cooks","cooked"),
    ("clean","cleans","cleaned"),("open","opens","opened"),
    ("close","closes","closed"),("help","helps","helped"),
    ("call","calls","called"),("smile","smiles","smiled"),
    ("go","goes","went"),("run","runs","ran"),("see","sees","saw"),
    ("give","gives","gave"),("take","takes","took"),("eat","eats","ate"),
    ("drink","drinks","drank"),("sing","sings","sang"),
    ("write","writes","wrote"),("drive","drives","drove"),
    ("buy","buys","bought"),("find","finds","found"),
    ("build","builds","built"),("send","sends","sent"),
]
OBJECTS = ["the answer","my friend","this book","the truth","your help",
    "the door","his name","the result","the noise","this idea","the music",
    "that movie","her voice","his work","the meeting","the question",
    "your offer","this gift","the news","the report","the message",
    "the menu","the puzzle","the contract","the project","the package",
    "the photo","that recipe","this song","the letter"]
NAMES = ["John","Mary","Alice","Bob","Susan","David","Linda","Mike","Anna",
    "Tom","Sarah","James","Emma","Daniel","Olivia"]


def gen_pool() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for v in VERBS:
        for o in OBJECTS[:9]:
            pairs.add((f"I {v[0]} {o}.", f"I {v[2]} {o}."))
    for v in VERBS:
        for o in OBJECTS[:8]:
            pairs.add((f"She {v[1]} {o}.", f"She {v[2]} {o}."))
    for v in VERBS:
        for o in OBJECTS[:5]:
            pairs.add((f"We {v[0]} {o} in the morning.",
                       f"We {v[2]} {o} in the morning."))
    for v in VERBS:
        for o in OBJECTS[:5]:
            name = NAMES[hash((v[0], o)) % len(NAMES)]
            pairs.add((f"{name} {v[1]} {o}.", f"{name} {v[2]} {o}."))
    for v in VERBS:
        for o in OBJECTS[:4]:
            pairs.add((f"They {v[0]} {o}.", f"They {v[2]} {o}."))
    for v in VERBS:
        for o in OBJECTS[:3]:
            pairs.add((f"You {v[0]} {o}.", f"You {v[2]} {o}."))
    for v in VERBS:
        pairs.add((f"He {v[1]} the door.", f"He {v[2]} the door."))
    return sorted(pairs)


def embed_two_ways(strs: list[str], model, tok, batch_size: int = 128):
    """Return (mean_pool, last_token) (N, D) arrays from a single forward pass."""
    mean_chunks, last_chunks = [], []
    with torch.inference_mode():
        for i in range(0, len(strs), batch_size):
            enc = tok(strs[i:i + batch_size], return_tensors="pt",
                      padding=True, truncation=True, max_length=64).to(DEVICE)
            res = model(**enc, output_hidden_states=True, use_cache=False)
            h = res.hidden_states[LAYER_INDEX]
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            mean_p = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            last_p = h[torch.arange(h.shape[0], device=DEVICE), last_idx]
            mean_chunks.append(mean_p.float().cpu().numpy())
            last_chunks.append(last_p.float().cpu().numpy())
            del res, h, mean_p, last_p, mask, last_idx, enc
    return (np.concatenate(mean_chunks, axis=0),
            np.concatenate(last_chunks, axis=0))


def run(pos_acts, neg_acts, W_dec, N, label, seed):
    seed_rng = np.random.RandomState(seed)
    sub = seed_rng.permutation(len(pos_acts))[:N]
    p_all = pos_acts[sub]
    n_all = neg_acts[sub]
    idx = seed_rng.permutation(N)
    n_train = int(0.8 * N)
    p_tr, n_tr = p_all[idx[:n_train]], n_all[idx[:n_train]]
    p_te, n_te = p_all[idx[n_train:]], n_all[idx[n_train:]]

    d = p_tr.mean(0) - n_tr.mean(0)
    d /= np.linalg.norm(d)
    mu = np.concatenate([p_tr, n_tr], axis=0).mean(axis=0)
    acc = float(np.concatenate([
        (p_te - mu) @ d > 0,
        (n_te - mu) @ d <= 0,
    ]).mean())

    c = W_dec @ d
    abs_c = np.abs(c)
    l1 = abs_c.sum()
    l0 = int((abs_c > 0.01 * abs_c.max()).sum())
    order = np.argsort(abs_c)[::-1]
    top10 = order[:10].tolist()
    top3 = float(abs_c[order[:3]].sum() / l1)

    if top3 > 0.7 and l0 < 10:
        verdict = "SPARSE / LEXICAL"
    elif top3 < 0.4:
        verdict = "DIFFUSE / AFFIX"
    else:
        verdict = "INCONCLUSIVE"
    return {"label": label, "N": N, "acc": acc, "l0": l0, "top3": top3,
            "top10": top10, "verdict": verdict, "d": d, "c": c}


def main() -> None:
    pool = gen_pool()[:800]
    print(f"[stage A] {len(pool)} past-tense minimal pairs")
    for p, n in pool[:3]:
        print(f"  present: {p}")
        print(f"  past:    {n}")

    print(f"\n[stage B] loading {MODEL_ID}")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    print(f"[stage B] load: {time.perf_counter() - t0:.2f}s")

    pos_strs = [p for p, _ in pool]
    neg_strs = [n for _, n in pool]
    t0 = time.perf_counter()
    pos_mean, pos_last = embed_two_ways(pos_strs, model, tok)
    neg_mean, neg_last = embed_two_ways(neg_strs, model, tok)
    torch.cuda.synchronize()
    print(f"[stage B] cached {len(pool) * 2}x2 residuals "
          f"in {time.perf_counter() - t0:.2f}s")

    sae_path = hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILE)
    sae = np.load(sae_path)
    W_dec = sae["W_dec"]
    print(f"[stage D] SAE W_dec {W_dec.shape}  from {SAE_REPO}/{SAE_FILE}\n")

    results = []
    for N in (200, 400, 800):
        results.append(run(pos_mean, neg_mean, W_dec, N, "MEAN", seed=42))
        results.append(run(pos_last, neg_last, W_dec, N, "LAST", seed=42))

    print("N     pool     acc        L0     top_3_mass   top-1 fid   verdict")
    print("-" * 78)
    for r in results:
        print(f"{r['N']:<5} {r['label']:<8} {r['acc']:.3f}    {r['l0']:>6}    "
              f"{r['top3']:.5f}    fid={int(r['top10'][0]):<6} {r['verdict']}")

    base_m = next(r for r in results if r["N"] == 200 and r["label"] == "MEAN")
    base_l = next(r for r in results if r["N"] == 200 and r["label"] == "LAST")
    print("\nstability vs N=200 baseline (same pool method):")
    for r in results:
        base = base_m if r["label"] == "MEAN" else base_l
        cs = float(r["d"] @ base["d"])
        ov = len(set(r["top10"]) & set(base["top10"]))
        print(f"  {r['label']}  N={r['N']:<4}  cos={cs:+.4f}  top10 overlap {ov}/10")

    m800 = next(r for r in results if r["N"] == 800 and r["label"] == "MEAN")
    l800 = next(r for r in results if r["N"] == 800 and r["label"] == "LAST")
    print(f"\ncross-pool agreement at N=800: cos={float(m800['d'] @ l800['d']):+.3f}  "
          f"top10 overlap {len(set(m800['top10']) & set(l800['top10']))}/10")

    print("\ntop 10 features at N=800 MEAN-POOL (signed c, |c|):")
    c800 = m800["c"]
    for fid in m800["top10"]:
        print(f"  fid={int(fid):5d}  c={c800[fid]:+.5f}  |c|={abs(c800[fid]):.5f}")


if __name__ == "__main__":
    main()
