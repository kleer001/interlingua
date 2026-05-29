"""GLUE Path 2 — negation re-probe with confound-free pooling.

The cross-category sweep (scripts/path2_categories.py) showed that negation's
recovered direction lands on a surface-`not` / discourse confound (fids 4667,
1041, 6810, 1178) under both mean-pool and last-token, even with a
length-matched frame. Hypothesis: the confound is the literal negator token's
neighborhood, not the negation operator's semantic scope.

This probe pools the residual over the OBJECT span only — the noun phrase that
is identical on both sides of the pair — excluding the `will` / `not` / verb
tokens. If negation is a distributed operator with semantic scope, the object
representation should still carry "this VP is negated" and separate without the
surface confound. If it still confounds, negation needs a non-template probe.

Comparison: object-span pool vs the whole-sentence mean-pool baseline, same
pairs, same Gemma 2 2B / layer-12 / 16k-SAE setup.

Run:
  HF_HUB_CACHE=/media/menser/fauna/interlingua/hf-cache \
  python scripts/path2_negation_reprobe.py
"""

from __future__ import annotations

import os
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
LAYER_INDEX = 13
CONFOUND_ZOO = {4667, 1041, 6810, 1178}

VERBS = [
    "walk",
    "talk",
    "ask",
    "work",
    "learn",
    "play",
    "watch",
    "finish",
    "start",
    "paint",
    "cook",
    "clean",
    "open",
    "close",
    "help",
    "call",
    "accept",
    "answer",
    "read",
    "send",
    "build",
    "find",
    "write",
    "buy",
]
OBJECTS = [
    "the answer",
    "the book",
    "the door",
    "the result",
    "the noise",
    "the music",
    "the meeting",
    "the question",
    "the news",
    "the report",
    "the message",
    "the menu",
    "the puzzle",
    "the contract",
    "the project",
    "the package",
    "the photo",
    "the song",
    "the letter",
    "the offer",
]
SUBJECTS = ["I", "We", "You", "They"]
NAMES = ["John", "Mary", "Alice", "Bob"]


def gen_pairs():
    out = set()
    for v in VERBS:
        for o in OBJECTS:
            for s in SUBJECTS:
                out.add((f"{s} will {v} {o}.", f"{s} will not {v} {o}.", o))
            for n in NAMES:
                out.add((f"{n} will {v} {o}.", f"{n} will not {v} {o}.", o))
    return sorted(out)


def find_span(sent_ids, obj_ids):
    """Return (start, end) of the contiguous obj_ids subsequence in sent_ids."""
    m = len(obj_ids)
    for i in range(len(sent_ids) - m, -1, -1):
        if sent_ids[i : i + m] == obj_ids:
            return i, i + m
    raise ValueError("object span not found")


def embed(strs, objs, model, tok, span_pool, batch_size=128):
    """span_pool=True: mean over object-span tokens. False: whole-sentence mean."""
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(strs), batch_size):
            bs = strs[i : i + batch_size]
            bo = objs[i : i + batch_size]
            enc = tok(bs, return_tensors="pt", padding=True, truncation=True, max_length=64).to(
                DEVICE
            )
            res = model(**enc, output_hidden_states=True, use_cache=False)
            h = res.hidden_states[LAYER_INDEX]
            am = enc["attention_mask"]
            if span_pool:
                mask = torch.zeros_like(am, dtype=h.dtype)
                ids = enc["input_ids"].tolist()
                for r, (row, obj) in enumerate(zip(ids, bo, strict=True)):
                    # Leading space so "the" tokenizes as in-context "▁the".
                    obj_ids = tok(" " + obj, add_special_tokens=False).input_ids
                    s, e = find_span(row, obj_ids)
                    mask[r, s:e] = 1.0
            else:
                mask = am.to(h.dtype)
            mask = mask.unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            chunks.append(pooled.float().cpu().numpy())
            del res, h, mask, enc
    return np.concatenate(chunks, axis=0)


def probe(pos, neg, W_dec, N, seed=42):
    r = np.random.RandomState(seed)
    sub = r.permutation(len(pos))[:N]
    p, n = pos[sub], neg[sub]
    idx = r.permutation(N)
    k = int(0.8 * N)
    d = p[idx[:k]].mean(0) - n[idx[:k]].mean(0)
    d /= np.linalg.norm(d)
    mu = np.concatenate([p[idx[:k]], n[idx[:k]]]).mean(0)
    acc = float(np.concatenate([(p[idx[k:]] - mu) @ d > 0, (n[idx[k:]] - mu) @ d <= 0]).mean())
    c = W_dec @ d
    ac = np.abs(c)
    order = np.argsort(ac)[::-1]
    top10 = order[:10].tolist()
    top3 = float(ac[order[:3]].sum() / ac.sum())
    return {
        "acc": acc,
        "top3": top3,
        "top10": top10,
        "c": c,
        "d": d,
        "l0": int((ac > 0.01 * ac.max()).sum()),
    }


def main():
    pairs = gen_pairs()
    N = min(len(pairs), 800)
    pairs = pairs[:N]
    pos = [p for p, _, _ in pairs]
    neg = [n for _, n, _ in pairs]
    objs = [o for _, _, o in pairs]
    print(f"[stage A] {N} negation pairs")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    ).eval()
    W_dec = np.load(hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILE))["W_dec"]

    t0 = time.perf_counter()
    res = {}
    for name, span in (("whole-sentence", False), ("object-span", True)):
        pp = embed(pos, objs, model, tok, span)
        nn = embed(neg, objs, model, tok, span)
        res[name] = probe(pp, nn, W_dec, N)
    torch.cuda.synchronize()
    print(f"[stage B] {time.perf_counter() - t0:.1f}s\n")

    print(f"{'pool':<16} {'acc':>6} {'top3':>7} {'L0':>6}  confound_in_top10  top-5 fids")
    print("-" * 92)
    for name in ("whole-sentence", "object-span"):
        r = res[name]
        zoo = sorted(set(r["top10"]) & CONFOUND_ZOO)
        top5 = ", ".join(str(int(f)) for f in r["top10"][:5])
        print(
            f"{name:<16} {r['acc']:>6.3f} {r['top3']:>7.4f} {r['l0']:>6}  "
            f"{str(zoo) if zoo else 'none':<17}  {top5}"
        )

    cs = float(res["whole-sentence"]["d"] @ res["object-span"]["d"])
    ov = len(set(res["whole-sentence"]["top10"]) & set(res["object-span"]["top10"]))
    print(f"\nwhole-sentence vs object-span: cos={cs:+.3f}  top10 overlap {ov}/10")
    obj_zoo = set(res["object-span"]["top10"]) & CONFOUND_ZOO
    print(
        "object-span de-confounded:"
        if not obj_zoo
        else f"object-span STILL confounded ({sorted(obj_zoo)})"
    )


if __name__ == "__main__":
    main()
