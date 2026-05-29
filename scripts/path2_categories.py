"""GLUE Path 2 cross-category runner.

Generalises the past-tense pilot (scripts/path2_pilot_tense.py) to a set of
Tier 1 grammatical contrasts. The Stage B->C->D->E pipeline is identical to the
pilot; only Stage A (minimal-pair generation) differs per category. The model
and SAE load once and every category reuses them, so the marginal cost per
category is one pair of forward passes (~6 s each on an RTX 3090).

Each category contrasts two poles (A vs B) that differ only in one grammatical
operator. The recovered direction d = mean(A) - mean(B); the SPARSE/DIFFUSE
verdict is sign-independent (it reads |c| over the SAE decoder), so pole order
only matters for reading the top-feature signs.

The cross-pool agreement (mean-pool vs last-token cos + top-10 overlap) is the
cleanliness diagnostic: high agreement means the signal is the operator; low
agreement signals either a surface confound (negation, even length-matched) or
an operator that is not at the last token (conjunction). The two-population
eyeball disambiguates them. See GLUE-TODO.md Path 2 for the calibrated gate.

Run:
  HF_HUB_CACHE=/media/menser/fauna/interlingua/hf-cache \
  python scripts/path2_categories.py
"""

from __future__ import annotations

import json
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
LAYER_INDEX = 13  # hidden_states[13] = output of transformer block 12

rng = np.random.RandomState(42)

# Shared vocabulary. Verbs are bare/3sg/past triples (reused for frames that
# need agreement). Count nouns carry regular + a few irregular plurals.
VERBS = [
    ("walk", "walks", "walked"),
    ("talk", "talks", "talked"),
    ("ask", "asks", "asked"),
    ("work", "works", "worked"),
    ("learn", "learns", "learned"),
    ("play", "plays", "played"),
    ("watch", "watches", "watched"),
    ("finish", "finishes", "finished"),
    ("start", "starts", "started"),
    ("paint", "paints", "painted"),
    ("cook", "cooks", "cooked"),
    ("clean", "cleans", "cleaned"),
    ("open", "opens", "opened"),
    ("close", "closes", "closed"),
    ("help", "helps", "helped"),
    ("call", "calls", "called"),
    ("accept", "accepts", "accepted"),
    ("answer", "answers", "answered"),
    ("read", "reads", "read"),
    ("send", "sends", "sent"),
    ("build", "builds", "built"),
    ("find", "finds", "found"),
    ("write", "writes", "wrote"),
    ("buy", "buys", "bought"),
]
NOUNS = [
    ("cat", "cats"),
    ("dog", "dogs"),
    ("book", "books"),
    ("car", "cars"),
    ("house", "houses"),
    ("tree", "trees"),
    ("box", "boxes"),
    ("table", "tables"),
    ("chair", "chairs"),
    ("door", "doors"),
    ("road", "roads"),
    ("phone", "phones"),
    ("song", "songs"),
    ("plate", "plates"),
    ("window", "windows"),
    ("garden", "gardens"),
    ("letter", "letters"),
    ("picture", "pictures"),
    ("bottle", "bottles"),
    ("ticket", "tickets"),
    ("child", "children"),
    ("man", "men"),
    ("woman", "women"),
    ("foot", "feet"),
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
NAMES = ["John", "Mary", "Alice", "Bob", "Susan", "David", "Linda", "Mike"]


def dedup(pairs):
    return sorted(set(pairs))


def gen_plural():
    """Number on the object noun; no verb-agreement change (object position)."""
    out = []
    for sg, pl in NOUNS:
        for subj in SUBJECTS:
            out.append((f"{subj} saw the {sg}.", f"{subj} saw the {pl}."))
            out.append((f"{subj} like the {sg}.", f"{subj} like the {pl}."))
        for name in NAMES:
            out.append((f"{name} found the {sg}.", f"{name} found the {pl}."))
            out.append((f"{name} wanted the {sg}.", f"{name} wanted the {pl}."))
    return dedup(out)


def gen_definiteness():
    """Indefinite 'a/an' vs definite 'the' on the object noun."""
    out = []
    for sg, _ in NOUNS:
        art = "an" if sg[0] in "aeiou" else "a"
        for subj in SUBJECTS:
            out.append((f"{subj} saw {art} {sg}.", f"{subj} saw the {sg}."))
            out.append((f"{subj} found {art} {sg}.", f"{subj} found the {sg}."))
        for name in NAMES:
            out.append((f"{name} bought {art} {sg}.", f"{name} bought the {sg}."))
            out.append((f"{name} opened {art} {sg}.", f"{name} opened the {sg}."))
    return dedup(out)


def gen_negation():
    """Affirmative vs negated, length-matched: only 'not' is inserted after a
    fixed 'will' aux. Same subject/verb/object on both sides."""
    out = []
    for v in VERBS:
        for o in OBJECTS:
            for subj in SUBJECTS:
                out.append((f"{subj} will {v[0]} {o}.", f"{subj} will not {v[0]} {o}."))
            for name in NAMES[:4]:
                out.append((f"{name} will {v[0]} {o}.", f"{name} will not {v[0]} {o}."))
    return dedup(out)


def gen_modality():
    """Certainty 'will' vs possibility 'might' — single-word swap, matched."""
    out = []
    for v in VERBS:
        for o in OBJECTS:
            for subj in SUBJECTS:
                out.append((f"{subj} will {v[0]} {o}.", f"{subj} might {v[0]} {o}."))
            for name in NAMES[:4]:
                out.append((f"{name} will {v[0]} {o}.", f"{name} might {v[0]} {o}."))
    return dedup(out)


def gen_conjunction():
    """Additive 'and' vs adversative 'but' joining two fixed clauses."""
    out = []
    for v in VERBS:
        for subj in SUBJECTS:
            out.append(
                (
                    f"{subj} tried, and {subj.lower()} {v[2]} the work.",
                    f"{subj} tried, but {subj.lower()} {v[2]} the work.",
                )
            )
        for name in NAMES:
            out.append(
                (
                    f"{name} tried, and {name.lower()} {v[2]} the work.",
                    f"{name} tried, but {name.lower()} {v[2]} the work.",
                )
            )
    return dedup(out)


CATEGORIES = [
    ("plural          (sg|pl)", gen_plural),
    ("definiteness    (a|the)", gen_definiteness),
    ("negation        (aff|neg)", gen_negation),
    ("modality        (will|might)", gen_modality),
    ("conjunction     (and|but)", gen_conjunction),
]


def embed_two_ways(strs, model, tok, batch_size=128):
    """Return (mean_pool, last_token) (N, D) arrays from one forward pass."""
    mean_chunks, last_chunks = [], []
    with torch.inference_mode():
        for i in range(0, len(strs), batch_size):
            enc = tok(
                strs[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(DEVICE)
            res = model(**enc, output_hidden_states=True, use_cache=False)
            h = res.hidden_states[LAYER_INDEX]
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            mean_p = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            last_p = h[torch.arange(h.shape[0], device=DEVICE), last_idx]
            mean_chunks.append(mean_p.float().cpu().numpy())
            last_chunks.append(last_p.float().cpu().numpy())
            del res, h, mean_p, last_p, mask, last_idx, enc
    return (np.concatenate(mean_chunks, axis=0), np.concatenate(last_chunks, axis=0))


def run(pos_acts, neg_acts, W_dec, N, label, seed=42):
    seed_rng = np.random.RandomState(seed)
    sub = seed_rng.permutation(len(pos_acts))[:N]
    p_all, n_all = pos_acts[sub], neg_acts[sub]
    idx = seed_rng.permutation(N)
    n_train = int(0.8 * N)
    p_tr, n_tr = p_all[idx[:n_train]], n_all[idx[:n_train]]
    p_te, n_te = p_all[idx[n_train:]], n_all[idx[n_train:]]

    d = p_tr.mean(0) - n_tr.mean(0)
    d /= np.linalg.norm(d)
    mu = np.concatenate([p_tr, n_tr], axis=0).mean(axis=0)
    acc = float(
        np.concatenate(
            [
                (p_te - mu) @ d > 0,
                (n_te - mu) @ d <= 0,
            ]
        ).mean()
    )

    c = W_dec @ d
    abs_c = np.abs(c)
    l1 = abs_c.sum()
    l0 = int((abs_c > 0.01 * abs_c.max()).sum())
    order = np.argsort(abs_c)[::-1]
    top10 = order[:10].tolist()
    top3 = float(abs_c[order[:3]].sum() / l1)

    if top3 > 0.7 and l0 < 10:
        verdict = "SPARSE/LEXICAL"
    elif top3 < 0.4:
        verdict = "DIFFUSE/AFFIX"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "label": label,
        "N": N,
        "acc": acc,
        "l0": l0,
        "top3": top3,
        "top10": top10,
        "verdict": verdict,
        "d": d,
        "c": c,
    }


def main():
    print(f"[stage B] loading {MODEL_ID}")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    print(f"[stage B] load: {time.perf_counter() - t0:.2f}s")

    sae_path = hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILE)
    W_dec = np.load(sae_path)["W_dec"]
    print(f"[stage D] SAE W_dec {W_dec.shape}  from {SAE_REPO}/{SAE_FILE}\n")

    print(
        f"{'category':<28} {'N':>5} {'acc_m':>6} {'acc_l':>6} {'L0':>5} "
        f"{'top3':>7} {'mxl_cos':>8} {'ov':>3}  verdict"
    )
    print("-" * 96)

    summary = []
    for name, gen in CATEGORIES:
        pool = gen()
        N = min(len(pool), 800)
        pos_strs = [p for p, _ in pool[:N]]
        neg_strs = [n for _, n in pool[:N]]
        t0 = time.perf_counter()
        pos_mean, pos_last = embed_two_ways(pos_strs, model, tok)
        neg_mean, neg_last = embed_two_ways(neg_strs, model, tok)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        rm = run(pos_mean, neg_mean, W_dec, N, "MEAN")
        rl = run(pos_last, neg_last, W_dec, N, "LAST")
        mxl = float(rm["d"] @ rl["d"])
        ov = len(set(rm["top10"]) & set(rl["top10"]))
        print(
            f"{name:<28} {N:>5} {rm['acc']:>6.3f} {rl['acc']:>6.3f} "
            f"{rm['l0']:>5} {rm['top3']:>7.4f} {mxl:>+8.3f} {ov:>2}/10  "
            f"{rm['verdict']}   ({dt:.1f}s, pool={len(pool)})"
        )
        summary.append((name, rm, rl, mxl, ov))

    print("\ntop-10 MEAN-POOL features per category (signed c):")
    for name, rm, _, _, _ in summary:
        c = rm["c"]
        feats = "  ".join(f"fid={int(f)}({c[f]:+.3f})" for f in rm["top10"])
        print(f"  {name:<28} {feats}")

    # Stage E artifact. All categories verdict DIFFUSE here, so each carries an
    # affix_direction summary (no sparse lexical_entries). Mirrors the spec's
    # function_lexicon_probed.json shape; consumed by Path 6 Stage 1.
    out = {
        "model": MODEL_ID,
        "layer": LAYER_INDEX,
        "sae": f"{SAE_REPO}/{SAE_FILE}",
        "pilot": "scripts/path2_categories.py",
        "n_categories": len(summary),
        "categories": {},
    }
    for name, rm, rl, mxl, ov in summary:
        key = name.split()[0]
        c = rm["c"]
        out["categories"][key] = {
            "label": name.strip(),
            "verdict": rm["verdict"],
            "N": rm["N"],
            "acc_mean": round(rm["acc"], 4),
            "acc_last": round(rl["acc"], 4),
            "top3_mass": round(rm["top3"], 5),
            "L0": rm["l0"],
            "mean_last_cos": round(mxl, 4),
            "mean_last_top10_overlap": ov,
            "top10_mean": [{"fid": int(f), "c": round(float(c[f]), 5)} for f in rm["top10"]],
        }
    path = "data/processed/function_lexicon_probed.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[stage E] wrote {path}")


if __name__ == "__main__":
    main()
