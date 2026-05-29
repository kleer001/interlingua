# GLUE Path 2 — pilot finding: tense is a distributed direction

`GLUE-TODO.md` Path 2 extracts grammatical-primitive directions from
minimal-pair activations and verdicts each as **lexical**, **diffuse /
affix**, or **inconclusive** against SAE decoder geometry. A pilot run on
past tense in Gemma 2 2B (layer 12, gemmascope-res-16k SAE) yields a
clean, robust direction with **DIFFUSE / AFFIX** verdict — and that
verdict is the linguistically meaningful answer, not a methodology
failure.

## Pipeline

- **Stage A** — 1190 unique template-generated minimal pairs of the form
  `(I VERB X, I VERB-ed X)` etc., across 34 verbs (regular + irregular)
  and 7 sentence patterns. Pool truncated to 800 for the sweep.
- **Stage B** — Gemma 2 2B layer-12 residual cache, two pool methods
  captured in one forward pass: full-sentence mean-pool and last-token
  position. 1600 forward inputs in ~1.5 s on an RTX 3090.
- **Stage C** — difference-in-means direction with 80/20 train/test
  split. Mean-pool: ≥ 0.99 accuracy at N ≥ 200. Last-token: 0.93–1.00
  accuracy at N = 200–800.
- **Stage D** — projection onto the 16384-feature SAE decoder
  (`google/gemma-scope-2b-pt-res`, layer 12, width 16k, average-l0-82).
- **Stage E** — verdict by `top_3_mass` and L0 thresholds.

## Verdict

| N   | Pool      | Held-out acc | top_3_mass | L0 (> 1% of max) | Verdict |
| --- | --------- | ------------ | ---------- | ---------------- | ------- |
| 200 | mean      | 1.000        | 0.0052     | 12530            | DIFFUSE |
| 200 | last      | 0.950        | 0.0057     | 12610            | DIFFUSE |
| 400 | mean      | 0.994        | 0.0053     | 12559            | DIFFUSE |
| 400 | last      | 0.950        | 0.0058     | 12326            | DIFFUSE |
| 800 | mean      | 0.997        | 0.0052     | 12583            | DIFFUSE |
| 800 | last      | 0.931        | 0.0058     | 12322            | DIFFUSE |

The diffuse-vs-sparse character is a property of the SAE basis, not of N.
Direction cosine ≥ 0.99 across all sample sizes vs the N = 200 baseline;
top-10 feature overlap is 10/10 for mean-pool and 9–10/10 for last-token.
4× more pairs gives an essentially identical direction. Cross-pool
agreement at N = 800: `cos(d_mean, d_last) = +0.87`, top-10 overlap 9/10.

## The 9 robust top features

Both pool methods rank these 9 features in the top 10. Auto-interp
labels (`gpt-4o-mini`) are content-biased and miss the grammatical
signature; the actual top-activating contexts (peak token marked in
**bold**) reveal a clean tense split.

| fid    | c (mean) | side    | gpt-4o-mini label              | What actually fires                                              |
| ------ | -------- | ------- | ------------------------------ | ---------------------------------------------------------------- |
| 13390  | −0.59    | past    | "recurring actions or events"  | **was** better, **acted** as, **was** in line                    |
| 12506  | −0.43    | past    | "guidelines and instructions"  | **didn't**, came **up** with, **occurred**                       |
| 3967   | −0.36    | past    | "reporting or presenting"      | **phoned**, **assessed**, **concluded**, **worked**, **noted**   |
| 785    | −0.33    | past    | "forms of the verb to be"      | **was** ×6 — past copula                                         |
| 3976   | −0.31    | past    | "the word 'did' and negation"  | **did** not, **did** get, **did** enjoy — past auxiliary         |
| 15888  | +0.53    | present | "action verbs"                 | **coincides**, **retrieved** (present passive)                   |
| 11758  | +0.44    | present | "monitoring"                   | **change**, **recruits**, **go**, **fall**, **lucks**            |
| 10018  | +0.44    | present | "ownership and possession"     | **owns**, **have**, **map**, **has**                             |
| 11118  | +0.43    | present | "actions related to work"      | **fly**, **do**, **grows**, **use**                              |

Tense is encoded as **two feature populations**, not a single signed
axis. 5 past-like features fire on past-form verbs and past auxiliaries
(`was`, `did`); 4 present-like features fire on present-form verbs and
the present auxiliaries `has`/`have`.

## What this means for Path 2's verdict system

The Path 2 spec describes the diffuse/affix bucket as:

> Diffuse / affix: ... Promote the *direction itself* to a grammatical
> operator. It becomes an affix in the morphology — the phonological form
> is assigned by the same anchor-pool / phonosemantic logic that names
> content features, but applied to the direction's nearest SAE feature
> for orthography.

Past tense in Gemma 2 2B fits that bucket exactly:
- the direction is robust (cos ≥ 0.99 across N, ≥ 0.87 across pool
  methods at N = 800),
- the top features are interpretable and grammatically coherent,
- there is no single "PAST" feature, but the *operator* lives in
  residual space and projects onto a small interpretable population.

The pilot establishes that the diffuse verdict can be a substantive
positive finding for a Tier 1 category — not a fallback for "didn't
work."

## What this means for the spec's sparsity gate

The current gate is `top_3_mass > 0.7 AND L0 < 10` for sparse / lexical.
For a 16k SAE with non-orthogonal decoder vectors, that threshold may
be unachievable for grammatical operators even when the direction is
real:

- An idealized lexical feature would have one large `c_i` and 16383
  near-zero others. In practice, decoder vectors are non-orthogonal, so
  any real direction projects with some magnitude onto thousands of
  features. L1 is dominated by the noise floor.
- The pilot's tense direction is clearly meaningful — 100% held-out
  accuracy at N = 200, interpretable top features — but
  `top_3_mass = 0.005`, far from 0.7.

A **population-check gate** may complement the concentration gate:

> Are the top-K features by |c|, for some K matched to expected operator
> dimension, interpretably aligned with the contrast category?

For tense, K = 9 with the population above is yes.

## Methodology lessons

1. **Capture-method robustness is a diagnostic.** When mean-pool and
   last-token recover similar directions (cos > 0.9, top-10 overlap ≥ 8),
   the category is well-encoded. When they disagree (cos < 0.5, overlap
   < 3), the direction is suspect. The pilot saw this on negation
   (cos = +0.30, overlap 1/10) vs clean past tense (cos = +0.93,
   overlap 9/10). The cross-category sweep (`scripts/path2_categories.py`,
   recorded in `GLUE-TODO.md` Path 2) refined the reading of the
   negation case: a length-matched negation frame (only `not` inserted)
   still disagrees (cos +0.23, overlap 1/10) and lands on the same
   confound features, so the cause is the literal negator's surface
   neighborhood, *not* sentence-length. And low coherence is not always
   a confound — a mid-sentence operator like conjunction (`and`/`but`)
   disagrees because the last token misses it, not because the direction
   is dirty. Use the two-population eyeball to tell the two apart.
2. **Auto-interp labels are content-biased.** `gpt-4o-mini` labelled
   `fid=13390` as "recurring actions or events" when its top contexts
   are simply past-form verbs. A human-eyeball pass on actual top
   activations is necessary to recover grammatical structure. Path 1
   step 4-5 is justified for the same reason.
3. **Sample size has diminishing returns past ~100 pairs.**
   Difference-in-means converges quickly; direction stability is
   cos ≥ 0.99 across N ∈ {200, 400, 800}. Spend authorship budget on
   pair *quality* and *category coverage*, not pair count per
   category.
4. **Stage B GPU compute is essentially free.** 800 pairs × 2 sides
   through Gemma 2 2B at layer 12, both pool methods captured, takes
   ~2 seconds on an RTX 3090. The bottleneck for Path 2 at scale is
   Stage A authorship, not Stage B compute.

## Reproducibility

- Script: `scripts/path2_pilot_tense.py`. Run:
  ```
  HF_HUB_CACHE=/media/menser/fauna/interlingua/hf-cache \
  python scripts/path2_pilot_tense.py
  ```
  Wall clock ~6 s on RTX 3090 (4 s model load + ~2 s embed + sub-second
  linalg + 3 N × 2 pool methods of Stage C/D/E).
- Model: `google/gemma-2-2b` (bfloat16, single GPU).
- SAE: `google/gemma-scope-2b-pt-res` layer 12 width 16k average-l0-82
  (`params.npz`), resolved via `huggingface_hub.hf_hub_download`.
- Layer index: 13 in `hidden_states` (= output of transformer block 12,
  matches `src/conlang/lab/embed_positions.py`).
- Cached Neuronpedia responses for the 9 features:
  `data/interim/neuronpedia/gemma-2-2b__12-gemmascope-res-16k__{fid}.json`.
