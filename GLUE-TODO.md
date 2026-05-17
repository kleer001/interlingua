# GLUE-TODO

**Goal:** discover the function-word / grammatical-glue layer of the conlang
*from the LLM*, not by hand. Keep the framing "this is the language inside a
translation."

**Companion docs:** `spec.md` v0.2 §4 Stage 3 (function-vector edges), §4
Stage 5 (transformation primitive), §7 (crystal-sparsity risk). This document
revives that line of work with a stronger extraction method, plus two cheaper
paths to run first.

**Hardware target:** local rig, beefy. Plan assumes ≥ 1× 24 GB GPU (24 GB
minimum for Gemma 2 9B bf16 + activations; 80 GB if you have it gives bigger
batches and lets you cache more activations in VRAM). Plan also assumes
plenty of disk — Path 2's activation cache for 10M tokens × multiple layers
runs into the hundreds of GB. SSD strongly preferred.

---

## Three paths, in order of cost

1. **Audit** the existing 1000-node lexicon for function-concept features
   that survived §6 but got miscategorized as class-11 abstracts.
2. **Probe** for grammatical primitives via supervised minimal pairs
   (residual-stream direction-finding → SAE projection).
3. **Re-extract** from attention-out SAEs (Gemma Scope ships them), which
   tend to surface syntactic / positional structure better than residual SAEs.

Path 1 is an afternoon. Path 2 is the main event. Path 3 is optional, gated
on whether (1) and (2) leave gaps.

All three are independent and can run in parallel if hardware permits — see
the parallelism notes at the end of each path.

---

## Path 1 — Audit the existing lexicon

### Inputs

- `data/processed/lexicon.json` — 1000 entries, classes assigned, stems
  built.
- `data/processed/regularized.json` — co-activation parent/sibling/near
  edges.
- `data/raw/features.jsonl` — Neuronpedia labels (original, pre-class).

### Steps

1. **Build a vocabulary of grammatical-concept indicator phrases.** One
   pattern per category. Starting set (regex; tune after first pass):

   | Category               | Indicator regex (case-insensitive)                                                                                |
   | ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
   | negation / polarity    | `\b(negation|negat(ed|ing|ion)|not\b|denial|absence of|lack of|refus|contradiction)\b`                            |
   | number (plural)        | `\b(plural(ity)?|multiple instances|collective|several\b.*items|enumeration of)\b`                                |
   | tense / past           | `\b(past (tense|events?)|previously|completed actions?|historical)\b`                                             |
   | tense / future         | `\b(future (tense|events?|reference)|upcoming|anticipated|will\b.*(occur|happen))\b`                              |
   | aspect / progressive   | `\b(ongoing|in progress|continuing actions?|continuous)\b`                                                        |
   | aspect / perfect       | `\b(completed|finished|having (done|been)|resultative)\b`                                                         |
   | person / 1st           | `\b(first[- ]person|speaker self[- ]reference|narrator\b)\b`                                                      |
   | person / 2nd           | `\b(second[- ]person|addressee|direct address)\b`                                                                 |
   | definiteness           | `\b(definite|specific (referent|entity)|previously mentioned|anaphoric)\b`                                        |
   | possession             | `\b(possession|ownership|belonging to|possessive)\b`                                                              |
   | spatial deixis         | `\b(proximal|distal|here vs there|spatial reference|location indicator)\b`                                        |
   | temporal deixis        | `\b(now\b|then\b|temporal (reference|deixis|adverb))\b`                                                           |
   | conjunction / additive | `\b(addition(al)?|in addition|also\b|conjunctive|listing items|enumerating)\b`                                    |
   | conjunction / contrast | `\b(contrast|opposition|adversative|however|nevertheless|despite)\b`                                              |
   | conjunction / causal   | `\b(caus(e|al|ation)|because|therefore|consequently|reason for)\b`                                                |
   | question / interrog.   | `\b(interrog(ative|ation)|question(s|ing)?|inquiry)\b`                                                            |
   | comparative            | `\b(compar(ative|ison)|more than|greater (degree|extent)|gradient)\b`                                             |
   | modality / necessity   | `\b(necessity|must\b|obligation|requirement|deontic)\b`                                                           |
   | modality / possibility | `\b(possibility|may\b|might\b|epistemic|uncertain(ty)?)\b`                                                        |
   | discourse marker       | `\b(discourse marker|hedging|topic shift|attention indicator|emphasis)\b`                                         |

2. **Tag matches.** First match wins, multi-tag allowed for borderline
   cases. Output: `data/interim/function_candidates_raw.json` with
   `{feature_id, label, current_class_id, provisional_grammatical_category}`.

3. **Compute co-activation promiscuity per candidate.** Function-glue
   features tend to fire across many semantic clusters, not within a tight
   field. For each candidate:
   - `n_distinct_clusters_in_neighbors` — count distinct HDBSCAN clusters
     among its top-k co-activation neighbors.
   - `entropy_over_neighbor_clusters` — Shannon entropy of the cluster
     distribution.
   Genuine glue should have **high cluster-entropy** and **few same-cluster
   neighbors**. Concrete features (an "apple" feature) will have low
   entropy.

4. **Cross-validate with Neuronpedia top-activating examples.** For each
   surviving candidate, pull top-20 activating contexts from the
   Neuronpedia cache. A "negation" feature should fire on diverse semantic
   contexts unified by polarity — not on a narrow topical field. This step
   is human-eyeball; flag the candidates the audit is most confident
   about.

5. **Decide the cut.** Drop candidates whose co-activation profile contradicts
   the regex tag (label says "contrast" but neighbors are all in one
   semantic field — probably a content feature with "contrast" in its
   description). Keep the rest as the seed function-lexicon.

### Output

- `data/processed/function_lexicon.json` — schema:
  ```json
  {
    "schema_version": 1,
    "source": "path-1-audit",
    "n_entries": 0,
    "entries": [
      {
        "feature_id": 12345,
        "label": "...",
        "grammatical_category": "negation",
        "cluster_entropy": 2.7,
        "neighbor_clusters": 11,
        "audit_confidence": "high|medium|low",
        "neuronpedia_url": "..."
      }
    ]
  }
  ```

### Parallelism

- Embarrassingly parallel over categories: 20-row table × 1000 features ≈
  20k regex evals. Trivial — just run it.
- The co-activation promiscuity computation is a dict lookup per
  candidate. Single process.
- The Neuronpedia cross-validation can be done in parallel across
  candidates if you want to pull top-activating contexts fresh from
  Neuronpedia — `asyncio` + a polite rate limiter (already conventional
  in `src/conlang/`).

### Done criteria

A `function_lexicon.json` with ≥ 30 high-confidence entries means the
audit alone gave you a usable function sub-lexicon. <10 entries means
glue mostly isn't in the current 1000-node slice; lean on Path 2.

---

## Path 2 — Probe for grammatical primitives

This is the v0.2 §5 transformation plan revived with a stronger method.
Crystals failed because they require the transformation to manifest as a
**sparse pairwise difference in SAE feature space**. We relax that: extract
the direction in **residual stream**, then project to SAE basis and *measure*
sparsity. Sparse → it's a feature (lexical glue). Diffuse → it's a direction
(affix glue).

### Minimal-pair categories

Same categories as Path 1's regex table, plus anything that path surfaced
weakly. For each category, generate ≥ 1000 minimal pairs.

**Generation strategies (run all three, dedupe):**

1. **Template + slot-fill.** Templates like `I {verb}` vs `I {verb}-ed`,
   with slots filled from a WordNet-derived word list. Cheap, controlled.
2. **FLORES-200 derived.** You already pull FLORES for co-activation
   (`src/conlang/edges/coactivation.py`). For each English sentence, run
   a stanza/spaCy parse and synthesize the minimal-pair edit
   (e.g., flip tense, add `not`, swap singular for plural). Naturalistic.
3. **LLM-generated.** Have a strong model write 1k diverse pairs per
   category given a 5-shot prompt. Best diversity, costs API tokens.

**Output**: `data/interim/minimal_pairs/{category}.jsonl` with
`{positive, negative, edit_type, source}`.

### Direction extraction

For each category, at each chosen layer:

1. Run Gemma 2 (both 2B and 9B — see model-sweep below) on each side of
   every pair. Cache residual-stream activations at the answer token (or,
   for categories where there's no obvious answer token, mean-pool over
   the differing span).
2. Compute the **difference-in-means** direction:
   `d = mean(act_positive) − mean(act_negative)`.
3. Normalize. Held-out validation: take 20% of pairs as test set, score
   classification accuracy of `<act, d> > 0` predicts "positive". Direction
   is valid if ≥ 85% test accuracy; below that, the category isn't cleanly
   encoded at that layer and gets dropped or reattempted at a different
   layer.

Optional refinement: contrast-consistent search (CCS) or logistic probe
on top of the difference-in-means direction. Worth running on the 5–10
highest-priority categories. Skip on first pass.

### SAE projection + sparsity gate

For each validated direction `d`:

1. Project onto each SAE's decoder basis: `c = SAE_decoder.T @ d`.
2. Compute sparsity stats:
   - `L0` — count of `|c_i| > threshold` (threshold: 1% of max).
   - `top_k_mass` — fraction of `||c||_1` carried by the top 3, 5, 10
     features.
   - `top_features` — feature ids and Neuronpedia labels for the top 10.

3. Classify the direction:
   - **Sparse / lexical**: `top_3_mass > 0.7` AND `L0 < 10`. Promote the
     top features to function-lexicon entries.
   - **Diffuse / affix**: otherwise. Promote the *direction itself* to a
     grammatical operator. It becomes an affix in the morphology — the
     phonological form is assigned by the same anchor-pool /
     phonosemantic logic that names content features, but applied to the
     direction's nearest SAE feature for orthography.
   - **Inconclusive**: top_3_mass between 0.4 and 0.7. Flag for manual
     inspection.

### Model sweep

Run both Gemma 2 2B and Gemma 2 9B. Reasons:
- 2B is the model the current pipeline targets — anything found here
  drops straight into the existing lexicon and the §4 Stage 3 multigraph.
- 9B has richer features and may surface categories 2B doesn't encode
  cleanly (or vice versa — sometimes simpler models concentrate function
  semantics more sharply). The cross-model agreement is itself a signal:
  a category that probes cleanly in both is high-confidence; only-in-9B
  is suspicious.

Gemma Scope SAEs exist for both. Match each model to its own SAE set.

### Layer sweep

For each model, probe **three layers**: early-middle, middle, late-middle.

- Gemma 2 2B: layers 6, 12, 20. (Middle is 12, which is what the rest
  of the pipeline uses.)
- Gemma 2 9B: layers 12, 20, 31.

Grammatical concepts often peak at a different layer than content
concepts — tense/aspect tends to be later than topical concepts. Sweep
catches this.

### SAE width sweep

Gemma Scope ships multiple widths (16k, 32k, 65k, 131k, 262k, 524k).
Wider SAEs catch more concepts but at proportional memory cost.

- **First pass**: 16k (matches current pipeline) + 65k. Two widths × 3
  layers × 2 models = 12 SAE projections per category.
- **If glue is diffuse at 16k/65k**: try 262k for the top 5 categories.
  Some "ought to exist" features only appear at wider widths.

### Output

- `data/interim/probes/{model}/{layer}/{category}.npz` — direction vector,
  validation accuracy, sample size.
- `data/interim/projections/{model}/{layer}/{width}/{category}.json` —
  L0, top_k_mass, top features.
- `data/processed/function_lexicon_probed.json` — final categorized
  output with category → {lexical_entries[], affix_direction?}.

### Parallelism

This is the meat of the parallelism budget. Layers of independence:

- **Across categories** (20+ items): fully independent after minimal-pair
  generation. Run in parallel processes; each owns a GPU slice or its own
  GPU on a multi-GPU box.
- **Across (model, layer, width)**: independent given cached activations.
- **Activation caching**: do one big forward-pass sweep over the
  combined minimal-pair corpus per model, hooking all three layers at
  once. Write activations to disk in shards. Downstream probes load
  shards — they don't re-run the model.
- **SAE projections** after the direction is found are pure
  matrix multiplies. Embarrassingly parallel.
- **Minimal-pair generation**: the three strategies (template / FLORES /
  LLM) are independent; run all three concurrently.

Sketch of execution plan, assuming a 4-GPU rig:

1. **Stage A (1 GPU each, 4 categories at a time, ~4 hours per model on
   1k pairs/category × 20 categories)**: minimal-pair generation. Run on
   CPU + LLM API in the background while you set up Stage B.
2. **Stage B (all GPUs, 1 model at a time)**: full activation cache
   sweep. Hook layers {6,12,20} for 2B, then swap to 9B and hook
   {12,20,31}. Write shards to disk. ~2 hours for 2B, ~6–8 hours for
   9B at 10M tokens.
3. **Stage C (parallel, CPU-bound)**: direction extraction + validation
   for every (model, layer, category) cell. ~150 cells; trivial to
   parallelize, each cell is a few minutes.
4. **Stage D (parallel, GPU-light)**: SAE projections. Load each
   relevant SAE once, project all directions through it.
5. **Stage E (single process)**: aggregate, classify (sparse/diffuse),
   emit `function_lexicon_probed.json`.

Use `joblib` or `concurrent.futures.ProcessPoolExecutor` — the project
already prefers stdlib + numpy. Avoid Ray / Dask unless you actually need
the distributed layer.

### Done criteria

- ≥ 15 categories with cleanly validated directions (≥ 85% held-out
  accuracy) across at least one (model, layer) combination.
- ≥ 5 categories classified as sparse/lexical (yield concrete lexicon
  entries).
- ≥ 5 categories classified as diffuse/affix (yield grammatical
  operators).

Anything less is a finding too — write it up. "We probed for X, the
model didn't encode it cleanly" is honest and on-brand.

---

## Path 3 — Attention-SAE re-extraction (optional)

Gemma Scope ships **attention-out SAEs** alongside residual SAEs.
Attention SAEs catch positional / syntactic / structural features more
than residual SAEs, which lean toward content concepts. Worth a pass if
Paths 1 and 2 leave categories uncovered — e.g., if "definiteness" or
"discourse marker" doesn't surface cleanly anywhere in Path 2.

### Steps

1. Re-run Stage 1 (`src/conlang/ingest.py`) against an attention SAE at
   the same target layer. Apply the same §6 filter rubric.
2. Skip Stage 2 dedup at first — feature character will be different
   enough that cosine dedup against the residual lexicon isn't
   meaningful. Treat the attention features as a separate node set.
3. Repeat Path 1's audit on the attention-feature labels. Function-glue
   features are more likely here.
4. Optionally merge attention + residual function lexicons by hand,
   collapsing duplicates where two features clearly describe the same
   grammatical concept.

### Output

- `data/processed/function_lexicon_attention.json` — same schema as
  Path 1.

### Parallelism

Single extraction job; not a parallelism story. Could run concurrently
with Path 2's Stage B if you have spare VRAM.

---

## Phonotactics for the function sub-lexicon

The Bantu-shaped `(C)V` constraint and ≥ 2-syllable word minimum were
bootstrap scaffolding. For function morphemes, loosen both:

- **Allow VC, CVC, and monosyllables.** Add a `function_word` mode to
  `src/conlang/phonology.py` that permits codas drawn from a safe subset
  of consonants (start with `n, m, s, l` — avoid stops in codas to keep
  the phonotactics gentle).
- **Drop the 2-syllable minimum for function words.** Closed-class items
  are short across nearly every natural language for a reason.
- **Phonosemantic schema doesn't apply.** Function entries don't carry
  CV1=cluster / CV2=parent / CV3=self. Surface form is assigned by the
  anchor-pool projection (Track A) or by a simple hash (Track B), using
  whichever pipeline the rest of the lexicon ends up using.

Concrete change: a `WordClass.FUNCTION` distinct from `WordClass.STEM`,
checked by `is_valid_word`. The class prefix machinery doesn't apply to
function words — they're prefixes-or-particles in their own right.

Update `docs/grammar.md`:
- New section "Function lexicon" after "Phonosemantic stems".
- Add the loosened phonotactics in a sub-section under "Phonology".
- Note explicitly that function entries were discovered via Paths 1–3,
  not designed.

---

## Cross-cutting deliverables

- `data/processed/function_lexicon.json` — merged output of Paths 1, 2,
  3. Single source of truth.
- `docs/grammar.md` updates — function lexicon section, loosened
  phonotactics.
- `docs/methodology.md` updates — new subsection on Path 2's probing
  method (minimal pairs → direction → SAE projection → sparsity gate).
- Site build (`src/conlang/site.py`) renders the function lexicon as a
  separate table with the grammatical category as a column.
- A `notebooks/glue_audit.ipynb` (Path 1) and `notebooks/glue_probe.ipynb`
  (Path 2) for the writeup. The probe notebook is where the "what the
  model says is glue" story lives — keep it readable.

---

## Open questions, to revisit after first pass

- Should the function-lexicon entries get **noun classes** when used as
  particles, or are they class-less? (Default: class-less; revisit if
  Path 2 turns up a definiteness operator that should pattern with a
  class.)
- For diffuse/affix directions, how do we choose **affix slot order** when
  multiple stack? (E.g., negation + plural + past on one stem.) Defer
  until at least three stackable affixes have been discovered.
- Do we want to **re-run Stage 5 regularization** including function
  features as a third edge type in the multigraph? Possibly; would let
  function features participate in parent/sibling structure where
  they have semantic content (e.g., the negation feature could be the
  parent of all polarity-related content features). Note this as a
  Stage 7 candidate.
- Track A (anchor-pool interpolation) vs Track B (deterministic Bantu)
  may need different function-word policies. Decide per track.

---

## Risk register (additions to spec.md §7)

- **Probe directions may not be sparse in any width.** Mitigation: the
  sparsity gate is the call. Diffuse directions are still useful — they
  become affixes, not lexicon entries.
- **Minimal-pair generation may leak content.** A naive "I walk / I
  walked" template might encode `walk`-the-content alongside tense.
  Mitigation: average across many lexical contexts; the direction-finding
  method already cancels content if the slot fill is diverse enough.
- **Category list is English-shaped.** We're probing for the
  grammatical categories English makes salient, which biases what we
  find. Mitigation: after the first pass, repeat the minimal-pair
  generation with templates seeded from typologically diverse
  languages (FLORES is multilingual; use parallel sentences) and check
  whether any new directions surface.
- **Audit regex is biased toward auto-interp's vocabulary.** Neuronpedia
  labels were generated by a different LLM with its own preferred
  description style. Iterate the regex table after the first run by
  eyeballing the misses.

---

## Suggested order of execution

1. Path 1 audit. 1 afternoon. Decision point: how much glue is already
   sitting in the lexicon?
2. Minimal-pair generation (Path 2 Stage A). Runs in the background
   while you read Path 1's output.
3. Activation cache sweep (Path 2 Stage B). Overnight job for 9B at full
   corpus.
4. Direction extraction + projection (Path 2 Stages C–E). One day.
5. Inspect outputs, decide whether Path 3 is needed.
6. Phonotactics update + site build.
7. Origin-story / methodology writeup. Last, after the artifact stabilizes.
