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

## Three paths plus two wildcards, in order of cost

1. **Audit** the existing 1000-node lexicon for function-concept features
   that survived §6 but got miscategorized as class-11 abstracts.
2. **Probe** for grammatical primitives via supervised minimal pairs
   (residual-stream direction-finding → SAE projection).
3. **Re-extract** from attention-out SAEs (Gemma Scope ships them), which
   tend to surface syntactic / positional structure better than residual SAEs.
4. **Platypus path** — unsupervised discovery of glue-shaped features
   without naming the target. Finds operators we don't have a name for.
5. **Lichen path** — discovery of operators that only function as
   *compounds* of features. Atomic-feature search misses them by
   construction.

Path 1 is the cheapest. Path 2 is the main event. Path 3 is optional, gated
on whether (1) and (2) leave gaps. Path 4 is the exploratory wildcard for
atomic operators. Path 5 is the wildcard for non-atomic ones — runs after
Path 4 because it reuses its infrastructure and pruning relies on Path 4's
candidates.

---

## Unit-of-analysis caveat (read before all paths)

Every path in this document commits to **SAE features as the unit of
analysis**. That's a choice, not a fact about the model. The model doesn't
compute on SAE features; it computes on residual-stream activations. SAEs
are a learned post-hoc decomposition, and they're decent at capturing some
phenomena (atomic content concepts) and worse at others (smeared
operators, compositional structure, polysemantic remainders).

Three levels of commitment are implicit in Paths 1–5:

| Commitment                              | Where it bites                                                                                                         |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| SAEs are the right basis                | A direction the SAE doesn't reconstruct sparsely (Path 2's "diffuse" outcome) gets bucketed as an affix, not missed entirely. So this commitment is partially relaxed already. |
| Operators are individual units          | Path 4 assumes a single feature has the operator behavior. Path 5 (below) relaxes this to *pairs and triples*.        |
| Compositional units don't exceed ~3     | Path 5 caps composition order at 3 for tractability. A "4-feature lichen" is searchable in principle but not in this plan. |

What we miss by these commitments:

- **Directions in raw residual-stream that no SAE-feature combination
  reconstructs sparsely.** Path 2's supervised probing finds these for
  named targets; nothing in Paths 1–5 finds them unsupervised. The
  truly open Path 6 would do unsupervised direction-discovery in raw
  activation space, decoupled from any SAE. See `POST-GLUE-SKETCH.md`
  §3 — that's the agenda for after the glue survey lands.
- **Large circuits.** If an operator is a 7-feature computational graph,
  no method in this doc finds it. Anthropic's circuit-tracer (already in
  `spec.md §4 Stage 3`) is the right tool; `POST-GLUE-SKETCH.md` §5
  sketches how it gets wired into glue discovery.
- **Distributed / no-locus operators.** Some grammatical behavior in
  modern models lives in attention patterns, not in residual-stream
  features. Path 3 (attention-out SAEs) catches the SAE-decomposable
  subset of this; `POST-GLUE-SKETCH.md` §6 covers the rest (attention
  patterns themselves as operators).

The post-glue sketch also covers subspace / nonlinear operators (§4),
context-conditioned operators (§7), and cross-layer trajectories (§8) —
all of which are outside the scope of the present document by design.

The point of naming these explicitly: when the writeup says "Gemma 2
encodes evidentiality" or "Gemma 2 does not encode mirativity," the
claim is "at the level of SAE features and 2-feature compounds, with
the procedures described." Not at every possible level. That caveat
isn't a hedge — it's the honest scope.

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
   pattern per category. Starting set below — this is the **English-shaped
   baseline** for the quick audit. The fuller typological / Ithkuil-style
   roster lives in the "Category roster" section after Path 1; Path 2
   probes the full roster. The audit's regex over Neuronpedia labels has
   highest precision on categories whose auto-interp descriptions are
   well-attested in English prose, which biases toward this baseline.

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

## Category roster — beyond English-shaped

Path 1's regex is English-vocabulary biased. Path 2 isn't, so its category
list shouldn't be either. The roster below is what Path 2 probes; it spans
five tiers from familiar to deliberately weird. Going wide here is the
point — the project's premise is that the model has a language-agnostic
concept space, and the right question isn't "does the LLM encode English
grammar" but "which of the world's grammatical distinctions, and which
distinctions no human language makes, does the LLM cleanly encode."

For every category: a probe that comes back with a validated direction is
a finding; a probe that doesn't is *also* a finding. "We looked for
mirativity and the model didn't separate surprise from new information"
is a real result and earns a line in the writeup.

### Tier 1 — English-shaped baseline (20 categories, see Path 1 table)

Negation, plural, past, future, progressive, perfect, 1st/2nd person,
definiteness, possession, spatial deixis, temporal deixis, additive /
contrastive / causal conjunction, interrogation, comparative, necessity,
possibility, discourse marker. Already in Path 1's table.

### Tier 2 — Typologically common, English-weak

These are well-attested in the world's languages but English marks them
poorly or not at all. Minimal-pair source defaults to **multilingual**
unless an English paraphrase contrast is clean.

| Category                         | Sketch                                                                         | Pair source             |
| -------------------------------- | ------------------------------------------------------------------------------ | ----------------------- |
| evidentiality / direct           | "I saw it" vs "I'm told" — direct witness contrast                             | English + multilingual  |
| evidentiality / hearsay          | reported speech, second-hand information                                       | English + multilingual  |
| evidentiality / inferential      | "must have rained" — inference from evidence                                   | English paraphrase      |
| mirativity                       | new / surprising information; speaker's just-learned-it stance                 | English paraphrase      |
| clusivity                        | inclusive vs exclusive 1pl ("we [with you]" vs "we [not you]")                 | multilingual (Tagalog…) |
| dual number                      | exactly two, distinct from plural                                              | multilingual (Slovene…) |
| paucal number                    | a few, distinct from plural                                                    | multilingual (Arabic…)  |
| addressee honorific              | formality toward listener (T/V, keigo addressee axis)                          | multilingual + register |
| referent honorific               | exalting the subject of speech (Japanese sonkeigo)                             | multilingual            |
| humble / self-lowering           | humbling the speaker (Japanese kenjōgo)                                        | multilingual            |
| volitionality                    | controlled vs uncontrolled action ("I fell" — on purpose? not?)                | English paraphrase      |
| telicity                         | bounded vs unbounded event ("ran" vs "ran a mile")                             | English templates       |
| pluractionality                  | verbal plural — action repeated / by many agents                               | English paraphrase      |
| tense remoteness                 | immediate / yesterday / distant past distinctions                              | multilingual (Bantu)    |
| egophoricity                     | privileged self-knowledge ("I know I'm tired" vs "she's tired")                | multilingual (Tibetan)  |
| cislocative / translocative      | toward speaker vs away from speaker (motion frame)                             | English paraphrase      |
| logophoricity                    | "she said she₍same₎ left" vs "she said she₍other₎ left"                        | English disambig.       |
| switch-reference                 | next clause same-subject vs different-subject                                  | English paraphrase      |
| obviation                        | proximate vs obviative 3rd person ("he₁" vs "the other he")                    | English disambig.       |
| animacy hierarchy                | animate vs inanimate referent in argument position                             | English templates       |
| inverse marking                  | when a low-animacy agent acts on a high-animacy patient                        | English templates       |
| ergative alignment cue           | intransitive subject patterns with patient, not agent                          | multilingual (Basque)   |
| alienable vs inalienable poss.   | "my hand" (body part) vs "my book" (acquired thing)                            | English templates       |
| numeral classifier — shape       | long / flat / round / sheet classifier semantics                               | multilingual (Mandarin) |
| numeral classifier — animacy     | human vs animal vs inanimate count                                             | multilingual (Japanese) |
| reflexive / reciprocal           | "they saw themselves" vs "they saw each other" vs "they saw them"              | English templates       |
| causative                        | productive causation: "X dies" → "Y makes X die"                               | English templates       |
| applicative                      | productive valency add: beneficiary / instrumental promoted to object          | multilingual (Bantu)    |
| topic vs subject                 | what we're talking *about* vs syntactic subject (Japanese wa vs ga)            | multilingual            |
| focus marker                     | what's new / contrastive ("it's the CAT that…")                                | English clefts          |
| middle voice                     | neither active nor passive (Greek-style middle)                                | English paraphrase      |
| antipassive                      | transitive subject preserved, patient demoted                                  | multilingual            |

### Tier 3 — Ithkuil-style abstract distinctions

Categories whose closest natural-language home is one or two outlier
languages, or that exist only in conlangs. Probing these is the highest-risk
part of the plan — the LLM may not have a clean direction for any of them.
That's interesting either way.

| Category                | Sketch                                                                         | Pair source         |
| ----------------------- | ------------------------------------------------------------------------------ | ------------------- |
| configuration           | single entity / set of similars / set of dissimilars / undifferentiated mass   | LLM-generated       |
| affiliation             | coincidental / associative / variative / cooperative arrangement of a set      | LLM-generated       |
| extension               | event touches a point / delimited region / whole referent                      | LLM-generated       |
| perspective             | monadic event / unbounded / generic-truth / abstract-conceptual                | LLM-generated       |
| phase                   | discrete / fluctuative / frequentative / fragmentative iterativity             | LLM-generated       |
| sanction (Ithkuil)      | speaker's force: assertive / presumptive / allegative / refutative             | LLM paraphrase      |
| validation (Ithkuil)    | basis for claim: observational / inferential / intuitive / reportative         | overlaps evidential |
| bias                    | speaker's emotional stance: skeptical / expectant / mocking / resigned         | LLM paraphrase      |
| illocution              | declarative / interrogative / directive / hortative / admonitive               | English templates   |
| stativity vs eventivity | describes a state vs describes a change                                        | English templates   |
| boundedness / countness | count vs mass distinction at the referent level                                | English templates   |

### Tier 4 — Discourse / pragmatic

Categories the model almost certainly encodes (these are *everywhere* in
training data) but that mostly aren't morphologized in IE languages.

| Category                  | Sketch                                                          | Pair source          |
| ------------------------- | --------------------------------------------------------------- | -------------------- |
| sentence-final softening  | yo / ne / ba / ma equivalents — Japanese, Mandarin              | multilingual         |
| confirmation-seeking      | "right?" / "tag question" / Japanese ne                         | English templates    |
| hedging gradient          | "definitely" → "probably" → "maybe" → "perhaps not"             | English templates    |
| politeness / face         | deferential / neutral / blunt / rude                            | LLM-generated        |
| empathy alignment         | whose POV the narration takes (Japanese giving/receiving verbs) | multilingual         |
| genre register            | formal-written / informal-spoken / technical / poetic           | corpus contrast      |
| narrative perspective     | first-person narrator / third limited / third omniscient        | corpus contrast      |
| direct vs indirect speech | "she said 'I'm tired'" vs "she said she was tired"              | English templates    |
| presupposition trigger    | "stopped X-ing" presupposes prior X-ing                         | English templates    |
| metalinguistic mention    | "the word X" vs use of X                                        | English templates    |

### Tier 5 — LLM-native / no clear linguistic home

Distinctions the LLM is known or strongly suspected to encode that don't
map to any one language's grammar. Some of these (sentiment, refusal) are
already documented in published mech-interp work; including them here is
honest about what an LLM "language" might prioritize that human languages
don't.

| Category               | Sketch                                                          | Pair source         |
| ---------------------- | --------------------------------------------------------------- | ------------------- |
| sentiment polarity     | positive / negative valence — well-attested SAE feature         | corpus + templates  |
| affect intensity       | calm / agitated / extreme regardless of polarity                | LLM-generated       |
| formality register     | known SAE feature in many models                                | corpus contrast     |
| factuality cue         | confident-fact vs speculation vs known-falsehood                | LLM-generated       |
| refusal / safety       | known SAE feature in instruction-tuned models                   | LLM-generated       |
| code-switching marker  | mid-utterance language shift                                    | multilingual corpus |
| self-reference (model) | "as an AI" / model-as-narrator features                         | corpus contrast     |
| instruction vs content | imperative-to-model vs content-to-process                       | LLM-generated       |
| length / verbosity     | concise vs expansive register cue                               | corpus contrast     |
| reasoning-chain marker | step-by-step / let's-think-about-this scaffolding               | corpus contrast     |

### Tier-aware execution

Run tiers in numerical order so Tier 1 validates the pipeline before
later tiers are trusted. Tier 4 (discourse / pragmatic) tends to peak
at later layers than Tier 1 — bias layer choice accordingly. Tier 3
is exploratory; expect a low hit rate.

---

## Path 2 — Probe for grammatical primitives

This is the v0.2 §5 transformation plan revived with a stronger method.
Crystals failed because they require the transformation to manifest as a
**sparse pairwise difference in SAE feature space**. We relax that: extract
the direction in **residual stream**, then project to SAE basis and *measure*
sparsity. Sparse → it's a feature (lexical glue). Diffuse → it's a direction
(affix glue).

### Minimal-pair categories

The full roster — Tiers 1 through 5 in the "Category roster" section
above. ~80 categories. For each, generate ≥ 1000 minimal pairs from
whichever generation strategies fit (see roster's "Pair source" column).

**Generation strategies (run all four, dedupe):**

1. **Template + slot-fill.** Templates like `I {verb}` vs `I {verb}-ed`,
   with slots filled from a WordNet-derived word list. Cheap, controlled.
   Works for Tier 1 + parts of Tier 2.
2. **FLORES-200 derived.** You already pull FLORES for co-activation
   (`src/conlang/edges/coactivation.py`). For each English sentence, run
   a stanza/spaCy parse and synthesize the minimal-pair edit (flip tense,
   add `not`, swap singular for plural, etc.). Naturalistic.
3. **LLM-generated.** Have a strong model write 1k diverse pairs per
   category given a 5-shot prompt. Best diversity. Indispensable for
   Tier 3 (Ithkuil-style abstract) and parts of Tier 4–5, where the
   distinction has no template.
4. **Multilingual minimal pairs (NEW).** For categories English doesn't
   mark morphologically — most of Tier 2 — pair source isn't English at
   all. Three sub-strategies:

   a. **Parallel-corpus paired sentences.** FLORES-200 ships parallel
      translations across 200 languages. For evidentiality, pull Tariana
      / Quechua / Turkish sentences that explicitly mark the distinction;
      pair the morphologically-marked positive with a same-meaning,
      different-mark negative *in the same language*. Probe the LLM with
      the foreign-language pairs. The model's concept space is multilingual
      — the direction we recover is the *concept*, not a language-specific
      morpheme.

   b. **Translation-aligned contrast.** A single English sentence paired
      with two translations into a marking language that differ only in
      the grammatical category of interest. ("I saw the dog" → Turkish
      direct-evidential vs Turkish inferential.) The English source
      anchors meaning; the contrast lives in the translation.

   c. **Code-switched English.** For categories where a marking language
      has a productive particle that's borrowable, generate
      English-with-particle pairs ("I saw it" vs "I saw it -mış"). Weak
      but fast; useful as a control.

   Multilingual minimal pairs are honest because the rest of the pipeline
   already premises on a language-agnostic concept space. They're also
   the only fair way to probe Tier 2 — generating English approximations
   biases the probe toward whatever English does encode.

**Output**: `data/interim/minimal_pairs/{tier}/{category}.jsonl` with
`{positive, negative, edit_type, source, source_language}`.

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

1. **Stage A** (1 GPU each, 4 categories at a time): minimal-pair
   generation. Run on CPU + LLM API in the background while you set up
   Stage B.
2. **Stage B** (all GPUs, 1 model at a time): full activation cache
   sweep. Hook layers {6,12,20} for 2B, then swap to 9B and hook
   {12,20,31}. Write shards to disk.
3. **Stage C** (parallel, CPU-bound): direction extraction + validation
   for every (model, layer, category) cell. ~150 cells; trivial to
   parallelize.
4. **Stage D** (parallel, GPU-light): SAE projections. Load each
   relevant SAE once, project all directions through it.
5. **Stage E** (single process): aggregate, classify (sparse/diffuse),
   emit `function_lexicon_probed.json`.

Use `joblib` or `concurrent.futures.ProcessPoolExecutor` — the project
already prefers stdlib + numpy. Avoid Ray / Dask unless you actually need
the distributed layer.

### Done criteria

Targets are tier-aware now that the roster is ~80 categories:

- **Tier 1 (~20 categories):** ≥ 15 validated. This is the floor — if Tier
  1 underperforms, the pipeline is broken, not the model.
- **Tier 2 (~30 categories):** ≥ 12 validated across at least one
  (model, layer). Cross-model agreement (validated in both 2B and 9B) is
  the highest-confidence subset; aim for ≥ 5 cross-validated.
- **Tier 3 (~11 categories):** ≥ 2 validated would be a real result.
  Zero is also a result — "Ithkuil-style distinctions don't separate
  cleanly in Gemma 2's concept space" is publishable as a negative
  finding.
- **Tier 4 (~10 categories):** ≥ 6 validated, mostly at later layers
  than Tier 1.
- **Tier 5 (~10 categories):** ≥ 5 validated. Sentiment / formality /
  refusal are nearly guaranteed; others test the breadth.

Across all tiers:
- ≥ 10 categories classified as **sparse/lexical** → concrete function
  lexicon entries.
- ≥ 10 categories classified as **diffuse/affix** → grammatical
  operators.

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

## Path 4 — Unsupervised glue discovery (the platypus path)

> A platypus is a mammal that lays eggs. It exists outside the
> taxonomy that was supposed to be exhaustive. Path 4 looks for the
> grammatical equivalent: directions in the model's activation space
> that *behave like* operators but don't map to anything we'd have
> thought to probe for.

Identify glue by **behavioral signature** rather than by labeled target.
A feature is glue-shaped if it:

- fires across many semantic clusters (promiscuous),
- is topic-independent (same feature in cooking texts, physics texts,
  dialogue),
- is positionally / structurally cued rather than topically cued (fires at
  boundaries, after specific structural positions, regardless of content),
- and — most importantly — **causes systematic transformations of
  output** when steered, not changes of topic.

The last criterion is the gold standard: steering by topical features
changes the subject; steering by operator features changes the grammar
around the subject.

### Stage 1 — Glue-signature filter

For every SAE feature (all ~16k, not just the 1000 in the lexicon —
this path looks at the whole space, including features that were
filtered out by §6):

1. **Cluster entropy.** For each feature, take its top-N activating
   contexts. Embed each context, cluster (HDBSCAN or k-means).
   `cluster_entropy = Shannon entropy of the cluster distribution`.
   High entropy = promiscuous = candidate glue.

2. **Topic invariance.** Pool the contexts of each feature into a
   "topic distribution" using off-the-shelf topic-model embedding.
   `topic_invariance = 1 - max_topic_mass`. Features whose activations
   spread across topics evenly score high.

3. **Positional / boundary score.** For each activation, record
   within-sentence position (normalized 0–1) and whether it's at a
   clause / phrase / discourse boundary (use a parser — stanza is
   fine). Aggregate:
   - `position_concentration`: variance of within-sentence position
     (low = positionally locked).
   - `boundary_fraction`: fraction of activations at structural
     boundaries (high = boundary-active).

4. **Token-vs-structure ratio.** For each activation, score how
   predictable the activation is from the surrounding *token identity*
   vs the surrounding *POS / dependency structure*. Train two small
   probes — one on token n-grams, one on POS / dep features — and
   compare their accuracy in predicting whether the feature fires.
   Operators predict better from structure than from token identity.

A **glue candidate** is a feature in the top quartile on
`cluster_entropy` AND top quartile on `topic_invariance` AND
(top quartile on `position_concentration` OR top quartile on
`boundary_fraction` OR `structure_probe_accuracy >
token_probe_accuracy`).

This identifies the set without naming what they encode. Expect a few
hundred candidates from a 16k-width SAE; the wider the SAE, the more
candidates, the more platypi.

### Stage 2 — Steering characterization

For each glue candidate, the question is: when you push this feature
around, what changes in the model's output?

Procedure per candidate:

1. Sample ~100 diverse prompts (mix of topics, registers, lengths) —
   reuse the FLORES-200 + LLM-generated prompts from Path 2.
2. For each prompt: generate the completion under three conditions:
   - **Baseline**: no intervention.
   - **Clamp ON**: set the feature's activation to high (e.g., 5× the
     99th percentile of its natural distribution) throughout the
     completion.
   - **Clamp OFF**: zero the feature's activation throughout.
3. Capture all three completions per prompt.
4. Compute the **steering signature** of the feature:
   - **Token-level shift**: which tokens change in probability between
     ON and OFF? Aggregate across prompts. Filter to grammatical /
     functional tokens (modals, particles, conjunctions, tense
     morphology, polarity, pronouns) vs content tokens. The ratio
     functional:content is the **grammaticality of the operator**.
   - **Structural shift**: parse all three completions, compare clause
     type / mood / valency distributions ON vs OFF.
   - **Embedding shift**: where does the completion embedding move?
     Magnitude of the shift = strength of the operator; direction
     should be consistent across prompts (if it's a real operator) or
     random (if it's noise).
   - **Consistency**: across the 100 prompts, how consistent is the
     transformation? Operators transform systematically; spurious
     features don't.

A candidate is **promoted to platypus** if:
- functional:content shift ratio ≥ some threshold (start: 2.0),
- structural distribution shifts in a consistent direction across
  prompts,
- embedding-shift direction has high cross-prompt cosine similarity
  (≥ 0.5).

Otherwise it gets demoted — it's a content feature that happened to look
promiscuous, or it's noise.

### Stage 3 — Cluster the platypi

The candidates that survive Stage 2 are operators, but each is a single
feature, not a category. To find *kinds of operators*, cluster them by
their steering signature:

1. Encode each platypus's signature as a vector: functional token shift
   profile + structural shift profile + embedding shift direction.
2. Cluster (HDBSCAN; the project already uses it).
3. Each cluster is a **candidate operator category**.

Two platypi in the same cluster transform output in similar ways —
they're likely two facets of the same underlying operator. A platypus
that's alone in its cluster is either a singleton operator or noise that
made it through Stage 2; flag for manual inspection.

### Stage 4 — Naming

For each cluster:

1. Pull the top-activating contexts of all member features.
2. Pull the steering effect summaries (what tokens shifted, what
   structures changed).
3. Pull the SAE auto-interp labels of member features (if any survived
   §6) — sometimes Neuronpedia already had a hint we ignored.
4. Hand-write (or LLM-assist) a working name. The name should describe
   the *transformation*, not the content: "boundary marker that promotes
   the following clause to topic position" not "topic-related feature."

If a cluster's effect doesn't fit any name you can come up with,
**preserve that**. Give it a placeholder code (`PLATYPUS-007`) and
write up the empirical signature without forcing it into a known
category. That's the genuinely-novel-discovery case and it should be
loud in the writeup.

### Stage 5 — Typology cross-check (Burke-and-Wills note)

Burke and Wills crossed the continent without learning from people who
already knew the route. The analogous failure mode here: discover a
"new" operator that's actually well-documented in WALS / SSWL /
Glottolog and we just hadn't named it because our roster was incomplete.

For each platypus cluster, *before* committing to "this is novel":

1. Pull the steering signature (token shifts + structural shifts).
2. Search WALS feature descriptions (192 features, covering most of
   typology) for close matches. WALS is downloadable as a JSON dump.
3. Search the existing Path 2 supervised directions — project the
   platypus's mean signature onto each known-category direction. High
   alignment = we re-discovered something we already named in Path 2,
   re-tag accordingly.
4. If no match in WALS and no match in Path 2: **this is the
   deliverable.** Keep the placeholder name, write up the empirical
   signature, document the search you did so the writeup can defend the
   "we looked, it isn't this" claim.

The genuinely-novel set is small by construction (most platypi will
turn out to be poorly-named known categories), and that's fine — even
zero genuine platypi is a finding about the shape of the model's
operator space.

### Output

- `data/interim/glue_candidates.json` — Stage 1 output. Schema:
  `{feature_id, cluster_entropy, topic_invariance, position_concentration,
  boundary_fraction, structure_vs_token_accuracy_gap, score}`.
- `data/interim/steering_signatures/{feature_id}.json` — Stage 2 output
  per candidate.
- `data/processed/platypi.json` — final output:
  ```json
  {
    "schema_version": 1,
    "source": "path-4-unsupervised",
    "clusters": [
      {
        "cluster_id": 7,
        "placeholder_name": "PLATYPUS-007",
        "working_name": "...",
        "member_feature_ids": [...],
        "steering_signature_summary": "...",
        "wals_search_result": "no match" | "matches WALS feature 81 (...)",
        "path2_alignment": "no match" | "near 'evidentiality / inferential' (cos=0.72)",
        "novelty_status": "novel | re-discovery | known-category"
      }
    ]
  }
  ```

### Parallelism

This is the heaviest path — generation cost dominates.

- **Stage 1** (signatures): single pass per feature using cached
  activations from Path 2 Stage B. Embarrassingly parallel across
  features. Disk-bound, not GPU-bound.
- **Stage 2** (steering): this is the expensive bit. 200–500 candidates
  × (ON / OFF / baseline) × 100 prompts × generation cost. Mitigations:
  - Cap candidates at top-500 by Stage 1 score before steering.
  - Short completions (50 tokens).
  - Aggressive batching — Gemma 2 9B with bf16 + a 24 GB GPU can run
    batches of 32 short completions at decent speed; 80 GB GPU handles
    128+.
  - Parallelize across candidates if you have multiple GPUs — each
    candidate's steering set is independent.
- **Stage 3** (cluster): single CPU job. Trivial.
- **Stage 4** (naming): human-in-the-loop, not parallelized. LLM-assist
  via a notebook with a small UI that shows top-activating contexts +
  steering examples side-by-side for each cluster.
- **Stage 5** (WALS / Path 2 cross-check): single CPU job. WALS as JSON
  is small (~10 MB). Path 2 alignment is a matmul.

### Done criteria

- ≥ 50 candidate features survive Stage 1's glue-signature filter
  (sanity check on the filter; if many fewer, the thresholds are too
  strict).
- ≥ 20 of those promote to platypi at Stage 2 (have systematic
  grammatical steering effects).
- ≥ 5 distinct clusters at Stage 3.
- ≥ 1 cluster survives Stage 5 as "no match in WALS, no match in
  Path 2." That's a linguistic platypus.

### Risks specific to Path 4

- **Stage 1's filter is itself a hypothesis.** "Glue looks like
  X / Y / Z" — if the model encodes operators in a *different* shape than
  promiscuous + topic-invariant + structural, Path 4 misses them.
  Mitigation: log the features that *almost* passed and inspect a sample
  for "this looks operator-y but didn't make our cut" — iterate the
  filter thresholds in a second pass.
- **Steering is fragile.** A feature that doesn't have a clean effect
  on output may still be a real operator that's downstream-suppressed
  by some other circuit. Mitigation: try steering at multiple
  coefficients (1×, 5×, 20×) and at multiple layers (Gemma 2 has 26;
  steer the same feature at L12 *and* L18 — sometimes the effect is
  cleaner one or the other). Don't rule a feature out on a single
  steering attempt.
- **Naming is the bottleneck.** With 5+ clusters and 20+ platypi,
  writing names that don't smuggle in pre-existing categories is hard.
  Mitigation: the notebook UI for Stage 4 should show the empirical
  signature first and human-suggested names second; the order matters
  for not biasing the namer.
- **"Novel" is unfalsifiable in some directions.** If a platypus
  separates contexts that no human has ever named the distinction
  between, we can't prove it's not just noise that happens to be
  consistent. Mitigation: require cross-model agreement (the same
  platypus signature must show up when the procedure is run on both
  Gemma 2 2B and 9B). A platypus that's only in one model is suspect;
  one that's in both is real.

---

## Path 5 — Compositional operator discovery (the lichen path)

> A lichen is two species in one functional unit — the fungus and the
> alga are inert apart, viable together. Path 5 looks for the
> grammatical analog: pairs (or triples) of SAE features whose joint
> activity has a clean operator effect that neither member produces
> alone.

A lot of mech-interp work (Anthropic's circuits, the Bricken et al.
monosemantic features work, attribution graphs) suggests that operator
behavior is often distributed across small clusters of features that
fire together but don't act in isolation. Negation, in some models,
decomposes into a "polarity feature" + "scope feature" pair where
neither alone flips output polarity but the conjunction does.

### What counts as a lichen

A feature pair (i, j) is lichen-shaped if all three hold:

1. **Co-activation**: i and j fire together more than independent
   chance predicts. (Cheap filter.)
2. **Synergy**: the *interaction* `a_i · a_j` is a significant
   predictor of some grammatical metric (next-token polarity / tense /
   register shift, choice of complementizer, etc.) *after controlling
   for `a_i` and `a_j` alone*. Knowing both gives more than knowing
   either.
3. **Joint steering > solo steering**: clamping both features together
   produces a systematic grammatical transformation that neither
   solo-clamp produces. This is the gold standard — it mirrors Path 4
   Stage 2, just on pairs.

A pair that passes (1) + (2) is a *candidate*. A pair that also passes
(3) is a *lichen*.

### Stage 1 — Candidate pair pruning

16k × 16k = 256M pairs. Cannot steer them all. Three filters cascade:

1. **Co-activation graph membership.** Reuse `data/interim/coactivation.npz`
   from spec §4 Stage 3. Take all edges (pairs with PMI above some
   threshold — start with the threshold the existing pipeline uses for
   sibling assignment). Drops 256M → maybe 1M.
2. **Glue-context restriction.** A lichen operator should fire in
   glue-shaped contexts. Restrict to pairs where at least one member is
   a Path 4 Stage 1 glue candidate, OR where the co-firing event itself
   has a glue signature (high cluster entropy in the contexts where both
   fire). Drops 1M → maybe 100k.
3. **Synergy regression pre-screen.** For each surviving pair, fit a
   tiny logistic regression on a small sample of held-out text:
   `P(grammatical_event_X) ~ a_i + a_j + a_i·a_j` for several
   grammatical events of interest (polarity flip in next clause,
   tense shift, mood shift, register shift). Keep pairs where the
   interaction coefficient is significant for at least one event after
   controlling for both main effects. Drops 100k → maybe 1k–5k.

### Stage 2 — Synergy / interaction test (full)

For each candidate pair from Stage 1, do the full synergy measurement:

1. Sample ~10k tokens. Record `(a_i, a_j, next-token distribution)` for
   each.
2. Fit two models: a baseline with main effects only, and a full model
   with the interaction term.
3. Measure information gain of the full model over the baseline
   (likelihood ratio or BIC delta).
4. Compute **partial-information decomposition** if you want to be
   precise: split the joint information `(a_i, a_j) → output` into
   unique-i, unique-j, redundant, synergistic components. A lichen has
   high *synergistic* mass relative to unique-i and unique-j.

Pairs with low synergy drop out. Expect to cut 1k–5k → 200–500
high-synergy pairs.

### Stage 3 — Joint steering

For each high-synergy pair, the same steering protocol as Path 4
Stage 2, but with four conditions instead of three:

- **Baseline**: no intervention.
- **Clamp i alone** (j at natural).
- **Clamp j alone** (i at natural).
- **Clamp both** at the joint coordinates suggested by Stage 2 (e.g.,
  the (a_i, a_j) values that maximize the synergistic effect).

Score each condition on the same metrics as Path 4 Stage 2 (functional
token shift, structural shift, embedding shift). A **lichen** is a pair
where the *joint-clamp* condition produces a transformation signature
that's:

- Significantly stronger than either solo-clamp condition.
- Not predicted by linear sum of the solo-clamp signatures (the joint
  effect is qualitatively different, not just bigger).
- Consistent across the ~100 prompts.

Pairs that fail any of those become *false lichens* — interesting on
their own (they're synergistic in prediction but not in causation) and
worth logging, but they don't promote.

### Stage 4 — Cluster and name

Same as Path 4 Stages 3–4 but on lichen pairs instead of atomic
platypi. The clustering vector is the lichen's joint-steering
signature. Each cluster is a candidate *compound operator category*.

Naming caveat: when writing the working name, describe the
*compound's* effect, not either member's. The whole point of a lichen
is that the description belongs to the unit, not to either species.

### Stage 5 — Cross-check vs Path 4, WALS, and the lexicon

Before claiming novelty:

1. **Vs Path 4 platypi**: does any lichen's joint-steering signature
   align with an atomic platypus signature? If so, this lichen is a
   compositional re-implementation of an operator the model also encodes
   atomically. Note it; don't double-count.
2. **Vs WALS**: same procedure as Path 4 Stage 5.
3. **Vs the existing lexicon**: occasionally a Path-1 lexicon entry's
   member features turn out to be lichen partners. That's a *finding
   about the lexicon* — it means the supervised entry has compositional
   internal structure.

### Triples and higher (deferred)

The same procedure extends to triples (a_i · a_j · a_k) and beyond.
Triples are tractable if the pair search yields a small set of
"lichen-prone" features — restrict triple search to combinations of
known lichen members. Pure cubic search (16k choose 3 ≈ 700B) is not
tractable; the gate is "at least two members already participate in
known lichens." Defer until pairs are mapped.

### Quick gesture — mycorrhizal hubs

Tangent to the lichen path, worth a parallel scan: **graph-centrality
features**. Features with high betweenness centrality in the
co-activation graph mediate between many semantic clusters without
being content themselves — the underground fungal network that lets
trees of different species share resources. Cheap to find (single graph
statistic, no steering). Test for operator behavior by steering the top
20. If they steer cleanly, they're operators of a different sort:
context-stabilizers or topic-bridges. If they don't, they're
multi-purpose intermediate computations and not glue.

This is a quick scan; don't build a path around it. Just run it and
note the top features in `data/interim/mycorrhizal_candidates.json`.
Promote any with clean steering to the Path 4 platypus list.

### Output

- `data/interim/lichen_candidates.json` — Stage 1 + 2 output. Schema:
  `{pair: [i, j], coactivation_pmi, synergy_score, synergy_event,
  interaction_coef, p_value}`.
- `data/interim/lichen_steering/{i}_{j}.json` — Stage 3 per-pair
  steering signatures.
- `data/processed/lichens.json` — final output:
  ```json
  {
    "schema_version": 1,
    "source": "path-5-compositional",
    "clusters": [
      {
        "cluster_id": 3,
        "placeholder_name": "LICHEN-003",
        "working_name": "...",
        "member_pairs": [[123, 456], [123, 789], ...],
        "steering_signature_summary": "...",
        "path4_alignment": "no match" | "...",
        "wals_search_result": "no match" | "...",
        "novelty_status": "novel | re-discovery-of-platypus | known-category"
      }
    ]
  }
  ```

### Parallelism

- **Stage 1**: graph filter is single-CPU on the existing coactivation
  data. Synergy pre-screen is per-pair; embarrassingly parallel across
  pairs.
- **Stage 2**: synergy regression per surviving pair; per-pair
  parallel.
- **Stage 3**: steering per pair, 4 conditions × 100 prompts. This is
  4x more generation than Path 4's per-feature steering. Run it after
  Path 4 finishes so GPU contention is sequential, not parallel.
- **Stage 4–5**: same as Path 4. CPU and human-in-the-loop.

### Done criteria

- ≥ 5k pairs survive Stage 1.
- ≥ 100 pairs pass the synergy threshold at Stage 2.
- ≥ 10 of those are confirmed lichens at Stage 3 (joint > solo,
  cross-prompt consistent).
- ≥ 2 distinct lichen clusters at Stage 4.
- ≥ 1 cluster survives Stage 5 as "no match to Path 4 platypi, no
  match to WALS, distinct from any lexicon entry."

### Risks specific to Path 5

- **Synergy is detectable but not steerable.** A pair may show strong
  joint statistical structure (Stage 2) without joint steering producing
  a clean effect (Stage 3). Could mean the model uses the synergy
  read-only — for internal computation, not as an output operator.
  That's interesting; log separately.
- **Pruning may miss lichens.** Two features could be lichen partners
  while not being co-activation graph neighbors (PMI threshold filters
  them out). Mitigation: do one cheap full-sweep round at very low PMI
  on a random ~10% sample; check whether any high-synergy pairs were
  below the threshold. If many, raise the threshold concern.
- **Pair-level naming is harder.** Atomic platypi at least have
  top-activating contexts to anchor naming. Lichens have *joint*
  contexts, which are sparser by construction. Mitigation: the Stage 4
  notebook should show contexts where both members co-fire above their
  individual 95th percentiles, not just where either fires.
- **Triples are real and out of scope.** If many operators are
  3-feature compounds, Path 5's pair-only search will mis-attribute or
  miss them. Mitigation: log the triples we're skipping by listing
  3-member co-activation triangles among lichen members; revisit in a
  v2.

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

- **Category list is no longer English-shaped, but the discovery method
  still is.** The roster spans typological + Ithkuil + LLM-native, but
  someone had to write it, and that someone (and the LLM that helped)
  thinks in human-language categories. The probe will not find a
  distinction nobody on Earth has named. Mitigation: accept the
  limitation, document it in the methodology writeup, and treat any
  category where multiple Tier-3 probes converge on overlapping top
  features as a hint that the model carves the space along a seam we
  didn't name explicitly. Worth a notebook of "what's left unaccounted
  for" — pick the validation-failed cells and inspect what direction
  *did* separate the pairs (if any), even if it wasn't the one we
  intended to find.
- **Multilingual probing assumes the concept space is genuinely
  language-agnostic.** Prior work supports this for content concepts; the
  jury is partially out for grammatical ones. Mitigation: cross-validate
  by probing the same category in multiple source languages and checking
  that the recovered directions cluster. Wide spread between source
  languages = the "concept" is partly a language-specific morphological
  pattern, not a shared concept; that's also a finding.
- **Audit regex is biased toward auto-interp's vocabulary.** Neuronpedia
  labels were generated by a different LLM with its own preferred
  description style. Iterate the regex table after the first run by
  eyeballing the misses.

---

## Suggested order of execution

1. Path 1 audit. Decision point: how much glue is already sitting in
   the lexicon?
2. Minimal-pair generation (Path 2 Stage A). Run in the background
   while you read Path 1's output.
3. Activation cache sweep (Path 2 Stage B). **This cache also powers
   Path 4 Stage 1**, so don't delete it.
4. Direction extraction + projection (Path 2 Stages C–E).
5. Path 4 Stage 1 (glue-signature filter). Reuses the cache from step 3.
6. Inspect Path 2 outputs, decide whether Path 3 is needed.
7. Path 4 Stages 2–5 (steering, clustering, naming, WALS cross-check).
   Run *after* Path 2 finishes so the WALS / Path-2 alignment check has
   targets to compare against.
8. Path 5 Stages 1–2 (lichen candidates + synergy). Reuses the
   coactivation graph and Path 4's glue candidates.
9. Mycorrhizal-hub scan. Can run parallel to (8).
10. Path 5 Stage 3 (joint steering). Run after Path 4's steering
    finishes — same GPUs.
11. Path 5 Stages 4–5 (cluster, name, cross-check).
12. Path 6 Stages 1–6 (synthesis: unify, collapse, type, induce
    combinator rules, promote false lichens, emit grammar spec).
    Mostly CPU; reuses Path 2's activation cache and Path 4/5
    steering signatures.
13. Phonotactics update + site build. Uses Path 6's core inventory.
14. Origin-story / methodology writeup. Last, after the artifact
    stabilizes. Surface Path 4, Path 5, and Path 6 findings
    prominently. Include the unit-of-analysis caveat verbatim in the
    methodology section.

---

## Path 6 — Particle grammar synthesis

> Paths 1–5 produce a tagged inventory of operators. This path turns
> that inventory into a *grammar*: a small typed alphabet of
> particles plus the combinator rules that govern how they sequence.
> The discovery work is done; what's left is collapsing redundancy
> across paths, applying inventory-size pressure, typing what
> survives, and inducing the combinatorial rules from real
> activations.

**Companion docs:** `spec.md` §5 (transformations as a primitive),
`POST-GLUE-SKETCH.md` §0 (Path 6 runs *before* the post-glue agenda
— it's the synthesis step that lets post-glue claim novelty against
a *closed* grammar rather than a heap of operators).

**Premise:** the parts are inert without the joints. Path 1 says
"negation is in the lexicon." Path 2 says "negation has a clean
direction." Path 4 says "this platypus steers polarity." Path 5
says "this lichen flips scope under polarity." Four entries,
plausibly one particle. The grammar can't ship with all four labels
intact, and a Toki Pona / CCG-style design needs both a small typed
alphabet *and* the combinator rules between elements — neither
falls out of Paths 1–5 directly.

### Stage 1 — Cross-path operator unification

Take all four upstream deliverables and find what's duplicated:

- `data/processed/function_lexicon.json` (Path 1)
- `data/processed/function_lexicon_probed.json` (Path 2)
- `data/processed/platypi.json` (Path 4)
- `data/processed/lichens.json` (Path 5)

For each entry across all four files, construct a common-space
steering signature:

- Path 1 entries: synthesize a signature by steering the entry's
  feature on the standard 100-prompt steering set (same prompts
  Path 4 used). Cheap — features already live in the lexicon.
- Path 2 sparse / lexical entries: same — steer the top SAE feature.
- Path 2 diffuse / affix entries: steer along the direction itself
  (residual-stream intervention, no SAE projection). Same 100 prompts.
- Path 4 platypi: reuse the existing signature.
- Path 5 lichens: reuse the existing joint-steering signature.

Every entry now has a signature vector in the same space.

Build an alignment graph:

1. Compute pairwise cosine similarity between all signatures.
2. Threshold; start at 0.55 and sweep θ ∈ {0.45, 0.55, 0.65, 0.75}.
3. Connected components of the thresholded graph are **unified
   operators**.

For each component:

- Canonical representative wins by validation accuracy (Path 2) or
  steering consistency (Paths 4/5); ties broken by agreement count.
- Other members become aliases. Not discarded — the alias list
  documents which atomic / compositional / supervised paths each
  operator showed up in. That redundancy is evidence of reality.

Expected shape: 60–120 raw upstream entries collapse to 30–50
unified operators.

**Output:** `data/interim/unified_operators.json`:
```json
{
  "schema_version": 1,
  "threshold_swept": [0.45, 0.55, 0.65, 0.75],
  "threshold_chosen": 0.55,
  "operators": [
    {
      "operator_id": "UOP-014",
      "canonical_source": "path-4 / platypus / cluster-3",
      "canonical_feature_id": 8421,
      "signature": [...],
      "validation_accuracy_path2": 0.91,
      "steering_consistency_path4": 0.74,
      "aliases": [
        {"source": "path-1", "entry_id": "..."},
        {"source": "path-2", "category": "negation"},
        {"source": "path-5", "lichen_id": "LICHEN-007"}
      ],
      "best_label": "polarity-flip",
      "agreement_count": 3
    }
  ]
}
```

### Stage 2 — Inventory-size pressure (MDL collapse)

Thirty-to-fifty operators is too many for a particle grammar. Apply
pressure.

1. Build a **grammatical-coverage test set**: ≈ 500 pairs of
   contrasting short texts where the only difference is one
   grammatical category (polarity, tense, scope, mood, register…).
   Reuse Path 2's minimal-pair corpus — it's already exactly this.
   Re-weight so each tier contributes proportionally to its
   discovery yield (otherwise Tier 1's category count dominates
   selection by sheer mass).
2. Define **coverage of an operator inventory I** as test-set
   accuracy of a sparse classifier using only I's steering
   directions as features.
3. Greedy forward selection: start from empty I, repeatedly add the
   operator that gives the largest coverage gain, stop when marginal
   gain falls below ε (start: 0.5% per added operator).
4. Sweep θ from Stage 1 jointly with ε; pick the (θ, ε) that
   maximizes coverage / inventory-size ratio. Threshold becomes
   data-chosen, not author-chosen.

Two outputs:

- **Core inventory**: operators selected before the stop condition.
  Target size: 6–14. The conlang's particle set.
- **Long tail**: unified operators not selected. Remain in the
  function lexicon as rare / register-specific items but don't get
  particle slots in the grammar.

If core inventory blows past 14, Gemma 2's operator space is
genuinely high-dimensional and the conlang has to be. Document and
proceed — the small-particle aesthetic isn't worth misrepresenting
the model. If it comes in under 6, the test set is too narrow —
expand it.

**Output:** `data/processed/core_particles.json` and
`data/processed/long_tail_operators.json`.

### Stage 3 — Particle typing (arity, scope, binding)

Each core particle needs a type signature. Three properties to
recover per particle, all measurable from activation data:

1. **Arity.** Steer the particle and measure structural scope of
   the effect:
   - Localized to firing token → unary (free-standing modifier).
   - Spans firing token + one neighbor → binary.
   - Spans firing token + clause → clause-scope.
   Use Path 4 Stage 2's structural-shift profile across positions;
   bucketize affected positions and read off the dominant scope.
2. **Binding direction.** At the firing token, where does the
   attention head most associated with the particle attend? Pull
   the attention pattern at the layer where the SAE feature lives.
   - Preceding-heavy → right-binding (modifies what came before).
   - Following-heavy → left-binding (modifies what comes after).
   - Self / diffuse → free / type-driven.
   Use the median across the particle's top-1000 activating contexts.
3. **Composition scope.** For binary / clause-scope particles, the
   embedding half-life under steered vs. baseline at positions
   t+1 … t+10. < 2 tokens: tight. 2–6: phrase. 7+: clause / sentence.

Particles that don't admit a clean type (no dominant attention
direction, diffuse scope) become **type-free** particles: documented
as exceptions, combine permissively. Expect 1–3 of these.

**Output:** `data/processed/core_particles_typed.json` — extends
`core_particles.json` with:
```json
{
  "operator_id": "UOP-014",
  "type": {
    "arity": "binary",
    "binding": "right",
    "scope": "phrase",
    "half_life_tokens": 3
  }
}
```

### Stage 4 — Combinator-rule induction

Find the small set of moves that govern how typed particles compose.

1. Run Gemma 2 over a 100k-token natural-text sample. Record, per
   token, which core particles are active above their steering
   threshold.
2. Extract sequences where ≥ 2 core particles co-fire within a
   5-token window. Expect a few thousand such windows.
3. For each (particle_A, particle_B) pair: compute relative-position
   distribution (does A precede B, follow B, appear at variable
   distance?).
4. Cluster windows by (particle types involved, relative positions,
   intervening token count). Each large cluster is a **construction**
   — a stable multi-particle pattern.
5. Read off the combinator rules implied by the constructions:
   - "Right-binding particle X composes with following content word
     Y; intermediate particles permitted up to depth 2."
   - "Particles X and Z always co-occur in an (X … Z) frame; the
     enclosed material is operand of both."
   - …

Target: a small CCG-style rule alphabet of 3–6 combinators. If
construction clusters yield one rule (everything is "apply"), the
grammar is purely typed and that's fine. If clusters yield 10+, the
grammar has constructional idiosyncrasies — document each, flag for
review.

**Output:** `data/processed/combinator_rules.json`:
```json
{
  "schema_version": 1,
  "rules": [
    {
      "rule_id": "C1",
      "name": "apply-right",
      "pattern": "particle{binding=right, arity=binary} + content",
      "instances_observed": 14210,
      "example_window_ids": [...]
    }
  ]
}
```

### Stage 5 — Promote the false lichens

Path 5 logs pairs that pass synergy (its Stage 2) but fail joint
steering (its Stage 3). The plan calls them "false lichens" and
shelves them. They're not false — they're candidate **gating
combinators**: pairs the model uses for internal selection, not for
output transformation.

For each false lichen:

1. Pull the activations of both members on the 100k-token natural-
   text sample from Stage 4.
2. Test: does the joint activity of (i, j) predict which *other*
   core particle fires in the following 1–5 tokens, controlling for
   either member alone? Logistic regression on each downstream
   particle as target.
3. If yes → **gating combinator**: when both fire, the model
   selects a specific downstream operator. Add to
   `combinator_rules.json` with `source: false-lichen-promotion`
   and the predicted downstream operator named in the rule.
4. If no → leave shelved.

Expect 5–20 false lichens; expect 1–5 to promote. Each promoted
combinator is a meta-grammar rule about how particles condition
each other.

### Stage 6 — Emit the grammar spec

Compile Stages 2–5 into a single human-readable + machine-readable
spec:

- `data/processed/grammar_spec.json` — full machine spec: core
  particles + types + combinator rules + constructions + long tail.
- `docs/grammar.md` — human-readable narrative drawing from the
  spec. Section order: particle inventory → typing system →
  combinator rules → worked examples (parse "word rules word
  rules word word" end-to-end so a reader can see the system
  work).

The grammar spec is the conlang's **closed grammar at v1**: any
utterance is parseable using only the rules in the spec, or it
isn't well-formed. Closure is aspirational and bounded by the SAE
basis; `POST-GLUE-SKETCH.md` queues the operators that may extend
v2 once raw-residual and circuit-level discovery lands.

### Parallelism

- **Stage 1**: per-entry steering for unification signatures. Mostly
  cached if Path 4 ran on the same prompt set. Embarrassingly
  parallel across entries.
- **Stage 2**: small numerical optimization, single CPU. Sweep over
  (θ, ε) is a few dozen evaluations — trivial.
- **Stage 3**: per-particle attention extraction. Reuses Path 2
  Stage B's activation cache + corresponding attention-pattern
  cache (add attention hooks to the cache sweep if not already
  captured — cheap to redo). Embarrassingly parallel across
  particles.
- **Stage 4**: single forward-pass over 100k tokens (cheap on
  Gemma 2 2B), then CPU clustering.
- **Stage 5**: per-pair regression on cached activations; CPU.
- **Stage 6**: writing.

Total compute footprint: small. This path is mostly CPU; the only
GPU work is Stage 3's attention extraction and Stage 4's natural-
text forward pass, both single-pass.

### Done criteria

- Core inventory of 6–14 particles, each with a complete type
  signature.
- Combinator-rule alphabet of 3–6 rules.
- ≥ 90% of the grammatical-coverage test set explained by the core
  inventory (Stage 2).
- ≥ 1 false lichen promoted to combinator rule (zero means the
  false-lichen channel was empty signal — a finding, not a failure
  of this path).
- A `grammar_spec.json` and `docs/grammar.md` draft that together
  let a reader parse "word rules word rules word word" without
  further guidance.

### Risks specific to Path 6

- **Unification threshold is hand-tuned.** Set cosine at 0.55 and
  you get one inventory; set it at 0.65 and you get a different
  one. Mitigation: sweep θ jointly with Stage 2's ε; the data
  picks the operating point, not the author.
- **Core-inventory size depends on the coverage test set.** Path 2's
  minimal pairs were built for direction-finding, not for
  inventory-selection. They over-represent Tier 1. Mitigation:
  re-weight the test set so each tier contributes proportionally
  to its discovery yield.
- **Typing assumes attention patterns are interpretable.** For some
  particles, the relevant computation is distributed across heads
  and no single head shows clean directional preference.
  Mitigation: fall back to the steered-effect propagation profile
  (Stage 3 item 3) and assign type-free status if even that's
  diffuse.
- **Combinator induction might find no structure.** If
  constructions cluster into 50+ singleton patterns, the model
  doesn't use particles compositionally — they're independent
  modifiers. Mitigation: that's a real finding; ship a typed-but-
  free grammar (one combinator, "apply") and document the result.
- **The "false lichens as gating combinators" hypothesis is
  testable but might be wrong.** Stage 5 has a sharp gate; if no
  pairs pass, leave them shelved. Don't force-promote.
- **Closure is aspirational.** v1 says "anything parseable with
  these rules"; v2 will need extension when post-glue surfaces
  operators that don't live in the SAE basis. Acknowledge the open
  boundary in `grammar.md` itself.
