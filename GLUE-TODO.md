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


All three are independent and can run in parallel if hardware permits — see
the parallelism notes at the end of each path.

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

Path 1 is an afternoon. Path 2 is the main event. Path 3 is optional, gated
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
  named targets; nothing in Paths 1–5 finds them unsupervised. A truly
  open Path 6 would do unsupervised direction-discovery in raw activation
  space, decoupled from any SAE. That's a research project of its own
  scale; flagged in "Open questions" rather than planned here.
- **Large circuits.** If an operator is a 7-feature computational graph,
  no method in this doc finds it. Anthropic's circuit-tracer (already in
  `spec.md §4 Stage 3`) is the right tool; this doc doesn't yet wire it
  in for glue discovery. Candidate addition for later.
- **Distributed / no-locus operators.** Some grammatical behavior in
  modern models lives in attention patterns, not in residual-stream
  features. Path 3 (attention-out SAEs) catches the SAE-decomposable
  subset of this; the rest is out of scope.

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

The tier doesn't change the probing method but it does change expectations
and ordering:

- **Tier 1** probes should mostly succeed (high-confidence target);
  failures here mean methodology bug, not absence.
- **Tier 2** is where the project earns its keep — these are real
  cross-linguistic distinctions, multilingual minimal pairs are honest,
  and any one of them surfacing cleanly is a payoff that hand-rolling
  could never claim.
- **Tier 3** is exploratory. Expect low hit rate. Each hit is a paper-worthy
  result on its own.
- **Tier 4** should mostly succeed but on later layers than Tier 1
  (discourse / pragmatic features tend to peak deeper).
- **Tier 5** is a sanity check and writeup material — knowing the LLM
  language privileges things human languages don't is itself the point.

Run tiers in this order so Tier 1's clean wins validate the pipeline
before you trust Tier 3's findings.

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

Anything less is a finding too — write it up. "We probed for X, the
model didn't encode it cleanly" is honest and on-brand. The negative
results in Tier 3 in particular are valuable: they tell you something
about the shape of the model's grammatical space that no successful
result can.

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

Paths 1–3 are confirmatory. Each one starts from a target the project
team (or the linguistic typology literature, or the Ithkuil community)
already named, and asks "does the model encode this." Path 4 inverts
the question: "what is the model encoding that *acts like* glue,
regardless of whether we have a name for it?"

The methodological pivot is identifying glue by **behavioral signature**
rather than by labeled target. A feature is glue-shaped if it:

- fires across many semantic clusters (promiscuous),
- is topic-independent (same feature in cooking texts, physics texts,
  dialogue),
- is positionally / structurally cued rather than topically cued (fires at
  boundaries, after specific structural positions, regardless of content),
- and — most importantly — **causes systematic transformations of
  output** when steered, not changes of topic.

The last criterion is the gold standard. Tense markers, negation,
politeness, definiteness all share it: when you manipulate them, content
stays but *form* shifts. Steering by topical features changes the
subject; steering by operator features changes the grammar around the
subject. That's the platypus test.

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
  features. ~16k features at ~1s each = a few hours single-threaded;
  minutes on a multi-process pool. Disk-bound, not GPU-bound.
- **Stage 2** (steering): this is the expensive bit. 200–500 candidates
  × (ON / OFF / baseline) × 100 prompts × generation cost. Mitigations:
  - Cap candidates at top-500 by Stage 1 score before steering.
  - Short completions (50 tokens).
  - Aggressive batching — Gemma 2 9B with bf16 + a 24 GB GPU can run
    batches of 32 short completions at decent speed; 80 GB GPU handles
    128+.
  - Parallelize across candidates if you have multiple GPUs — each
    candidate's steering set is independent.
  - Estimated cost on a 4-GPU rig: ~12–24 hours for 500 candidates ×
    300 completions. Run overnight.
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

Zero genuine platypi is still a publishable result: "Gemma 2's operator
space, after unsupervised search and typology cross-check, decomposes
into known grammatical categories" — that's a strong claim about
model–language alignment and worth saying out loud.

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

Path 4 promotes a feature to a platypus only if *clamping that feature
alone* produces systematic grammatical change. A feature that's inert
solo but functions in compound gets dropped at Path 4 Stage 2. That's
the gap.

Lichens matter because the model probably uses them. A lot of mech-interp
work (Anthropic's circuits, the Bricken et al. monosemantic features
work, attribution graphs) suggests that operator behavior is often
distributed across small clusters of features that fire together but
don't act in isolation. Negation, in some models, decomposes into a
"polarity feature" + "scope feature" pair where neither alone flips
output polarity but the conjunction does. We won't know which categories
are lichen-shaped in Gemma 2 until we look.

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

This is a 1-day scan; don't build a path around it. Just run it and
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
  Estimated cost on a 4-GPU rig for 500 lichen candidates: 1–2 nights
  of steering.
- **Stage 4–5**: same as Path 4. CPU and human-in-the-loop.

### Done criteria

- ≥ 5k pairs survive Stage 1.
- ≥ 100 pairs pass the synergy threshold at Stage 2.
- ≥ 10 of those are confirmed lichens at Stage 3 (joint > solo,
  cross-prompt consistent).
- ≥ 2 distinct lichen clusters at Stage 4.
- ≥ 1 cluster survives Stage 5 as "no match to Path 4 platypi, no
  match to WALS, distinct from any lexicon entry."

Zero genuine novel lichens is again a finding: "Gemma 2's operator
behavior decomposes into atomic features and known compositional
categories." Strong claim. Worth saying out loud.

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

- **Probe directions may not be sparse in any width.** Mitigation: the
  sparsity gate is the call. Diffuse directions are still useful — they
  become affixes, not lexicon entries.
- **Minimal-pair generation may leak content.** A naive "I walk / I
  walked" template might encode `walk`-the-content alongside tense.
  Mitigation: average across many lexical contexts; the direction-finding
  method already cancels content if the slot fill is diverse enough.
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

1. Path 1 audit. 1 afternoon. Decision point: how much glue is already
   sitting in the lexicon?
2. Minimal-pair generation (Path 2 Stage A). Runs in the background
   while you read Path 1's output.
3. Activation cache sweep (Path 2 Stage B). Overnight job for 9B at full
   corpus. **This cache also powers Path 4 Stage 1**, so don't delete it.
4. Direction extraction + projection (Path 2 Stages C–E). One day.
5. Path 4 Stage 1 (glue-signature filter). Reuses the cache from step 3.
   A few hours of CPU.
6. Inspect Path 2 outputs, decide whether Path 3 is needed.
7. Path 4 Stages 2–5 (steering, clustering, naming, WALS cross-check).
   Overnight for steering, then a day for human-in-the-loop naming. Run
   *after* Path 2 finishes so the WALS / Path-2 alignment check has
   targets to compare against.
8. Path 5 Stages 1–2 (lichen candidates + synergy). Reuses the
   coactivation graph and Path 4's glue candidates. A day of CPU.
9. Mycorrhizal-hub scan. Half a day; parallel to (8).
10. Path 5 Stage 3 (joint steering). Run after Path 4's steering
    finishes — same GPUs. 1–2 nights.
11. Path 5 Stages 4–5 (cluster, name, cross-check). A day.
12. Phonotactics update + site build.
13. Origin-story / methodology writeup. Last, after the artifact stabilizes.
    The writeup needs to surface Path 4 and Path 5 findings prominently
    — that's where "the language inside a translation" earns its
    strongest claim. The unit-of-analysis caveat goes in the
    methodology section verbatim; honesty about scope is the difference
    between a fun conlang and a defensible artifact.
