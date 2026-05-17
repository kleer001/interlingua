# POST-GLUE-SKETCH

**Status:** Sketch, not a plan. Pre-implementation. Reads after
`GLUE-TODO.md` has run end-to-end and produced its glue lexicon. The
question this document opens is: *what did we miss by committing to SAE
features as the unit of analysis, and how would we find it?*

**Companion docs:** `GLUE-TODO.md` (Paths 1–5), `spec.md` v0.2 (overall
pipeline), `prior-work.md` (the methodological substrate).

**Working title for this phase:** _post-glue discovery_ — until the glue
survey lands and the actual gaps are visible, the agenda below is
provisional. Sections will get reorganized, dropped, or expanded based
on what Paths 1–5 surface.

---

## 0. Why this document exists

Paths 1–5 commit to **SAE features** as the unit of analysis. Path 5
relaxes "atomic" to "pair or triple," but the basis is still SAE. The
"Unit-of-analysis caveat" in `GLUE-TODO.md` names three things this
leaves outside scope:

1. Directions in raw residual stream that no SAE-feature combination
   reconstructs sparsely.
2. Large circuits — multi-feature, multi-layer computational graphs.
3. Distributed / no-locus operators living in attention patterns or
   other components.

This document sketches how to attack each of those, plus a few
adjacent open questions. None of it should run until the glue survey
is done — the glue survey produces the **comparison set** that lets
post-glue discovery claim novelty (or admit re-discovery).

The biology metaphor continues. Path 4 found platypi (unclassifiable
atomic individuals); Path 5 found lichens (compound organisms with
joint function); Path 5's side scan found mycorrhizal hubs (mediators).
The paths below look for:

- **Dark biota** — organisms living in basins we don't normally
  sample (raw activation space, not the SAE basis).
- **Colonial organisms** — operators living in subspaces, not points.
- **Ecosystems** — multi-layer computational graphs that act as units.
- **Pheromone networks** — attention patterns as operators.
- **Shapeshifters** — operators whose form depends on context.

The metaphors are flavor. The methodology is what matters.

---

## 1. What we know after Paths 1–5 (placeholder)

This section gets filled in after the glue survey runs. It should
record, at minimum:

- Number of audited / probed / discovered / lichen-bound operators per
  category, per tier.
- Categories that didn't separate cleanly in any path.
- Fraction of "high-glue-signature" SAE features whose steering
  signature *didn't* match any known operator at Path 4 Stage 5.
- Fraction of co-activation graph edges that turned out to be lichen
  partners.
- Coverage gaps the writeup admits exist.

The post-glue paths target the gaps. Don't write this section
speculatively; write it from data.

---

## 2. What's still presupposed (expanded)

`GLUE-TODO.md`'s caveat names six things. Restating with sharper edges,
because each maps to a path below:

| Presupposition                              | What it forecloses                                                    | Path that relaxes it |
| ------------------------------------------- | --------------------------------------------------------------------- | -------------------- |
| SAE features are the right basis            | Operators in directions no single SAE feature reconstructs            | Path 6               |
| Operators are linear / 1-dimensional        | Operators living in 2D rotations, manifolds, polytopes                | Path 7               |
| Operators are at most 3-feature compositions| Larger circuits, multi-layer computational graphs                     | Path 8               |
| Operators live in residual stream           | Operators in attention patterns, MLP-out, layernorm, embedding tables | Path 9               |
| Operators are fixed directions              | Context-conditioned operators (different form in different inputs)    | Path 10              |
| Discovery happens at a single layer         | Cross-layer trajectories, iterative computation                       | Path 11 (speculative)|

Each row is a hypothesis the glue survey can't test. Whether any of
them is *worth* testing depends on what the glue survey leaves missing.

---

## 3. Path 6 — Unsupervised raw-residual direction discovery

> **Dark biota.** Most life on Earth lives in conditions we don't
> normally sample — deep ocean, deep crust, hydrothermal vents.
> Path 6 looks for operators in the raw residual stream, not filtered
> through any SAE basis.

This is the centerpiece of post-glue discovery. The premise: SAEs are
a *learned compression* of activation space. They optimize for sparse
reconstruction on a training corpus. Anything the corpus underweights,
or anything that doesn't decompose sparsely into the SAE's dictionary,
is invisible to Paths 1–5. Path 6 looks at the activations directly.

### 3.1 Caching strategy

Reuse the activation cache from `GLUE-TODO.md` Path 2 Stage B if it's
still on disk. If not, re-cache:

- Same corpus (FLORES-200 + LLM-generated minimal pairs + a large
  Wikipedia / Gutenberg sample for broader coverage).
- Multiple layers — start with the same {6, 12, 20} for Gemma 2 2B and
  {12, 20, 31} for 9B, but Path 6 may want a denser layer sweep
  (every 3 layers) since operators may live anywhere.
- Both models.

Disk: dense cache for 10M tokens × multiple layers × bf16 runs into
the hundreds of GB. Use streaming-friendly formats (HDF5 chunked, or
zarr) — Path 6's methods process in batches, no need to load all at
once.

### 3.2 Decomposition methods (run in parallel, compare results)

The point isn't to pick one decomposition — it's to use *several* and
see which directions show up across methods. A direction that surfaces
under PCA *and* sparse dictionary learning *and* causal discovery is
high-confidence, regardless of which method's basis it lives in.

1. **PCA** — boring, but the right baseline. Captures
   maximum-variance directions. If operators are high-variance
   (negation often is), they should appear in top components. Cheap.
2. **ICA** — Independent Component Analysis. Captures statistically
   independent directions, which is closer to what an operator is
   ("independent of content"). Computationally heavier than PCA but
   still single-pass.
3. **Sparse dictionary learning, non-SAE.** k-SVD, MOD, or OMP-based
   dictionary learning. Different priors than SAEs (which use ReLU +
   L1 sparsity). Specifically try:
   - k-SVD with explicit L0 sparsity constraint.
   - Convolutional sparse coding (captures positional structure).
   - Group-sparse coding (forces groups of correlated atoms — finds
     subspace operators that linear sparsity misses).
4. **Diffusion / score-based decomposition.** Recent work has framed
   activation manifolds in terms of score functions. Sketch only —
   needs a separate study to spec properly.
5. **Causal direction discovery.** The most principled approach: find
   directions whose perturbation has consistent grammatical effect on
   output, with no reference to any decomposition. Procedure:
   - Sample a direction d (random initialization).
   - For each of ~100 prompts, perturb residual at the chosen layer by
     ±α·d and measure output shift.
   - Optimize d to maximize *consistency* of the grammatical shift
     across prompts and *minimize* topical change.
   - Restart many times to find multiple directions.
   - This is essentially "operator-search by gradient descent" with no
     basis commitment. Expensive but answers the question directly.

### 3.3 Validation

Each candidate direction goes through Path 4 Stage 2's steering
protocol unchanged. The validation is method-agnostic: clamp the
direction ON / OFF, measure functional-vs-content token shift and
structural shift across diverse prompts. A direction is an operator
if steering produces systematic grammatical transformation.

The output is a set of operator directions in raw residual stream,
each with:
- Its discovery method (which decomposition surfaced it).
- Its steering signature.
- Its projection onto the SAE basis (sparsity, top features, mass).
- Its alignment with Path 4 platypi and Path 5 lichens.

### 3.4 Two interesting outcomes

a) **A direction surfaces in raw space, projects diffusely onto the
   SAE basis, and steers cleanly.** This is the "SAE missed an
   operator" case — proof that Paths 1–5 had a blind spot. Each such
   direction is a finding.

b) **A direction surfaces in raw space, projects *sparsely* onto SAE
   features that were filtered out by §6 or that have low Neuronpedia
   confidence.** This is the "SAE saw it but the audit dropped it"
   case — a methodological loss in §6's filter rubric. Document the
   missed features; consider relaxing the filter.

### 3.5 Parallelism

- **PCA / ICA**: incremental algorithms exist; can stream over
  activation cache without loading it all. CPU-bound on a 256-core
  box, ~hours.
- **Dictionary learning**: per-layer parallelism; each layer can run
  on a separate GPU.
- **Causal direction discovery**: most expensive; parallelize across
  the multi-restart loop. Each restart is independent. GPU-bound.
- **Steering validation**: same parallelism story as Path 4 Stage 2.

### 3.6 Done criteria

- ≥ 5 raw-space directions validated as operators (clean steering
  signature, cross-prompt consistent).
- ≥ 1 direction with diffuse SAE projection (the "SAE missed it"
  case). If zero, that's a strong claim about SAE coverage — worth
  saying.
- ≥ 1 direction that matches Path 4 platypi or Path 5 lichens (the
  "we re-discovered our own finding via a different method" case —
  validates both methods).

---

## 4. Path 7 — Subspace and nonlinear operators

> **Colonial organisms.** A Portuguese man o' war is several
> organisms operating as one; an operator in a 2D subspace is a
> rotation, not a translation. Path 7 looks for operators that aren't
> 1-dimensional directions.

The Tegmark group's parallelogram crystals were an instance of this —
they looked for *transformations* (function vectors) encoded as
directions. The crystals failed in SAE basis (sparsity gate); they
may not fail in raw residual or in 2D subspaces.

### 4.1 Subspace operators

A 2D subspace operator has the form: rotation by angle θ around a
fixed axis in a 2D plane. Clamping any single direction in the plane
doesn't capture the operator; you need to perturb the rotation itself.

Discovery procedure:

1. Find 2D planes in residual stream where:
   - One axis is content-like (varies with topic).
   - The other axis is grammatical (varies with operator state).
2. The plane is identifiable by: variance along one axis correlates
   with topic; variance along the orthogonal axis correlates with
   grammatical state.
3. Validate by *rotating* test inputs through the plane and checking
   output behavior — the rotation should produce systematic
   grammatical shifts.

This is conceptually similar to the "linear representation
hypothesis" vs "non-linear representation" debate in mech-interp. The
empirical question is open. Path 7 contributes a probe.

### 4.2 Polytope and manifold operators

More speculative. Some operators may live on curved manifolds — e.g.,
a "register" operator that traces a curve through neutral → formal →
ceremonial states. Linear directions approximate the curve poorly.

Procedure sketch:
- For each candidate grammatical category, fit a low-dimensional
  manifold (UMAP / Isomap / autoencoder bottleneck) to activations
  conditioned on the category state.
- Test whether the manifold is well-approximated by a line (linear
  operator) vs a curve / surface (nonlinear).
- If nonlinear: characterize the manifold and probe its dimensions.

### 4.3 Connection to Tegmark crystals

The crystal method searched for transformations as **pairwise
differences in SAE basis**. The two methods Path 7 should try, in
order:

- Re-run the crystal search on **raw residual differences** instead of
  SAE feature differences. The LDA distractor projection still
  applies. Loses the sparsity-as-confidence signal but gains coverage.
- Search for crystals at the **subspace level**: pairs of 2D subspaces
  whose offset is consistent across many (a, b, c, d) parallelograms.

### 4.4 Done criteria

- ≥ 1 validated subspace operator (rotation in a 2D plane producing
  consistent grammatical shift).
- ≥ 1 crystal recovered in raw space that wasn't recovered in SAE
  space.

A null result here is also informative: "operators in Gemma 2's
residual stream are well-approximated by linear directions" is a
falsifiable claim worth establishing.

---

## 5. Path 8 — Circuit-level operator discovery

> **Ecosystems.** A circuit is a coordinated multi-feature
> multi-layer computational graph. The operator behavior emerges from
> the interactions; no single piece carries it.

This is where Anthropic's circuit-tracer (already in `spec.md §4
Stage 3`) earns its keep for the glue question. The glue survey
didn't use circuit-tracer at all — Path 5's lichens are
single-layer 2-feature compounds. A real circuit can span 5–20
features across 3–10 layers.

### 5.1 Procedure

1. **Seed**: take the highest-confidence operators from Paths 1–7
   (atomic, lichen, raw, subspace). Each seed is a "this is an
   operator we believe in" anchor.
2. **Attribution-trace upstream and downstream.** Use circuit-tracer
   to find features and components that causally contribute to the
   seed feature's activation, and features the seed feature causally
   affects.
3. **Identify the circuit**: the subgraph that, when intact, supports
   the operator behavior, and when severed at any single edge,
   degrades it.
4. **Test by ablation**: knock out individual nodes of the circuit
   and measure how much the operator behavior degrades. A "core"
   circuit has nodes whose ablation kills the operator; a "redundant"
   circuit has many parallel paths and degrades gracefully.
5. **Catalog circuits per operator.** Some operators will turn out
   to be one core feature plus an attribution halo (not very
   interesting). Others will be genuine multi-feature circuits
   (interesting).

### 5.2 What we learn

- **For named categories**: how is "negation" actually computed in
  Gemma 2? Is it a single feature firing, or a 7-feature circuit
  across layers 4–18? The circuit shape is itself a finding.
- **For platypi / lichens**: are they isolated, or are they faces of a
  larger circuit? An atomic platypus that turns out to be the readout
  node of a deep circuit is more meaningful than one that floats
  alone.
- **For coverage**: are there *circuits without operator labels* — i.e.,
  Gemma 2 has clearly-coordinated multi-feature subgraphs whose
  function we can't name? These are circuit-level platypi.

### 5.3 Cost

Circuit-tracer is the most expensive tool in the stack. Realistic
target: trace circuits for the top ~30 operators (across all paths).
Don't try to trace everything. Each trace produces a graph; visualize
in the Stage 6 site as a per-operator detail page.

### 5.4 Done criteria

- ≥ 30 operators with traced circuits.
- ≥ 5 operators whose circuit involves a feature *not in the
  operator's own definition* — i.e., the operator turns out to depend
  on a feature it doesn't include. These are the most interesting
  ecosystem findings.
- ≥ 1 multi-feature circuit with no nameable operator at its readout
  (circuit-level platypus).

---

## 6. Path 9 — Attention-pattern operators

> **Pheromone networks.** Ants coordinate via chemical trails that
> aren't any single ant. Attention patterns aren't any single
> feature.

Path 3 in `GLUE-TODO.md` uses attention-out SAEs — the *outputs* of
attention layers decomposed into features. Path 9 looks at the
*patterns themselves*: which positions attend to which, by which
heads, conditioned on what.

### 6.1 Why

Some grammatical operators are realized as attention patterns.
Long-distance agreement (subject-verb across embedded clauses),
anaphora resolution, parenthetical handling — these are head-level
operations in many models, not feature-level.

### 6.2 Procedure sketch

1. For each attention head in each layer, characterize its function
   by minimal-pair contrast (Path 2 style). What does this head do
   that no other head does?
2. Categorize heads by function: copy heads, induction heads,
   coreference heads, syntactic-agreement heads, name-tracking heads.
3. For grammatical operators that didn't fully resolve in Paths 1–8:
   check whether the work is being done by attention heads instead of
   residual-stream features.
4. Ablation: knock out heads (individually and in combinations) and
   measure grammatical capability of the model. Combine with
   activation patching to confirm causal role.

### 6.3 Prior art

Head categorization is a well-developed corner of mech-interp.
Olsson et al. on induction heads; the various IOI papers; Anthropic's
attention work. Path 9 builds on these, doesn't reinvent.

### 6.4 Done criteria

- ≥ 20 heads categorized by function in Gemma 2.
- ≥ 3 grammatical operators traced to specific head sets.
- ≥ 1 operator whose head set has no analog in published head
  taxonomies (an attention-level platypus).

---

## 7. Path 10 — Dynamic / context-conditioned operators

> **Shapeshifters.** An operator that takes one form in formal
> contexts and another in informal ones isn't a fixed direction —
> it's a mapping from context to direction.

The premise: not all operators are static directions. Some may be
**direction-valued functions of context** — the "polarity flip"
direction might be different in legal text than in casual dialogue.
If so, Paths 1–8 see a smeared / inconsistent direction and
under-attribute to the operator.

### 7.1 Procedure sketch

1. For each operator that didn't validate cleanly in any earlier
   path (low cross-prompt consistency at Path 4 Stage 2, for
   example): hypothesize it's context-dependent.
2. Partition the test corpus by context (genre, register, topic
   cluster).
3. Re-run the direction-finding within each partition.
4. If the within-partition directions are clean but the cross-partition
   directions diverge — the operator is dynamic. Characterize the
   context → direction mapping.

### 7.2 Why this is the most speculative path

It requires assuming the operator exists despite earlier-path
failure. The risk is finding patterns in noise. Mitigation: require
the partition-wise directions to *steer cleanly within each
partition*. A dynamic operator passes the steering test conditional
on context; a noisy non-operator doesn't pass any context's test.

### 7.3 Done criteria

- ≥ 1 operator confirmed as context-dependent (clean within-partition
  steering, divergent across partitions, where the divergence is
  systematic and interpretable).
- ≥ 0 is acceptable: "Gemma 2's grammatical operators are
  context-stable" is a finding.

---

## 8. Path 11 — Cross-layer trajectories (speculative; deferred further)

> **Migration patterns.** Some operators may not live in any one
> layer — they're trajectories of an idea through the model's depth.

The premise: an operator might be realized as a sequence of
transformations across layers. Early-layer "raw polarity," middle-layer
"polarity-bound-to-subject," late-layer "polarity-as-output-bias."
Probing at any single layer sees only one slice.

### 8.1 Procedure sketch

1. Pick an operator with clean atomic signatures at multiple layers
   (per Path 4 results).
2. Track its evolution across layers: how does the direction rotate
   / change magnitude / acquire structure as you move deeper?
3. Characterize the trajectory. If consistent across many operators,
   there's a "language stack" — each layer does a specific job in
   the realization of any operator.

### 8.2 Why deferred

This path overlaps with the field's open question about whether
features "iterate" through layers or whether each layer has
independent features. The empirical answer isn't settled. Don't run
Path 11 until the field's tools mature; the cost of premature attack
is high and the payoff is uncertain.

### 8.3 Done criteria (aspirational)

- A characterization of the layer-by-layer trajectory for ≥ 5
  operators.
- Evidence for or against "layer stack" hypothesis.

---

## 9. Cross-cutting infrastructure

All post-glue paths share requirements:

- **Direct model hooks.** TransformerLens, nnsight, or
  Anthropic's circuit-tracer (which uses its own hooks). Decide
  early which framework — switching mid-path is expensive.
- **Activation cache infrastructure.** Path 6 will likely require a
  re-cache (denser layer sweep). Plan for ~1 TB of disk.
- **Steering / intervention infrastructure.** Reuse what Path 4 built;
  generalize to support direction-clamping, subspace-rotation, and
  attention-head ablation.
- **Comparison framework.** Every path needs a way to compare its
  findings against Paths 1–5 findings and against earlier post-glue
  paths. Build a `data/processed/operators_registry.json` that all
  paths read from and write to, with provenance per operator
  (discovered by which path, when, with what method).

---

## 10. Decision criteria — when to run which path

Don't run any of this until the glue survey lands. Once it does,
decisions to make:

| Glue-survey finding                                                | Recommended next path |
| ------------------------------------------------------------------ | --------------------- |
| Many categories failed validation in Path 2                        | Path 6 first — maybe the SAE basis was wrong; raw-space probing may find the missed directions. |
| Path 4 yielded many platypi but they cluster oddly                 | Path 8 — trace their circuits; the cluster structure may make sense at the circuit level. |
| Path 5 yielded few lichens (synergy is rare)                       | Path 7 — operators may be in subspaces, not point pairs. |
| Operators that *steer inconsistently* across prompts               | Path 10 — they may be context-dependent. |
| Long-distance grammatical phenomena underrepresented in findings   | Path 9 — attention is the obvious culprit. |
| Glue survey found lots of clean atomic operators with clean circuits | Path 11 — try cross-layer; the model may have a deep stack worth mapping. |
| Glue survey found roughly what we expected                         | Stop here. Publish the conlang. Post-glue is for residual mysteries, not curiosity tax. |

The decision is empirical, not prescribed. The glue survey's actual
output dictates which paths earn their compute.

---

## 11. Risk register (post-glue specific)

- **Method proliferation without method validation.** Running PCA,
  ICA, dictionary learning, causal discovery, subspace probes, and
  attention-head probes produces lots of "candidate operators." Each
  method has its own false-positive profile. Mitigation: require any
  operator claim to pass *steering validation* (Path 4 Stage 2),
  regardless of which method discovered it. The validation is the
  arbiter; the methods are just candidate generators.

- **Compute cost balloons fast.** Path 6 alone could 10x the glue
  survey's compute. Mitigation: each path has a cap (number of
  candidates to validate, hours of GPU). If the cap is hit before
  done criteria are met, that's the result — don't extend.

- **Findings become unfalsifiable.** "We found a dynamic operator
  whose form depends on context" is hard to falsify if the test is
  permissive. Mitigation: pre-register the done criteria. Don't
  weaken them mid-experiment.

- **The conlang doesn't care.** All of post-glue is methodological
  rigor. The conlang artifact itself can ship with just the glue
  survey results — post-glue is for the methodology writeup and for
  honesty about coverage. Don't let post-glue block shipping.

- **Mech-interp moves fast.** The decompositions and tools available
  in 12 months will differ from today's. Mitigation: write
  decomposition-method-agnostic code where possible; treat each
  method as a plugin.

---

## 12. Biology framing summary

| Path | Biological analog          | Methodological move                                            |
| ---: | -------------------------- | -------------------------------------------------------------- |
|    1 | Audit of existing finds    | Re-survey the known catalog for misclassified specimens        |
|    2 | Field survey with checklist| Supervised probing for named categories                        |
|    3 | Different ecological niche | Switch SAE flavor (attention-out instead of residual)          |
|    4 | Platypi                    | Atomic individuals with unclassifiable behavior                |
|    5 | Lichens + mycorrhizal hubs | Compound organisms; mediator species                           |
|    6 | Dark biota                 | Discovery in unsampled basins (raw activation space)           |
|    7 | Colonial organisms         | Coordinated multi-cell units (subspaces, manifolds)            |
|    8 | Ecosystems                 | Multi-organism dynamics (circuits across layers)               |
|    9 | Pheromone networks         | Coordination signals (attention patterns)                      |
|   10 | Shapeshifters              | Context-dependent forms                                        |
|   11 | Migration patterns         | Trajectory-level life cycles (cross-layer)                     |

The biology is mnemonic, not load-bearing. The methodology is what
holds up.

---

## 13. Open questions for this document

- Does Path 6 need a fresh activation cache, or can it reuse Path 2's?
  Depends on whether Path 2 caches enough layers densely enough.
  Revisit after Path 2 lands.
- Which framework: TransformerLens vs nnsight vs circuit-tracer's
  own hooks? Probably all three for different paths. Worth a
  one-page comparison doc.
- Is Path 8 (circuit-tracer) actually feasible at the cost we
  estimated? Circuit-tracer demos on Gemma 2 are tractable for
  toy circuits; multi-operator catalog-building at scale is
  unproven. Get a cost estimate from the circuit-tracer team.
- Does any of this need a second model (not Gemma 2)? Cross-model
  agreement is a strong validator (per Path 4 risk register). Adding
  Llama or Mistral to the comparison would multiply cost ~3x. Maybe
  worth it for the top ~10 operators only.
- Does the conlang artifact change at all based on post-glue
  findings? Probably yes for operators that surface in raw space
  (Path 6) and would get phonological forms — but for circuits and
  cross-layer trajectories, probably no (they're methodology, not
  lexicon). Decide per finding.

---

## 14. Versioning this sketch

This document is a sketch until the glue survey lands. After that:

- Section 1 ("What we know after Paths 1–5") gets written from data.
- Sections 3–8 (Paths 6–11) get re-prioritized based on what gaps
  the glue survey actually leaves.
- Section 10 (decision criteria) becomes a real decision document.
- Sections that don't get run get deleted, not kept around as
  ghosts. Pruning is a feature.

Don't promote this to a full spec until you're committed to running
at least one of its paths. Sketches that become specs without first
becoming experiments tend to ossify.
