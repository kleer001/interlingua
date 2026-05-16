# Project Spec: Conlang from LLM Internals

**Status:** Living document. v0.2 — pre-implementation, post-prior-work-survey.
**Companion:** `prior-work.md` — read first; this spec assumes familiarity with the landscape it documents.
**Working title for the language:** _undecided_ (named after v1 pipeline runs; see §12).
**Deliverable:** A full conlang documentation site (origin story, methodology writeup, grammar sketch, lexicon browser, example texts). Local Python project, static site at the end.

## Changelog from v0.1

- §0 framing tightened to acknowledge we're building on a mature substrate, not pioneering it.
- §4 Stage 3 reworked: use Anthropic's open-source circuit-tracer rather than rolling our own attribution code.
- §4 Stage 4 reworked: the "does this have the shape of a language" question has been partially answered by prior work for the *substrate*; what we test is whether the substrate fits our *purpose*. References to Tegmark group, ConceptViz, Google PAIR added.
- §4 Stage 5 substantially revised: parallelogram crystals (function vectors encoding semantic transformations like king:queen::man:woman) are added as a candidate primitive relation type. This was not in v0.1.
- §7 risk register: added crystal-sparsity risk and LDA-distractor-projection risk based on Tegmark group's finding that initial searches yielded "mostly noise."
- §9 open questions: refined based on what prior work has now answered.
- §11 (new): repos to study before writing code.

---

## 0. The Idea in One Paragraph

Multilingual LLMs develop an internal "concept space" in middle layers — a shared, language-agnostic representation that input languages encode into and output languages decode from. Sparse autoencoders (SAEs) decompose model activations into a dictionary of features, many of which are multilingual (the same feature fires for the same concept across languages). The geometric structure of those features has been shown to encode semantic relations — most strikingly, parallelogram "crystals" that capture transformations like (man:woman::king:queen). This project takes that feature dictionary and its relational geometry as raw material and asks: if we treat the features as the lexicon of a "language" the model thinks in, regularize the relational structure between them, and attach phonemes, what conlang falls out? The deliverable is a documented, browsable artifact — not a research paper, but a real language with origin lore.

What's new in this framing (vs. v0.1): the *substrate* — features, their topology, their hierarchical clustering — is well-studied. The *interpretive frame* (treating it as a lexicon, designing a relational schema, assigning phonemes, shipping a conlang) is the novel contribution.

## 1. The LIDAR Analogy (Load-Bearing)

This project mirrors the pipeline of turning a messy LIDAR scan into a watertight quad mesh. The analogy dictates stage ordering and failure modes:

- Raw scan = SAE feature set (dense, noisy, redundant, with artifacts).
- Denoising = filtering low-confidence and non-semantic features.
- Downsampling = merging feature-splits via cosine deduplication.
- Normal estimation / surface reconstruction = building the relational multigraph.
- Mesh cleanup = pruning orphans, fixing broken edges, sanity-checking topology.
- Quad remeshing = imposing a regular relational schema (canonical relations).
- UV mapping / texturing = phonology and naming.

**Critical rule:** Do not texture an unclosed mesh. Stages 1–5 must produce something with the shape of a language before any phonology is assigned. If topology comes out ugly, adjust extraction — don't paper over with naming.

## 2. Core Design Commitments

| # | Decision | Why |
|---|----------|-----|
| 1 | **Source model: Gemma 2 + Gemma Scope SAEs** | Open weights, open SAEs at multiple layers/widths, Neuronpedia auto-interp labels. Also: Tegmark group's "Geometry of Concepts" used Gemma-2-2b SAEs — we can directly build on their findings. |
| 2 | **Scope of v1: topology-first, full conlang as eventual deliverable** | Prove the mesh is closed before texturing. |
| 3 | **Final artifact: full documentation site** | Browsable conlang plus methodology writeup. |
| 4 | **Relations: multigraph with multiple edge types** | Cosine for dedup, co-activation for fields, attribution for hierarchy. **New (v0.2):** function vectors (parallelogram crystal directions) for semantic transformations. |
| 5 | **Scan one middle layer; design data model to admit more layers later** | Corroborated by Tegmark group's "galaxy-scale" finding that anisotropy is steepest in middle layers. |
| 6 | **Unit of analysis: features with high-confidence Neuronpedia auto-interp labels** | Smaller, cleaner starting set. |
| 7 | **Antonymy is compositional, not primitive** | SAE features don't obviously encode antonymy. **Nuance (v0.2):** Tegmark group's parallelogram crystals show *some* transformations are encoded as directions — possibly including negation. If so, we may have more options than v0.1 implied. Revisit during Stage 4. |
| 8 | **Filter out culturally/temporally specific features** | Want something that feels like a human language. Filter rubric in §6. |
| 9 | **Deliverable scope: full doc site** | Lexicon browser + grammar + examples + origin story + methodology writeup. |
| 10 | **Local Python project; static site built at the end** | No web app during exploration. |
| 11 | **Phonology: inspired by a real language family (Polynesian or Bantu, TBD)** | Borrowed inventory gives sound consistency for free. |
| 12 | **Language naming: deferred until after v1 pipeline runs** | Name it for what it turns out to be. |

## 3. Out of Scope (for now)

- Multi-model comparison (though cross-model feature universality work suggests it may be valuable for v2).
- Training our own SAEs (use Gemma Scope pretrained).
- Multi-layer / cross-layer feature tracking.
- Hierarchical SAE training (HSAE, Tree SAE, etc. — interesting v2 direction but uses different methodology).
- Steering experiments / using the conlang to manipulate the model.
- Translation tooling between the conlang and natural languages.
- Phonology before lexicon stabilizes.
- A web app for exploration. The final site is static.

## 4. Pipeline Stages

Six stages. Stages 1–2 are the vertical slice (§5). Stages 3–6 sketched; details solidify as earlier stages produce real data.

### Stage 1: Ingestion

- Pull Gemma Scope SAE for chosen middle layer (specific layer TBD — see §9; Tegmark group used the 2B residual stream SAEs).
- Pull Neuronpedia auto-interp labels and confidence scores.
- Filter to features above a confidence threshold (start: top 200 for vertical slice; scale up later).
- Output: list of feature IDs with labels, confidence scores, decoder vectors.

### Stage 2: Deduplication

- Compute pairwise cosine similarity on decoder vectors.
- Cluster near-duplicates (threshold TBD — inspect distribution first). HDBSCAN is a known-working choice from prior LessWrong work on Gemma-2B feature families.
- Pick a representative per cluster.
- Output: deduped node set.
- **Vertical slice ends here.** See §5.

### Stage 3: Edge Construction

Three (now four) edge types:

- **Cosine edges** (cheap): already from Stage 2; semantic-neighborhood edges.
- **Co-activation edges** (medium cost): corpus through Gemma, log SAE activations, pairwise co-activation. Threshold and add.
- **Attribution edges** (expensive, best causal signal): **use Anthropic's open-source circuit-tracer** (https://github.com/safety-research/circuit-tracer) rather than rolling our own. Their demo notebook already covers multilingual analysis on Gemma-2-2b. Neuronpedia also hosts a no-install graph generator.
- **Function-vector edges** (NEW in v0.2): per Tegmark group's finding, compute pairwise difference vectors between SAE features and cluster them (K-means after LDA distractor projection). Each resulting cluster corresponds to a *semantic transformation* — e.g., "gender-flip," "royal-status," "negation." These transformations become a separate edge type ("X is the [gender-flip] of Y") that may serve as primitive relations in Stage 5.

Output: multigraph with up to four edge types per node pair.

### Stage 4: Topology Inspection

**Human-driven stage. Claude Code produces visualizations; project owner judges.**

- Visualize the graph. Force-directed layout for relational structure; UMAP of feature vectors as a second view; Ball Mapper or HDBSCAN cluster overlay as a third.
- Compare against prior visualizations: Google PAIR's "Mapping LLMs with Sparse Autoencoders" explorable provides a pre-built clustered map of all 16,384 Gemma Scope features at one layer. ConceptViz (open source) offers a six-view system for the same SAEs.
- Look for: clean semantic fields (clusters), hierarchy emerging from co-activation and attribution, parallelogram crystals after LDA distractor projection, orphans, holes.
- **Decision gate (revised v0.2):** does this substrate support a *lexicon-shaped* artifact? Prior work has already established that meaningful structure exists; the question is whether the structure has the *resolution and coverage* needed for a language design. Specifically:
  - Are there enough distinct semantic fields to be a real lexicon (target: 500–5,000 nodes after dedup)?
  - Do the parallelogram crystals cover a substantial fraction of semantically-meaningful node pairs, or are they sparse curiosities?
  - Does the co-activation hierarchy yield a coherent parent/sibling structure for most nodes?
- If yes → Stage 5. If no → adjust filters/thresholds/layer.

### Stage 5: Regularization (Quad Remeshing)

**Substantially revised in v0.2.**

Define a canonical relational schema. Tentative starting set, in priority order:

- **transformation** (NEW): for nodes participating in a parallelogram crystal, encode the function vector as a typed relation. Example: if there's a crystal {man, woman, king, queen}, the crystal contributes both `gender_flip(man) = woman, gender_flip(king) = queen` and `royalty(man) = king, royalty(woman) = queen`. These named transformations are the strongest candidate for *primitive* (rather than compositional) relations in our grammar.
- **parent** (most-upstream attribution edge per node): if circuit-tracer attribution yields a clean DAG.
- **siblings** (co-activation cluster members): semantic-field membership.
- **near** (cosine neighbors not already parent/sibling): residual similarity.
- **opposite**: handled compositionally per commitment 7 by default, but if a "negation" transformation crystal emerges cleanly in Stage 4, it may become a primitive relation instead. **This is a v0.2 hedge** — the original v0.1 plan was firm on compositional negation; we now keep it as default but allow data to override.

Resolve conflicts (e.g., a node with two equally-strong parents needs a rule).

Output: regularized graph where every node has a defined role in the schema.

**Implication for grammar (Stage 6):** if transformations are primitive, the morphology becomes richer than originally planned. Esperanto's `mal-` (compositional negation) was the v0.1 reference point; if v0.2 allows multiple primitive transformations, we lean toward a system with a small inventory of transformation affixes (gender, evaluative, scale, etc.) — closer to a language with productive derivational morphology than to Esperanto's single-prefix antonymy.

### Stage 6: Phonology and Site Build

- Choose phonology source family (Polynesian or Bantu — decide based on lexicon size and target word-length distribution).
- Define phoneme inventory.
- Define morphological affixes — must support productive negation/inversion (commitment 7) and, if Stage 5 adopts primitive transformations, the full set of those.
- Assign forms to nodes. Rules TBD; likely a mix of: phonetic similarity reflects semantic similarity (cosine-near nodes share phonemes), morphological parents share roots with children, transformation crystals share affixes.
- Build the static site:
  - Origin story (written last).
  - Methodology writeup (this spec + prior-work, narrativized).
  - Grammar sketch.
  - Lexicon browser (searchable, with relationships visualized per entry).
  - Example texts (canonical short passages or original).

## 5. Vertical Slice (Milestone 1)

Goal: Run Stages 1–2 end-to-end on a tiny subset. Produce a navigable visualization of ~200 features and their cosine relationships. Decide whether the shape is promising before investing further.

Tasks for Claude Code:

1. Set up the Python project per §8.
2. Write loader for Gemma Scope SAE (via `sae_lens`).
3. Write Neuronpedia client for labels and confidence scores.
4. Filter to top 200 features by confidence.
5. Compute pairwise cosine on decoder vectors.
6. Produce a graph visualization (networkx + pyvis, or plotly).
7. Save intermediate data to disk in a documented format.

Done criteria:

- Project owner can open the visualization, see ~200 nodes with cosine edges, hover/click for labels, and form an opinion.
- All intermediate artifacts saved to disk.

Out of scope for the slice: co-activation, attribution, function vectors, phonology, the site, anything past visualization.

## 6. The Filter Rubric (Commitment 8)

Write down before applying; refine against real data, not in advance.

**Starting rubric:**

Exclude: proper nouns; post-1900 cultural references; programming-language syntax and code tokens; URLs, emails, formatting artifacts; tokenizer/positional features; features whose Neuronpedia label is too vague to commit to ("various words").

Keep: abstract concepts (justice, change, similarity); perceptual categories (color, texture, motion); emotional states; social relations and roles; physical actions and processes; grammatical and discourse functions (negation, hedging, contrast); natural-world categories (animals, plants, weather, geography) even when specific, as long as not culturally narrow.

Decision rule for ambiguous cases: if a competent human linguist sketching basic vocabulary for a new language would plausibly include this concept, keep it.

Will need revision after Stage 1 runs. Update this section as decisions get made.

## 7. Risk Register

- **Attribution graphs may be tangled in middle layers.** Mitigation: Stage 4 decision gate; fallback to co-activation hierarchical clustering. Anthropic's circuit-tracer demos suggest this is tractable.
- **High-confidence Neuronpedia features may be biased toward easy-to-label concepts.** Mitigation: accepted for v1; document as a known limitation.
- **Feature-splitting may not cluster cleanly.** Mitigation: inspect cosine distribution in Stage 2 before picking a threshold; HDBSCAN (LessWrong prior art) is a known-working alternative to fixed-threshold clustering.
- **Lexicon size wrong** (too small to feel like a language, too large to texture). Mitigation: tunable confidence threshold; target 500–5,000 nodes after dedup.
- **No clean antonyms.** Mitigated by commitment 7 (compositional) by default; may be promoted to primitive if a clean negation crystal emerges.
- **Aesthetic drift during filtering.** Mitigation: write the rubric (§6) before filtering; log every filter decision; review logs for bias.
- **NEW (v0.2): Parallelogram crystal structure may be too sparse to be load-bearing for Stage 5.** Tegmark group's initial search found "mostly noise" before LDA distractor projection. Even after projection, only a fraction of pairwise differences cluster into recognizable transformations. Mitigation: in Stage 4, explicitly measure the *coverage* of crystal structure — what fraction of high-confidence nodes participate in at least one crystal. If <30%, fall back to v0.1's plan where transformations are not primitive.
- **NEW (v0.2): LDA distractor projection introduces methodological complexity.** Crystal-finding requires identifying and projecting out "distractor" directions like word length. This is a hyperparameter-laden step. Mitigation: start by replicating the Tegmark group's published procedure on the same model; only deviate after replication works.

## 8. Project Structure (Suggested)

```
conlang-from-internals/
├── spec.md                    # this document
├── prior-work.md              # companion landscape doc
├── pyproject.toml
├── README.md
├── src/
│   └── conlang/
│       ├── __init__.py
│       ├── ingest.py          # Stage 1
│       ├── dedupe.py          # Stage 2
│       ├── edges/
│       │   ├── cosine.py
│       │   ├── coactivation.py
│       │   ├── attribution.py # wraps circuit-tracer
│       │   └── crystals.py    # NEW: parallelogram function vectors
│       ├── topology.py        # Stage 4 helpers
│       ├── regularize.py      # Stage 5
│       ├── phonology.py       # Stage 6
│       └── site.py            # Stage 6 site build
├── data/
│   ├── raw/                   # SAE handles, Neuronpedia caches
│   ├── interim/               # deduped nodes, edge sets
│   └── processed/             # final lexicon, regularized graph
├── notebooks/                 # exploration, topology inspection
├── site/                      # final static site output
└── tests/
```

## 9. Open Questions (To Resolve Before Stage 3)

- Which specific Gemma 2 model size? (2B is fast and well-studied — Tegmark group used it. 9B has richer features. Recommend 2B for v1.)
- Which specific middle layer? (Tegmark group's published experiments used 2B residual stream; check which layer they focused on and start there.)
- Confidence threshold for Neuronpedia filtering — pick after seeing the distribution.
- Cosine threshold for deduplication — pick after seeing the distribution.
- What corpus for co-activation in Stage 3? (FLORES-200 for multilingual; Wikipedia for monolingual breadth; Project Gutenberg for clean prose.)
- Crystal-detection method: replicate Tegmark group's K-means-after-LDA, or experiment with alternatives?
- Crystal coverage threshold for promoting transformations to primitive relations in Stage 5: tentatively 30% of nodes participating in at least one crystal. Adjust after Stage 4 inspection.

## 10. Versioning This Spec

Update this file as decisions get made. Specifically:

- §6 (filter rubric) will be refined after Stage 1.
- §4 Stage 3 details solidify after running Anthropic's circuit-tracer demo.
- §4 Stage 4 will get a results section after the first inspection.
- §4 Stage 5 details (which transformations get primitive status) will be decided based on crystal coverage in Stage 4.
- §5 will be replaced with a new "Milestone 2" section once the vertical slice lands.
- §2 commitments should only change with deliberation — they're the project's north star.

## 11. Repos to Study Before Writing Code

These are listed in `prior-work.md` but called out here because they directly inform implementation:

- **feature-geometry** (Tegmark group, MIT) — https://github.com/ejmichaud/feature-geometry — read this *before* implementing Stage 3's function-vector edges or Stage 4's crystal-coverage analysis. Their parallelogram-detection code is the starting point.
- **ConceptViz** — https://github.com/Happy-Hippo209/ConceptViz — six-view visual analytics system for Gemma Scope on Gemma-2-2b. May save substantial work in Stage 4 visualization, or at least inform the design.
- **Anthropic circuit-tracer** — https://github.com/safety-research/circuit-tracer — Stage 3 attribution. Run the demo notebook before scoping our own attribution work.
- **SAE Lens** — https://github.com/jbloomAus/SAELens — the standard loader. Stage 1 dependency.
- **Google PAIR explorable** — https://pair.withgoogle.com/explorables/sae/ — already has a hierarchically clustered, LLM-labeled map of all 16,384 Gemma Scope features at one layer. Check if backing data is accessible; if so, it's a head start on Stage 2.

Reading these first is not optional. The original v0.1 plan implicitly assumed we'd be building these tools; v0.2 assumes we'll be building *on top of* them.
