# Methodology

The lexicon was extracted in six stages, each a small CLI in the
`conlang` package. Each stage's intermediate output lives under
`data/interim/` or `data/processed/` and can be inspected independently.

## Stage 1 — Ingest

`python -m conlang.slice` (orchestrates Stages 1–4)

Loads the Gemma Scope SAE
(`gemma-scope-2b-pt-res-canonical / layer_12/width_16k/canonical`) and the
bulk Neuronpedia explanation dump for the matching feature source
(`gemma-2-2b / 12-gemmascope-res-16k`, 16,395 explanations). Filters per
spec §6 (Neuronpedia description exists; non-boilerplate; defined decoder
direction) and takes the first 1000 that pass. The slice is deterministic
given the dump.

Output: `data/raw/features.jsonl` (1000 `{feature_id, label}` records) and
`data/raw/decoder_vecs.npy` (1000 × 2304, float32).

## Stage 2 — Dedupe + cluster

Cosine similarity over decoder vectors gives a 1000 × 1000 matrix.
Distribution: median ≈ 0, 99th percentile ≈ 0.085. Most SAE decoder
vectors are near-orthogonal — that's the SAE doing its job.

Dedup is HDBSCAN over the cosine distance matrix
(`min_cluster_size=5`). 838 of the 1000 features end up as noise
singletons; seven multi-member clusters emerge, three of which read as
coherent semantic groups (discourse markers, news content, physical
attributes). The HDBSCAN labels feed both the regularization step and
the phonosemantic CV1 assignment.

## Stage 3 — Edges

Three edge types were attempted; two ship:

- **Cosine edges** (always available, from Stage 2). Threshold for viz:
  0.10. 1375 edges at this threshold on the 1000-feature slice.
- **Co-activation edges** (`python -m conlang.run_coactivation
  --use-flores --n-per-lang 1000`). For each pair of slice features,
  count how often they fire above an activation threshold on the same
  token. Normalize to PMI. The corpus is FLORES-200 dev across six
  languages (eng, fra, deu, spa, zho, jpn); ~5,982 sentences, ~189,824
  tokens through the residual stream of layer 12. 86 edges with PMI ≥ 5
  on the slice; 28,250 at PMI ≥ 3.
- **Function-vector crystal edges** — *attempted, dropped.* Per Tegmark
  et al.'s observation that activation differences for labeled pairs
  encode named transformations, we replicated the parallelogram-crystal
  result on Gemma 2 2B layer 12 raw hidden states (silhouette 0.53 on 12
  relations; antonymy separates cleanly as a 13th class). The bridge
  from those crystals to Gemma Scope SAE decoder vectors failed the §7
  distinctiveness-margin gate (0% of slice nodes at margin ≥ 0.05 on
  either the 12- or 13-relation set). Per Commitment 7, the
  `transformation` primitive does not exist in the regularized graph;
  negation is handled compositionally by the morphology instead.

## Stage 4 — Decision gate

Three load-bearing questions, asked of the slice before continuing:

| Question                                                          | Result            |
| ----------------------------------------------------------------- | ----------------- |
| Enough distinct semantic fields (target 500–5000)?                | Yes — ~990 nodes  |
| Crystals cover ≥ 30% of nodes at margin ≥ 0.05?                   | No — 0%           |
| Co-activation yields parent/sibling structure?                    | Yes               |

Two of three green. The remaining red is exactly the failure mode spec
§7 anticipated, so the spec's mitigation (compositional negation) is in
force.

## Stage 5 — Regularize

`python -m conlang.regularize`

Each node is assigned a `parent` (its highest-PMI co-activation
neighbor), a `siblings` list (its other HDBSCAN cluster members), and a
`near` list (its top cosine neighbors not already covered). 981 of the
1000 nodes have a parent after the FLORES-scale co-activation pass; 162
have at least one sibling. The output is
`data/processed/regularized.json` — the input that the lexicon stage
consumes.

The schema deliberately has no `transformation` field. See Commitment 7.

## Stage 6 — Phonology, lexicon, site

Three small modules:

- `conlang.phonology` — the inventory + the affix table. Pure data and
  composition.
- `conlang.lexicon` — keyword class assignment (regex over the
  Neuronpedia label) + phonosemantic stem generation + applies the class
  prefix and negation per node. Output:
  `data/processed/lexicon.json` (1000 entries).
- `conlang.site` — renders the lexicon to a single self-contained HTML
  file (`data/processed/lexicon.html`).

The keyword class heuristic is intentionally biased toward five active
singular classes (1 human, 5 natural, 7 tool/process, 9 animal/language,
11 abstract). The plural classes are unused in this slice because the
Neuronpedia labels don't carry plural markers in a way that's reliable
to detect. The resulting class distribution is:

```
1=61   5=8   7=220   9=6   11=705
```

That heavy weight on class 11 is honest: most of what an SAE feature
*is* is an abstract concept-detector.

## Reproducibility

The shipped CLIs are deterministic given the same SAE checkpoint and the
same Neuronpedia dump. The FLORES download is from the FAIR-hosted
tarball
(`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`); the
older HF `facebook/flores` dataset relies on a loading script that is no
longer supported in current `datasets`. The HF mirror
`Muennighoff/flores200` has the same problem.

To rebuild from scratch:

```bash
source .venv/bin/activate
python -m conlang.slice \
    --sae-release gemma-scope-2b-pt-res-canonical \
    --sae-id layer_12/width_16k/canonical \
    --neuronpedia-model gemma-2-2b \
    --neuronpedia-source 12-gemmascope-res-16k \
    --top-n 1000 --dedup-method hdbscan
python -m conlang.run_coactivation --use-flores --n-per-lang 1000
python -m conlang.regularize
python -m conlang.lexicon
python -m conlang.site
```

Times on an RTX-4090-class GPU: stages 1–2 finish in under a minute;
co-activation on FLORES takes a few minutes; the rest are seconds.

## Supplementary: circuit-tracer attribution graphs

We considered using Anthropic's
[circuit-tracer](https://github.com/safety-research/circuit-tracer) to
build attribution edges into the Stage 3 multigraph. We did not. The
reason is that circuit-tracer operates on per-layer MLP **transcoders**
(`mntss/gemma-scope-transcoders`), which are a different feature space
from the residual-stream SAE
(`gemma-scope-2b-pt-res-canonical layer_12/width_16k`) that the lexicon
is built on. Cross-space alignment would require its own §7-style
distinctiveness gate, and we chose not to take on that research surface
at this time.

What we did do is run circuit-tracer on three canonical prompts as
**parallel supplementary material**: attribution graphs that illustrate
the kind of structure that exists *in* Gemma 2 (2B) without claiming
that structure is identical to what the SAE-based lexicon captures.

| Prompt                                             | Nodes | Links  | Download                                                |
| -------------------------------------------------- | ----: | -----: | ------------------------------------------------------- |
| `The capital of state containing Dallas is`        |   417 |  6,565 | [dallas-texas.json](static/circuit_tracer/dallas-texas.json) |
| `Two plus two is`                                  |   333 |  5,906 | [english-math.json](static/circuit_tracer/english-math.json) |
| `Le chat est assis sur le`                         |   546 | 17,432 | [french-cat.json](static/circuit_tracer/french-cat.json)     |

Each graph is a circuit-tracer attribution graph at `node_threshold=0.6`,
`edge_threshold=0.85`, built on the `mntss/gemma-scope-transcoders`
per-layer transcoders. Load them into circuit-tracer's local viewer
(`circuit-tracer serve --data_dir <dir>`) or the Neuronpedia
attribution-graph viewer to interact.

The driver is `external/run_circuit_tracer.py`. First run downloads ~3 GB
of transcoder weights into the HF cache; subsequent runs are seconds plus
attribution time (~30 s per prompt on a 24 GB GPU).

## Reproducible walkthrough

→ [Walkthrough notebook (executed)](static/walkthrough.html)

A single Jupyter notebook (`notebooks/walkthrough.ipynb`) loads every
stage's on-disk artifacts and renders the same summary numbers reported
here. The HTML export above is a frozen snapshot. To re-execute against
fresh artifacts:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb
jupyter nbconvert --to html --output-dir docs/static notebooks/walkthrough.ipynb
```
