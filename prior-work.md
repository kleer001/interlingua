# Prior Work and Source Pointers

**Companion to `spec.md`.**

**Editorial rule (v0.3):** every entry below points to runnable code, downloadable data, or an interactive resource. Verified live. Papers without an associated code/data drop have been moved to a "Background reading" section at the end with bare arXiv links and no commentary — they're context, not tooling.

## TL;DR — Where this project sits in the landscape

The substrate is mature. Multilingual concept space in middle layers and SAE features encoding concepts are both well-established (Wendler 2024, Anthropic 2025, Tegmark 2024). The technical core of our pipeline (Stages 1–4) overlaps heavily with existing tools. The novel parts are the *interpretive frame* (treating topology as lexicon), Stage 5's regularization into a fixed relational schema, Stage 6's phonology, and the conlang artifact.

The **single most consequential reference** for our project is Li/Tegmark et al.'s "Geometry of Concepts" (§1 below) — they ran exactly the topology analysis we're planning, on Gemma-2-2b SAEs, found parallelogram crystals encoding semantic relations like king:woman::queen:woman, and released the notebooks. If only one link gets read before writing code, it's that repo.

---

## 1. SAE Feature Topology — Direct Code for Stages 1–4

### Li, Michaud, Baek, Engels, Sun, Tegmark — "The Geometry of Concepts: Sparse Autoencoder Feature Structure" (MIT, 2024)

Studied Gemma-2-2b SAE features at three scales: atomic "crystals" (parallelograms encoding semantic transformations after LDA distractor projection), brain-scale "lobes" (math/code spatial modularity), galaxy-scale anisotropy (steepest in middle layers). Published as MDPI Entropy 27/4. **This is the work to build on.**

- Paper (arXiv): https://arxiv.org/abs/2410.19750
- Paper (MDPI/Entropy, open access): https://www.mdpi.com/1099-4300/27/4/344
- **Code:** https://github.com/ejmichaud/feature-geometry (Jupyter notebooks, 99% Python)
- Three subdirectories matching the paper's three scales:
  - https://github.com/ejmichaud/feature-geometry/tree/main/atom (crystal/parallelogram search)
  - https://github.com/ejmichaud/feature-geometry/tree/main/brain (modularity analysis)
  - https://github.com/ejmichaud/feature-geometry/tree/main/galaxy (eigenvalue / point-cloud structure)

### ConceptViz — Li, Wen, Jiang et al. (IEEE VIS 2025)

Six-view visual analytics system for exploring SAE features. Identification → Interpretation → Validation pipeline. Uses Gemma Scope SAEs on Gemma-2-2b residual stream across all 26 layers (16,384 features). Open source with documented tutorial.

- Paper (arXiv): https://arxiv.org/abs/2509.20376
- **Main repo:** https://github.com/Happy-Hippo209/ConceptViz
- Backend (Python): https://github.com/Happy-Hippo209/ConceptViz-backend
- Frontend (Next.js): https://github.com/Happy-Hippo209/ConceptViz-frontend
- **Tutorial site with screenshots and videos:** https://happy-hippo209.github.io/ConceptViz/

### "HDBSCAN is Surprisingly Effective at Finding Interpretable Clusters of the SAE Decoder Matrix" (Lim, Tantia, Sinem — LessWrong, Oct 2024)

Clustered SAE decoder vectors on GPT-2 Small (jbloom-resid-pre SAEs) and Gemma-2-2b (Gemma Scope 16k) using HDBSCAN on cosine similarity, visualized via UMAP. Found interpretable feature families: cardinal directions, polysemes of "side," days of week, cooking, etc. >90% of features get classified as "noise" but the remaining clusters are highly coherent. Direct method ancestor for our Stage 2.

- **Post (full method + results, no separate repo):** https://www.lesswrong.com/posts/Dc2w5kHXksSBcjNTs/hdbscan-is-surprisingly-effective-at-finding-interpretable
- The authors promised to open-source code "soon" in Oct 2024; not yet released as far as I can find. The post itself contains enough detail to reimplement.

### Google PAIR — "Mapping LLMs with Sparse Autoencoders" (Hussein, Raval, Reif et al., October 2025)

Interactive explorable that maps all 16,384 features at one layer of Gemma Scope, with hierarchical clustering and LLM-generated cluster labels. You can hover over features to see labels and search.

- **Interactive page:** https://pair.withgoogle.com/explorables/sae/
- Note: the underlying clustered data is not (publicly) downloadable from the page — it's embedded in the visualization. Useful as a reference visual when judging our own Stage 4 output.

---

## 2. The "Latent Language" Thread — Foundational Substrate

### Wendler, Veselovsky, Monea, West — "Do Llamas Work in English? On the Latent Language of Multilingual Transformers" (EPFL, 2024)

Layer-by-layer tracking of Llama-2 embeddings. Three phases: early → concept space (English-leaning) → output language.

- Paper (ACL): https://aclanthology.org/2024.acl-long.820/
- Paper (arXiv): https://arxiv.org/abs/2402.10588
- **Code:** https://github.com/epfl-dlab/llm-latent-language
  - Translation experiments notebook: https://github.com/epfl-dlab/llm-latent-language/blob/main/Translation.ipynb
  - Cloze experiments notebook: https://github.com/epfl-dlab/llm-latent-language/blob/main/Cloze.ipynb
  - Llama wrapper: https://github.com/epfl-dlab/llm-latent-language/blob/main/llamawrapper.py
- **Precomputed latents on HuggingFace:** https://huggingface.co/datasets/wendlerc/llm-latent-language (Llama-2 7B/13B/70B)

### Anthropic — "On the Biology of a Large Language Model" + "Circuit Tracing" (Lindsey, Ameisen et al., March 2025)

Used SAE features and attribution graphs on Claude 3.5 Haiku. The multilingual section is the directly relevant case study for us.

- Biology paper (main): https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- **Multilingual circuits section (direct anchor):** https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-multilingual
- Methods paper (companion): https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Blog overview: https://www.anthropic.com/research/tracing-thoughts-language-model
- Interactive attribution graph viewer (multilingual antonym example): linked from the biology paper's multilingual section

### Anthropic's Open-Source Circuit Tracer (Hanna, Piotrowski, Lindsey, Ameisen — May 2025)

For Stage 3 attribution. Multilingual representations in Gemma-2-2b is one of the published demos.

- Blog announcement: https://www.anthropic.com/research/open-source-circuit-tracing
- **Library:** https://github.com/safety-research/circuit-tracer (redirects to decoderesearch fork; both work)
- **Main tutorial notebook:** https://github.com/safety-research/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb
- Gemma-2-2b demo notebook: https://github.com/safety-research/circuit-tracer/blob/main/demos/gemma_demo.ipynb
- Attribution demo: https://github.com/safety-research/circuit-tracer/blob/main/demos/attribute_demo.ipynb
- Intervention demo: https://github.com/safety-research/circuit-tracer/blob/main/demos/intervention_demo.ipynb
- **Hosted graph generator (no install):** https://www.neuronpedia.org/gemma-2-2b/graph
- Frontend code: https://github.com/anthropics/attribution-graphs-frontend

---

## 3. Multilingual SAE Features — Code Releases

### SAE-LAPE — Andrylie et al. (July 2025) — "Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages"

Method based on feature activation probability for identifying language-specific features. Useful inverse of our project — gives us a way to *filter out* language-specific features and keep language-agnostic ones.

- Paper: https://arxiv.org/abs/2507.11230
- **Code:** https://github.com/LyzanderAndrylie/language-specific-features
- **Interactive visualizations (identified features in Llama 3.2 1B):** https://lyzanderandrylie.github.io/language-specific-features/

### Deng et al. (May 2025) — "Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders"

Independent contemporaneous work with a monolinguality metric for SAE features.

- Paper: https://arxiv.org/abs/2505.05111
- **Code:** https://github.com/Aatrox103/multilingual-llm-features

---

## 4. Data and Tooling — Stage Dependencies

### Gemma Scope (Gemma 2 SAEs) — Google DeepMind

Open JumpReLU SAEs on all layers/sub-layers of Gemma 2 2B and 9B (plus select layers of 27B), both base (`pt`) and instruction-tuned (`it`).

- Landing page: https://huggingface.co/google/gemma-scope
- Technical report (arXiv): https://arxiv.org/abs/2408.05147
- **2B residual stream SAEs (likely starting point):** https://huggingface.co/google/gemma-scope-2b-pt-res
- **Specific canonical SAE example** (layer 20, 16k width, L0=71): https://huggingface.co/google/gemma-scope-2b-pt-res/tree/main/layer_20/width_16k/average_l0_71
- 9B residual stream SAEs: https://huggingface.co/google/gemma-scope-9b-pt-res
- 2B MLP SAEs: https://huggingface.co/google/gemma-scope-2b-pt-mlp
- 2B attention SAEs: https://huggingface.co/google/gemma-scope-2b-pt-att
- Interactive demo (Neuronpedia): https://www.neuronpedia.org/gemma-scope
- Minimal loader code (Python, from the model card):

  ```python
  from sae_lens import SAE  # pip install sae-lens
  sae, cfg_dict, sparsity = SAE.from_pretrained(
      release="gemma-scope-2b-pt-res-canonical",
      sae_id="layer_20/width_16k/canonical",
  )
  ```

### Gemma Scope 2 (Gemma 3 SAEs and transcoders) — Google DeepMind

Newer release for Gemma 3. SAEs at 25%/50%/65%/85% depth across 270M/1B/4B/12B/27B. Cross-layer transcoders for 270M and 1B (which would matter if we ever do multi-layer in v2).

- Landing page: https://huggingface.co/google/gemma-scope-2
- 27B IT SAEs and transcoders example: https://huggingface.co/google/gemma-scope-2-27b-it

### Neuronpedia — auto-interp labels and feature browser

Where our Stage 1 labels come from. Free with rate limits; some endpoints need an API key.

- Main site: https://neuronpedia.org
- Gemma Scope landing: https://www.neuronpedia.org/gemma-scope
- Docs: https://docs.neuronpedia.org
- Python client (PyPI): https://pypi.org/project/neuronpedia/
- Platform source code: https://github.com/hijohnnylin/neuronpedia
- Feature URL pattern: `https://neuronpedia.org/{model}/{layer-source}/{index}`
- JSON API pattern: `https://www.neuronpedia.org/api/feature/{model}/{layer-source}/{index}`
- Example feature page: https://neuronpedia.org/gpt2-small/6-res_scefr-ajt/650
- Example JSON API call: https://www.neuronpedia.org/api/feature/gpt2-small/6-res_scefr-ajt/650

### SAE Lens — standard SAE loader

- **GitHub:** https://github.com/jbloomAus/SAELens
- Docs: https://jbloomaus.github.io/SAELens/
- Tutorial — loading and analyzing pretrained SAEs: linked from the GitHub README
- Training tutorial notebook: https://github.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb

### FLORES-200 — multilingual parallel corpus (for Stage 3 co-activation, if used)

- GitHub: https://github.com/facebookresearch/flores/tree/main/flores200
- Hugging Face: https://huggingface.co/datasets/facebook/flores
- Newer FLORES versions (Open Language Data Initiative): https://github.com/openlanguagedata/flores and https://www.oldi.org

---

## 5. NSM — Hand-Built Prior Art

Theory only; no code. Useful as a benchmark — if our extracted lexicon doesn't include rough analogs of WANT, KNOW, GOOD, BAD, etc., something is off.

- Wikipedia overview: https://en.wikipedia.org/wiki/Natural_semantic_metalanguage
- Griffith University "What is NSM" with the full 65-prime table: https://intranet.secure.griffith.edu.au/schools-departments/natural-semantic-metalanguage/what-is-nsm
- Griffith downloads page (paper PDFs, prime charts): https://intranet.secure.griffith.edu.au/schools-departments/natural-semantic-metalanguage/downloads
- Research portal (1,100+ publications, searchable by explication): https://nsm-approach.net/

**Note:** The Griffith URLs are on `intranet.secure.griffith.edu.au` and appear public to search engines, but may eventually require auth or move. The Wikipedia article reproduces the prime list.

---

## 6. LLMs Doing Conlang Work — Top-Down (Inverse of Our Direction)

### ConlangCrafter (Alper, Yanuka, Girves, Beguš — August 2025)

Top-down LLM pipeline for conlang generation: phonology → morphology → syntax → lexicon → translation. Not what we're doing, but their phonology/grammar generation may save Stage 6 work.

- Paper: https://arxiv.org/abs/2508.06094
- **Project page with examples:** https://conlangcrafter.github.io

(I did not find a code repo for ConlangCrafter; the project page is the public artifact.)

---

## 7. Background Reading (No Code/Data)

These are papers worth knowing about but lack actionable releases. Bare links, no commentary.

- "What Language(s) Does Aya-23 Think In?": https://arxiv.org/abs/2507.20279
- "High-Dimensional Interlingual Representations of LLMs": https://arxiv.org/abs/2503.11280
- "Beyond English-Centric LLMs: What Language Do Multilingual LMs Think in?": https://arxiv.org/abs/2408.10811
- "Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders" (Harrasse et al.): https://arxiv.org/abs/2511.10840
- Anthropic "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Anthropic "Towards Monosemanticity": https://transformer-circuits.pub/2023/monosemantic-features/
- "Towards Universal Semantics with Large Language Models" (DeepNSM): https://arxiv.org/abs/2505.11764
- "Sparse Autoencoders Reveal Universal Feature Spaces Across LLMs": https://arxiv.org/abs/2410.06981
- "Visual Exploration of Feature Relationships in SAEs with Curated Concepts" (Ball Mapper): https://arxiv.org/abs/2511.06048
- "Navigating the Concept Space of Language Models" (Concept Explorer): https://arxiv.org/abs/2603.23524
- "SAE feature geometry is outside the superposition hypothesis": https://www.alignmentforum.org/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis
- "Topological Data Analysis and Mechanistic Interpretability": https://www.lesswrong.com/posts/6oF6pRr2FgjTmiHus/topological-data-analysis-and-mechanistic-interpretability
- Hierarchical SAE training papers (HSAE, Tree SAE, MetaSAE): https://arxiv.org/abs/2506.01197, https://arxiv.org/abs/2602.11881, https://arxiv.org/abs/2605.07922, https://arxiv.org/abs/2509.22033

---

## 8. Synthesis

Three claims the project depends on:

1. **Shared, language-agnostic concept space in middle layers.** Strong evidence (§2). Wendler, Anthropic biology paper, Harrasse.
2. **SAE features encode interpretable, often language-agnostic concepts.** Strong evidence (§3 background + Scaling Monosemanticity in §7).
3. **The relational topology is rich enough to support a lexicon.** Substantial structural evidence (§1, especially Tegmark) — parallelogram crystals encode semantic transformations. But no prior work has translated this into a fixed-arity grammar suitable for language design. *That's our contribution.*

What would tell us we're wrong: if the parallelogram crystals from the Tegmark group turn out to be sparse — covering only a small fraction of features — then our Stage 5 plan to use them as primitive relations falls apart. The Tegmark paper itself notes their initial search found "mostly noise" before LDA distractor projection. We should measure crystal coverage explicitly in Stage 4 before committing Stage 5 to it.

---

## 9. To Add as Project Progresses

- Specific Gemma Scope layer choice and reasoning.
- Crystal coverage statistics from our Stage 4 inspection.
- Whether Neuronpedia features turned out useful or junk in our domain filter.
- Any further prior work discovered during implementation.
- The eventual project's own writeup.
