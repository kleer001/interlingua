# interlingua

A bottom-up conlang whose lexicon is derived from sparse-autoencoder features
of Gemma 2 (2B). 1000 nodes, Bantu-shaped phonology, productive compositional
negation. Built to test a thesis: that *the units of meaning a transformer has
already learned* can be surfaced as a vocabulary.

The pipeline is documented end-to-end. The lexicon is reproducible from the
SAE checkpoint, the Neuronpedia explanation dump, and a small multilingual
corpus. Nothing in the surface forms was hand-picked.

## What's here

- [Origin](origin.md) — what this language is, in its own frame.
- [Grammar](grammar.md) — phonology, morphology, the eleven noun classes,
  and the single productive negation prefix.
- [Methodology](methodology.md) — how the lexicon was extracted, from
  Gemma Scope features to surface form.
- [Lexicon](lexicon.md) — the full 1000-entry table, sortable and searchable.

## Spec

Design constraints, decisions, and the failure-mode mitigations live in
`spec.md` (v0.2) and `prior-work.md` (v0.3). The single most load-bearing
commitment — *Commitment 7: no `transformation` primitive in the graph* —
is what makes the negation productive instead of opaque, and what keeps the
schema honest about what the model's geometry actually supports.
