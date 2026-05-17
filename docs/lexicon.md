# Lexicon

The full 1000-entry lexicon is a single self-contained HTML page.

→ [Open the lexicon (lexicon.html)](static/lexicon.html)

The table supports:

- **Filter** — type any substring to narrow by stem, surface form,
  negation, class name, or English label.
- **Sort** — click any column header. The Class and Feature columns
  sort numerically.

## Columns

- **Stem** — the bare 3-to-5-syllable phonosemantic stem, before any
  prefix. Sibling features (same HDBSCAN cluster) share the first
  syllable; PMI-related features share the second.
- **Surface** — the stem with its class prefix applied. This is the
  citation form, what you would write or say.
- **Negation** — the surface form with the productive `si-` negation
  prefix. The semantic complement.
- **Class** / **Class domain** — the noun class assigned by the keyword
  heuristic (see [Methodology](methodology.md)).
- **Label (Neuronpedia)** — the English description of what the
  underlying SAE feature detects. This is provenance, not a translation.
- **Feature** — the SAE feature id, for cross-referencing with
  Neuronpedia.

## Why the labels are imperfect

The English column comes from Neuronpedia's bulk explanation dump. These
explanations are LLM-generated descriptions of what each SAE feature
appears to fire on. They are useful as a rough handle, but a single
English phrase rarely captures what an SAE feature actually represents.
The conlang's surface forms are intentionally not glosses of these
labels — they are derived from the model's *geometry* (clusters and
co-activation), not from the labels themselves.
