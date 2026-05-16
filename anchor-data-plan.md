# Anchor Data Collection Plan

Companion to `anchor-pool-sketch.md`. The sketch is the conceptual structure; this is the data acquisition plan.

## How far can we go?

**Languages.** The realistic upper bound is set by PanLex: ~9,000 language varieties, free and open license, API and monthly dumps. We don't actually want all 9,000 — beyond ~100 typologically diverse languages, the marginal anchor adds nothing meaningful to tessellation density and increases curation cost linearly.

Target: **~100 languages**, weighted for typological coverage (one or two per major family branch) plus a few extras for languages with especially rich documented ideophone systems (Japanese, Korean, Bantu, Mon-Khmer, Quechua, Khoisan).

**Onomatopoeic concepts.** Far more than the ~20 animals in the sketch. The full inventory across categories:

| Category | Count | Examples |
|---|---|---|
| Mammals | ~12 | dog, cat, cow, pig, sheep, horse, lion, wolf, bear, mouse, monkey, donkey |
| Birds | ~8 | rooster, hen, duck, owl, crow, songbird, dove, eagle |
| Reptiles/amphibians | ~3 | snake, frog, lizard |
| Insects/arthropods | ~5 | bee, cricket, mosquito, fly, cicada |
| Marine | ~3 | whale, dolphin, seal |
| Water sounds | ~6 | drip, splash, gurgle, trickle, wave, pour |
| Fire/wind/weather | ~8 | crackle, howl, whistle, rustle, thunder, rain pitter-patter, hail |
| Hard impact | ~8 | bang, crash, thud, smash, slam, knock, snap, crack |
| Soft impact | ~5 | plop, splat, thump, pat, squish |
| Resonant | ~5 | clang, gong, ding, dong, ring |
| Mechanical/electronic | ~10 | tick, beep, buzz, whir, hum, click, clack, zip, ping, vroom |
| Human non-verbal | ~12 | laugh, cry, sneeze, cough, snore, hiccup, sigh, gasp, yawn, groan, scream, whistle |
| Movement | ~8 | whoosh, swoosh, zoom, swish, glug, slosh, flap, thunder (of hooves) |
| Texture/eating | ~6 | crunch, slurp, smack, gulp, scratch, scrape |
| Exclamation/affect | ~10 | ow, ouch, ah, oh, wow, ugh, eww, hmm, mm-hmm, huh |

**Total: ~110 distinct onomatopoeic concepts.**

At ~100 languages × ~110 concepts = ~11,000 potential anchor entries. Realistically, 50–70% will have documented onomatopoeia in any given language (mechanical and modern sounds have sparser coverage in smaller languages). Expect **5,000–7,000 actual anchor entries** after collection.

Each entry has multiple semantic attributes (the sketch's snake-cloud structure), so the effective number of points in the Voronoi tessellation is more like **30,000–60,000**. Plenty of density for any reasonable semantic embedding dimension.

## Why not all 9,000 PanLex languages?

We could. But beyond ~100 typologically diverse languages:

- Anchor sharpness saturates — adding language #200 changes the cross-linguistic mean by epsilon
- Curation cost is linear in language count
- The variance signal (how much languages disagree on a sound) is already well-sampled
- Many smaller languages have sparse onomatopoeia documentation, requiring direct elicitation we can't do

Better strategy: typological coverage > raw count. Sample one language per Glottolog major branch, supplement with ideophone-rich languages.

## Data sources

### Tier 1 — Aggregated, machine-readable

- **PanLex** — https://panlex.org. ~9,000 languages, ~25M lexemes. Free and open. API at https://dev.panlex.org. Monthly snapshots in CSV/JSON. Onomatopoeia entries scattered through dictionary sources but extractable. Project of The Long Now Foundation.
- **Wiktionary** — https://en.wiktionary.org. Cross-language onomatopoeia entries with IPA transcriptions. Scriptable via the Wiktionary API or dumps. The "Translations" section on individual onomatopoeia pages is a goldmine.
- **Wikipedia "Cross-linguistic onomatopoeias"** — https://en.wikipedia.org/wiki/Cross-linguistic_onomatopoeias. Curated seed list, ~50 languages × dozens of sounds.

### Tier 2 — Specific databases

- **CHIDEOD** (Chinese Ideophone Database) — Van Hoey & Thompson 2020. Open citation; database itself may require contacting authors.
- **Bzzzpeek** — http://bzzzpeek.com. Children's site with audio recordings of animal sounds across many languages. Useful for verification.
- **The Ideophone blog** — https://ideophone.org. Mark Dingemanse's research blog with examples from Siwu, Japanese, and many other languages.

### Tier 3 — Language-specific scholarship

- Japanese: Hamano (1986), Akita & Pardeshi (2019)
- Korean: Sohn (1999)
- Bantu: Doke (1935), Childs (1994), Dingemanse on Siwu
- Quechua: Nuckolls on Pastaza Quichua
- Various: scattered grammars in Mouton Grammar Library

### Tier 4 — Direct LLM elicitation (with caution)

For languages with sparse documented onomatopoeia, query a multilingual LLM. This is reasonable for major languages where the model has seen comparable data; risky for under-resourced languages where the model may hallucinate. Use only as supplement, flag all LLM-derived entries for manual review.

## Tooling

- **Epitran** — https://github.com/dmort27/epitran. Open-source orthography-to-IPA converter, supports ~100 languages out of the box. Maintained by David Mortensen at CMU.
- **PHOIBLE 2.0** — https://phoible.org. Open phonological feature database; each IPA segment maps to a vector of articulatory features (place, manner, voicing, etc.). CLDF format. Use for phonological feature extraction after Epitran conversion.
- **lingpy** — https://lingpy.org. Computational historical linguistics tools, includes IPA handling and phonological feature manipulation.

## Aggregation pipeline

For each `(concept, language)` pair:

1. **Lookup.** Query PanLex API + Wiktionary scraping + Wikipedia cross-language list. Take the union, mark sources.
2. **IPA normalization.** Some sources have IPA; others have native orthography. For orthography, run Epitran. Where Epitran is missing the language, fall back to hand-transcription or skip.
3. **Phonological feature extraction.** Each IPA phoneme → articulatory feature vector via PHOIBLE.
4. **Projection onto our inventory.** Each source phoneme → nearest neighbor in our 10C/5V grid by articulatory-feature distance.
5. **Attribute tagging.** For each anchor concept, get a list of semantic attributes. Start from a hand-curated hypothesis list (the snake example in the sketch); supplement and validate against the LLM's actual feature clustering once Stage 4 data is in.

Manual verification on 10–20% of entries — sampled by language, by category, and by edge cases (rare IPA segments, orthographic ambiguity).

## What we don't need

- **Audio recordings.** We're doing symbolic phonology, not acoustic synthesis. Bzzzpeek is useful for verification but not as primary data.
- **Usage frequency within each language.** We want the canonical form, not the usage profile.
- **Etymology.** Onomatopoeia is presumed motivated; etymological history is irrelevant.
- **Native speaker fluency confirmation.** We're using these as anchor points, not building a usable creole.
- **Exhaustive coverage in any one language.** Even partial coverage in 100 languages dominates exhaustive coverage in one.

## Open questions

- **Which 100 languages exactly.** Could be governed by: Glottolog typological-distance sampling, top speaker count, ideophone-richness weighting, or a mix. Probably typological sampling with ~20 high-ideophone supplements.
- **Dialect handling.** Mandarin vs Cantonese disagree on onomatopoeia; British vs American English diverge. Treat as separate entries with linked anchors, or merge with a parent-language label?
- **Competing onomatopoeia within one language.** English has both "bark" and "yap" for dogs. Probably keep both as separate anchors with intra-language variance signal.
- **PanLex license specifics.** They state "free and open" — https://panlex.org/license/ has the actual terms. Need to verify CC-BY-style attribution requirements for our use.

## How this fits the project

The anchor pool is a parallel workstream:

- **Independent of LLM extraction** (Stages 1–4). Can run in parallel.
- **Ready to plug into Stage 6** the moment Stage 5 produces a semantic-feature graph ready for phonology assignment.
- **Reusable.** The pool is the kind of artifact other projects could cite or build on; in fact aggregating a clean (concept × language × onomatopoeia × IPA × features) table at ~100 languages may itself be a side-publication.

## Phasing

1. **Seed.** Scrape Wikipedia's cross-linguistic onomatopoeias table. Get ~50 languages × ~50 concepts as a starting set with manual verification. Tests the pipeline.
2. **Expand languages.** Add the PanLex pull, push toward ~100 languages × same ~50 concepts.
3. **Expand concepts.** Add the full ~110-concept inventory across whatever languages have coverage.
4. **Tier 3 supplement.** Pull from scholarly sources for ideophone-rich languages, especially for concept categories where Tier 1 is thin.
5. **LLM supplement** for gaps in major languages; flag all entries for review.
6. **Attribute tagging.** Run against Stage 4 data when available.
