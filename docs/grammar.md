# Grammar

## Phonology

The inventory grows thoughtfully. Minimalism is not the goal — fidelity to
the hidden language between languages is. We bias toward sounds present in
most natural languages so any generated stem stays pronounceable without
specialist training, and we expand the inventory when iconicity or
expressive coverage would otherwise be lost in translation. The shape
remains Bantu-leaning because that family gave the best starting point, not
because we are bound to it.

**Vowels (5):** `a e i o u`

**Single consonants (16):** `p t k b d g m n s z f v l r w y`

**Nasal digraphs (2):** `ny ng` — single phonemes despite the two-letter
spelling.

**Prenasalized onsets (4):** `mp mb nt nd` — these are not single
phonemes; they arise from class-9 N-prefix sandhi.

**Syllable template:** `(C)V`. Onsets may be a single consonant, a nasal
digraph, or one of the four prenasalized clusters. No codas. No glide
insertion. No tone.

**Word minimum:** ≥ 2 syllables. Every entry in the lexicon satisfies
this after class prefixing and (optional) negation prefixing.

## Morphology

Eleven noun classes, mostly in singular/plural pairs. Composition order:

```
[negation?] + [class prefix] + [stem]
```

Productive negation is a single prefix, `si-`, that attaches outside any
class prefix. There is no special form for negated nouns versus negated
verbs versus negated adjectives — per spec v0.2 §7 Commitment 7, all
negation is the same productive operation. `si-` elides its vowel before
a vowel-initial form.

### The eleven classes

| Class | Prefix | Domain                                | `paka` + class | negation       |
| ----: | -----: | ------------------------------------- | -------------- | -------------- |
|     1 |   `mu-`| human, singular                       | `mupaka`       | `simupaka`     |
|     2 |   `ba-`| human, plural                         | `bapaka`       | `sibapaka`     |
|     3 |    `u-`| plant/object, singular                | `upaka`        | `supaka`       |
|     4 |   `mi-`| plant/object, plural                  | `mipaka`       | `simipaka`     |
|     5 |   `li-`| fruit/paired thing, singular          | `lipaka`       | `silipaka`     |
|     6 |   `ma-`| fruit/paired thing, plural            | `mapaka`       | `simapaka`     |
|     7 |   `ki-`| tool/thing, singular                  | `kipaka`       | `sikipaka`     |
|     8 |   `vi-`| tool/thing, plural                    | `vipaka`       | `sivipaka`     |
|     9 |    `N-`| animal/language, sg (homorganic / yi-)| `mpaka`        | `simpaka`      |
|    10 |   `zi-`| animal/language, plural               | `zipaka`       | `sizipaka`     |
|    11 |   `lu-`| long thing / mass / abstract          | `lupaka`       | `silupaka`     |

In the current lexicon, only the singular classes (1, 5, 7, 9, 11) are
populated. Plural assignment is left for actual use: any node can be
re-prefixed with its plural counterpart at speech time.

### Class 9 sandhi

Class 9 is the only class with allomorphy. It takes one of two surface
forms depending on the stem's initial sound:

- **Prenasalized stop:** `m` before `p` or `b`, `n` before `t` or `d`.
  Produces `mp-`, `mb-`, `nt-`, `nd-`.
- **`yi-`** (or just `y-` before a vowel) for every other onset, including
  `r`. The pure homorganic outcome (`ng+k`, `n+s`, `n+w`, `n+r`, …) would
  produce phonotactically illegal clusters in this inventory, so we use a
  vowel-bearing allomorph instead.

This is the one place where the inventory and the morphology had to
negotiate. It is documented because the negotiation matters.

### Inventory candidates under review

Tracked so we keep an honest eye on what gets lost in translation:

- **`h`** — breath/whisper iconicity has no current home. Cheap to add
  (most languages have some [h] or aspirated stops; main exceptions are
  French and Italian), but acoustically weak and easily dropped in casual
  speech. Deferred until the iconicity work shows the gap matters.
- **`ch` / `j`** (affricates `tʃ`, `dʒ`) — punctate-sharp iconicity is
  already partly covered by `t k s`. Affricates would add digraph
  orthography, new minimal-pair confusions (`ch`/`sh`, `j`/`y`), and
  homorganic-sandhi work in class 9. Deferred on diminishing returns.
- **`r`** — added. Canonical carrier for rolling/rumbling iconicity, and
  no good substitute. Cost: `r` is the most variable consonant across
  world languages (trill, tap, approximant), and many L1s conflate r/l;
  the inventory commits to a tap-or-approximant tolerant reading rather
  than a specific articulation.

### Vowel hiatus

When a prefix-final vowel meets a stem-initial vowel, the prefix vowel
simply drops. There is no glide insertion. This was a deliberate KISS
choice: glide insertion would have meant maintaining a `/w/` vs `/y/`
table by quality of the preceding and following vowels, and the affix
table is the easier-to-read deliverable.

A consequence is that `mu + andu`, `mi + andu`, and `ma + andu` all
collapse to `mandu`. The lexicon entries are still unique because their
stems differ; class disambiguation is by context.

## Phonosemantic stems

> **Cutover in progress.** The hash scheme below is the live Track-B
> baseline (see `semanticphonology.md`). Phase 0 of the cutover has
> frozen the substrate at **N = 2000** features
> (`data/processed/substrate-v1-n2000.parquet`) and the anchor pool at
> v1 (`data/processed/anchors-v1.parquet`). Phase 2 will replace the
> CV1/CV2/CV3 hash with anchor-interpolated stems; until then, the
> 1000-entry hash lexicon shipped at `lexicon-pre-cutover` (git tag)
> remains the canonical Track-B deliverable.

Every stem in the lexicon is built from three syllables (occasionally
four, in collision-resolution rounds):

- **CV1** encodes the HDBSCAN cluster the feature belongs to. Siblings
  (cluster co-members) share their first syllable.
- **CV2** is a hash of the highest-PMI co-activation parent's feature id.
  Two features with the same parent share their second syllable.
- **CV3** is a hash of the feature's own id. Uniqueness within the
  (CV1, CV2) bucket.

When CV1+CV2+CV3 is not unique across the lexicon, the resolver appends a
CV4 derived from a salted hash, and keeps salting until the global stem
set is unique. In the 1000-entry slice the longest stem is five syllables.

The point is: phonological similarity between two surface forms is *not*
incidental. It encodes structure in the model's geometry.
