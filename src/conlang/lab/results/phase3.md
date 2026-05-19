# Phase 3 measurements

> Source of truth: `semanticphonology.md` §3 in the interlingua repo root.
> This file is a frozen copy so the result travels with the apparatus.
> Re-sync manually if the master log changes.

All measurements: N=2000 substrate, NW phonological-distance kernel,
Gemma 2 (2B) residual mean-pool, 10,000 random feature-pairs (seed=0).
The `layer` column is the index into `hidden_states[i]` (layer 13 =
output of transformer block 12).

| Date | Anchors | Embed text | Layer | ρ | p |
|---|---|---|---|---|---|
| 2026-05-18 | 63 concept | `english_seeds[0]` ("hiss") | 13 | **0.0365** | 2.65e-4 |
| 2026-05-18 | 63 concept | `english_seeds[0]` | 21 | 0.0348 | 4.92e-4 |
| 2026-05-18 | 1574 attribute | `f"{slug} ({seed}) :: {attr}"` | 13 | 0.0292 | 3.46e-3 |
| 2026-05-18 | 63 concept | `english_seeds[0]` | 9 | 0.0197 | 4.86e-2 |
| 2026-05-18 | 1574 attribute | `f"{slug} :: {attr}"` | 13 | 0.0178 | 7.62e-2 |
| 2026-05-18 | 63 concept | `english_seeds[0]` | 25 | 0.0174 | 8.18e-2 |
| 2026-05-18 | 63 concept | `english_seeds[0]` | 17 | 0.0154 | 1.24e-1 |
| 2026-05-18 | 63 concept | `english_seeds[0]` | 5 | 0.0079 | 4.30e-1 |

## Conclusion

Hypothesis falsified at this scale. Best measured ρ across every
configuration tried — anchor count from 63 to 1574, embed format from
bare seed through slug + attribute to slug + seed + attribute, Gemma
layers 5 through 25 — is ρ = 0.0365, against the §3 cutover threshold
of 0.15 and the §5 strong-claim threshold of 0.20.

`build_stem()` stays as the deterministic hash. The
`lexicon-pre-cutover` tag at commit `af1c965` remains the live
deliverable. The §5 question *consonant directionality is the
load-bearing claim* answers **no, consonant features alone are not
enough**, at least at this model scale and substrate granularity.

Reopen on: a different model, a different substrate, a different
distance kernel, or a richer phonology.
