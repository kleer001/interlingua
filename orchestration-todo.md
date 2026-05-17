# Orchestration TODO

The meta-doc. Read this first when you come back to the project and forgot
where you were. Everything below points at other docs — this one stays
short on purpose.

**Last refreshed:** 2026-05-17, on branch
`claude/consonant-directionality-research-IgZcw`.

---

## 1. Status snapshot

| Workstream | Doc(s) | Status | What's left |
| --- | --- | --- | --- |
| Pipeline spec | `spec.md` v0.2, `prior-work.md` | living | revisit after Stage 6 cutover and the GLUE survey land |
| Stage 1 — ingest | `src/conlang/ingest.py`, `slice.py` | shipped | N is currently 1000; bump to 2000 in Phase 0 of `semanticphonology.md` |
| Stage 2 — dedupe + HDBSCAN | `src/conlang/dedupe.py` | shipped | nothing |
| Stage 3 — cosine + coactivation | `src/conlang/run_coactivation.py` | shipped | circuit-tracer attribution still external; crystals/function-vector bridge landed negative (see README) |
| Stage 4 — decision gate | manual | passed for v1 | re-run after substrate refresh |
| Stage 5 — regularize | `src/conlang/regularize.py` | shipped | nothing |
| Stage 6 Track B — Bantu/hash lexicon | `src/conlang/phonology.py`, `lexicon.py`, `site.py`, `docs/grammar.md` | shipped — but the hashes are the bad choice | replaced by cutover per `semanticphonology.md` |
| Stage 6 Track A — anchors pipeline | `src/conlang/anchors/`, `anchor-pool-sketch.md`, `anchor-data-plan.md` | infrastructure shipped (5,338 forms × 84+ langs × ~100 concepts, 10C/5V projection done) | not yet wired into lexicon — that's the cutover |
| Stage 6 — semantic phonology cutover | `semanticphonology.md` | **planned, not started** | Phase 0 (freeze + N=2000), Phase 1 (interpolation lib), Phase 2 (cutover), Phase 3 (validation) |
| Phonology inventory | `docs/grammar.md` | shipped with `r` added | `h`, `ch`, `j` deferred as candidates |
| GLUE survey (function-word discovery) | `GLUE-TODO.md` | **planned, not started** | Path 1 (cheap audit) → Path 2 (main event) → Paths 3–5 as gated |
| Post-GLUE discovery | `POST-GLUE-SKETCH.md` | sketch only | unblocked after GLUE survey ships |
| Anchor pool growth | `concepts.py`, `anchor-pool-sketch.md` §"Open questions" | mostly animal sounds today (~100 concepts) | grow to ~150 before N=5k, ~300 before N=10k; next tiers are weather, human-body, mechanical, ideophones |
| Anchor data acquisition | `anchor-data-plan.md` | Phases 1, 3, 6 shipped; Phase 2 redirected to Epitran enrichment; Phases 4, 5 deferred | deferred phases unblock if v1 shows gaps in specific concept categories |
| Deliverables (Track B) | `docs/static/lexicon.html`, `docs-site/`, walkthrough notebook | shipped; GitHub Pages deploy automated | regenerate after cutover |

**One-line summary:** the SAE pipeline is shipped end-to-end through a hash-based
Track-B lexicon; the real Track-A interpolation cutover and the GLUE
survey are the two big open workstreams; everything else is either
done, deferred for cause, or downstream of those two.

---

## 2. Doc map

When to open what.

- **`README.md`** — pipeline overview, how to run it, repo layout. Read for orientation.
- **`spec.md`** — the 6-stage spec. Read to know what each stage is supposed to do.
- **`prior-work.md`** — the methodological substrate this builds on. Read once.
- **`CLAUDE.md`** — project conventions (commits, style, boundaries). Read once.
- **`anchor-pool-sketch.md`** — Track A design (Sibson natural-neighbor interpolation). The blueprint Stage 6 was supposed to follow.
- **`anchor-data-plan.md`** — data-acquisition phases for the anchor pool. Status table for what's shipped vs deferred.
- **`semanticphonology.md`** — the active cutover plan. Phases 0–3 that replace hash-based stems with anchor-pool interpolation. **Open this when working on Stage 6.**
- **`GLUE-TODO.md`** — function-word discovery, Paths 1–5. Long. Open when starting that workstream.
- **`POST-GLUE-SKETCH.md`** — Paths 6–11 (dark biota, etc). Don't open until GLUE survey lands.
- **`docs/grammar.md`** — published phonology + morphology spec. Update on Stage 6 changes.
- **`docs/methodology.md`** — published methodology writeup. Update after major findings.
- **`docs/origin.md`** — origin lore. Written last per spec §12.
- **`orchestration-todo.md`** (this file) — refresh after any structural change.

---

## 3. Dependency graph

Plain-text "A → B" means A unblocks B / A is prerequisite for B.

```
Stages 1–5 (shipped)
    │
    ├──→ Stage 6 Track B (shipped, but doomed)
    │        │
    │        └──→ GLUE-TODO Path 1 audit
    │                   (audits the existing 1000-entry lexicon)
    │
    ├──→ Anchor pool infrastructure (shipped)
    │        │
    │        └──→ semanticphonology Phase 0 (freeze substrate + anchors)
    │                   │
    │                   ├──→ semanticphonology Phase 1 (interpolation lib)
    │                   │           │
    │                   │           ├──→ semantic_neighbors.py query module
    │                   │           │       (answers "closest concept to X" today)
    │                   │           │
    │                   │           └──→ semanticphonology Phase 2 (cutover)
    │                   │                       │
    │                   │                       └──→ Phase 3 (validation, Spearman ρ)
    │                   │                                   │
    │                   │                                   └──→ v1 site rebuild
    │                   │
    │                   └──→ anchor pool growth (concepts.py expansion)
    │                               │
    │                               └──→ N scale-up past 2000 (5k / 10k)
    │
    └──→ GLUE-TODO (independent track)
              │
              ├──→ Path 1 audit (cheap) ──────┐
              ├──→ Path 2 probing (main event)│
              ├──→ Path 3 attention-SAEs (gated on 1–2)
              ├──→ Path 4 platypus (reuses Path 2's cache)
              └──→ Path 5 lichen (reuses Path 4)
                              │
                              └──→ POST-GLUE-SKETCH (Paths 6–11)
                                          │
                                          └──→ v2 phonology revisit (if Path 6 surfaces
                                                  raw-space directions that want forms)
```

Notable independencies:
- The GLUE survey is independent of the Stage 6 cutover. Can run in parallel on a different branch.
- `semantic_neighbors.py` (the query interface from `semanticphonology.md`
  §6 step 3) is the cheapest thing that answers "closest concept to
  'justice'" — it lands between Phase 1 and Phase 2 and unlocks
  exploratory questions before the lexicon is regenerated.

---

## 4. Critical paths

Two reasonable destinations; pick one per session.

### Path A — Ship a better v1 (the Stage 6 fix)

The hash-based stems are the worst thing in the artifact today. Fixing them is
the highest-leverage move per unit of time.

1. `semanticphonology.md` Phase 0 — freeze substrate, bump N to 2000, land
   `phonological_distance()` with tests. Half a day.
2. Phase 1 — build `src/conlang/interpolate.py` standalone. Sibson weights,
   anchor mixer, discretization, phonotactic gate. 1–2 days.
3. *(Side ship)* `src/conlang/semantic_neighbors.py` — thin wrapper over
   Phase 1 that answers query-style questions. Half a day.
4. Phase 2 — atomic cutover commit. Regenerate the 1000-entry lexicon (or
   2000 if Phase 0 bumped N), update `docs/grammar.md`. 1 day.
5. Phase 3 — validate (Spearman ρ ≥ 0.4, anchor recall ≥ 70%, hand audit).
   1 day, mostly inspection.
6. Rebuild site + walkthrough notebook, commit deliverables. Trivial.
7. **v1 ships.** Tag it.

### Path B — Research depth (the GLUE survey)

Independent of Path A. Bigger commitment, more interesting findings, longer
horizon.

1. `GLUE-TODO.md` Path 1 — audit the existing lexicon for glue features hiding
   in class 11. Half a day.
2. Path 2 Stages A–E — minimal-pair generation, activation caching, direction
   extraction. **Days to weeks**, GPU-bound. Cache is reused by Path 4 — do
   not delete.
3. Decide whether Path 3 (attention-SAEs) is worth the cost based on what
   Path 2 surfaces.
4. Path 4 Stages 1–5 — unsupervised platypus discovery. Reuses Path 2 cache.
5. Path 5 Stages 1–5 — lichen (compositional operators). Reuses Path 4.
6. Mycorrhizal hub scan (parallel to Path 5).
7. Phonotactics update for newly-named glue operators, site rebuild.
8. **`POST-GLUE-SKETCH.md` opens.** Decide which of Paths 6–11 actually go.

Path A and Path B don't conflict. If you have parallel compute, run Path B's
Stage 2 cache job in the background while you work Path A.

---

## 5. Pick-your-next-task decision tree

When you sit down and have N minutes:

- **15 minutes.** Read this doc. Maybe pick a small `semanticphonology.md`
  Phase 0 sub-bullet (e.g. add the `phonological_distance` stub + tests).
- **An hour or two.** Land all of Phase 0 (substrate snapshot, anchor
  snapshot, fossil tag, distance metric). Or do `GLUE-TODO.md` Path 1's
  prep work (write the regex-and-keyword filter that flags candidate
  function features in the existing lexicon).
- **A focused day.** Build `interpolate.py` (Phase 1). Or run Path 1
  audit end-to-end and write up findings.
- **A focused week.** Phases 1–2 of `semanticphonology.md` plus the
  `semantic_neighbors.py` module — that's Path A through cutover, with
  validation pending the following week.
- **A research arc (weeks).** GLUE survey Paths 1–5, in order.

When in doubt: do Path A. Hash-based stems are the load-bearing weakness
in the artifact today.

---

## 6. Things that look like work but are actually maintenance

Don't conflate with the critical paths above:

- Site auto-deploys via GitHub Actions on every push that touches `docs/`
  or `mkdocs.yml`. No manual deploy step. (See `.github/workflows/deploy-pages.yml`.)
- The walkthrough notebook is regenerated via
  `python notebooks/build_walkthrough.py` + nbconvert. Run after any
  Stage 6 change that affects example output.
- Anchor pool growth: only do this when something downstream is gated on it
  (e.g. scaling N past 2000 per `semanticphonology.md` §"N is a knob").
- `panphon` / `epitran` install in this environment is currently broken
  (unicodecsv build failure on Python 3.11). If you need them, switch to
  Python 3.10 or build wheels from source. Not blocking for any planning
  work, only for execution.

---

## 7. When to update this file

Refresh §1 (status snapshot) and §3 (dependency graph) when:

- A workstream changes status (planned → in-progress → shipped → deferred).
- A new dependency surfaces that re-orders the critical path.
- A new doc lands or an old one is retired.

The body of every other doc is the source of truth. This file is the
index. Keep it short — if it grows past ~400 lines, you're putting
content here that belongs in the workstream docs themselves.
