# Orchestration TODO

**Audience:** a Claude Code instance running locally on the user's
machine — the box that has GPU, a populated `/media/menser/fauna/`,
and history of having pulled and run pieces of this pipeline before.
The user does not remember exactly how far the local state has
advanced. **You** are expected to find out.

**Operator preference:** verify before suggesting. Don't assume the
remote's "shipped" claim matches local artifacts — they can drift in
either direction (local has run things never pushed; remote has plans
the local hasn't pulled).

This doc has three parts:

1. **Reconcile** — sync with origin so the planning docs you're
   reading match the planning docs on disk.
2. **Investigate** — concrete commands to discover local pipeline
   state, with interpretations.
3. **Decide** — a decision tree keyed to what you found.

Followed by short workstream pointers and a status snapshot you can
fill in from your investigation.

---

## 1. Reconcile with remote

Run these in order. Don't push or rewrite history without user consent.

```bash
git status
git branch --show-current
git fetch origin
git log --oneline -20 origin/main..HEAD          # local-only commits
git log --oneline -20 HEAD..origin/main          # remote-only commits
git log --oneline -20 origin/claude/consonant-directionality-research-IgZcw 2>/dev/null
```

Interpretation:

- **Working tree dirty.** Stash or commit before pulling. If the dirty
  state is the user's in-progress work, ask before doing anything.
- **On `main`, remote has new branch with `semanticphonology.md` and
  `orchestration-todo.md`.** That branch is the active research
  thread (consonant directionality / Stage 6 cutover plan). Decide
  with the user whether to check it out or merge it.
- **On `main`, local is ahead of `origin/main`.** Local ran something
  that wasn't pushed. Read the commit messages to find out what before
  doing anything.
- **Detached HEAD or unfamiliar branch.** Stop, ask the user where
  they want to be.

Read the three meta-docs once you're on the right ref:

- `spec.md` v0.2 — the 6-stage pipeline (what *should* exist on disk)
- `semanticphonology.md` — the active replan for Stage 6 (what
  *will* be built; replaces the hash scheme)
- `anchor-pool-sketch.md` — what Track A was always supposed to be
- `anchor-data-plan.md` — Phases 1–6 of anchor data collection, with
  an implementation-status table

---

## 2. Investigate local pipeline state

All artifact paths are under `/media/menser/fauna/interlingua/`
(`FAUNA_ROOT` in `src/conlang/__init__.py`). The repo's `data/` is a
symlink there. Don't assume; check.

### 2.1 Disk layout

```bash
ls -la /media/menser/fauna/interlingua/
ls -la /media/menser/fauna/interlingua/data/{raw,interim,processed}/ 2>/dev/null
ls -la /media/menser/fauna/interlingua/anchoring/{raw,interim,processed}/ 2>/dev/null
du -sh /media/menser/fauna/interlingua/{data,anchoring,hf-cache}/ 2>/dev/null
```

If any of these directories are missing or empty, the corresponding
stage hasn't run on this box.

### 2.2 Pipeline-state file checklist

For each file: existence answers "did this stage run", mtime answers
"how recently", and contents answer "with what parameters". Use
`stat -c '%n %s bytes %y'` if you want all three at once.

| Stage | File to look for | What it tells you |
| --- | --- | --- |
| 1 | `data/raw/features.jsonl` | exists → Stage 1 ran; `wc -l` gives N (was 1000; v1 target is 2000) |
| 1 | `data/raw/decoder.npy` (or similar from `save_node_set`) | exists → SAE decoder vectors persisted |
| 1 | `data/raw/flores200_dataset/` | exists → FLORES corpus downloaded for Stage 3 |
| 2 | `data/interim/sim_matrix.npy` | exists → pairwise cosine computed |
| 2 | `data/interim/sim_summary.json` | inspect: threshold distribution and dedup choices |
| 2 | `data/interim/hdbscan_labels.npy` | exists → HDBSCAN clusters assigned |
| 3 | `data/interim/coactivation/pmi.npy` | exists → PMI computed (the expensive GPU step) |
| 3 | `data/interim/coactivation/{cofire,fires}.npy` | inputs to PMI; if missing, PMI was rebuilt without them |
| 3 | `data/interim/coactivation/{top_pairs,summary}.json` | inspect: what edges survived thresholds |
| 3 | `data/interim/crystals/` | the function-vector crystal attempt — per `README.md`, this "landed negative" but the cache may still exist |
| 3 | `data/processed/crystal_bridge/` | crystal bridge outputs; same status as above |
| 5 | `data/processed/regularized.json` | exists → parent/sibling/near schema applied; this is the input to Track B Stage 6 |
| 6B | `data/processed/lexicon.json` | exists → Track B (hash-based) lexicon was generated |
| 6B | `data/processed/lexicon.html` and/or `docs/static/lexicon.html` | the deliverable for Track B |
| 6B | `data/processed/slice.html` | the cytoscape visualization |
| 6A | `anchoring/raw/wikipedia/` | exists → Phase 1 of anchor pipeline ran |
| 6A | `anchoring/processed/anchors-seed.{jsonl,csv}` | Phase 1 anchor seed output |
| 6A | `anchoring/processed/anchors-v*.jsonl` | Phase 3/6 anchor outputs; the highest version number is current |
| 6A | `anchoring/processed/signatures-v*.jsonl` | per-concept phonological signatures (input to interpolation when Phase 1 lands) |

Quick one-liner to scan all of it:

```bash
find /media/menser/fauna/interlingua -maxdepth 4 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.npy' -o -name '*.parquet' -o -name '*.html' \) \
  -printf '%TY-%Tm-%Td %s\t%p\n' 2>/dev/null | sort
```

### 2.3 Cross-check against repo deliverables

Two committed deliverables encode what the project last shipped:

```bash
ls -la docs/static/lexicon.html docs/static/walkthrough.html
# both should be present (≈230 KB and ≈315 KB respectively per current main)
grep -c '<tr><td' docs/static/lexicon.html
# row count gives you the lexicon size that was last published
```

If `docs/static/lexicon.html` differs from `data/processed/lexicon.html`,
something was regenerated locally without committing. Diff before
deciding what to do with it.

### 2.4 Sanity check the env

```bash
which python && python --version
python -c "import sae_lens, torch, transformers; print(sae_lens.__version__, torch.__version__)"
python -c "import panphon, epitran" 2>&1 | head -2
nvidia-smi | head -15
```

`panphon` and `epitran` install was broken in the remote container
(unicodecsv build failure on Python 3.11). If they import cleanly on
this box, the local env is healthy enough to run the anchor pipeline
end-to-end.

---

## 3. Decide what's next

Walk the decision tree based on what §2 turned up. Stop at the first
branch that matches.

1. **Pipeline never ran here** (`data/raw/features.jsonl` missing).
   Either this isn't the box the user thinks it is, or the data dir is
   on a different mount. Ask before doing anything.

2. **Stages 1–5 ran, Track B `lexicon.json` exists, anchor pipeline
   ran to at least `anchors-v1.jsonl`.** This is the normal "shipped
   v0.x" state. The active workstream is the Stage 6 cutover —
   `semanticphonology.md`. Recommended sequence:

   1. Confirm with the user they want to do Path A (Stage 6 cutover)
      and not Path B (GLUE survey).
   2. Start Phase 0 of `semanticphonology.md`: snapshot substrate to
      `data/processed/substrate-v1-n2000.parquet`, snapshot anchors
      to `data/processed/anchors-v1.parquet`, tag and commit
      `docs/static/lexicon.html` as the fossil, land
      `phonological_distance()` next to `project_ipa()` with tests.
   3. **Decide whether to bump N from 1000 to 2000 before Phase 0
      finishes.** Per `semanticphonology.md` §"N is a knob", the
      bottleneck is the co-activation forward pass which is roughly
      N-flat, so the bump is cheap in compute. The compute cost lives
      in re-running Stage 3 (coactivation) on the new feature set.
      Confirm with user.

3. **Stages 1–5 ran, but `lexicon.json` is missing.** Some intermediate
   work stalled. Run `python -m conlang.lexicon` to rebuild Track B,
   then evaluate per (2).

4. **Anchor pipeline didn't run beyond Phase 1** (no `anchors-v1.jsonl`,
   only `anchors-seed.jsonl`). Phase 3 + 6 of `anchor-data-plan.md`
   need to land before the cutover has anything to interpolate from.
   `anchor-data-plan.md` §"Implementation status (2026-05)" has the
   phase table; pick up wherever it shows incomplete.

5. **Coactivation never ran, no `pmi.npy`.** Stages 1–2 only. This is
   pre-v1 state. Either complete Stage 3 or ask the user whether they
   moved to a different feature graph (e.g., circuit-tracer outputs,
   though README says crystals "landed negative").

6. **GLUE survey artifacts present** (anything under
   `data/interim/glue/` or `data/processed/glue-*`). The user already
   started Path B. Open `GLUE-TODO.md` §"Suggested order of execution"
   and find the next step.

7. **`substrate-v*.parquet` exists.** Phase 0 of the cutover already
   ran. Move to Phase 1 — build `src/conlang/interpolate.py` per
   `semanticphonology.md` §"Phase 1 — Interpolation infrastructure".
   If Phase 1 also exists (`src/conlang/interpolate.py` present), move
   to Phase 2 cutover.

---

## 4. Workstream pointers

Once you know where you are, here's where to read:

- Cutover plan, with phases, acceptance criteria, and the
  hash-penalty rationale → `semanticphonology.md`.
- Track A blueprint (Sibson natural-neighbor interpolation) →
  `anchor-pool-sketch.md`.
- Anchor data collection phases and what shipped vs deferred →
  `anchor-data-plan.md`.
- Track B current grammar (the version about to be replaced) →
  `docs/grammar.md`.
- Function-word discovery research (independent of the cutover) →
  `GLUE-TODO.md`. Don't open unless doing Path B.
- After-the-GLUE-survey agenda → `POST-GLUE-SKETCH.md`. Don't open
  until GLUE survey lands.
- Master spec (rarely changes) → `spec.md` v0.2.
- Methodological substrate (read once) → `prior-work.md`.
- Project conventions → `CLAUDE.md`.

---

## 5. Status snapshot — fill in from your investigation

Replace each `(check)` with what §2 told you. Re-commit when you do.

| Workstream | Reported state | Local state | Notes |
| --- | --- | --- | --- |
| Stage 1 ingest (N=?) | shipped @ N=1000 | (check `wc -l data/raw/features.jsonl`) | |
| Stage 2 dedupe | shipped | (check `data/interim/hdbscan_labels.npy`) | |
| Stage 3 coactivation | shipped | (check `data/interim/coactivation/pmi.npy` mtime) | |
| Stage 3 crystals | landed negative per README | (check `data/processed/crystal_bridge/`) | left as a fossil |
| Stage 5 regularize | shipped | (check `data/processed/regularized.json`) | |
| Track B lexicon | shipped, hash-based | (check `data/processed/lexicon.json`) | doomed by cutover |
| Anchor pipeline Phases 1, 3, 6 | shipped per `anchor-data-plan.md` | (check `anchoring/processed/`) | |
| Anchor pipeline Phase 2 (PanLex) | redirected to Epitran enrichment | n/a | see `anchor-data-plan.md` |
| Stage 6 cutover Phase 0 | planned | (check for `substrate-v*.parquet`) | next thing to land |
| Stage 6 cutover Phase 1 | planned | (check for `src/conlang/interpolate.py`) | |
| Stage 6 cutover Phase 2 | planned | n/a | atomic with grammar.md edit |
| Stage 6 cutover Phase 3 | planned | n/a | needs Spearman metric script |
| GLUE survey | planned, not started | (check `data/interim/glue/`) | independent of cutover |
| Concept inventory growth | ~100 concepts | `wc -l src/conlang/anchors/concepts.py` or count entries | gate condition for N > 2000 |
| Deliverables | shipped @ N=1000 hash | (check `docs/static/lexicon.html` row count) | regenerate after cutover |

---

## 6. Update protocol

This file is the index, not a re-statement of the workstream docs.
Refresh §5 whenever a workstream changes state. Refresh §3 when a
new decision branch becomes worth describing. If §1–4 start drifting
toward 400+ lines, content is in the wrong place — push it back into
the workstream docs.

The single most important thing this file does: **make the local
instance verify reality before acting**. Everything else is
secondary.
