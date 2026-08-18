# CLAUDE.md — Codebase Audit (not a new phase)

**Purpose:** Phases 1 through 5 are all built and tested, but no one has looked at the codebase as a whole with an eye toward cleanliness rather than correctness. This is a **review and report task, not a rewrite task** — find what's redundant, dead, or messy; do not delete or restructure anything without listing it first. The user wants to see the list and decide what actually gets removed, the same posture as every other decision in this project (nothing ships without review).

**Do not touch test behaviour or measured results.** Every number in `PROJECT_LOG.md`/`EVALUATION.md` is tied to specific code having run a specific way — a "cleanup" that silently changes what a function does would invalidate results this project has spent five phases establishing. If a genuine bug is found (not a style issue — an actual defect), flag it separately and do not fix it inline without saying so explicitly, the same discipline Phase 5 used for the grounding-checker's `12%` defect (documented and left failing, not silently patched).

## 1. Dead code and unused artifacts

- Any function, module, or file that nothing imports or calls — check with actual usage analysis (grep for references), not assumption. Ablation-era code (`forecasting/lstm_small.py`, `forecasting/baseline_ridge.py`, `forecasting/iod.py`, etc.) is **deliberately kept** as evidence per prior phases' own stated reasoning — do not flag these as dead without re-reading why they were kept.
- `__pycache__` directories, stale `.pyc` files, anything that shouldn't be tracked but might have been accidentally committed — check against `.gitignore` for gaps.
- Duplicate or superseded evidence files in `models/`, `retrieval/`, `crop_impact/`, `evaluation/` — e.g. any file whose contents were fully superseded by a later phase and is no longer cited from anywhere current (cross-check against `EVALUATION.md`'s citations before flagging — a file might look old but still be the cited source of record).

## 2. Redundant / duplicated logic

- Any calculation pattern repeated across `forecasting/`, `heat/`, `crop_impact/` that could be a shared helper instead — but only flag this if extracting it wouldn't blur the deliberate independence between Drought and Heat's feature sets (the project's own design principle keeps them separate on purpose; don't undo that in the name of DRY).
- Any place `check_grounding()`'s logic appears to have been reimplemented rather than imported — Phase 4 and 5 were both explicitly instructed to reuse it unmodified; verify that held.

## 3. The local-database/state concern

The user flagged a specific worry: "database bana gaya, ek baar download karke wapas open karo to remove ho jaata hai" — investigate and report on:

- What state is git-ignored vs. tracked (`data/raw/`, `data/processed/`, `models/*.pkl`/`.h5`, the ChromaDB persistent directory under `retrieval/`) — confirm exactly what a fresh `git clone` will and won't have, and what commands must be re-run to regenerate it (`fetch_data`, `train`, `retrieval.build`, etc.).
- Whether the ChromaDB persistent store's path is relative and portable (works the same after a clone to a different machine/folder) or has any absolute-path or machine-specific assumption baked in.
- Document this plainly in a new `SETUP_FROM_CLEAN.md` — the exact sequence of commands someone needs to run after cloning to get from empty repo to a working system, so "download and reopen" never again means "and now it's broken."

## 4. Requirements files

- `requirements-phase1.txt`, `-phase2.txt`, `-phase3.txt` exist separately — check whether Phase 4/5 added dependencies not captured anywhere, and whether a single consolidated `requirements.txt` would now serve better than three partial files, now that the phases aren't being built in isolation anymore.

## Output

A single `AUDIT_REPORT.md` — for each finding: what it is, where it is, why it's flagged, and a clear **recommendation with reasoning**, not just a list of file paths. Group into "safe to remove," "needs a decision," and "found a real bug — not fixed, flagging only." Nothing gets deleted or changed as part of this task except writing `SETUP_FROM_CLEAN.md` and, if genuinely warranted, consolidating the requirements files (state clearly if you did this).
