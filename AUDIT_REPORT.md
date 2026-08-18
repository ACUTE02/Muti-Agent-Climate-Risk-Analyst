# Codebase audit — Phases 1 to 6

Review-and-report only. **Nothing in this audit was deleted, refactored or
fixed.** Two exceptions were sanctioned by the audit brief and are stated
plainly at the end: `SETUP_FROM_CLEAN.md` (already written during Phase 6, before
this audit ran) and the requirements consolidation, which was **assessed and not
performed** — the reasoning is in §5.

Findings are grouped as the brief asked: safe to remove, needs a decision, and
one real bug flagged but deliberately not fixed.

**Headline:** the codebase is in better shape than the brief anticipated. The
suspicion of accumulated dead code and duplicated evidence files did not survive
contact with usage analysis — there is exactly **one** genuinely dead function in
the whole repository, and **zero** removable evidence files. The real problems
are elsewhere: a health check that can lie, a metric formula copy-pasted six
times, and five requirements files where there should be one or two.

---

## Method

Not assumption — actual usage analysis:

- Every module in the eight packages was checked for Python importers using a
  pattern covering all four import styles (`from pkg.mod import`,
  `import pkg.mod`, `from pkg import mod`, and `pkg.mod` attribute access). The
  first pass missed `from pkg import mod` and produced four false orphans; the
  corrected pass produced three, all explained below.
- Every top-level function was counted against a concatenation of all `.py` and
  `.md` files in the repo.
- Every tracked evidence file was cross-checked against `EVALUATION.md`,
  `PROJECT_LOG.md`, `README.md` and all source — and where a file was not cited
  by name, its *numbers* were checked against the summaries that are cited,
  before concluding anything.
- Third-party imports were extracted with `ast` and diffed against every
  `requirements*.txt`.

---

## 1. Safe to remove

### 1.1 `outlooks_for()` — the only genuinely dead function

**Where:** `retrieval/outlooks.py:103`

```python
def outlooks_for(agent: str, force: bool = True) -> list[dict]:
    """Only the outlooks tagged as relevant to a given agent."""
```

**Why flagged:** referenced nowhere — not by any module, test, script or
document. The orchestrator calls `fetch_outlooks()` directly and passes every
outlook through, attributing each by name; per-agent filtering was designed and
then never wired up.

**Second-order problem if it were ever used:** it defaults to `force=True`, so
every call would bypass the cache and re-fetch both IMD documents over the
network. A caller reaching for what looks like a cheap filter would get two HTTP
requests per invocation.

**Recommendation: remove it.** Four lines, no callers, no evidence value — it
documents no result and backs no number. If per-agent filtering is wanted later
it is trivially rewritten, and rewritten with a sane cache default. This is the
only code in the repository I would remove without further discussion.

### 1.2 Two stale `.gitignore` entries

**Where:** `.gitignore`

| Entry | Status |
|---|---|
| `reports/eval/*.json.tmp` | `reports/` has never existed in this repo |
| `*.chroma/` | no `*.chroma` directory exists; the real store is `retrieval/chroma_store/`, ignored separately three lines later |

**Recommendation: remove both.** Zero risk — they match nothing. Minor
housekeeping, but a `.gitignore` that describes a directory layout the project
never had is misleading to a reader trying to understand what state exists.

### 1.3 Nothing else

No tracked `__pycache__`, no tracked `.pyc`, no tracked `.joblib` / `.keras` /
`.parquet`. The ignore rules and what is actually tracked agree completely. This
was checked, not assumed.

---

## 2. Needs a decision

### 2.1 The skill-score formula is written out six times

**Where:** `forecasting/evaluate.py:50`, `forecasting/t1_model.py:40`,
`heat/model.py:72-75` (four times in one dict), `heat/phase11.py:54-62`.

Every one computes `1 - rmse / rmse_baseline`.

**Why this matters more than ordinary DRY.** Phase 1.3 refactored `evaluate.py`
specifically so all three drought variants scored through one
`score_predictions()` — its own stated reasoning was that *"the comparison is
only meaningful if the numbers are computed identically."* The Heat agent then
grew its own copies of the same formula.

And Heat 1.1's worst bug was precisely a baseline defect:
`climatology_prediction()` and `persistence_prediction()` hardcoded
`config.HEAT_TARGET` instead of following the run's actual target, producing an
apparent skill of **+0.44 / +0.63** that collapsed to −0.28 / +0.02 once fixed.
A single shared, tested `skill_score(rmse, rmse_baseline)` would not have
prevented that specific bug — it was in the baseline *prediction*, not the
division — but it is the same family of problem, and centralising the metric is
the cheapest available guard against the next one.

**Does this violate the Drought/Heat independence principle?** No, and this is
worth being precise about. That principle is about **features** — the two agents
deliberately share no predictors. A skill-score division is not a feature; it is
a metric definition, and having two definitions of one metric is a liability,
not independence.

**Recommendation: extract a shared helper — but not as part of this audit.**
Every one of these call sites produced a number that is published in
`PROJECT_LOG.md` or `EVALUATION.md`. Touching them means re-running the affected
evaluations and confirming the numbers are unchanged, which is a change with a
verification burden, not a tidy-up. **Needs your decision**, and if taken it
should be its own change with a before/after numeric comparison — the same
discipline Phase 1.6 used when it re-ran both IOD variants in one process rather
than comparing against numbers on disk.

### 2.2 API-key detection is implemented three different ways

**Where:**

| Site | Method |
|---|---|
| `retrieval/embed.py:52` `get_api_key()` | the real one — loads `.env`, checks all three env var names, raises `MissingAPIKey` |
| `api/app.py:216-221` | ad-hoc: `os.environ` check, then substring `"API_KEY" in .env text` |
| `scripts/setup.py:156-163` | the same ad-hoc pattern again |

The `.env` mechanism itself is sound — `get_api_key()` loads it, with a manual
parser fallback if `python-dotenv` is absent. The problem is that two callers
reimplemented the *detection* rather than calling it. See §4 for the bug this
produces.

**Recommendation:** collapse both ad-hoc checks onto `get_api_key()` in a
`try/except MissingAPIKey`. This is a small change but it is a **behaviour**
change to `/health`, so it belongs with the bug fix in §4 and needs your
go-ahead.

### 2.3 `tests/test_evaluation.py` has its own structural-token stripper

**Where:** `tests/test_evaluation.py:34-44` (`CROSS_REFS`), conceptually parallel
to `orchestrator/grounding.py:41-50` (`STRUCTURAL_PATTERNS`).

Both answer "which numbers in this text are not claims". The test strips
`Phase 5`, `§5` and list markers before handing text to grounding's own
extractor.

**Recommendation: leave it.** Moving `Phase N` stripping into
`STRUCTURAL_PATTERNS` would change what `check_grounding()` does to **real
generated reports** — a report saying "Phase 5 measured 0.9286" would stop having
its `5` checked. That is a live behaviour change to the project's most
safety-critical component, made to satisfy a style preference, and it would
invalidate the Phase-5 precision/recall measurement. The duplication is the
correct trade here. Flagged so a future reader knows it was considered.

### 2.4 Grounding reuse — verified, no action

Phases 4 and 5 were both instructed to reuse `check_grounding()` unmodified.
**They did.** Every consumer imports from `orchestrator.grounding`:
`crop_impact/tool.py`, `evaluation/checker_eval.py`, `orchestrator/graph.py`,
and both test modules. No number-extraction regex exists anywhere outside
`grounding.py` except the cross-reference stripper in §2.3, which is a
pre-filter, not a reimplementation. Nothing to do — recorded because the brief
asked for verification.

---

## 3. Investigated and found clean — do **not** remove

These were the audit's main suspicions. Each was checked properly and none holds.

### 3.1 No dead modules

Three modules have no Python importer. All three are runnable scripts whose
output is cited evidence:

| Module | Why it stays |
|---|---|
| `forecasting/lstm_small.py` | Phase 1.3 ablation. Produced `metrics_lstm_small_*.json`, whose val_loss figures **0.9999** and **1.0237** are quoted verbatim in `PROJECT_LOG.md` Phase 1.3 |
| `heat/phase11.py` | Heat 1.1's 36-cell grid. Produced `models/heat_phase11.json`, cited in `EVALUATION.md` §2 |
| `retrieval/evaluate.py` | Produced `retrieval/eval_results.json`, cited in `EVALUATION.md` §3; documented in README's run sequence |

### 3.2 No removable evidence files — this is the significant negative finding

Eleven tracked files under `models/` are cited nowhere **by filename**. The
tempting conclusion is that they are stale. They are not: each is the **primary
record** behind a number that is cited. Verified by matching their contents
against the summaries, not by reading their names:

| File | Evidence it backs |
|---|---|
| `metrics_ridge_*.json` | Phase-1.3 ablation numbers in `region_comparison.md` — `+0.0844`, `−0.1961` and others match exactly |
| `metrics_lstm_small_*.json` | the small-LSTM row of the same table, plus `0.9999` / `1.0237` in the log |
| `training_history_rajasthan.json` | Phase 1.1's diagnostic retrain. Contains exactly **9 epochs**, loss **0.872 → 0.844** — the precise figures the log quotes for "train_loss dropped only 3%" |
| `training_history_lstm_small_*.json` | the Phase-1.3 loss curves behind "val_loss dipped below the climatology benchmark on 28 of 30 epochs" |
| `heat_metrics_*.json` | numbers appearing verbatim in `heat_region_comparison.md` (`−0.0525`, `−0.1610`, `−0.1631`), which README cites |
| `test_forecast_plot_*.png` | Phase 1's own Definition of Done artifact |

**Recommendation: keep every one.** The `models/` directory is not bloated; it
is the raw record underneath the published summaries. Deleting a file whose
numbers appear in a cited `.md` would leave the summary standing with nothing
behind it — precisely the "cited to a source that does not contain it" failure
this project caught in Phase 3 and has been guarding against since.

---

## 4. Found a real bug — flagged, **not fixed**

### `/health` can report an API key that does not work

**Where:** `api/app.py:216-221`, and the same pattern at `scripts/setup.py:161-163`.

```python
env_file = fconfig.REPO_ROOT / ".env"
key_present = env_file.exists() and "API_KEY" in env_file.read_text(...)
```

**The defect:** this is a substring search for `API_KEY` anywhere in the file's
text. It returns `True` for all of these:

- a commented-out line: `# GEMINI_API_KEY=old-key`
- an empty value: `GEMINI_API_KEY=`
- an unrelated provider's key: `OPENAI_API_KEY=sk-...`

In each case `/health` reports `api_key_present: true` and `status: ok`, and then
`POST /report` fails, because the real resolver — `get_api_key()` — checks the
three specific variable names for a non-empty value and raises otherwise.

**Severity:** a health check that passes without examining the thing it claims to
check. That is the exact failure shape this project has now caught four times —
the Heat-1.1 baseline that scored against the wrong target, the Phase-3 vacuous
grounding pass over zero numbers, the Phase-5 percent/fraction false negative,
and now this. It is my own Phase-6 code, and it is the least defensible instance
of the four, because the correct implementation already existed and was simply
not called.

In `scripts/setup.py` the same check has a milder but still wrong effect: with a
malformed `.env` the ChromaDB build step runs and fails partway instead of being
skipped up front with a clear message.

**The fix, not applied:**

```python
try:
    from retrieval.embed import get_api_key
    key_present = bool(get_api_key())
except Exception:
    key_present = False
```

**Not fixed here, per the brief** — bugs are flagged, not silently patched, the
same posture Phase 5 took with the grounding checker's `12%` false negative. Say
the word and it is a five-line change in two files plus a test asserting
`/health` agrees with `get_api_key()`.

---

## 5. Requirements files — assessed, **not consolidated**

Current state, and it is worse than when the brief was written because Phase 6
added two more:

| File | Pins | Purpose |
|---|---|---|
| `requirements-phase1.txt` | 11 | forecasting + tensorflow |
| `requirements-phase2.txt` | 5 | retrieval |
| `requirements-phase3.txt` | 2 | orchestrator |
| `requirements-phase6.txt` | 3 | fastapi/uvicorn/httpx |
| `requirements-api.txt` | 16 | **standalone** runtime set for the Docker image |

**Phases 4 and 5 added no new dependencies** — verified by AST-extracting every
third-party import in the repo and diffing against all five files. Two gaps
surfaced, both minor:

- **`pydantic`** — imported directly by `api/schemas.py`, declared nowhere. It
  installs as a FastAPI transitive dependency so nothing breaks today, but
  relying on someone else's pin for something you import directly is fragile.
  **Recommend declaring it.**
- **`python-dotenv`** — imported by `retrieval/embed.py`, declared nowhere. This
  one is genuinely optional: the import sits in a `try/except ImportError` with a
  hand-rolled `.env` parser as fallback, so the code works without it. **Recommend
  leaving it undeclared** and adding a one-line comment saying the omission is
  deliberate.

**Why I did not consolidate.** The brief permits it, and a single
`requirements.txt` is tempting. But `requirements-api.txt` is not a subset — it
is a deliberate, load-bearing *exclusion*: it omits TensorFlow (~600 MB) and
matplotlib because those are imported only by `train.py`, `lstm_small.py` and
`evaluate.py`, the LSTM path that has no skill and serves no forecast. Collapsing
everything into one file would either put TensorFlow back into the Docker image
or quietly break LSTM retraining.

**Recommended shape — two files, not one and not five:**

| File | Contents |
|---|---|
| `requirements.txt` | everything needed to **run** the system and its tests — today's `requirements-api.txt` plus `pytest` and `pydantic` |
| `requirements-training.txt` | `-r requirements.txt` plus `tensorflow` and `matplotlib`, for re-running the LSTM ablations |

That change touches `README.md`, `SETUP_FROM_CLEAN.md` and the `Dockerfile`, so
it is a small coordinated edit rather than a file move. **Needs your decision.**

---

## 6. `SETUP_FROM_CLEAN.md` — already written

The brief's §3 asked for this document. It was written during Phase 6, before
this audit ran, and it covers everything asked: exactly what a fresh clone lacks,
which command regenerates each artifact, and whether the ChromaDB store is
portable.

Confirming the two specific questions the brief raised:

- **Is the Chroma store path portable?** Yes. Paths are computed at import time
  from `__file__` (`retrieval/config.py`), and the persisted store was scanned
  byte-wise for baked-in absolute paths — **zero hits**. A store built at `E:\…`
  works after cloning to any other drive or directory.
- **What actually breaks on a fresh clone?** Not ChromaDB. The real breakage is
  that `forecast_drought_risk()` loads `scaler_*.joblib` and `spi_params_*.joblib`
  at request time, and **neither `t1_model` nor `recursive` writes them** — both
  call `prepare_dataset(..., save=False)`. Only `train.py` and `evaluate.py`, the
  LSTM path, persist them. On the development machine those files existed purely
  as a leftover side effect of Phase-1 LSTM training. `scripts/setup.py` step 3
  fixes this, and regeneration was verified **byte-identical** by SHA-256.

No second version of this document was written.

---

## Summary of recommendations

| # | Finding | Recommendation | Needs your call? |
|---|---|---|---|
| 1.1 | `outlooks_for()` dead | Remove | no — clearly safe |
| 1.2 | Two stale `.gitignore` entries | Remove | no — match nothing |
| 2.1 | Skill formula written six times | Extract a shared helper, as its own change with before/after numbers | **yes** |
| 2.2 | API-key detection done three ways | Collapse onto `get_api_key()` | **yes** (with §4) |
| 2.3 | Test-side structural stripper | Leave as is | no |
| 2.4 | Grounding reuse | Verified correct, no action | no |
| 3.1 | Three importer-less modules | Keep — all cited evidence producers | no |
| 3.2 | Eleven "uncited" evidence files | Keep all — primary records behind cited summaries | no |
| 4 | `/health` can report a non-working key | **Fix** — flagged, not applied | **yes** |
| 5 | Five requirements files | Consolidate to two | **yes** |

Nothing was changed by this audit.
