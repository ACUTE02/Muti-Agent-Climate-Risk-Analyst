# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 1.4: Is the Barmer t+1 signal real, or one lucky test window?

**Why this phase exists:** Phase 1.3 found one genuinely positive number — Ridge (linear) at Barmer, 1-month-ahead only, Skill Score +0.1314, R² +0.244 — against a pre-set +0.1 threshold. But it's one result from one fixed test period (2020–2024, 58 windows). Before investing in specializing a model around it, check whether it holds up across *other* historical periods too, or whether 2020–2024 just happened to be a favorable slice. This is a validation step, not a tuning step — do not touch the model, features, or architecture in this phase.

---

## 1. Rolling-window robustness check

Re-run the existing Ridge pipeline (same alpha-selection process, same features, same SPI-3/ENSO setup — nothing new) across **four independent historical windows** instead of just the current one, for **both regions**, evaluating **t+1 only** (that's the horizon in question):

| Window | Train | Val | Test |
|---|---|---|---|
| A (existing, Phase 1.3) | 1980–2015 | 2016–2019 | 2020–2024 |
| B | 1980–2010 | 2011–2014 | 2015–2019 |
| C | 1980–2005 | 2006–2009 | 2010–2014 |
| D | 1980–2000 | 2001–2004 | 2005–2009 |

For each window: refit `month_stats`/`spi_params` (per Rule A — train-only, same discipline as every prior phase, just re-applied to that window's own train partition), refit the Ridge alpha grid on that window's train/val split, evaluate t+1 skill on that window's test set. ONI is a shared external series — no refitting needed for it, just slice to the relevant dates.

Save results to `models/rolling_window_check.json` — one row per (region, window) with RMSE, R², Skill Score, matching the existing metrics format.

## 2. Verdict — read the numbers plainly

Report a small table: 8 skill scores (2 regions × 4 windows). Then answer directly:

- **If Barmer's t+1 skill is positive in at least 3 of the 4 windows** — that's real, recurring signal, not a fluke. Proceed to Section 3 (specialize a dedicated model).
- **If it's positive in only window A (the original one) or scattered/inconsistent** — that supports "noise from one favorable test period." Report this plainly in `region_comparison.md` and `PROJECT_LOG.md`, and stop here — don't specialize a model around something that doesn't replicate.

Do not round up "positive in 2 of 4" to "mostly confirmed" — be as strict here as every prior phase has been about not overselling a marginal result.

## 3. Only if confirmed: build a dedicated 1-month-ahead model

If (and only if) Section 2 confirms a real, recurring signal:

- Build a model — start with Ridge again, since linear already outperformed both LSTMs at this horizon — trained to predict **t+1 only**, not jointly with t+2/t+3. A model that only has to get one horizon right, instead of splitting capacity across three (where two show no skill), may do better than the joint model's t+1 slice did.
- Also worth trying: a shorter lookback window than 60 months for this specific model — if the signal is short-term persistence rather than long seasonal memory, a 6- or 12-month window might work as well or better with less noise from irrelevant older history. Try lookback ∈ {12, 24, 60} months, pick by validation performance across the same 4 windows from Section 1, not just window A.
- Check Rajasthan too at t+1 — it was close (+0.0844 in the original test), might clear the bar with a dedicated model even if it didn't jointly.

## Stopping rule

This phase has two possible endpoints, both are acceptable outcomes, and either one ends the phase:

1. **Signal doesn't replicate** → document plainly, close out, move to Heat Stress as planned. This is not a failure — catching a false lead before building on it is exactly the discipline this project has kept throughout.
2. **Signal replicates and a dedicated model is built** → report its cross-window performance honestly (not just its best window), update the resume-facing framing to include this as a genuine result, then move to Heat Stress.

Do not add a third round of iteration after this regardless of outcome — same discipline as Phase 1.3.

## Definition of Done

- [ ] `models/rolling_window_check.json` — 8 (region × window) results
- [ ] Explicit verdict stated: signal confirmed or not, with the count (e.g., "positive in 3/4 windows")
- [ ] If confirmed: dedicated t+1 model, its cross-window results, lookback choice justified by validation data not guesswork
- [ ] `region_comparison.md` and `PROJECT_LOG.md` updated either way
- [ ] `.gitignore` extended for any new evidence files

## When done

Report back the 4-window table and the verdict. Whichever way it lands, this is the actual final gate before committing Phases 1 through 1.4 and moving to Heat Stress.
