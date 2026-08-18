# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 1.6: Add the Indian Ocean Dipole (IOD) as a second exogenous feature

**Why this reopens a formally closed phase:** Phase 1.5 declared the Forecasting Agent closed with an explicit "no Phase 1.6" rule. This phase is a deliberate, narrow exception — not a re-tuning of anything already tested. IOD is a real, literature-documented driver of the Indian monsoon and drought variability, independent of ENSO, and it has never been in the feature set. Every other angle (target definition, architecture, site, horizon method) has already been tested exhaustively across Phases 1-1.5; this is the one genuinely untested predictor. Treat this the same as the Heat Stress Agent's SPI-3 cross-feature test: one bounded, single-hypothesis phase, report honestly, no chaining.

**Scope discipline:** this phase adds exactly one thing — the IOD feature — to the existing, already-validated pipeline. Do not touch the target definition, the model type, the regions, or the horizon methodology. If it helps, keep it. If it doesn't, remove it and say so plainly.

---

## 1. Fetch the Dipole Mode Index (DMI)

- Source: NOAA PSL, same family as the existing ONI fetch — `https://psl.noaa.gov/data/timeseries/month/data/dmi.had.long.data` (a CSV variant also exists at `dmi.had.long.csv` if easier to parse; use whichever parses more reliably, but inspect the raw file directly before trusting either, the same defensive discipline Phase 1.1 used for ONI — do not assume a header/missing-value format without checking).
- Monthly values, degrees Celsius, coverage from 1870 to near-present — slice to the project's 1980-2024 range.
- **Sanity-check against known events before trusting the parser** (same pattern as ONI's Dec 1997 / Dec 2010 check):
  - November 1997 ≈ **+1.55** (strong positive IOD, coincided with the 1997-98 El Niño)
  - October 2016 ≈ **-0.84** (negative IOD)
  - November 2019 ≈ **+1.78** (one of the strongest positive IOD events on record)
  - If the fetched values don't land close to these, stop and debug the parser before proceeding — do not train on a misparsed feature.
- Save as `models/iod_series.parquet`, alongside the existing `models/oni_series.parquet`.

## 2. Add it as a feature — nothing else changes

- Add `iod` (and optionally `iod_lag1`, mirroring how `oni` is used) to the existing `FEATURES` list in both the Drought Agent's joint 3-horizon model and the dedicated t+1/t+2/t+3 models from Phase 1.4/1.5.
- Do not remove `oni` — the point is to test whether IOD adds information *beyond* what ENSO already provides, not to replace it. If IOD and ONI turn out to be highly correlated in this data (check and report the correlation), say so — that would explain a small or zero marginal benefit even if IOD itself is meaningful.

## 3. Re-run the existing evaluation exactly as before — same rigor, same windows

- Re-run the dedicated t+1 model selection process (lookback ∈ {12, 24, 60}, alpha grid, same as Phase 1.4 Section 3) with `iod` added, across the same four windows (A/B/C/D) and both regions.
- Also re-run t+2/t+3 direct models from Phase 1.5 with `iod` added.
- Report **before/after**, directly comparable to the existing numbers in `region_comparison.md`:

| Region | Horizon | Skill without IOD (existing) | Skill with IOD (new) | Change |
|---|---|---|---|---|
| rajasthan | t+1 | +0.2622 | ? | ? |
| rajasthan | t+2 | +0.0766 | ? | ? |
| rajasthan | t+3 | -0.0145 | ? | ? |
| barmer | t+1 | +0.2053 | ? | ? |
| barmer | t+2 | +0.0438 | ? | ? |
| barmer | t+3 | -0.0489 | ? | ? |

- Use the same 4-window robustness check as Phase 1.4 — a single-window improvement is not enough to adopt the change; it needs to hold up the same way the original t+1 signal did.

## Decision rule

- If IOD **improves** mean test skill at a horizon **and** the improvement holds in at least 3 of 4 windows (same bar as every prior confirmation in this project): keep it, update `forecast_drought_risk()` to use the IOD-augmented model for that horizon, update `region_comparison.md` and `PROJECT_LOG.md` with the new numbers.
- If IOD makes **no meaningful difference or hurts**: report that plainly, drop the feature, keep the existing Phase 1.5 model as-is. This is a legitimate, useful negative result — "we checked IOD, a real documented driver, and it didn't add anything beyond ENSO for these two specific sites" is itself worth recording, not a failure.
- Either way: **this closes the question for real this time.** No Phase 1.7. If IOD doesn't help, the Drought Agent's forecasting ceiling (validated 1-month, weak 2-month, none at 3-month) is the final, thoroughly-tested answer.

## Definition of Done

- [ ] `models/iod_series.parquet` fetched, sanity-checked against the three reference values above
- [ ] Correlation between `iod` and `oni` in this data reported (context for interpreting the result either way)
- [ ] Dedicated t+1/t+2/t+3 models re-run with IOD added, same 4-window process as Phase 1.4/1.5
- [ ] Before/after comparison table completed and reported honestly
- [ ] Decision made per the rule above and applied consistently (either adopted with updated tool output, or explicitly rejected and reverted)
- [ ] `region_comparison.md` and `PROJECT_LOG.md` updated with the final outcome
- [ ] `.gitignore` extended for the new IOD artifact file

## When done

Report the before/after table, the IOD-ONI correlation, and the final decision. Whatever the outcome, this is the last word on the Drought Agent's feature set.
