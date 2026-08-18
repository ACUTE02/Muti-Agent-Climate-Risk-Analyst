# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Heat Stress Agent — Phase 1.1: extremes-focused target + a real cross-risk feature

**Why this phase exists:** Heat Phase 1 tested one specific hypothesis (monthly *mean* Tmax anomaly, heat-only features) and got a clean, honest null. Before accepting "no skill" as final, two genuinely untested, well-motivated ideas remain — not re-tuning the same thing, two different hypotheses:

1. A monthly *mean* dilutes short heat-wave spells (5-10 days) into a 30-day average. An extremes-focused target might retain signal the mean threw away.
2. Heat Phase 1 deliberately excluded drought/rainfall features to keep the test clean. Land-atmosphere feedback (dry soil → less evaporative cooling → hotter days) is documented in the climate literature and was never tested here.

This is a **one-shot, bounded phase** — same discipline as every prior phase in this project. Report honestly whatever it shows, do not chain into further tuning.

---

## 1. Two new candidate targets (build both, evaluate both, keep whichever if either works)

Keep Phase 1's `heat_anomaly` (monthly mean) as the reference point — do not discard it, report the new targets alongside it.

**1a. `heat_extreme` — max single-day Tmax anomaly within the month.**
- For each month, take the single hottest day's Tmax, standardize it the same way as Phase 1 (train-only calendar-month climatology of *daily* Tmax — note this is deliberately the daily climatology, not the monthly-mean climatology Phase 1 had to switch to, because this target is itself a daily-scale quantity).
- Check its distribution the same way Phase 1 checked `heat_anomaly` (skew, kurtosis, tail percentiles vs. normal) — report honestly, don't assume it behaves the same way.
- This is still continuous, so it plugs into the exact same Ridge pipeline as Phase 1 — no new modeling machinery needed.

**1b. `heatwave_day_count` — monthly count of IMD-criteria heat wave days (already computed in Phase 1 for the operational indicator; now also use it as a forecast target).**
- This is zero-inflated (most months outside April-June are 0) — say so explicitly and handle it honestly:
  - Report what fraction of train months are zero.
  - A skill score against *overall* climatology (which will itself be close to zero for 8 of 12 months) can look artificially good or bad — also report performance restricted to the pre-monsoon months (April, May, June) where the count is actually usually nonzero, as a second, more meaningful cut.
  - Ridge is not the natural model for a non-negative count, but keep methodology consistent with the rest of this project: try Ridge first (predicting a continuous approximation, clip negative predictions to 0 at evaluation time), report R²/RMSE/skill honestly including where clipping helps or hides problems. Do not reach for a Poisson/count model in this pass — that's a real methodology upgrade, save it for a follow-up only if Ridge on this target shows enough promise to justify it.

## 2. New feature: antecedent dryness (SPI-3)

- Add `spi3` (already computed and saved by the Drought Agent, same region, same dates) and `spi3_lag1`, `spi3_lag3` as additional predictor columns in the heat feature set, for both new targets **and** re-run the original `heat_anomaly` target with this feature added, for a clean before/after comparison.
- This is a real cross-agent dependency — document it plainly in code and in `PROJECT_LOG.md`: the Heat Stress Agent now reads a Drought Agent artifact. Guard it with a test that fails loudly if the SPI-3 file is missing or misaligned in dates, rather than silently producing NaNs.
- Report whether adding it changes anything, for all three targets (`heat_anomaly` with SPI-3, `heat_extreme`, `heatwave_day_count`) — most likely case is it helps some, all, or none; report whichever is true.

## 3. Evaluation — same rigor as every prior phase

- Same split as Phase 1 (train 1980-2015, val 2016-2019, test 2020-2024), same regions, same Ridge/alpha-grid/lookback-grid process.
- Same three-way baseline comparison Phase 1 established: vs. climatology, vs. persistence, and now also report **vs. Phase 1's original `heat_anomaly` result** as a fourth reference point, so it's clear whether the new targets/features are actually an improvement or just a different flavor of null.
- Same labels: `>= +0.1` validated, `0 to 0.1` weak/directional, `<=0` no skill.
- For `heatwave_day_count`, additionally report a plain-language sanity check: for a specific recent hot month (e.g. May 2024, both regions — already known from Phase 1's sanity check to have real heat wave days), what did the model predict vs. what actually happened?
- **Winter sanity check, for all three targets:** Rajasthan/Barmer Tmax essentially never approaches the 40°C plains threshold in Dec-Feb, so both the observed and predicted values for these months should sit near zero (near-zero `heat_extreme`/`heat_anomaly`, `heatwave_day_count` = 0). Confirm this explicitly with a reported table for a couple of recent winters — don't just assume it; if a target or model produces a nontrivial predicted/observed heat signal in winter, that's a bug to flag, not a result to report as-is.

## Stopping rule

Three hypotheses tested this phase: extreme-day target, count target, SPI-3 feature (and their combinations). Whatever the honest result:
- If **any** combination clears the +0.1 bar and replicates sensibly (do not need the full 4-window robustness check yet — that's a justified follow-up only if something here looks real, same sequencing as the Drought Agent's Phase 1.3 → 1.4 → 1.5 progression) — report which one, and stop iterating beyond this phase.
- If **nothing** clears it — report that plainly too. At that point, three independent target definitions (mean, extreme, count) and two feature sets (heat-only, heat+drought) will have been tried. That is a legitimate, thorough basis to call Heat Stress forecasting genuinely unskillful with the available zero-cost data, keep only the operational IMD heat wave counter, and move on. Do not open a Phase 1.2 chasing this further.

## Explicitly out of scope for this phase — a conditional follow-up only

An idea worth recording rather than testing yet: splitting each month into two 15-day halves and predicting each half's extreme separately (a finer-resolution, staged approach, similar in spirit to the Drought Agent's recursive/direct comparison). **Do not build this in this phase.** Standard ML/experimental discipline applies here the same way it did for the Drought Agent's Phase 1.3 → 1.4 → 1.5 progression: establish whether a signal exists at the simpler, coarser resolution first; only add resolution/complexity once there's a real baseline signal to refine. Adding this now, before any of Phase 1.1's three hypotheses have shown a signal, would be over-engineering against noise. If (and only if) `heat_extreme` or another target here clears the skill bar, the half-month split becomes a well-justified next question for a follow-up phase — record that trigger condition in `PROJECT_LOG.md` when this phase closes, whichever way it lands, so the idea isn't lost.

## Definition of Done

- [ ] `heat_extreme` built, train-only, distribution checked and reported
- [ ] `heatwave_day_count` built as a target (reusing the existing operational computation), zero-inflation reported, pre-monsoon-only cut reported separately
- [ ] `spi3`/`spi3_lag1`/`spi3_lag3` merged in as features, with a loud test guarding date alignment against the Drought Agent's artifacts
- [ ] All three targets × {heat-only features, heat+SPI-3 features} evaluated at t+1/t+2/t+3, both regions, vs. climatology, persistence, and Phase 1's original result
- [ ] `models/heat_region_comparison.md` extended with this phase's full table (not replacing Phase 1's numbers — append)
- [ ] `PROJECT_LOG.md` updated honestly with the outcome, whichever way it lands
- [ ] Explicit final statement: does anything about Heat Stress forecasting clear the bar, or is the null result now well-established across three target definitions and two feature sets
- [ ] Winter (Dec-Feb) sanity check reported for both regions — predicted and observed values near zero, confirmed not assumed
- [ ] `PROJECT_LOG.md` records the 15-day-split idea and its trigger condition (only pursue if a target here clears the skill bar), whichever way this phase lands

## When done

Report the full comparison table across all target/feature combinations, the SPI-3 cross-agent dependency test result, and a plain final verdict on whether Heat Stress forecasting works with this data or not. This is the last round on this question either way — no Phase 1.2.
