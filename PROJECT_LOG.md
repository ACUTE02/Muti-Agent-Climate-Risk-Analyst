# Project Log — Multi-Agent Climate Risk Analyst

Running record of what was built, what was decided, what went wrong, and how it was resolved — updated at every phase transition and every real scope decision. This is the "walk me through your project" answer key for interviews, and the source of truth if a later phase needs to know why an earlier one made a particular call.

---

## Phase 0 — Setup (Aug 17, 2026)

- Environment: Python **3.11.9** chosen over the also-installed 3.10.11 (EOL Oct 2026, one month away) and 3.14.5 (not yet supported by TensorFlow 2.21). Sits inside every library's supported range for the whole project (TensorFlow, LangGraph, LangChain-core, ChromaDB, google-genai).
- Repo: `E:\Muti-Agent-Climate-Risk-Analyst`, git initialized, no commits yet by design — held until the codebase reaches a clean, reviewed checkpoint.
- Architecture-level decisions locked in before coding started: Gemini 2.5 Flash (not 1.5 Pro — not in current free tier) as the LLM; Hugging Face Spaces as primary deploy target (AWS EC2 dropped — free-tier policy changed Jul 2025); ChromaDB baked into the Docker image at build time (no zero-cost host offers persistent disk).

## Phase 1 — Forecasting Agent, first pass (Aug 17, 2026)

**Built:** Leak-free LSTM pipeline on Open-Meteo data for Rajasthan (Jaipur centroid, lat 26.9/lon 75.8), 1980–2024. 60-month input window predicting SPI at t+1/t+2/t+3. Fixed two known bugs carried over from an earlier prototype (`E:\LSTM`): scaler fit before train/test split, and two-sided `seasonal_decompose` causing look-ahead bias. 22 tests passed.

**Problem hit:** No skill vs. climatology (Skill Score ≈ −0.004, R² negative, flat-line prediction).

**Root cause found:** the "SPI" target was a naive rainfall z-score — skewed, one-sided, and mathematically incapable of reaching the "Severe" (< −1.5) threshold. This was a genuine methodology bug (real SPI requires a gamma-distribution fit), not just a modeling choice, and it's also a subtler leakage risk the original tech spec's own reference code carried (`month_stats` computed on the full series instead of train-only).

**Decision:** fix the target methodology and add an exogenous predictor before accepting a result, rather than shipping a null model. → Phase 1.1.

## Phase 1.1 — Real SPI-3 + ENSO exogenous feature (Aug 17, 2026)

**Built:** Replaced the naive SPI proxy with real SPI-3 (McKee 1993 method — gamma distribution fit per calendar month, train-only, transformed through the normal CDF; handles zero-rainfall months via a mixed q+gamma model). Added NOAA's Oceanic Niño Index (ENSO indicator) as a feature — fetched defensively (raw file format inspected before parsing, not assumed) and validated against two known historical events (Dec 1997 El Niño = +2.39, Dec 2010 La Niña = −1.54, both confirmed against the raw data) before trusting the parser. 35 tests passed.

**Result:** SPI-3 confirmed properly standard-normal (train mean +0.005, std 0.998, min −3.18 — Severe now reachable; empirical tail percentiles 14.8%/5.9% vs. theoretical 15.9%/6.7%, a good match). Skill Score still ≈0 (+0.0011 avg). RMSE dropped (1.331 → 1.068) but this was mostly a target-rescaling effect, not skill appearing — stated plainly rather than presented as a win.

**Problem hit again:** the fix alone didn't produce skill. Diagnosed with real numbers rather than guessing: ONI's actual lead-0..3 correlation with local SPI-3 measured at only +0.09 to +0.11 overall, ≈0.00 in monsoon months (Jun–Sep) alone. Ran a diagnostic retrain capturing the full loss curve: train_loss dropped only 3% across 9 epochs (0.872→0.844, still below the ~1.0 pure-mean baseline — the model does fit some in-sample autocorrelation, likely from SPI-3's own lag features) while val_loss was best at epoch 1 and rose monotonically every epoch after. Conclusion at the time: the in-sample pattern the model finds does not generalize — a predictor-limitation result, not a fixable training bug. (This conclusion was revisited and tested properly in Phase 1.3 below, rather than just accepted.)

## Phase 1.2 — Second region: Barmer, and a clean 3-way comparison (Aug 17, 2026)

**Built:** Added Barmer (Thar Desert, lat 25.75/lon 71.38) as a second, fully independent region — chosen from a researched shortlist of four drought-prone candidates (`candidate_regions.md`) specifically because it's a genuinely harsher, more drought-variable site than Jaipur. Jaisalmer was ruled out for this purpose after research showed Rajasthan's desert belt trending wetter in recent years, most pronounced there. Same pipeline, same architecture, same hyperparameters as Rajasthan's Phase 1.1 run — nothing tuned per region, by design, so the comparison isolates one variable: does a harsher site produce more forecast skill under an identical setup? Rajasthan's Phase 1.1 model was not retrained, only renamed — re-evaluation reproduced its metrics byte-identically, confirming no regression. 83 tests passed.

**Result:** Barmer confirmed as the harsher site (SPI-3 record min −4.75 vs. Jaipur's −3.18; 17.1% probability of a bone-dry 3-month window in December vs. Jaipur's 5.7%) — genuinely more real drought signal available. Despite that, same flat-line prediction, same training pathology. Skill Score −0.0053 (Rajasthan: +0.0011) — both at/below zero.

**Problem still not resolved:** three independent angles (target definition, exogenous predictor, site choice) now agreed on "no skill," but the user correctly pushed back that "no amount of tuning would help" had never actually been tested — only inferred from one architecture's behavior. That pushback was right and led to Phase 1.3.

## Phase 1.3 — Architecture ablation: is it really a data problem, not a model problem? (in progress)

**Why:** the Phase 1.1 diagnosis that "tuning wouldn't help" was an inference, not a tested fact — the original LSTM has ~123,000 trainable parameters against only 358 training sequences (over 300 parameters per example), which is itself a plausible independent cause of the flat-line failure, separate from whether real signal exists in the data.

**Built (CLAUDE_phase1.3.md, dispatched):** a one-shot, bounded ablation — (1) a Ridge (linear) regression baseline on the same flattened features, to test whether *any* signal exists at all, independent of deep-learning failure modes; (2) a deliberately much smaller LSTM (16 units, single layer, lr=1e-4, more patience) to test whether the original model was simply too large/fast to train stably on this little data. Both evaluated on both regions, compared honestly against the existing LSTM(128→64) and climatology numbers in an extended `region_comparison.md`. Explicit stopping rule set: report whichever way it lands, don't iterate further after this.

**Status:** awaiting results.

---

## Scope Decision — Multi-Risk-Type Expansion (Aug 17, 2026)

The original tech spec described 5 climate risk types (Drought, Flood, Heat Stress, Agricultural, Coastal/Sea Level), but the Phase 0–6 build plan had implicitly narrowed to Drought only, without this being explicitly surfaced or decided — caught when the user asked directly "what about the other risk types?"

**Decision:** expand to at least 2 independent risk types — Drought (in progress) plus **Heat Stress**, chosen as the next addition specifically because it reuses infrastructure already built for Drought (same Open-Meteo source already provides temperature data, same regions, same pipeline pattern — smallest incremental cost of the three unbuilt risk types). Flood Risk and Coastal/Sea Level remain explicitly out of scope for now — to be documented in the README as "future work," not silently dropped. Sequencing: Heat Stress work starts only after Phase 1.3 (Drought ablation) is reviewed and closed out — not in parallel.

**Design principle set for Crop Impact Agent (Phase 4):** it must be built generically, not hardcoded to Drought/SPI only. It should consume whichever risk signals exist in the system's shared state at the time (currently Drought; Heat Stress once built; potentially Flood later) and, per crop/region/time, determine which risk factor is actually dominant before calculating yield loss — rather than assuming drought is always the cause. This was the user's own architectural suggestion and fits the existing LangGraph shared-state design without rework.

**Also flagged, not yet resolved:** a complete real-world "Agricultural Risk" would need Flood data too (excess rain also damages crops, not just drought) — acknowledged as a known limitation of the current (drought-only, soon drought+heat) Crop Impact Agent scope, to be revisited if Flood Risk ever gets added.

## Phase 1.3 — Architecture ablation (Aug 18, 2026)

**Why:** Phase 1.1 concluded "predictor limitation, not architecture" from one architecture's behaviour. That was an inference, not a tested fact. This phase tested it with two pre-committed variants — a Ridge linear baseline and a deliberately small, slow, patient LSTM(16) at lr 1e-4 — one shot, no iterating on results.

**Built:** `forecasting/baseline_ridge.py` (multi-output Ridge on flattened 60×12 = 720-feature windows, alpha chosen from the fixed grid {0.1, 1, 10, 100, 1000} by validation MSE — alpha=10 won for both regions) and `forecasting/lstm_small.py` (1,907 trainable parameters against 358 training sequences, vs the main model's ~123k). `evaluate.py` refactored so all three variants score through one `score_predictions()` function — the comparison is only meaningful if the numbers are computed identically. 83 tests still pass.

**Result — the Phase 1.1 claim was partly wrong, and the correction matters.** Phase 1.1 stated validation never beat the climatology baseline "even once". True for the big model; **false for a small one.** The LSTM(16)'s val_loss dipped below the climatology benchmark on **28 of 30 epochs at Jaipur** (best 0.9999 vs 1.0237) and **37 of 40 at Barmer** (0.8153 vs 0.8338). Small, systematic, not a fluke. The small model also beat the big one on test at Jaipur (RMSE 1.055 vs 1.068, skill +0.0132 vs +0.0011). So capacity *was* mildly hurting — 300+ parameters per training example was as unreasonable as it looked.

**But it does not rescue the forecast.** Best averaged skill across all six model/region combinations is **+0.0194** (Ridge, Barmer), far under the +0.1 threshold committed to in advance. Everything still sits at the climatology baseline.

**The one genuine positive:** Ridge at Barmer reaches **skill +0.1314 and R² +0.244 at t+1 alone**, decaying to +0.003 at t+2 and −0.067 at t+3; Ridge at Jaipur shows the same shape (+0.084 at t+1, −0.196 by t+3). Real one-month-ahead signal exists and a *linear* model extracts it better than either LSTM. Caveat stated in `region_comparison.md` and repeated here: that is one horizon out of 18 numbers in the table, on 58 test windows — it could be noise, and it was not the pre-registered metric.

**Conclusion, four angles now agreeing on the shape of the problem:** target definition (1.1), exogenous predictor (1.1), site choice (1.2), and architecture (1.3) all say the 3-month horizon carries no exploitable signal in these predictors. The refinement 1.3 adds is that the *horizon*, not just the predictors, is a binding constraint — and that the honest follow-up, if there is one, is one month ahead and linear, not three months ahead and deep. Stopping here per the phase's own stopping rule.

**Not yet done:** git commit still held pending review — this is the intended checkpoint covering Phases 1 → 1.3.

## Phase 1.4 — Is the t+1 signal real? (Aug 18, 2026) — **first genuine positive result**

**Why:** Phase 1.3 surfaced one number above the pre-set +0.1 bar: Ridge, Barmer, t+1 only, skill +0.1314. One result from one fixed test period. Before building on it, check whether it replicates elsewhere in the record.

**Step 1 — rolling-window check.** Re-ran the *same* Ridge pipeline (nothing new — same features, target, alpha grid) on four independent windows, t+1 only, both regions. Each window refits its own `month_stats`, `spi_params` and scaler on its own train partition, so B/C/D are as leak-free as A.

| Region | A (2020–24) | B (2015–19) | C (2010–14) | D (2005–09) | positive | mean |
|---|---|---|---|---|---|---|
| rajasthan | +0.0844 | +0.0163 | +0.0425 | −0.0082 | **3/4** | +0.0337 |
| barmer | +0.1314 | +0.1359 | +0.0354 | +0.0419 | **4/4** | +0.0861 |

**Verdict: confirmed.** Barmer positive in 4 of 4 (the rule required ≥3), Rajasthan 3 of 4. Not a lucky slice — though note the magnitudes outside window A/B are modest, and window B (+0.1359) is where the effect is strongest.

**Step 2 — dedicated t+1 model.** Ridge trained on t+1 alone rather than jointly with two horizons that show no skill. Lookback selected from {12, 24, 60} months by **mean validation skill across all four windows**; test sets never consulted for selection. 12 months won clearly at both regions (+0.306/+0.283 val skill vs +0.151/+0.159 for 60) — the signal is short-term persistence, not long seasonal memory, which is why the 60-month window the whole project had been using was actively diluting it.

| Region | lookback | A | B | C | D | positive | mean test skill |
|---|---|---|---|---|---|---|---|
| rajasthan | 12mo | +0.1594 | +0.3681 | +0.2303 | +0.3092 | **4/4** | **+0.2667** |
| barmer | 12mo | +0.2132 | +0.2441 | +0.1754 | +0.2890 | **4/4** | **+0.2304** |

**Validity check (not in the phase spec, added because the result looked too good to report unexamined):** SPI-3 is a 3-month accumulation, so SPI-3 at t+1 shares two of its three months with SPI-3 at t — part of this task is nowcasting, not forecasting. Measured how much: naive persistence ("next month equals this month") earns only +0.1009 skill at Rajasthan and +0.0268 at Barmer, and the dedicated model beats persistence by +0.1867/+0.2049 mean, in 4 of 4 windows at both regions. The overlap explains a minority of the skill; the rest is real. This also explains the horizon cliff cleanly: t+1 overlaps the target accumulation by 2 months, t+2 by 1, t+3 by 0 — exactly matching where skill vanishes.

**Result:** the project's first replicated, above-threshold result — **mean test skill +0.27 (Rajasthan) and +0.23 (Barmer) at 1-month lead, positive in 4 of 4 independent windows, beating both climatology and persistence.** The honest framing is narrow and specific: a *linear* model, at *one month* lead, with a *12-month* lookback, on a target that is partly already-observed. It is not the 3-month deep-learning forecast the original spec asked for — that one genuinely has no skill, and four phases of evidence say so.

**Stopping here** per the phase's own rule: no third round of iteration. Next is Heat Stress.

## Phase 1.5 — Recursive vs. direct + honest per-horizon confidence (Aug 18, 2026) — **Forecasting Agent closed**

**Leakage bug found and fixed first.** Parametrizing the split windows in Phase 1.4 left two statistics pointing at the *standing* window instead of each window's own train partition: the anomaly baseline (fixed 1981–2010, but window D trains only to 2000 and tests 2005–2009) and the IQR outlier bounds (defaulted to 2015). Windows C and D were affected; A and B were not. Fixed by clipping the baseline to the window's train range and passing each window's own `train_end`, and locked down with a catch-all test — *delete everything after a window's test period and every feature inside it must be unchanged*, which fails if any statistic reaches forward. Phase 1.4 was re-run on the corrected pipeline; both verdicts survived, individual numbers moved in both directions (Rajasthan 3/4 → 4/4, Barmer 4/4 → 3/4; dedicated t+1 model mean skill +0.2667 → **+0.2622** and +0.2304 → **+0.2053**).

**Recursive vs. direct, both regions, both remaining horizons, four windows each:**

| Horizon | Region | Approach | A | B | C | D | mean |
|---|---|---|---|---|---|---|---|
| t+2 | rajasthan | recursive | −0.0520 | +0.0421 | +0.1304 | +0.0298 | +0.0376 |
| t+2 | rajasthan | **direct** | +0.0529 | +0.0890 | +0.0918 | +0.0728 | **+0.0766** |
| t+2 | barmer | recursive | +0.0703 | −0.0136 | +0.0093 | +0.0413 | +0.0268 |
| t+2 | barmer | **direct** | +0.0855 | +0.0224 | +0.0419 | +0.0252 | **+0.0438** |
| t+3 | rajasthan | recursive | −0.3736 | −0.2676 | −0.0390 | −0.1839 | −0.2160 |
| t+3 | rajasthan | **direct** | −0.0209 | −0.0067 | −0.0314 | +0.0011 | **−0.0145** |
| t+3 | barmer | recursive | −0.0264 | −0.1324 | −0.0941 | −0.0931 | −0.0865 |
| t+3 | barmer | **direct** | −0.0026 | −0.0852 | −0.0087 | −0.0989 | **−0.0489** |

**Direct wins all four cells** — no split decision to report. Recursion is roughly neutral at t+2 and actively harmful at t+3 (Rajasthan −0.216 vs −0.015), which is what compounding reconstruction error looks like: each step inverts a predicted SPI-3 back into implied rainfall to rebuild the lag and rolling features, and by step three the window is mostly the model's own output. The recursive path also holds ONI flat at its last observed value — stated as a limitation, not hidden; a live system with a real ENSO forecast could do better.

**Final per-horizon verdict, now reported by the tool at runtime rather than hardcoded:**

| Region | t+1 | t+2 | t+3 |
|---|---|---|---|
| rajasthan | **+0.2622 validated** (direct, 12mo) | +0.0766 weak/directional (direct, 12mo) | −0.0145 no skill (direct, 24mo) |
| barmer | **+0.2053 validated** (direct, 12mo) | +0.0438 weak/directional (direct, 12mo) | −0.0489 no skill (direct, 12mo) |

`forecast_drought_risk()` now serves predictions from the per-horizon Ridge models the measurements actually favoured — the LSTM has no skill at any horizon and no longer answers — and returns `horizon_confidence` with each horizon's measured skill score, method, lookback, and a label derived from the +0.1 bar this project has used since Phase 1.3. The t+3 label says so explicitly: *"no skill — shown for context only, do not rely on this figure."*

**Forecasting Agent, final state:** one validated result (1-month lead, linear, 12-month lookback, ~+0.21 to +0.26 skill, replicated across four independent windows, beating both climatology and persistence), one weak/directional horizon, one horizon with nothing. 121 tests. That is the whole honest picture, and it is what Phases 2–6 may cite.

---

## Heat Stress Agent — Phase 1 (Aug 18, 2026) — second risk type, no skill

**Why this exists:** second independent risk type per the standing scope decision. Reuses the Open-Meteo source, the same two regions, and the same pipeline pattern; shares no features with the Drought Agent by design.

**Built:** `heat/` package — `target.py` (climatology + IMD heat wave criteria), `dataset.py`, `model.py`, `tool.py`. Open-Meteo pull extended with `temperature_2m_max`/`min`, cached to a new daily parquet per region (the drought monthly cache gained two columns; verified the committed drought numbers are bit-identical afterwards — window-A t+1 RMSE/skill unchanged to 4dp for both regions). ONI reused, not refetched.

**Lesson 1 applied — target built properly first, not after two wasted phases.** `heat_anomaly` = standardised monthly Tmax anomaly, climatology fit train-only, **distribution-checked before any modelling**: Jaipur skew −0.049 / excess kurtosis +0.626, Barmer −0.255 / +0.758, empirical p5/p95 within 0.19 of a standard normal's, both flagged approximately normal. It also reaches both tails (min < −2, max > +2), the exact property the Phase-1 SPI proxy lacked. No transform needed. *Deviation stated:* the spec described the climatology as mean/std of **daily** Tmax; standardising a monthly mean by a daily std would divide by 3–5× too much spread and make the normality check unpassable by construction, so it is fit on the monthly-mean series — which is what makes it a genuine z-score.

**Operational indicator (kept separate from the regression target):** IMD plains heat wave criteria — Tmax ≥ 45 °C, or (≥ 40 °C and ≥ +4.5 °C departure); severe at ≥ 47 °C or +6.4 °C; spells need ≥ 2 consecutive days. `normal_Tmax` is a day-of-year climatology pooled over ±7 days across the train-clipped 1981–2010 baseline (~450 samples/day: a bare DOY mean has ~30 samples and a 0.6 °C standard error, large next to a 4.5 °C threshold; a flat monthly normal misassigns month-edge days by >2 °C during April–May warming). **Documented adaptation:** IMD requires the criteria at ≥2 stations in a subdivision; this is a single grid point, so these are indicative single-point counts, not IMD declarations. Sanity-checked against the May 2016 national-record event — both regions register heat wave days and a spell, peaking on **19 May 2016**, the actual Phalodi record date.

**Lesson 2 applied — started at Ridge, no speculative LSTM.** Joint 3-horizon Ridge, alpha grid by validation MSE, lookback from {12, 24, 60} by validation skill (Jaipur chose 24mo, Barmer 60mo).

**Result: no skill at any horizon, either region.**

| Region | Lookback | t+1 | t+2 | t+3 |
|---|---|---|---|---|
| rajasthan | 24mo | −0.0525 | −0.1610 | −0.1631 |
| barmer | 60mo | −0.0209 | −0.0172 | −0.0118 |

Validation had looked mildly promising (+0.06 to +0.11) and test did not follow — an honest validation/test gap, not a target artifact this time: the target was checked first and is near-normal.

**Persistence, computed up front as instructed, and it changes the reading.** Ridge beats persistence everywhere (+0.08 to +0.35) — but persistence is *far worse than climatology* here (−0.14 to −0.61), because monthly Tmax anomalies barely persist month to month. That is the mirror image of drought, where SPI-3's 3-month accumulation made persistence genuinely strong at t+1. So "beats persistence" is close to meaningless for heat; climatology is the only baseline that matters, and Ridge loses to it. Computing persistence up front is what made this visible immediately rather than a phase later.

**Stopped here** per the phase's stopping rule — no LSTM (the precondition, Ridge skill > 0 at some horizon, was never met), no tuning chain. `forecast_heat_stress_risk()` is implemented and returns measured per-horizon skill; every horizon is labelled "no skill — shown for context only" and every risk flag reads "not rated — no measured skill at this horizon". The heat wave day counts *are* served, since they are observations rather than forecasts. 163 tests.

**Honest project-level read:** two risk types, one validated result between them (drought, 1-month lead, linear). Four of six drought phases and this heat phase all ended in "no skill, here's why". That is the expected shape of this work, and the record shows the reasoning rather than a curated highlight.

---

## Phase 1.6 — Indian Ocean Dipole tested and rejected (Aug 18, 2026)

**Why a formally closed phase reopened:** Phase 1.5 declared the Forecasting Agent closed with an explicit "no Phase 1.6". This was a deliberate, narrow exception — IOD is a real, literature-documented driver of Indian monsoon variability, independent of ENSO, and it was the one predictor never tested. One bounded hypothesis, nothing else touched.

**Fetching the DMI, and a discrepancy worth recording.** NOAA PSL publishes exactly one DMI series (HadISST1.1); the `.data` and `.csv` endpoints are the same numbers and the other candidate URLs 404. Its amplitudes run smaller than the DMI figures usually quoted: this series gives Nov 1997 = **+1.279** and Nov 2019 = **+0.835**, where the phase spec's reference values were ≈+1.55 and ≈+1.78. That is a difference of SST product, not a parse error — verified by checking the raw bytes and the alternative endpoints before proceeding, exactly as the spec instructed. The events were therefore pinned by **sign and rank** instead of absolute value: Nov 1997 is the single highest month of 1980–2024, Oct 2019 the second highest with Nov 2019 at the 98th percentile, Oct 2016 negative. A constant scale factor is irrelevant downstream anyway, since features are min-max scaled.

**IOD–ONI correlation:** r = **+0.386** over 1980–2024 (+0.340 on train alone, Spearman +0.343). Related but far from redundant — so a null result here is not simply "IOD is ENSO in disguise".

**Method:** both variants re-run in the same process rather than comparing against numbers on disk, so the "without IOD" column comes from exactly the code that produced the "with IOD" column. Same target, model type, regions, windows, lookback grid {12, 24, 60}, alpha grid, and selection-by-validation rule.

| Region | Horizon | Without IOD | With IOD | Change | Windows improved | Adopt |
|---|---|---|---|---|---|---|
| rajasthan | t+1 | +0.2622 | +0.2389 | −0.0233 | 1/4 | no |
| rajasthan | t+2 | +0.0763 | +0.0774 | +0.0011 | 2/4 | no |
| rajasthan | t+3 | −0.0145 | −0.0305 | −0.0160 | 1/4 | no |
| barmer | t+1 | +0.2053 | +0.1933 | −0.0120 | 2/4 | no |
| barmer | t+2 | +0.0496 | +0.0392 | −0.0104 | 2/4 | no |
| barmer | t+3 | −0.0489 | −0.0757 | −0.0268 | 0/4 | no |

**Decision: rejected.** Five of six cells got worse. The one nominal gain (+0.0011 at rajasthan t+2) held in only 2 of 4 windows against the project's standing 3-of-4 bar — the same bar that confirmed the original t+1 signal. Two extra features on ~350 training sequences cost more in variance than they return in information.

**Reverted properly:** `iod` is not in `FEATURES`, and `prepare_dataset` merges the DMI only when a caller explicitly asks for those columns, so the default path is byte-identical to what Phase 1.5 measured and committed — verified: window-A t+1 RMSE/skill unchanged to 4dp for both regions after the revert. `forecasting/iod.py` and `iod_check.py` stay in the tree with `models/iod_comparison.json` as the evidence, and a test asserts the rejection cannot drift back silently.

**Note on the baseline column:** t+2 reads +0.0763/+0.0496 here versus +0.0766/+0.0438 in Phase 1.5, because Phase 1.5 scored direct models on the horizon-3 window set for like-for-like comparison against recursion, while this module scores each horizon on its own window set. Both columns here use the identical path, so the comparison stands — this is precisely why both variants were re-run rather than one being read off disk.

**This closes the Drought Agent's feature set for real.** Validated at 1 month, weak at 2, none at 3 — five angles tested (target definition, architecture, site, horizon method, exogenous predictors) and that is the final answer. No Phase 1.7.

---

## Heat Stress Agent — Phase 1.1: extremes target, count target, SPI-3 feature (Aug 18, 2026)

**Why:** Heat Phase 1 tested one hypothesis (monthly *mean* Tmax anomaly, heat-only features) and got a clean null. Two motivated ideas remained untested — a mean dilutes 5–10 day spells into a 30-day average, and land-atmosphere feedback (dry soil → less evaporative cooling → hotter days) had been deliberately excluded. Three hypotheses, one bounded pass.

**A false positive caught before it was reported.** The first run showed `heat_extreme` at skill **+0.44 (Jaipur) and +0.63 (Barmer)** — after every model in this project had sat at zero. That is a red flag, not a result, so it was checked rather than written up. Cause: `climatology_prediction()` and `persistence_prediction()` in `heat/model.py` hardcoded `config.HEAT_TARGET` instead of following the run's actual target, so `heat_extreme` (which averages ≈ +1.4) was being scored against a baseline predicting `heat_anomaly` (≈ 0). The baseline was not even predicting the right quantity. Fixed to follow `ds.target`; the "skill" collapsed to −0.28 / +0.02. **A wrong baseline is the most flattering bug available in forecasting work** — the only defence is disbelieving good news.

**Results after the fix — nothing clears the bar:**

| Region | Target | Features | Lookback | t+1 | t+2 | t+3 | mean |
|---|---|---|---|---|---|---|---|
| rajasthan | `heat_anomaly` | heat-only | 24mo | −0.0525 | −0.1610 | −0.1631 | −0.1255 |
| rajasthan | `heat_anomaly` | heat+SPI3 | 60mo | −0.0038 | −0.0026 | −0.0078 | −0.0047 |
| rajasthan | `heat_extreme` | heat-only | 60mo | −0.2883 | −0.2779 | −0.2762 | −0.2808 |
| rajasthan | `heat_extreme` | heat+SPI3 | 60mo | −0.0510 | −0.0490 | −0.0472 | −0.0491 |
| rajasthan | `heatwave_day_count` | heat-only | 12mo | +0.0220 | +0.0075 | +0.0061 | +0.0119 |
| rajasthan | `heatwave_day_count` | heat+SPI3 | 12mo | +0.0088 | −0.0141 | −0.0116 | −0.0056 |
| barmer | `heat_anomaly` | heat-only | 60mo | −0.0209 | −0.0172 | −0.0118 | −0.0166 |
| barmer | `heat_anomaly` | heat+SPI3 | 60mo | −0.0115 | −0.0011 | +0.0007 | −0.0040 |
| barmer | `heat_extreme` | heat-only | 60mo | +0.0169 | +0.0227 | +0.0217 | +0.0204 |
| barmer | `heat_extreme` | heat+SPI3 | 60mo | +0.0110 | +0.0245 | +0.0236 | +0.0197 |
| barmer | `heatwave_day_count` | heat-only | 60mo | +0.0269 | +0.0225 | +0.0215 | +0.0236 |
| barmer | `heatwave_day_count` | heat+SPI3 | 60mo | +0.0378 | +0.0213 | +0.0200 | +0.0264 |

Best cell anywhere: **+0.0378** (barmer / `heatwave_day_count` / heat+SPI3 / t+1), against a +0.1 bar.

**Hypothesis 1 — extremes target: no.** `heat_extreme` is worse than the mean at Jaipur and indistinguishable at Barmer. Its distribution check also flagged something worth recording: standardising the monthly *max* by the *daily* climatology (as the phase specified, correctly, since it is a daily-scale quantity) produces a target centred at **+1.37 / +1.44 with std ≈ 0.6** — normal in shape but nowhere near *standard* normal, because the hottest day of any month sits well above that month's daily mean by construction. The `approximately_normal` flag was tightened to separate shape from location/scale, since shape alone had called this "normal".

**Hypothesis 2 — count target: no.** 92.4% of train months have zero heat wave days (75.2% even within April–June), so skill against an all-months climatology is flattered by eight easy months a year. The April–June-only cut gives +0.005 to +0.032 — still nothing. The plain-language check is the clearest statement of the failure: for May 2024, the model predicted **0.7–0.9 heat wave days** where **8 (Jaipur) and 7 (Barmer)** actually occurred. Ridge on a zero-inflated count also produced 12–29% negative raw predictions before clipping, which is the expected mismatch of model family to data; per the phase spec, a Poisson/count model was *not* reached for, since nothing here justifies the upgrade.

**Hypothesis 3 — SPI-3 antecedent dryness: it helps, but only by making bad models less bad.** Adding it moved Jaipur's `heat_anomaly` from −0.126 to −0.005 and `heat_extreme` from −0.281 to −0.049 — large relative moves, all of them from "clearly worse than climatology" toward "equal to climatology", never past it. It is regularisation, not signal.

**Cross-agent dependency, now real and guarded.** The Heat Agent reads the Drought Agent's SPI-3 via `heat/dataset.load_drought_spi3()`, rebuilt through the Drought pipeline for the *same* window so the gamma parameters stay fit on that window's train partition. The alignment guard fired on the first run — the Drought frame starts 1981-01 (it drops 12 months for its lag-12 warm-up) against the Heat frame's 1980-01 — and was then taught to distinguish expected leading truncation (trim) from genuine drift or interior gaps (raise loudly). Four tests cover it.

**Winter sanity check (Dec–Feb, 14 test months), confirmed not assumed:** `heat_anomaly` observed −0.05 / +0.33, predicted −0.32 to +0.06 — near zero as required. `heatwave_day_count` observed exactly 0.000 in every winter month, predicted 0.03–0.26 days (Barmer/heat+SPI3 peaking at 0.97) — small but not exactly zero, the expected artifact of a linear model that cannot represent a hard floor. `heat_extreme` observed **+1.45 / +1.65** in winter, which is the construction offset described above rather than a spurious winter heat signal.

**Verdict: Heat Stress forecasting does not work with this data.** Three target definitions (mean, extreme, count) × two feature sets (heat-only, heat+drought) × two regions × three horizons — 36 measured cells, best +0.0378. The null is now well-established, not merely observed once. What survives is the **operational IMD heat wave day counter**, which is observation rather than forecast and is reliable: it correctly identifies the 19 May 2016 national-record event and May 2024's spells at both sites. `forecast_heat_stress_risk()` keeps serving those counts with every forecast horizon labelled "no skill". No Phase 1.2.

**Recorded for the future (trigger condition, per the phase spec):** splitting each month into two 15-day halves and predicting each half's extreme separately is a reasonable next question — **but only if some heat target first clears the +0.1 skill bar at monthly resolution.** It did not, so adding resolution now would be refining noise. If a future phase ever produces a genuine monthly-resolution heat signal, that is the moment to revisit the half-month split.

---

## Heat Stress Agent — Phase 1.2: interface trimmed (Aug 18, 2026)

Housekeeping, not a new result. Phase 1.1 closed the forecasting question (36 measured cells, best +0.0378 against a +0.1 bar), but `forecast_heat_stress_risk()` was still returning three no-skill prediction fields behind "do not rely on this figure" labels. Serving three flavours of a result that does not work is dead weight on a live interface.

**Changed:** the tool now returns observations only — `region`, `month`, `heatwave_days`, `severe_heatwave_days`, `had_heatwave_spell`, `max_tmax_c`, plus `forecast_available: False` and a `note` pointing at this log. The predicted fields and `horizon_confidence` were **removed**, not relabelled. It also gained an optional `month` argument so callers can ask about any historical month rather than only the latest, and it no longer loads a model, scaler or manifest at all — it reads daily observations and applies the IMD criteria.

**Deliberately untouched:** `heat/target.py`, `heat/model.py`, `heat/phase11.py`, every `models/heat_*` evidence file, and the Phase 1/1.1 log sections. The negative result was rigorously established and the code that established it still has passing tests — what got trimmed is only what the live interface serves.

`tests/test_heat_tool.py` was rewritten (not deleted) against the smaller contract, including a test asserting no forecast field can reappear. No other call sites existed — checked before changing the shape. 193 tests pass.

---

## Phase 2 — Retrieval Agent (RAG) (Aug 18, 2026)

**Built:** a bounded, curated RAG corpus over Gemini embeddings (`gemini-embedding-001`, 768-dim, correct `task_type` per side) and a locally-persisted ChromaDB store, plus a `retrieve_context()` tool that returns citations, not just text. Two indexed document types, deliberately different in purpose: Type A (domain reference — real government/institutional documents establishing authoritative definitions and methodology) and Type B (this project's own evidence — `PROJECT_LOG.md` and both region-comparison files, so the system can cite its own measured numbers instead of inventing them). A third, unindexed Type C (IMD's live seasonal and extended-range outlooks) is fetched fresh at report time and kept as whole excerpts, clearly attributed to IMD, never blended with the project's own results.

**Source verification caught a real dead link.** `https://ndma.gov.in/Natural-Hazards/Drought` and every plausible variant (`/Droughts`, `/drought`, `/Natural-Hazards`) returned a hard 404 — NDMA no longer publishes a drought hazard page at that path. Dropped per the phase's own verify-before-include rule, recorded in `retrieval/corpus_manifest.json` under `verified_dead_and_excluded` alongside its replacement, and guarded by a test so it cannot quietly reappear. Every surviving Type A PDF was checked for genuine extractable text rather than scanned-image noise before inclusion (1,227-2,044 chars/page, alpha ratio 0.74-0.81).

**Type C finding, reported honestly rather than smoothed over:** `seasonal_forecast.php` turned out to be a navigation hub, not a bulletin — its extractable text is menus plus a rotating press-release marquee, with the real seasonal outlooks living in linked PDFs. The manifest says so rather than implying a full outlook exists where it doesn't. The extended-range forecast PDF is a genuine bulletin and works as intended.

**The drought blind spot, found and closed.** The first build ran with four Type A documents, all heat/agromet-focused — the dropped NDMA drought page had left the domain-reference half of the corpus with *zero* drought sources, while the eval set happened to contain zero drought-domain queries. That combination would have reported a perfect precision score while never testing half the domain this project is actually about. Closed by adding two verified sources: **NDMA's National Disaster Management Guidelines: Management of Drought** (108pp, NIDM-hosted since NDMA's own link is dead) and **NIH Roorkee's SPI methodology paper** (10pp, documenting the same gamma-fit McKee et al. 1993 method this project implements in `forecasting/split.py`). Two eval queries were added to target them, and two tests now enforce the invariant permanently: every Type A source must be targeted by some query, and both drought sources must be present. Same family of error as the Heat 1.1 hardcoded baseline — a metric that looks fine because it never examined the thing that was missing.

**Final corpus:** 9 documents, **224 chunks** — 181 Type A across six documents (IMD Heat Wave FAQ, NDMA Heat Wave page, NDMA Drought Guidelines, NIH Roorkee SPI methodology, IMD GKMS Agromet SOP, ICAR-ATARI Jodhpur Agro-Advisory) and 43 Type B across `PROJECT_LOG.md` and both region-comparison files. Header-aware chunking for Type B (keeps a result table with the caveat that qualifies it — `+0.0766` stays attached to "weak/directional, not validated" rather than being split into a misleading fragment, pinned by a test); paragraph-packing at ~2,000 chars with 200-char overlap for Type A.

**Rate limiting was not theoretical.** Every one of the seven embedding batches hit a 429 and recovered through exponential backoff; progress is written to disk per batch, so a failed build resumes rather than restarting. Exactly the discipline the Open-Meteo fetch taught in the Heat phase.

**Evaluation (12 hand-authored queries, written before the index existed and not revised against its results):** precision@5 = **1.00** overall — 1.00 for Type A (7 queries), 1.00 for Type B (5 queries), MRR 1.00, every correct source ranked first. Run *without* the `doc_type` filter, the harder setting, since a live Orchestrator will not know in advance which half of the corpus holds an answer.

**Read that number skeptically, because a perfect score usually means an easy test — and it does here.** Nine documents with barely-overlapping vocabularies make "which document" a soft problem. What this honestly establishes is that the plumbing is correct (task_type handling, index, metadata filtering, citations), not that retrieval is robust on ambiguous queries. A stronger check was run alongside: does the retrieved *text* contain the answer, not merely the right document? `+0.0766` appears in the top-5 text for the 2-month skill query, "no skill" for the 3-month reliability query, `47`/`6.4` for the severe heat wave threshold, and "gamma" for the SPI methodology query — that is the property the Synthesis agent will actually depend on.

**Not built, deliberately:** RAGAS-style faithfulness/relevance evaluation remains Phase 5. This was a bounded first pass run once; no chunking or embedding choice was revised after seeing the eval results.

**Status:** complete. The drought Type A gap that this entry previously tracked as open is closed.


## Phase 3 — Orchestrator + Synthesis Agent (Aug 18, 2026)

**Built:** a LangGraph state graph (`parse_request → call_tools → fetch_type_c → synthesize → verify_grounding → finalise`) that routes a natural-language request across the project's three tools using Gemini 2.5 Flash **function calling** — not hand-written keyword matching — then writes the report. The synthesis prompt lives in `orchestrator/prompts/synthesis.md` so it is reviewable rather than buried in a Python string. Type C (IMD's live outlooks) is always fetched. A deterministic fallback router covers the case where function calling returns nothing, so an LLM hiccup degrades to "call the obvious tools" instead of an empty report.

**The grounding checker is the load-bearing piece, and it is deliberately not an LLM** — an LLM checking an LLM shares the failure mode being checked. `check_grounding()` extracts every number from the report and verifies it against the tool outputs and retrieved chunks. Tolerance rule: a report may round a source value to its own precision (`+0.26` against `+0.2622` passes) with a 5e-5 absolute allowance for trailing-digit noise; anything materially different is flagged. Structural tokens (years, `t+1` horizon labels, model names, list markers) are stripped first. A test asserts the module contains no LLM call at all.

**It caught a real fabrication during testing — the headline result of this phase.** Asked for a 4-month drought forecast (which does not exist), Gemini correctly declined the forecast, then explained SPI by reciting the standard McKee classification bands — *"SPI values between -0.99 and 0.99 are near normal, -1.0 to -1.49 moderately dry"* — and cited them to the NIH Roorkee SPI methodology document.

Those numbers appear **nowhere in the corpus**: not in the retrieved chunks, not in any indexed document (verified by direct search). They came from the model's training data and were attributed to a real source that does not contain them. They are also *correct in the real world*, which makes this the most dangerous class of error — plausible, well-formed, and untraceable. A careful human reviewer would very likely have accepted it.

**How the pipeline behaved:** attempt 1 flagged `-0.99` and `-1.49`; the retry prompt named them and asked for removal; attempt 2 dropped `-0.99` but kept `-1.49`. Having exhausted its one regeneration, the graph attached the unverified-figures warning banner rather than returning a clean-looking report, and logged both attempts. Exactly the designed behaviour, and the reason the banner exists instead of a silent pass. Preserved in `orchestrator/grounding_caught_sample.json`.

**Fix applied:** the prompt had a category gap — it forbade inventing *results* but said nothing about reciting *reference tables*. Added an explicit rule against stating classification bands or numeric thresholds not present in the sources, with instructions to describe SPI qualitatively (negative drier, positive wetter) when no band table was retrieved. Re-run afterwards: grounded, 19 numbers checked, zero unverified.

**Three further report-quality problems that only surfaced by actually running it**, each fixed in the prompt: skill scores were being cited to `PROJECT_LOG.md` when they came from the *tool output* (grounded but mis-attributed — it sends a reader somewhere the number is not); the report claimed "IMD's current outlook was unavailable" when both fetches had **succeeded** and were merely off-topic (a false statement about provenance); and raw floats were pasted as `0.20994198322296143`. A fourth was caught in the last run: "no skill" was glossed as *"no better than random chance"*, when the measured meaning is no better than **climatology** — a much stronger baseline. Overstating a failure is as inaccurate as understating it, so the prompt now defines the labels precisely.

**A vacuous-pass bug in my own pipeline, found and fixed.** When synthesis failed (see quota note below), the report came back empty and `verify_grounding` reported `grounded: True` with `total_checked: 0` — a metric that passes because it examined nothing, the same shape as the Heat-1.1 baseline bug and the Phase-1.4 leak. Now an empty report returns `grounded: False, report_missing: True`, and `finalise` emits an explicit "REPORT NOT GENERATED" banner naming the cause instead of an empty string.

**Live results, before the quota ran out:** all three end-to-end scenarios plus the adversarial one passed — 20 tests green including 8 live ones. Routing was correct in every case (drought-only → drought + retrieve; both-risks → both forecast tools + retrieve; impossible → drought + retrieve, no heat tool). "No skill" labels survived verbatim into the report text, IMD content was separately attributed, and the impossible request was declined explicitly for both the 4/6-month horizons and the unsupported Jaisalmer district, with no invented figures.

**Free-tier constraint, stated plainly:** gemini-2.5-flash allows **20 `generate_content` requests per day**, and each scenario costs two (routing + synthesis). Testing exhausted it. The live tests are therefore opt-in behind `RUN_LIVE_ORCHESTRATOR=1` rather than running on every suite invocation; the 11 offline checker tests — including the corrupted-report and number-fragment attacks — run always. Chat calls now retry on transient 429s with the same discipline as the embedding build, though a daily cap cannot be backed off away.

**221 tests pass, 10 skipped** (the live orchestrator scenarios).

---

## Phase 3.1 — Model deprecation (gemini-2.5-flash → gemini-3.6-flash) and a content-format bug, found and fixed (Aug 19, 2026)

**Why this happened:** re-running Phase 3's live orchestrator tests the next day (fresh Gemini API key) hit an immediate, unrelated failure — `gemini-2.5-flash` returned `404 NOT_FOUND: This model ... is no longer available to new users`. This is a real, external event (Google deprecated the model for new API keys sometime between Aug 17–19, 2026), not a bug in this project's code. Confirmed via web search: Google's current flash-tier model is **Gemini 3.6 Flash** (announced Jul 21, 2026). `orchestrator/config.py`'s `CHAT_MODEL` was updated from `"gemini-2.5-flash"` to `"gemini-3.6-flash"`.

**Free-tier assumption corrected.** The project's original architecture note assumed ~15 RPM / 1,500 RPD for the free tier. The user's actual key, checked live in Google AI Studio's rate-limit dashboard, shows every current flash model (2.5, 3, 3.6, 3.7) capped at **5 RPM / 250K TPM / 20 RPD** — Google tightened the free tier considerably since the project's Aug 17 architecture decision. This is now the number planning is done against; the standing quota note in this log and in `tests/test_orchestrator.py` should be read as "~20 requests/day", not 1,500.

**A second, real bug surfaced once the model was swapped — not a fabrication.** Re-running the live suite against `gemini-3.6-flash` produced 4 failures in `check_grounding()`, flagging hundreds of small, near-random unverified numbers (`"9", "7", "8", "6446", "8984", ...`) per report, with `total_checked` in the thousands instead of the usual ~20. Investigated rather than assumed: `graph.py`'s `synthesize()` node had `report = response.content if isinstance(response.content, str) else str(response.content)`. Gemini 3.x's LangChain integration returns `response.content` as a **list of content blocks** (not a plain string) — likely including a non-text metadata/signature block alongside the actual text block. The `str(response.content)` fallback path stringified the *entire* Python list, including that metadata block's contents, straight into what the grounding checker treated as report text. The metadata block's contents look like a long digit-heavy token, and the number-extraction regex faithfully flagged every fragment of it as an unverified "claim" — the report text itself was fine throughout; this was a text-extraction bug, not a synthesis quality problem.

**Fix applied:** added `_extract_text(content)` to `graph.py`, which walks the block list and keeps only blocks with `type == "text"`, discarding everything else, with the old plain-string case still handled for backward compatibility. `synthesize()` now calls `report = _extract_text(response.content)` instead of the previous ternary. Applied by the user directly, reviewed against the diagnosis before running.

**Verification — confirmed fixed.** Offline tests (`pytest tests/ -v`, no live calls) — 221 pass, 10 skipped, unaffected by any of the above. Live suite re-run after the model swap but *before* this fix: 17 passed, 4 failed, matching exactly the diagnosis above (all 4 failures were the same `check_grounding` false-positive pattern, not a routing or coverage failure — `test_drought_only_request_calls_the_drought_tool`, `test_both_risks_request_calls_both_forecast_tools`, `test_no_skill_label_survives_into_the_report`, `test_type_c_is_attributed_to_imd_separately`, `test_unsupported_region_is_declined_not_invented`, and `test_type_c_is_always_fetched` all passed even before the fix, since they don't depend on `check_grounding`). **After applying `_extract_text()`, the full live suite was re-run and all 21 tests pass**, including the three grounding checks and the adversarial "declines a horizon it cannot forecast" test — zero false-positive unverified numbers. One informational warning noted, not a defect: `gemini-3.6-flash` "uses fixed sampling defaults" — this project's `TEMPERATURE = 0.2` setting in `orchestrator/config.py` is silently ignored by this model. Not investigated further this session; if report tone/consistency ever becomes a concern, revisit whether a different sampling control is exposed for Gemini 3.x models.

**Lesson, consistent with this project's pattern:** a suspicious result (thousands of "unverified numbers" instead of the usual handful) was investigated by reading the actual log (`orchestrator/grounding_failures.jsonl`) rather than assumed to be either "the new model just hallucinates more" or "the checker is broken" — the same discipline as the Heat-1.1 baseline bug and the Phase-1.4 leak: read the evidence before writing the conclusion.

---

## Phase 4 — Crop Impact Agent (optional)
*Not started. Scope updated above — must be generic across whichever risk types exist, not drought-only.*

## Phase 5 — Evaluation Suite
*Not started.*

## Phase 6 — API, Container, Deploy
*Not started.*
