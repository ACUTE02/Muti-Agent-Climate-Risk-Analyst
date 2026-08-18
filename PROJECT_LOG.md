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

## Phase 4 — Crop Impact Agent (Aug 19, 2026)

**Built:** `crop_impact/` — a hybrid agent in the order the phase spec fixed. `dominance.py` decides deterministically which risk binds yield (no LLM, guarded by a test); `yield_impact.py` reads a sourced coefficient table (no LLM, no arithmetic); **one** Gemini call writes the plain-language explanation of results it is handed and is never asked for a number; the Phase-3 `check_grounding()` then verifies that text. `assess_crop_impact` is wired into the orchestrator's routable tools, so "what's the impact on wheat in Rajasthan" routes here through the same function calling as the other three.

**The sourcing result is the honest headline, and it is mostly a null.** Four candidate sources were checked properly rather than cited on faith, and three were rejected:

| Source | Verdict |
|---|---|
| ICAR-ATARI Jodhpur Agro-Advisory *(named in the spec, already in the RAG corpus)* | **Rejected.** It is full of percentages for pearl millet and wheat — +19.2% yield for advisory adopters, Rs.9909 input cost saving, 27.93% "risk aversion" — and every one is a gain from *adopting agro-advisories*, not a climate-driven loss. Citing any of them as a drought or heat impact would have been the Phase-3 misattribution error in a new place. |
| NDMA Drought Guidelines *(in corpus)* | **Rejected.** Describes the crop-loss *assessment procedure* (annewari / paisewari / girdawari) and states no crop-specific loss percentage. |
| Manual for Drought Management 2016 (DAC&FW) | **Rejected.** 202 pages, downloaded and text-extracted locally with the project's own Phase-2 `pypdf` path rather than trusted to a summary. No SPI classification table and no crop coefficient. Its prominent "33%" refers to the share of India's cropped area under 750 mm rainfall — a number that looks usable and is about something else entirely. |
| ScienceDirect geospatial pearl-millet paper | **Rejected on two counts.** The full text is paywalled (HTTP 403), and this project does not cite what it has not read; and the "20–25% yield loss" figure as surfaced describes loss that could be *saved* by switching to stress-tolerant cultivars — a gain from an intervention, not the loss caused by a drought severity. |

**One coefficient survived: wheat × heat = 5.6%**, from the Union Minister of State for Agriculture's statement to the Rajya Sabha (4 Apr 2025) on the 2021–22 NWPZ wheat season at +5.5 °C. Verified by reading the source directly. It is stored with its verbatim quote, its exposure definition, and a caveat that it is a **single anchor point, not a slope** — 5.6% at +5.5 °C does not license "1.02% per degree", and the lookup enforces a `match_band` so it is never applied outside the exposure range it was measured in.

**Three gaps are recorded rather than filled** — bajra × drought, bajra × heat, wheat × drought. The bajra × drought gap is the awkward one and is stated as such: **drought is the risk type with the validated forecast, and it is the one with no sourced coefficient.** The closest quantified findings (ICRISAT's 0.9% per day of earlier stress onset, 0.7% per 1% irrigation deficit) are conditioned on irrigation deficit under controlled trials, not on SPI-3; converting them to an SPI-3 band would have been inventing the coefficient in all but name.

**A real mismatch found by measurement, not assumed.** The first working version judged heat candidacy by IMD heat wave days — and returned "no heat" for every warm wheat season on record. Cause: IMD's plains criteria gate on Tmax ≥ 40 °C, which February and March at these sites almost never reach. Measured directly: the warmest February at Jaipur in the whole record (**2006, +5.35 °C mean Tmax departure**) records **zero** IMD heat wave days, and so does the record March of 2022 (+3.68 °C). The day counter is a summer indicator; wheat's grain-filling window is Feb–March. Judging terminal heat by it is a metric that passes because it cannot see the thing it is measuring — the same shape as the Heat-1.1 baseline bug and the Phase-3 vacuous pass. Fixed by adding a second candidacy route on monthly mean Tmax departure, with **severe set to +4.0 °C specifically to match the sourced coefficient's own band** so the coefficient is never applied outside it. The +3.0 °C moderate threshold has no source and is flagged in code as a Phase-5 calibration target, the same posture as `spi_to_risk_score`.

**What `dominant_risk()` actually decided, three real cases:**

| Request | Decision | Yield impact |
|---|---|---|
| wheat / rajasthan / **2006-02** | **heat**, severe — +5.35 °C departure, via the departure route | **5.6%**, sourced and cited |
| wheat / rajasthan / **2022-03** | **heat**, moderate — +3.68 °C | **none** — "the table's only sourced coefficient was measured at severe severity… no number rather than one scaled down from a different severity band" |
| wheat / rajasthan / **2024-03** | **none dominant** — 0 heat wave days, −0.08 °C departure | none — nothing binding to look up |

**A structural limitation this phase surfaced and did not hide.** The two risk types describe different time frames: drought is a forecast (t+1…t+3) and heat is an observation of a month that has already happened. So for a **future** month heat is always "unknown" and can never be dominant, and for a **past** month the drought forecast cannot apply. A genuine head-to-head dominance comparison is therefore reachable in unit tests with fixtures but not in a live single call — the tie-break rule (equal severity → the observation wins, because a fact outranks an estimate) is correct and tested, but rarely exercised live. Stated here rather than left for a reader to discover.

**The t+3 gate holds, and it is the phase's most important property.** A t+3 SPI-3 of −2.5 is severe on the project's own thresholds and still declares nothing, because that horizon is labelled "no skill". Five phases of evidence say that number is worthless; letting it drive a crop assessment would have laundered it into something that looks like a finding. Tested directly, alongside "weak/directional may corroborate but never decide".

**Grounding needed no changes.** `check_grounding()` was reused completely unmodified on the new output shape and passed on the first live run — 6 numbers checked, zero unverified, including the 5.6% and the +5.35 °C. A captured run is checked in at `crop_impact/narrative_sample.json`. The model reproduced the caveat unprompted ("must not be scaled or converted into a per-degree rate") and correctly attributed the figure to the Rajya Sabha statement rather than to a retrieved document.

**Quota cost, against the corrected 5 RPM / 20 RPD budget:** `assess_crop_impact` costs **one** generate_content request (the narrative; no retry loop inside the tool). A full orchestrator report that routes to it costs **3** in the best case — routing + crop narrative + main synthesis — and **up to 5** if the main synthesis needs its one regeneration. The two live crop tests cost 2 and are gated behind the existing `RUN_LIVE_ORCHESTRATOR=1`, not a second variable.

**252 tests pass, 12 skipped.** 31 of the new tests are offline and free; 2 are live.

**Honest read:** the deterministic machinery is the deliverable and it works — the dominance rule, the skill gating, the band enforcement, and the refusal to invent. What it mostly returns is "no sourced yield-impact estimate available", because that is what the literature actually supports for these crops and risks at zero cost. One real coefficient, three recorded gaps, and a system that says so plainly is the correct outcome of the phase's own stopping rule.

## Phase 5 — Evaluation Suite (Aug 19, 2026)

**Built:** `EVALUATION.md` (the consolidated scorecard) plus an `evaluation/` package with two genuinely new measurements and a test suite that holds the scorecard to the same standard the system's own reports are held to.

### 1. The scorecard, and a real inconsistency it surfaced

`EVALUATION.md` consolidates every measured result in the project, each cited to the file it was measured in. `tests/test_evaluation.py` parses the document section by section, extracts every number, and verifies it appears in that section's cited sources — **reusing `orchestrator/grounding.py`'s own matching primitives rather than writing a second regex**, since "does this number appear in that source" is exactly the problem the checker already solves.

Building it surfaced one genuine discrepancy, which is the kind of finding this phase existed to produce. **The Phase 1.4 per-window table in this log holds pre-leak-fix numbers that were never restated.** Phase 1.5 found a leakage bug affecting windows C and D, re-ran, and corrected the *means* (+0.2667 → +0.2622, +0.2304 → +0.2053) — but left the per-window figures alone. The log says rajasthan C **+0.2303** / D **+0.3092**; `models/metrics_t1_ridge.json` says **+0.2403** / **+0.2809**. Same for barmer: logged C +0.1754 / D +0.2890, actual +0.1546 / +0.2269. The Phase 1.4 entry is not wrong as a *historical record* of that run, and Phase 1.5 does say the numbers moved — but a reader skimming could quote +0.3092 as current. `EVALUATION.md` uses post-fix values throughout and documents the discrepancy explicitly rather than quietly correcting it.

### 2. The grounding checker's own precision and recall — and a real defect found

The checker is the most safety-critical piece in this project and had never been **measured**, only exercised. On a hand-labelled adversarial set of **14 reports / 40 labelled numbers**, fully offline:

| Metric | Value |
|---|---|
| Precision | **1.0** |
| Recall | **0.9286** |
| F1 | **0.963** |
| Cases fully correct | 13 / 14 |

The labelling discipline matters as much as the number: `score_case()` **raises** if a case extracts a token that is not labelled either fabricated or grounded, so the label set cannot silently drift behind the fixtures and flatter the result. It fired immediately on the first run and caught three unlabelled tokens of my own.

**A false positive that turned out to be my mislabel, not a checker error.** The first run flagged `-1.0` in the McKee-band case as wrongly-flagged. On inspection `-1.0` genuinely appears nowhere in that case's sources — it is part of the same recited band table as `-0.99` and `-1.49`. The label was wrong, not the checker; corrected, with the correction noted in the case itself rather than silently applied.

**A false negative that is a real defect, reported rather than patched.** `12%` — an invented yield percentage, exactly the Phase-4 failure mode — passes the checker. Cause, diagnosed precisely rather than guessed: an integer percentage is compared against its fraction form at the report token's own precision, so `12%` → `0.12` → rounds at zero decimals to `0.0` → matches any source containing a zero. Tool outputs are full of legitimate zeros (`heatwave_days: 0`). **Any invented integer percentage below 50% is currently accepted whenever the sources contain a zero.**

Not fixed in this phase, deliberately. Phase 5's own stated non-goals forbid tuning a measured component, and fixing it here would mean the precision/recall above no longer describe the code that was measured. The failing case stays in the set, `PERCENT_FRACTION_DEFECT` documents the likely fix, and a test asserts the defect stays *documented and failing* — so if someone fixes the checker the test fails loudly, forcing a re-measurement and a scorecard update rather than a silent improvement.

### 3. Faithfulness and answer relevance — the one LLM-judges-LLM exception

`check_grounding()` verifies every **number**; it says nothing about whether a **claim** is entailed. This measures that gap, and it is the only place in the whole project where an LLM judges an LLM's output.

**The exception is deliberate and bounded.** Everywhere else the rule is the opposite — the grounding checker and `dominant_risk()` are plain Python precisely because an LLM checking an LLM shares the failure mode being checked. It holds here because the quantity is different in kind: verifying a discrete fact is mechanical, judging entailment of a claim is not. The two are complementary, and this one is labelled lower-trust in the module docstring, in the results JSON, and in the scorecard: **where they disagree, the mechanical checker wins.** A test asserts that labelling stays present.

**Set size: 3 items, not the 10–15 the spec sketched — and the sizing was done before any test was written.** One item costs 2 calls (routing + synthesis), a third if synthesis retries, a fourth if it routes to `assess_crop_impact`, plus 1 judge call: 3–5 each. Ten to fifteen items is 30–75 calls, two to four days of quota on one key. Three fits one day with margin. This was a **single-day, single-key run with no results discarded**. A test asserts `HELD_OUT_SET_SIZE × CALLS_PER_ITEM_WORST ≤ DAILY_REQUEST_BUDGET`, so the set cannot silently outgrow the budget it claims to fit.

| Metric | Value |
|---|---|
| Mean faithfulness | **0.9867** |
| Mean answer relevance | **1.0** |
| Unsupported claims across all items | 1 |
| Mechanical grounding clean on every item | yes (27, 19 and 28 numbers checked) |
| Routing correct on every item | yes |

**The one unsupported claim is not a hallucination, and reporting it as one would be wrong.** The judge flagged the report for stating that skill score is `1 - RMSE_model/RMSE_climatology`. That formula is in no retrieved chunk — so the judge is technically right — but it comes from `orchestrator/prompts/synthesis.md`, which *instructs* the model to define the labels precisely, a rule added in Phase 3 after "no skill" was glossed as "no better than random chance". **The judge sees tool outputs and retrieved chunks but not the system prompt**, so definitions the prompt legitimately supplies score as unsupported. That is a limitation of this evaluation setup, not of the report. Recorded rather than corrected into a cleaner-looking 1.0.

**Coverage gap, stated up front in `eval_requests.json` rather than discovered later:** with three slots, heat-only is not a standalone item; it is exercised indirectly through the crop-impact item. The three chosen — drought reliability, crop impact, and a three-way impossible request (six-month horizon + unsupported district + unsupported crop) — prioritise the paths where fabrication is most likely.

### Honest read

Two of the three deliverables came back clean and one came back with a defect — which is the right ratio for a phase whose job is to look. The scorecard's own closing section lists what it does **not** claim, including that the grounding checker is complete: it has a measured, documented false-negative class, unfixed as of this scorecard. **277 tests pass, 12 skipped.** The 25 new evaluation tests are entirely offline; the live faithfulness run is a script, not a test, so the suite verifies the measurement without repeating its cost.

## Independent verification note (Aug 19, 2026)

Phase 4 and Phase 5's claims were independently checked from the Cowork session that has been reviewing this project throughout — not re-taken on trust. Specifically verified against the actual files on disk: `crop_impact/yield_impact_table.json`'s sourced coefficient and the four sources checked/rejected; the wheat×heat 5.6% figure against the original Rajya Sabha statement (fetched and quoted directly — matches verbatim); `crop_impact/dominance.py`'s two-route heat-severity logic and the `+4.0 °C` severe threshold's match to the yield table's own exposure band; `dominance_rule.md`'s t+3 gate; `evaluation/checker_eval_results.json`'s precision/recall numbers and the documented `12%` false-negative case; and the Phase 1.4 per-window discrepancy claim directly against `models/metrics_t1_ridge.json` (rajasthan window D: log says +0.3092, file says +0.2809 — confirmed exactly as reported). No discrepancies found between what was reported and what the code/data actually show.

## Phase 6 — Local API + Container (Aug 19, 2026)

**Built:** `api/` (FastAPI over the real orchestrator), `scripts/setup.py` (clone → working API in one command), `SETUP_FROM_CLEAN.md`, `Dockerfile` + `.dockerignore`, `requirements-api.txt`. **No frontend** — per the Aug 19 scope change below, that is built separately with a design tool. No deployment configuration of any kind.

### The "database disappears on a fresh clone" complaint — root-caused, and it was not ChromaDB

The reported symptom pointed at the vector store. The store turned out to be the *healthy* part:

- **ChromaDB is portable.** Paths are computed at import time from `__file__`, and the persisted store was scanned byte-wise for baked-in absolute paths — **zero hits**. A store built at `E:\...` works after cloning anywhere. It is git-ignored because it is regenerable and 5.9 MB, not because it is fragile.
- **The actual breakage is in the forecasting artifacts, and it is worse than it looks.** `forecast_drought_risk()` loads `scaler_<region>.joblib` and `spi_params_<region>.joblib` at request time. **Neither `t1_model` nor `recursive` writes them** — both call `prepare_dataset(..., save=False)`. The only modules that persist them are `forecasting.train` and `forecasting.evaluate`: the **LSTM** path. So the obvious sequence (fetch → t1_model → recursive) yields a tracked manifest, Ridge models that load fine, and a tool that dies on a missing scaler at the first drought question. On this machine those four files existed **only as a leftover side effect of Phase-1 LSTM training** — for a model that has no skill and no longer answers anything.

That is the kind of dependency that survives indefinitely on the machine where it was born and breaks for everyone else. `scripts/setup.py` step 3 fixes it directly by calling the project's own `prepare_dataset(region, save=True)`, so setting up no longer requires training a model nobody uses.

**Verified byte-identical, not merely "equivalent":** deleting all four artifacts and re-running step 3 reproduces the same SHA-256 for each (`aab2201574c82ec4`, `bfd0060532fe1e02`, `8951594869b31732`, `f2379d5dcffd8ef7`). A fresh clone gets the artifacts every phase actually measured with — a regeneration that silently produced a *different* scaler would have shifted every forecast while looking like it worked.

`--check` runs exactly what `GET /health` runs, so setup and health cannot disagree about readiness; a test asserts it.

### The API — thin wrapper, honesty as structure

`POST /report` calls `orchestrator.graph.analyse()` directly — the same function the tests exercise, with no separate demo path. The API layer only reshapes; it never re-decides.

The response **promotes this project's honesty mechanisms to structured fields**, because a client should never parse English to find out whether a figure was verified:

- **Grounding** arrives as `clean` / `warning` / `not_generated`, with the unverified numbers listed and an explanation. `not_generated` exists specifically so the Phase-3 vacuous-pass shape (zero numbers checked reported as success) cannot reappear at the API boundary.
- **Per-horizon labels stay distinct.** `validated`, `weak/directional` and `no skill …` are never collapsed server-side into one confidence number — that single line of code would undo five phases of honesty. The `reliable` boolean follows the measured *label*, not a threshold on the skill score, and a test pins it: a horizon with skill 0.99 and a non-validated label still reads unreliable.
- **Missing data is stated, not omitted.** `forecast_available: false`, "no sourced yield-impact estimate available", a no-skill horizon, and a failed IMD outlook fetch all travel as explicit flags with reasons attached.

**Quota is first-class, and the count is real rather than estimated.** A tally was added at `invoke_with_backoff` — the single chokepoint every chat call in the project passes through — so `GET /quota` reports actual usage, retries included. Exhaustion returns **429 with a specific explanation** naming the budget and what a report costs, never a generic 500 and never a 200 with an empty report. Embedding calls are deliberately excluded from the tally: they bill against a separate quota, and conflating them would misreport both.

`/health`, `/quota`, `/examples` and `/evaluation` make no LLM calls. A test asserts `/health` never touches Gemini by reading its source — the same guard as `test_checker_is_not_an_llm`. It is the endpoint a container health check hits every 30 seconds; if it ever cost a call, a 20/day budget would be gone in ten minutes.

### The image leaves TensorFlow out

`requirements-api.txt` is deliberately not the union of phases 1–3. TensorFlow (~600 MB) and matplotlib are imported **only** by `train.py`, `lstm_small.py` and `evaluate.py` — the LSTM path. Verified by loading the whole API path and inspecting `sys.modules`: only scikit-learn appears. Neither the API nor `scripts.setup` touches Keras, so the image omits it. The LSTM code and its evidence stay in the tree; they are simply not runtime dependencies of a system whose forecasts are all linear.

**Build-time secret, not an ARG.** Two of this phase's rules collide: the ChromaDB index must be baked in at build time (no zero-cost host offers persistent disk), and the API key must never be baked into the image. Building the index needs a key. Resolved with a BuildKit secret mount — an `ARG` would persist the key in image history, which is precisely what the second rule forbids. A keyless build is a supported outcome: the index step is skipped, the build still succeeds, and the image serves `/health` as `degraded`.

### Tests

**301 pass, 13 skipped.** 24 new API tests, all offline — the orchestrator is monkeypatched so response shaping, error handling and the honesty fields are tested without spending a single call. One live smoke test sits behind the existing `RUN_LIVE_ORCHESTRATOR=1`, and it treats a 429 as a *pass* condition (skip with the reported detail), because the API correctly reporting exhaustion is correct behaviour, not a failure. The live smoke test was run once and passed.

### Docker — actually built and actually run, not just written

`docker build` and `docker run` were both executed and verified end-to-end rather than assumed.

**The first build failed, on a bug in the Dockerfile's own verification step** — and it is a good one to record because it is invisible on the happy path. The line read `raise SystemExit(1) if blocking else print(...)`, which Python parses as `raise (SystemExit(1) if blocking else print(...))`. When nothing is blocking, that evaluates `print(...)` → `None` and then raises `None` → `TypeError: exceptions must derive from BaseException`. So the guard failed **precisely when everything was fine**, which is the opposite of what a guard should do. It was spotted by reading the file while the build was still running and fixed before the rebuild; the container-side setup step underneath it had already succeeded completely.

**Second build: succeeded.** Image `climate-risk-analyst:latest`, **1.42 GB** — the whole `scripts.setup` sequence ran inside the image, fetching Open-Meteo and NOAA data, fitting the runtime artifacts, training the Ridge models and embedding the corpus. The embedding build hit rate limits repeatedly during the image build (`rate limited, waiting 20s / 40s`) and recovered through the existing backoff, exactly as it does on the host — the resumable-per-batch design from Phase 2 earning its keep in a new context.

**Run: verified working.** `docker run -p 8000:8000 -e GEMINI_API_KEY=...`:

| Check | Result |
|---|---|
| `GET /health` | `status: ok`, index ready, **zero** missing artifacts, key read from the runtime env |
| `GET /examples` | all 5 examples, both regions, both crops |
| `GET /evaluation` | scorecard served, 14,362 chars, checker summary incl. the known defect |
| `POST /report` | **HTTP 200**, routing correct, grounding **clean — 23 numbers checked, zero unverified** |

The report's per-horizon fields came back exactly as designed: t+1 `+0.2053 validated reliable=True`, t+2 `+0.0438 weak/directional reliable=False`, t+3 `-0.0489 no skill reliable=False`, with a `missing_data` flag naming t+3 as unreliable. **The quota tally read `calls_used_today: 2`** for that one report — matching the predicted typical cost exactly, which is the first independent confirmation that the chokepoint tally is accurate rather than plausible.

**One honest discrepancy worth recording.** The index built *inside the image* contains **249 chunks** (181 Type A + 68 Type B) against the committed manifest's **224** (181 + 43). Nothing is wrong: Type B is this project's own evidence — `PROJECT_LOG.md` and the region comparisons — and the log has grown with every phase since Phase 2 measured it. The corpus is rebuilt from current files at build time, so a container built today indexes today's log. The Phase-2 retrieval precision figures were measured on the 224-chunk index and are **not** re-validated by this build. The host manifest was checked afterwards and is untouched at 224.

**Nothing was deployed and no platform config exists.** No Hugging Face frontmatter, no Cloud Run YAML, no registry push, no credentials. The API key reached the build only through a BuildKit secret mount and reached the runtime only through `-e`; it is in no layer and no file. The project is exactly as portable as before — just packaged.

## Codebase audit (dispatched, not a numbered phase)

`CLAUDE_audit.md` dispatched (Aug 19, 2026) — a report-only review of Phases 1-5 for dead code, redundant evidence files, and duplicated logic, plus a specific investigation into a reported "the local database disappears after a fresh clone/download" problem (almost certainly the git-ignored `data/`/`models/`/ChromaDB paths not being obviously regenerable) — to be documented in a new `SETUP_FROM_CLEAN.md`. Explicitly scoped as find-and-report, not delete-and-restructure: nothing gets removed without the user reviewing the list first.


## Phase 6 scope change (Aug 19, 2026)

`CLAUDE_phase6.md` updated (from Cowork session) — frontend is no longer part of Phase 6. User is building the frontend separately with a design tool (Claude Design). Phase 6 is now backend-only: FastAPI (`/report`, `/health`, `/evaluation`, new `/examples`), Docker, setup automation. Section 2 kept as a reference-only "API contract" note (grounding status, honesty labels, missing-data flags as explicit structured fields; CORS enabled for local frontend dev) so the API still matches what an external frontend needs, but no UI files should be created under this phase. This change was made before Claude Code (the dispatched session) reported back on Phase 6, so it should apply to that work going forward — if that session already produced a frontend, it needs to be dropped per this update.

Codebase audit (`CLAUDE_audit.md`) status unchanged — still dispatched, no results/`AUDIT_REPORT.md` on disk yet as of this entry. Timing of that session is not controlled from here; check the Claude Code session directly for progress.


## Frontend design iteration (Claude Design tool, Aug 19, 2026)

Frontend is being built by the user directly in Claude Design (not by Claude Code — see Phase 6 scope change above). Cowork session verified the API contract against `api/schemas.py` before advising on the mockup:

- `ReportRequest` actually has 4 optional filters: `region`, `risk_types`, `month`, `crop` — confirmed by reading `api/schemas.py` and `api/app.py` directly, not assumed from the spec.
- `month` requires `"YYYY-MM"` format (e.g. `"2006-02"`), not a bare month name — a month-name-only dropdown ("August") would send a value the backend can't parse correctly.
- `crop` (`"bajra"` / `"wheat"`) is a real, meaningful filter — advised adding it to the mockup, and moving `month` to an optional/advanced section (with a year field, not just a month-name list) since most live queries don't need it and it's mainly used for historical crop-impact examples (e.g. the Feb 2006 wheat example).
- Region/crop dropdowns in the mockup were corrected earlier to match the system's real scope: only `rajasthan` and `barmer` regions (both in Rajasthan state), only `bajra` and `wheat` crops — the design tool had initially invented Punjab/Maharashtra/Karnataka examples that the backend does not support.

Audit (`AUDIT_REPORT.md`) status re-checked directly on disk at this point — still not present. `CLAUDE_audit.md` (the dispatched spec) is the only audit-related file on disk; the audit task has not produced output yet.


## Codebase audit — complete, independently verified (Aug 19, 2026)

`AUDIT_REPORT.md` landed. Report-only as instructed — nothing deleted/changed except the report itself; test suite still `301 passed, 13 skipped` per the report (not re-run independently this pass).

**Verified directly against files on disk from the Cowork session (not taken on trust):**
- `outlooks_for()` in `retrieval/outlooks.py:103` — confirmed zero callers anywhere in the repo (grepped). Genuinely dead, safe to remove.
- The two stale `.gitignore` lines (`*.chroma/`, `reports/eval/*.json.tmp`) — confirmed present at lines 7-8; confirmed they don't match the project's real paths (actual store is `retrieval/chroma_store/`, already separately gitignored).
- **Real bug confirmed byte-for-byte:** `api/app.py:221` does `"API_KEY" in env_file.read_text(...)` — a plain substring search on `.env`, so it reports `api_key_present: true` for a commented-out line, an empty value, or an unrelated key like `OPENAI_API_KEY`. Confirmed the correct helper (`get_api_key()`) already exists in `retrieval/embed.py` and just isn't called from `api/app.py` or `scripts/setup.py:161`. Not fixed yet, per the audit's own "flag, don't fix inline" discipline — same as Phase 5's grounding-checker defect.
- Requirements: confirmed 5 files present (`requirements-phase1/2/3/6.txt`, `requirements-api.txt`) as the report describes.

**Two decisions still open, not yet made:**
1. Extract the skill-score formula (`1 - rmse/rmse_baseline`, duplicated 6x across `evaluate.py`, `t1_model.py`, `heat/model.py`, `heat/phase11.py`) into one shared function — needs a before/after numeric check since every call site produced a published number.
2. Split requirements into `requirements.txt` (run+test) + `requirements-training.txt` (adds TF/matplotlib) instead of the current 5 files.

Not committed. Uncommitted tree is now three units of work: Phase 5, Phase 6, and this audit.


## Phase 7 dispatched (Aug 19, 2026) — audit fixes + real frontend, bundled

User approved all 4 audit-fix items in one go and asked to bundle them with the frontend build so Claude Code does both together. `CLAUDE_phase7.md` dispatched, covering two independent parts (kept separately labelled for PROJECT_LOG/commit purposes):

- **Part 1 (audit fixes):** the `api_key_present` substring-search bug (fix using existing `get_api_key()`), removing the dead `outlooks_for()` + 2 stale `.gitignore` lines, consolidating the 6x-duplicated skill-score formula (**explicit numeric before/after check required — any mismatch must be reported, not silently resolved**, since every call site's number is already published), and collapsing the 5 requirements files into `requirements.txt` + `requirements-training.txt`.
- **Part 2 (real frontend):** the user's Claude-Design export (`design_export/` — now committed into the repo: `AgriRisk Query Assistant.dc.html`, `support.js`, `_ds/nocturne/`) is a visual prototype only — its "Ask" button doesn't call any backend, it shows hardcoded fake data (`CANNED` object, which still references Punjab/Maharashtra/Karnataka — regions this project doesn't support, a leftover from the design tool's earlier mistake). Spec instructs building a real `frontend/` (plain HTML/JS/CSS) matching that visual design but wired to the actual API (`/report`, `/examples`, `/quota`, `/health`), with no fake data shipped.

Cowork session's own review of the export (inspected directly, not assumed): confirmed it uses a proprietary `x-dc`/`{{ }}` template runtime requiring `support.js`, and confirmed the `ask()` handler only flips local state — no `fetch()` call anywhere in the exported code.
