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

## Phase 2 — Retrieval Agent (RAG)
*Not started.*

## Phase 3 — Orchestrator + Synthesis
*Not started.*

## Phase 4 — Crop Impact Agent
*Not started. Scope updated above — must be generic across whichever risk types exist, not drought-only.*

## Phase 5 — Evaluation Suite
*Not started.*

## Phase 6 — API, Container, Deploy
*Not started.*

## Heat Stress Agent (second risk type)
*Not started — queued to begin after Phase 1.3 closes out.*
