# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Heat Stress Agent — Phase 1 (second risk type)

**Context:** the Forecasting (Drought) Agent is closed as of Phase 1.5 — see `PROJECT_LOG.md` for the full record. This phase starts the second independent risk type, per the standing scope decision: at least 2 risk types, Heat Stress chosen because it reuses the existing Open-Meteo source, the same two regions, and the same pipeline pattern.

**Carry over these lessons from the Drought Agent rather than re-discovering them (see `PROJECT_LOG.md` Phases 1–1.5 for the full story):**
1. A naive proxy target (plain z-score on a skewed variable) silently produced a flat-line, no-skill model for two full phases before the fix was traced to the target definition itself, not the model. Build the *real*, distribution-checked target from day one this time — Section 1 below exists specifically to avoid repeating that cycle.
2. The original 123k-parameter LSTM was ~300 parameters per training example and mildly hurt performance versus a much smaller model. Do not default to a large LSTM here. Start directly with Ridge (linear) as the primary candidate, per Phase 1.3's finding that linear beat both LSTM variants at short horizons on this data volume. Only try an LSTM if Ridge shows genuine signal and a nonlinear model might extract more of it — do not build one speculatively.
3. Every train-only statistic (means, stds, normals, outlier bounds) must be scoped to the *current* train partition, not a fixed date range — Phase 1.4/1.5 found a real leak exactly here when windows were parametrized. If this phase ever adds multiple historical windows later, apply that lesson from the start instead of re-finding it.
4. Report whatever the numbers say, including "no skill." Four of six phases on the Drought Agent ended in "no skill, here's why" before Phase 1.3/1.4 found the one real result. That is normal, not a failure state.

---

## 1. Define the Heat Stress target properly (do this before any modeling)

Do **not** use a plain mean-Tmax z-score as the primary target without checking it first — that is the exact mistake Phase 1 made with drought. Instead:

**1a. Fetch daily max temperature.** Extend the existing Open-Meteo historical fetch (same endpoint already used for precipitation, same two regions/coordinates already in `REGIONS`) to also pull `temperature_2m_max` (and `temperature_2m_min`, cheap to also have) for the same 1980–2024 date range. This is one additional variable on an existing call, not a new integration.

**1b. Build the primary regression target: standardized monthly Tmax anomaly.**
- Compute a train-only calendar-month climatology: mean and std of daily Tmax for each calendar month (Jan..Dec), fit on the train partition only — same discipline as `month_stats` for drought.
- Monthly target value = (monthly mean of daily Tmax − climatological mean for that calendar month) / climatological std for that calendar month.
- **Check the distribution before trusting it**, the way Phase 1.1 checked SPI-3 against theoretical percentiles: plot/report train-set skewness and kurtosis, and compare empirical tail percentiles (5th/95th) against a standard normal's. Tmax is generally much closer to normal than rainfall, so this step is expected to pass quickly — but confirm it, don't assume it. If it's meaningfully skewed, note that honestly and consider a simple transform (log, Box-Cox) before proceeding — do not silently ship a skewed "z-score" the way Phase 1's first SPI attempt did.
- Name this target `heat_anomaly` in code and outputs.

**1c. Build a secondary, non-modeled operational indicator: monthly heatwave day count.** This is for the tool's risk flags, not the regression target — keep the two separate, do not conflate an operational threshold count with a continuous forecast target.
- IMD's real heat wave definition (verified against `mausam.imd.gov.in` sources), adapted for a single grid point instead of "2 stations in a subdivision" (state that adaptation explicitly in code comments — it is a real, documented simplification, not hidden):
  - Plains threshold applies to both Rajasthan and Barmer (neither is a hill station).
  - A day qualifies as a **heat wave day** if: actual Tmax ≥ 45°C, **or** (Tmax ≥ 40°C **and** Tmax − normal_Tmax ≥ 4.5°C).
  - A day qualifies as a **severe heat wave day** if: actual Tmax ≥ 47°C, **or** (Tmax ≥ 40°C **and** Tmax − normal_Tmax ≥ 6.4°C).
  - `normal_Tmax` = train-only climatological mean Tmax per calendar day-of-year (or per calendar month if day-of-year is too noisy with the available history — pick whichever is stable, state the choice). Reuse this project's existing 1981–2010 baseline period for the normal, for consistency with the anomaly baseline already built for drought — do not introduce a second, different normal period without a reason.
  - IMD's real rule also requires ≥2 consecutive qualifying days before calling it an actual heat wave (not just a hot day) — implement this too: report both "heat wave days in month" (raw qualifying-day count) and "had ≥1 heat wave spell this month" (boolean, requires the 2-consecutive-day rule).
- Source: [IMD heat wave criteria via NDMA](https://ndma.gov.in/Natural-Hazards/Heat-Wave) and [IMD FAQ](https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf) — cite these in code comments/docs, not just here.

## 2. Features

Reuse the existing causal feature-engineering pattern (lag features, rolling means — all causal, no `center=True`, no two-sided decomposition), applied to `heat_anomaly`:
- `heat_anomaly` lag-1, lag-2, lag-3, lag-12
- rolling mean of `heat_anomaly` over trailing 3 and 6 months (causal)
- `month_sin`, `month_cos` (exactly known in advance, same as drought)
- `oni` (already fetched for drought — reuse the same series, do not refetch). Do not assume it helps; the link between ENSO and Indian pre-monsoon heat is weaker and less established in the literature than ENSO-monsoon-drought. Test it, let the ablation decide, same as every other feature in this project.
- Do **not** include SPI-3 or rainfall features as predictors here unless a later phase explicitly wants a joint drought+heat model — keep this first pass a clean, independent heat signal test, matching how Barmer was added as an independently-variable comparison in Phase 1.2.

## 3. Modeling — start where Phase 1.3 ended up, not where Phase 1 started

- Primary candidate: **Ridge regression** on flattened lookback windows, alpha grid `{0.1, 1, 10, 100, 1000}` chosen by validation MSE — same process as `forecasting/baseline_ridge.py`, reuse that code path rather than rewriting it.
- Try lookback ∈ {12, 24, 60} months, picked by validation skill — same process as Phase 1.4 Section 3. Given Phase 1.4's finding that the drought signal was short-term persistence, do not assume the same is true for heat; let validation choose.
- Predict 3 horizons jointly first (t+1, t+2, t+3), same as the project's original setup, to get a first honest read across all three — do not pre-narrow to t+1 only before measuring.
- Only build a small LSTM (16 units, lr=1e-4, generous patience, same as `forecasting/lstm_small.py`) if Ridge shows skill_score > 0 at any horizon and there's a reason to think a nonlinear model could do better — do not build one speculatively "just in case," per lesson 2 above.
- Split: chronological, same convention as the original Drought Agent (train through ~2015–2019 depending on final choice, val, test 2020–2024) — pick specific years and state them, do not leave the split implicit.

## 4. Evaluate honestly

- Skill Score = `1 - RMSE_model / RMSE_climatology`, identical definition used throughout this project.
- Also compute a **persistence baseline** (predict next month's `heat_anomaly` = this month's) up front this time, not as an afterthought — Phase 1.4 only added this after the fact and found it mattered. Report Ridge-vs-persistence directly alongside Ridge-vs-climatology.
- Apply the same `+0.1` → "validated", `0 to 0.1` → "weak/directional", `<=0` → "no skill" labels already established, for consistency across both risk types.
- Report per-horizon (t+1/t+2/t+3), not just an average — the Drought Agent's whole story was in the per-horizon breakdown, don't average it away here either.
- Both regions, side by side, in a `models/heat_region_comparison.md` file analogous to `region_comparison.md`.

## 5. Tool integration

Add `forecast_heat_stress_risk(region)` alongside the existing `forecast_drought_risk(region)`, same response shape:

```python
return {
    "region": region,
    "predicted_heat_anomaly": [t1, t2, t3],
    "horizon_confidence": [
        {"horizon": "t+1", "skill_score": <measured>, "method": "ridge"|"lstm", "label": <derived>},
        ...
    ],
    "heatwave_days_this_month": <int, from the operational indicator, not the model>,
    "had_heat_wave_spell": <bool>,
    "risk_flags": [...],
}
```

Only serve horizons that actually clear "no skill" territory with something meaningful in `risk_flags`; a `"no skill — shown for context only"` label is fine to still return, matching how the Drought Agent handles t+3.

## Stopping rule

This is a first pass, not open-ended. If Ridge shows no skill at any horizon after a proper target and a fair feature set (this phase's whole point is to not repeat Phase 1's naive-target mistake), report that plainly and stop — do not chain into open-ended tuning. If it shows skill, report it honestly with the same rigor as the Drought Agent's Phase 1.3/1.4 (does it replicate on more than the current test window? — that becomes a follow-up phase, not this one, same sequencing discipline as before: get the first honest read, then decide if a validation phase is warranted).

## Definition of Done

- [ ] `temperature_2m_max`/`temperature_2m_min` fetched and merged into the existing pipeline, both regions
- [ ] `heat_anomaly` target built train-only, distribution checked (skew/kurtosis, tail percentiles vs. normal) and reported, not assumed
- [ ] Operational heat wave day count built per the adapted IMD criteria above, cited to source
- [ ] Ridge baseline (and small LSTM only if warranted) evaluated at t+1/t+2/t+3, both regions, against climatology **and** persistence
- [ ] `models/heat_region_comparison.md` written with the full honest picture
- [ ] `forecast_heat_stress_risk()` implemented and wired to whichever model/horizon combination the measurements actually favour
- [ ] Tests: leak-free target construction (train-only stats), heat wave day logic against known hot dates as a sanity check (e.g. verify May/June 2016 Phalodi-area extreme heat period registers as a heat wave day if in range — use whatever known extreme date the region's own data confirms), `.gitignore` extended for new artifacts
- [ ] `PROJECT_LOG.md` updated with this phase's result, honestly, whatever it turns out to be

## When done

Report back the per-horizon, per-region skill table (Ridge vs. climatology vs. persistence), the target's distribution check, and the operational heat wave day counts for a recent year as a sanity check. Same rigor, same honesty, same stopping discipline as the Drought Agent.
