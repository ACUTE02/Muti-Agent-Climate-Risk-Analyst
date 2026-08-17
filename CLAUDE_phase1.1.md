# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 1.1: Fix SPI methodology + add ENSO exogenous feature

This supersedes nothing structural from Phase 1 — the repo layout, leak-free split, scaler discipline, model architecture, and LangGraph tool schema all stay. This phase fixes one real methodology bug and adds one new feature to give the model an actual chance at skill. Do not touch Retrieval/Orchestrator/Synthesis/Crop/API — still out of scope.

**Ruling on your two Phase 1 questions, for the record:**
1. **Window boundary rule — your fix stands, it was correct.** The original §6 "no window crosses a partition boundary" was unsatisfiable given a 60-month window vs. a 48-month VAL split, and was the wrong rule anyway. The real requirement (keep, it's already in your `test_no_leakage.py`): a window's *target* must fall inside its assigned partition, and a window's *inputs* must never include a date at or after that target's date. Reaching back into an earlier partition for input history is fine — that's just past data, identical to what a live deployment would have.
2. **SPI/"Severe never fires" — confirmed real bug, fixed below.** Not a calibration nicety; the z-score-of-rainfall proxy isn't real SPI and it's a defensible root cause of the flat-line prediction (skewed one-sided target pushes an MSE-trained network toward the bulk of the distribution).

---

## 1. Replace the SPI calculation with real SPI-3 (McKee et al. 1993 method)

Real SPI is not a z-score of raw rainfall. It's computed by fitting a gamma distribution to *accumulated* rainfall per calendar month, then transforming through the normal CDF — this is what makes it approximately standard-normal with usable tails in both directions, and it's what the tech spec is actually citing when it names "SPI."

**Step 1 — 3-month accumulation (causal).** Add a new column, distinct from the existing `roll3_mean` feature:
```python
df["rainfall_roll3_sum"] = df["rainfall_mm"].rolling(3, min_periods=3).sum()
```
This is what "SPI-3" means — a 3-month accumulation ending at each month, matching the tech spec's "90-day horizon" / kharif-season framing (SPI-1 on raw monthly rainfall is a different, noisier index — don't use it as the target).

**Step 2 — fit gamma parameters per calendar month, TRAIN ROWS ONLY** (same Rule A discipline as `month_stats` in Phase 1):
```python
# for each calendar month m in 1..12, using only TRAIN partition rows:
vals = train_df.loc[train_df.index.month == m, "rainfall_roll3_sum"].dropna()
q_m = (vals == 0).mean()                          # probability of a dry 3-month window
nonzero = vals[vals > 0]
alpha_m, _, beta_m = scipy.stats.gamma.fit(nonzero, floc=0)   # MLE, location fixed at 0
```
Save `{month: {"q": q_m, "alpha": alpha_m, "beta": beta_m}}` to `models/spi_params.joblib` — fit on train only, exactly like `month_stats.joblib`.

**Step 3 — transform (apply the TRAIN-fit params to train/val/test alike):**
```python
def to_spi3(x, month, params):
    p = params[month]
    if x <= 0:
        h = p["q"]
    else:
        h = p["q"] + (1 - p["q"]) * scipy.stats.gamma.cdf(x, a=p["alpha"], scale=p["beta"])
    h = min(max(h, 1e-6), 1 - 1e-6)                # clip before ppf to avoid ±inf
    return scipy.stats.norm.ppf(h)
```
This produces the new `spi3` column, which **replaces `spi` as the model TARGET**. Also keep `spi3` in the input `FEATURES` list (its own lagged/autoregressive signal is legitimate input, same principle as the existing lag features) — drop the old naive `spi` column entirely, but keep `anomaly` (the existing 1981–2010 baseline z-score) as-is, it was never the buggy part.

Updated `config.py`:
```python
FEATURES = [
    "rainfall_mm", "spi3", "anomaly", "month_sin", "month_cos",
    "rainfall_mm_lag1", "rainfall_mm_lag3", "rainfall_mm_lag6", "rainfall_mm_lag12",
    "roll3_mean", "roll12_sum", "oni",             # oni added in Section 2
]
TARGET = "spi3"
```

**Required test — `tests/test_spi_gamma_fit.py`:**
- Assert `spi_params.joblib` was fit using only rows within the train date range (mirror the existing `month_stats` leakage test pattern).
- Assert the resulting `spi3` series for the *train* period has mean ≈ 0 and std ≈ 1 within a loose tolerance (±0.3) — sanity-checks the gamma fit actually normalized the distribution instead of silently failing.
- Assert `spi3` reaches below −1.5 for at least one historical month in the dataset (confirms the "Severe" flag is reachable — this was impossible before the fix and must be possible after).

---

## 2. Add ENSO (Oceanic Niño Index) as an exogenous feature

Local rainfall history alone has weak 1–3-month forecast skill for South Asian monsoon variability — this is well documented, not a modeling failure. ENSO phase is the best-known, freely available predictor: El Niño winters reliably correlate with weak Indian monsoon / elevated drought risk.

**Fetch it defensively — do not assume the file format, verify it.** Try, in order:
1. `https://psl.noaa.gov/data/correlation/oni.data` — NOAA's standard PSL ASCII timeseries format (typically: header line with start/end year, then one row per year of 12 monthly values, a trailing missing-value sentinel like `-99.9`, plus metadata lines at the end). **Inspect the raw downloaded text yourself before writing a parser** — don't hardcode column positions from memory or assumption.
2. If that source or format doesn't work: `https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/` — a seasonal table indexed by overlapping 3-month season codes (DJF, JFM, FMA, …) per year. If you use this source, map each season to its center month using the standard convention (DJF→Jan, JFM→Feb, FMA→Mar, …, NDJ→Dec) to build a monthly series.

**Mandatory validation before trusting the parsed series — write `tests/test_enso_sanity.py`:**
```python
# Well-established, citable NOAA values — if these don't hold, the parser is wrong, stop and re-inspect the raw file.
assert oni_series.loc["1997-12"] > 2.0     # peak of the strong 1997-98 El Niño
assert oni_series.loc["2010-12"] < -1.0    # strong 2010-11 La Niña
```
Do not proceed to retraining until this test passes — an unvalidated ENSO parse is worse than no ENSO feature, since a wrong parse silently injects noise or leakage-shaped artifacts.

Merge onto the monthly climate dataframe by (year, month) — no train/val/test asymmetry needed here: ONI is computed by NOAA from global SST reanalysis, entirely independent of the local rainfall series being modeled, so using its published value directly at every date carries no leakage risk (unlike `month_stats`/`spi_params`, which are statistics *of this project's own target series* and must be train-only).

One forward-looking note, not a Phase 1.1 blocker: NOAA's most recent 1–2 months of ONI are provisional and get revised — irrelevant for this historical backtest (1980–2024), but worth a code comment for Phase 6 (live inference will need to handle a possibly-revised or missing latest-month value).

---

## 3. Retrain and re-evaluate

Keep the model architecture, hyperparameters, split windows, and training callbacks **exactly as they were in Phase 1** — the only things changing are the target (`spi` → `spi3`) and the feature set (+`oni`, −old `spi`). This isolates whether the fix helped, rather than conflating it with an architecture change.

Re-run evaluation (Phase 1 §8) and overwrite `models/metrics.json` and `models/test_forecast_plot.png`. Report the new RMSE/MAE/R²/Skill Score per horizon + averaged, same format as before — honestly, even if it's still not great. If Skill Score is still ≈0 after this fix, that's a legitimate, reportable finding (and would point toward needing IOD as a second exogenous feature, or a longer lookback — flag it, don't force a better number).

---

## 4. Updated Definition of Done

- [ ] `models/spi_params.joblib` exists, fit on train rows only (asserted by test)
- [ ] `spi3` reaches below −1.5 somewhere in the historical record (Severe flag is reachable)
- [ ] `models/oni_series.parquet` (or equivalent cached copy) exists, with `test_enso_sanity.py` passing against the two known historical ENSO events
- [ ] Model retrained on the new target/feature set, `models/metrics.json` and `models/test_forecast_plot.png` regenerated
- [ ] `forecast_drought_risk()` tool still returns the exact schema from Phase 1 §9 (only its internal SPI values are now real SPI-3)
- [ ] Update `spi_to_risk_score()` / flag thresholds in `tool.py` if the new SPI-3 distribution's practical range differs meaningfully from the old proxy's — check empirically against the train-set SPI-3 distribution rather than assuming the old thresholds still fit
- [ ] `pytest tests/ -v` passes, including the two new tests
- [ ] Commit to git once this passes — nothing has been committed yet per your last report; this is a good, coherent checkpoint to do it

## 5. When done

Report back: new metrics.json contents, confirmation both new tests pass with what values they asserted, and the new forecast plot. That review gates Phase 2 (Retrieval Agent).
