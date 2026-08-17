# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 1.2: Add a second region (Barmer) — multi-region support

This is a scope addition, not a rewrite: Rajasthan/Jaipur stays exactly as it is (same architecture, same hyperparameters, same features, same SPI-3/ENSO pipeline from Phase 1.1 — untouched). Barmer is added alongside it as a second, independently-trained region using the identical pipeline. Keeping everything else unchanged is deliberate: it turns this into a clean comparison — does a genuinely more drought-variable location (per `candidate_regions.md`) show any more forecast skill than Jaipur did, under the exact same setup? That answer matters more than the raw metrics for either region alone.

Barmer centroid: **25.75°N, 71.38°E**. Caveat carried over from `candidate_regions.md`: Rajasthan's desert belt has a documented recent trend toward more rainfall — don't be surprised if 2020–2024 looks less classically "drought" than Barmer's historical reputation suggests. Report the numbers honestly either way, same discipline as Phase 1/1.1.

---

## What changes, file by file

**`forecasting/config.py`** — add the second region to the existing registry (this was already designed to be extensible, per Phase 1 §3):
```python
REGIONS = {
    "rajasthan": {"lat": 26.9, "lon": 75.8, "label": "Rajasthan (Jaipur centroid)"},
    "barmer":    {"lat": 25.75, "lon": 71.38, "label": "Barmer (Thar Desert)"},
}
DEFAULT_REGION = "rajasthan"   # unchanged
```

**`forecasting/fetch_data.py`** — no code change needed (already takes `lat`/`lon` as arguments). Run it once per region, saving to region-named files: `data/raw/rajasthan_raw.parquet` (already exists, don't re-fetch) and `data/raw/barmer_raw.parquet` (new).

**`forecasting/clean.py` / `split.py`** — no logic changes (already operate on whatever dataframe they're given). Run once per region, output to `data/processed/{region}_clean.parquet`. **New naming requirement** (Phase 1 left this ambiguous — fixing it now): always suffix processed files with the region name, not a bare `clean.parquet`, so both regions can coexist on disk.

**ONI data — do not refetch or duplicate.** ENSO is a global index, not region-specific. Reuse the existing `models/oni_series.parquet` for both regions' feature sets.

**`forecasting/train.py`** — run once per region. Output artifacts must be region-suffixed so neither region overwrites the other:
```
models/lstm_drought_model_rajasthan.keras   (already exists from Phase 1.1 — do not retrain unless you want to confirm reproducibility)
models/lstm_drought_model_barmer.keras       (new)
models/scaler_rajasthan.joblib / scaler_barmer.joblib
models/spi_params_rajasthan.joblib / spi_params_barmer.joblib
```
Architecture, hyperparameters, callbacks, `FEATURES`, `TARGET` — all identical to Phase 1.1 for both regions. Do not tune per region; that would break the comparison.

**`forecasting/evaluate.py`** — run once per region:
```
models/metrics_rajasthan.json / metrics_barmer.json
models/test_forecast_plot_rajasthan.png / test_forecast_plot_barmer.png
```
Add one more line to each region's evaluation output: after computing both regions' `metrics.json`, write a short `models/region_comparison.md` (2-3 lines) stating both regions' averaged RMSE/R²/Skill Score side by side, plainly. No spin — if both show ~0 skill, say so; if Barmer shows real skill and Jaipur didn't, say that too.

**`forecasting/tool.py`** — `forecast_drought_risk(region: str = "rajasthan")` needs to load the correct region-suffixed artifacts based on the `region` argument instead of hardcoded filenames:
```python
def forecast_drought_risk(region: str = "rajasthan") -> dict:
    if region not in REGIONS:
        raise ValueError(f"Unsupported region '{region}'. Supported: {list(REGIONS)}")
    model = load_model(f"models/lstm_drought_model_{region}.keras")
    scaler = joblib.load(f"models/scaler_{region}.joblib")
    spi_params = joblib.load(f"models/spi_params_{region}.joblib")
    # oni_series.parquet is shared, load once regardless of region
    ...
```
Schema returned stays exactly as Phase 1 §9 specified — just add the resolved `region` label in the response (already part of the schema) so callers can confirm which region's model actually answered.

**Tests** — parametrize over both regions rather than duplicating test files. Most of your existing tests (`test_no_leakage.py`, `test_spi_gamma_fit.py`, `test_clean.py`) check general pipeline properties, not Jaipur-specific values, so this is mechanical:
```python
import pytest
from forecasting.config import REGIONS

@pytest.mark.parametrize("region", list(REGIONS))
def test_scaler_fit_train_only(region):
    ...
```
`test_enso_sanity.py` doesn't need parametrizing — ONI is shared, one check covers both regions.

**`.gitignore`** — the current `models/*` + explicit negations (`!models/metrics.json`, `!models/test_forecast_plot.png`, `!models/training_history.json`) need to become glob patterns now that filenames are region-suffixed:
```
models/*
!models/metrics_*.json
!models/test_forecast_plot_*.png
!models/training_history_*.json
!models/region_comparison.md
```
Don't forget this — it's exactly the kind of thing that silently breaks (either untracked evidence files, or an accidentally-tracked 1.5MB model) if missed.

---

## Definition of Done

- [ ] `data/raw/barmer_raw.parquet`, `data/processed/barmer_clean.parquet` exist
- [ ] Barmer model/scaler/spi_params trained with architecture/hyperparameters identical to Rajasthan's Phase 1.1 run
- [ ] `models/metrics_barmer.json`, `models/test_forecast_plot_barmer.png` exist; Rajasthan's Phase 1.1 files untouched
- [ ] `models/region_comparison.md` states both regions' results side by side, honestly
- [ ] `forecast_drought_risk("rajasthan")` and `forecast_drought_risk("barmer")` both return the Phase 1 §9 schema correctly, using the right region's artifacts
- [ ] `forecast_drought_risk("nowhere")` (or any unsupported region) raises a clear error rather than silently loading the wrong model
- [ ] Tests parametrized over both regions, all passing
- [ ] `.gitignore` updated to the glob patterns above — verify with `git status` that exactly the intended files are trackable, same check as before
- [ ] `PROJECT_LOG.md` gets a Phase 1.2 entry — same format as Phase 1/1.1

## When done

Report back: both regions' `metrics.json`, `region_comparison.md`, and confirmation both are backed by architecture-identical training runs. That's the last review gate before Phase 2 (Retrieval Agent) — and before the first git commit, which should now cover Phases 1 through 1.2 as one settled checkpoint.
