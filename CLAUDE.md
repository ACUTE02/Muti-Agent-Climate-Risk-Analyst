# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 1 of 7: Forecasting Agent (LSTM, leak-free, India rainfall/drought data)

You are working **only on Phase 1** of a larger multi-agent LangGraph system. Do not build the Retrieval Agent, Orchestrator, Synthesis Agent, Crop Impact Agent, FastAPI layer, or Docker/deploy config in this phase — those are Phases 2–6 and will be handed to you separately, each with its own CLAUDE.md. Your only job here is a working, leak-free, evaluated LSTM forecasting model wrapped as a callable LangGraph tool.

Read this whole file before writing any code. It is intentionally precise — follow the exact specs given (feature names, architecture, thresholds) rather than improvising, since later phases depend on this phase's output shape.

---

## 0. Context (why this phase exists)

There is a prior prototype at `E:\LSTM\LSTM.py` trained on `0-2017-yearly.csv`. It proves the author can build/train/save an LSTM, but it has bugs that make it unusable as-is:

1. **Scaler leakage** — `MinMaxScaler` was fit on the full series (train+test) before splitting.
2. **Look-ahead bias** — `seasonal_decompose()` is two-sided by default, so its "trend" output at time *t* already reflects values from *after* time *t*.
3. **Wrong data** — the CSV is U.S. Drought Monitor category data (D0–D4/W0–W4), not India rainfall, and the model uses one feature only.

Do not reuse that script's logic. Rebuild from scratch per the spec below. You may reuse the general Keras training-loop *mechanics* (`.fit()`, `.predict()`, plotting) as a style reference only.

---

## 1. Environment

**Use Python 3.11.** Rationale (verified August 2026, do not re-litigate this):
- TensorFlow 2.21.0 supports Python 3.10–3.13.
- LangGraph 1.2.11 and LangChain-core 1.5.x support 3.10–3.13/3.14.
- ChromaDB (needed in Phase 2) supports 3.9+.
- Python 3.10 reaches **end-of-life in October 2026** — one month from now — so don't start a new project on it even though it's already installed and proven on this machine.
- Python 3.14 is **not yet supported by TensorFlow** — do not use it.
- 3.11 is the version installed on this machine that sits safely inside every library's supported range.

```powershell
# from the repo root (E:\Muti-Agent-Climate-Risk-Analyst)
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-phase1.txt
```

Create `requirements-phase1.txt` at the repo root:

```
tensorflow==2.21.0
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
joblib>=1.4
requests>=2.32
matplotlib>=3.9
langchain-core>=1.5,<2.0
pytest>=8.0
```

Do **not** add `climate-indices` for this phase — it's an unmaintained, fragile dependency and SPI is fully computable with the manual per-month z-score method specified below (Section 4). Only reach for it later if a more exotic index (PDSI) becomes necessary.

Do **not** add `langgraph` or `chromadb` yet — not needed until Phase 2/3. Keep this phase's dependency footprint minimal.

---

## 2. Repo structure to create

```
Muti-Agent-Climate-Risk-Analyst/
├── requirements-phase1.txt
├── forecasting/
│   ├── __init__.py
│   ├── config.py          # region registry, feature list, hyperparams — single source of truth
│   ├── fetch_data.py       # Open-Meteo API pull
│   ├── clean.py            # leak-free cleaning + feature engineering
│   ├── split.py             # chronological split + scaling
│   ├── train.py              # model build + train + save
│   ├── evaluate.py            # test-set metrics + sanity-check plot
│   └── tool.py                 # LangGraph @tool wrapper (the Phase-1 deliverable)
├── data/
│   ├── raw/                     # fetched, untouched API responses (.parquet)
│   └── processed/                # cleaned, feature-engineered (.parquet)
├── models/
│   ├── lstm_drought_model.keras
│   ├── scaler.joblib
│   ├── month_stats.joblib        # per-month SPI stats fit on TRAIN ONLY (see Section 4)
│   └── metrics.json
└── tests/
    ├── test_clean.py
    └── test_no_leakage.py
```

---

## 3. Data acquisition

Use the **Open-Meteo Historical Weather API** — no key, no registration, no rate-limit risk for this project's scale.

```python
# forecasting/fetch_data.py
def fetch_openmeteo_india(lat: float, lon: float,
                           start: str = "1980-01-01",
                           end: str = "2024-12-31") -> pd.DataFrame:
    """
    Daily temp/precip/soil-moisture for one lat/lon, resampled to monthly.
    Returns columns: temp_c, rainfall_mm, soil_moisture — indexed by month-start date.
    """
```

Default region (put in `forecasting/config.py` as a dict so more regions can be added in later phases without touching pipeline code):

```python
REGIONS = {
    "rajasthan": {"lat": 26.9, "lon": 75.8, "label": "Rajasthan (Jaipur centroid)"},
}
DEFAULT_REGION = "rajasthan"
```

Save the raw pull to `data/raw/{region}_raw.parquet` before any cleaning — never re-fetch during development; cache it.

---

## 4. Cleaning & feature engineering — leak-free rules (read carefully)

This is the part the prototype got wrong twice. Follow these rules exactly:

**Rule A — Split before you fit anything.** Compute the chronological split (Section 5) *first*. Every statistic used to transform a column (scaler min/max, SPI per-month mean/std, anomaly baseline) must be fit on the train partition only, then applied to val/test. This includes a subtlety the original tech-spec's own reference code got wrong: `df.groupby("month")[target].agg(["mean","std"])` for SPI, if computed on the *whole* dataframe, leaks val/test values into the SPI calculation for val/test rows. Fix: compute `month_stats` from **train rows only** (grouped by calendar month 1–12), save it (`models/month_stats.joblib`), and apply those same train-derived per-month stats to transform val and test rows.

**Rule B — No two-sided smoothing.** If any rolling/trend feature is used, it must be strictly causal: `df[col].rolling(window, min_periods=1).mean()` with `center=False` (the default) is fine — it only looks backward. Never set `center=True`. Do not use `statsmodels.seasonal_decompose` (it's two-sided by default and was the source of the prototype's look-ahead bug).

**Rule C — Anomaly baseline is a fixed historical window, not "whole series."** The 1981–2010 baseline period is a valid fixed reference (it's a subset of the train range 1980–2015), so `baseline = df[target]["1981":"2010"]` is fine as-is — just don't extend this pattern to statistics computed over the full series.

Implement:

```python
# forecasting/clean.py
def clean_india_climate(df: pd.DataFrame, target: str = "rainfall_mm") -> pd.DataFrame:
    """Interpolate gaps<3mo -> IQR outlier replace (causal rolling median) ->
    anomaly (1981-2010 baseline) -> cyclical month encoding -> lag features (1,3,6,12) ->
    causal rolling features (roll3_mean, roll12_sum). Does NOT compute SPI — that
    requires train-only stats, so it happens in split.py after the split (Rule A)."""

def compute_spi(df: pd.DataFrame, target: str, month_stats: pd.DataFrame) -> pd.Series:
    """Apply already-fit (train-only) per-month mean/std to compute SPI for any subset."""
```

Final feature set (exact column names the model consumes — keep this list in `config.py`):

```python
FEATURES = [
    "rainfall_mm", "spi", "anomaly", "month_sin", "month_cos",
    "rainfall_mm_lag1", "rainfall_mm_lag3", "rainfall_mm_lag6", "rainfall_mm_lag12",
    "roll3_mean", "roll12_sum",
]
TARGET = "spi"   # the model predicts future SPI, matching the report's "Pred SPI" field
```

---

## 5. Chronological split & scaling

Fixed windows (no shuffling, ever):

```python
TRAIN = slice("1980", "2015")
VAL   = slice("2016", "2019")
TEST  = slice("2020", "2024")
```

1. Split raw cleaned df into train/val/test by date.
2. Fit `month_stats` on train only → compute SPI for all three partitions using train-derived stats (Rule A).
3. Fit `MinMaxScaler` on train `FEATURES` only → transform val/test with the same scaler.
4. `joblib.dump()` both `scaler` and `month_stats` to `models/`.

---

## 6. Sequence framing (multi-step forecast)

The system needs a **3-month-ahead forecast** (matches the report format: predictions for month *t+1, t+2, t+3*), not a single-step prediction like the prototype.

- Input window: 60 months (5 years) of `FEATURES`, shape `(60, 11)`.
- Target: SPI at `t+1, t+2, t+3` → shape `(3,)`.
- Build sliding windows separately within train/val/test (do not let a window span across a split boundary).

---

## 7. Model architecture

```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(60, len(FEATURES))),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(3),                       # SPI at t+1, t+2, t+3
])
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
```

Training:
- `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`
- `ModelCheckpoint("models/lstm_drought_model.keras", monitor="val_loss", save_best_only=True)`
- `epochs=100, batch_size=32, validation_data=(X_val, y_val)`
- Save in **`.keras` format**, not `.h5` (legacy).

---

## 8. Evaluation (held-out test set, 2020–2024)

Compute and print, per horizon (t+1, t+2, t+3) and averaged across all three:

- RMSE, MAE, R² (`sklearn.metrics`)
- Skill Score vs. climatology baseline: `1 - (RMSE_model / RMSE_climatology)`, where the climatology baseline predicts the train-period monthly mean SPI (≈0 by construction, so this mostly tests whether the model beats "always predict normal").

Target ranges to report against (these are targets to aim for and log, not hard gates — real data may land differently; log the true numbers either way):
- R² > 0.80 on test set
- RMSE < 0.5 SPI units
- Skill Score > 0.30

Write all of this to `models/metrics.json`. Also produce `models/test_forecast_plot.png`: predicted vs. actual SPI for the test period (the "forecast sanity check" from the tech spec).

---

## 9. LangGraph tool wrapper (the Phase-1 deliverable)

```python
# forecasting/tool.py
from langchain_core.tools import tool

@tool
def forecast_drought_risk(region: str = "rajasthan") -> dict:
    """
    Forecasts drought risk (SPI, 3-month horizon) for a supported Indian region.
    Loads the trained model/scaler/month_stats, pulls the latest 60 months of
    features for the region, and returns a structured forecast.
    """
    # returns exactly this schema:
    return {
        "region": region,
        "predicted_values": [float, float, float],   # SPI at t+1, t+2, t+3
        "risk_score": float,                           # 0-10, see formula below
        "risk_flags": [str, str, str],                  # "Normal" | "Moderate" | "Severe" per month
        "model_rmse_test": float,                        # from metrics.json, for citation in the report
    }
```

Risk score formula (tunable — flag in code comments that Phase 5 will calibrate this against real NDMA drought records):

```python
def spi_to_risk_score(avg_predicted_spi: float) -> float:
    return round(min(10, max(0, -avg_predicted_spi * 4 + 2)), 1)
```

Flag thresholds per month: `spi >= -1.0` → `"Normal"`; `-1.5 <= spi < -1.0` → `"Moderate"`; `spi < -1.5` → `"Severe"`.

---

## 10. Tests — write these, they must pass

`tests/test_no_leakage.py` must assert, at minimum:
- `scaler.data_min_` / `scaler.data_max_` match values computed directly from the train partition only (not train+val+test).
- `month_stats` (loaded from `models/month_stats.joblib`) was computed only from rows within the train date range — assert its index/values match a fresh groupby on the train slice alone.
- No sliding window in the training set contains dates from val/test, and vice versa.

`tests/test_clean.py`: after `clean_india_climate()`, assert zero NaNs in the output, and assert `month_sin`/`month_cos` are within `[-1, 1]`.

Run with `pytest tests/ -v` — all tests must pass before this phase is considered done.

---

## 11. Definition of Done

- [ ] `data/raw/rajasthan_raw.parquet` and `data/processed/rajasthan_clean.parquet` exist
- [ ] `models/lstm_drought_model.keras`, `scaler.joblib`, `month_stats.joblib` exist
- [ ] `models/metrics.json` has RMSE/MAE/R²/Skill Score per horizon + averaged, computed on the 2020–2024 test set only
- [ ] `models/test_forecast_plot.png` exists and visually shows predicted vs. actual tracking reasonably
- [ ] `forecast_drought_risk()` tool callable standalone (not yet wired into a graph — that's Phase 3) and returns the exact schema in Section 9
- [ ] `pytest tests/ -v` passes, including both leakage assertions
- [ ] No `center=True` rolling, no `seasonal_decompose`, no scaler/month_stats fit outside the train partition anywhere in the codebase

## 12. When done

Report back: the final metrics.json contents, a screenshot/description of test_forecast_plot.png, and confirmation all tests pass. That review gates the start of Phase 2 (Retrieval Agent).
