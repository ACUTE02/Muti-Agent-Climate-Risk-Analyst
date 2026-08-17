# Climate Risk Analyst

A multi-agent system that answers climate-risk questions for India (e.g. *"drought
risk for Rajasthan 2025–26"*) by combining:

- **RAG** over public climate PDFs (IPCC, NOAA, IMD)
- An **LSTM rainfall forecast** used as a callable tool
- A **synthesis agent** that produces a citation-backed risk report

Full design doc: [`ClimateRiskAnalyst_TechSpec_Revised.docx`](./ClimateRiskAnalyst_TechSpec_Revised.docx)
(original: [`ClimateRiskAnalyst_TechSpec.docx`](./ClimateRiskAnalyst_TechSpec.docx)).

## v1 scope

Deliberately narrow: **one risk type (drought), two regions**, fully evaluated and
deployed — not five risk types across eight states half-finished. See
[`CLAUDE.md`](./CLAUDE.md) for the engineering rules this project holds itself to
(data-leakage guards, no placeholder metrics, sourced-or-labeled-heuristic domain
constants).

## Status

**Phases 1 → 1.5 of 7 complete — Forecasting Agent, closed.** A leak-free LSTM predicts
SPI-3 at t+1/t+2/t+3 and is wrapped as a standalone LangChain tool. Phase 1.1
replaced the Phase-1 z-score SPI proxy with real gamma-fit SPI-3 (McKee et al. 1993)
and added NOAA's Oceanic Niño Index as an exogenous predictor; Phase 1.2 added
Barmer as an independently-trained second region; Phase 1.3 ablated the architecture
against a Ridge baseline and a small LSTM; Phase 1.4 validated the one positive
result across four historical windows and built a dedicated 1-month-ahead model;
Phase 1.5 settled t+2/t+3 (direct beat recursive everywhere) and made the tool
report measured per-horizon confidence. Phases 2–6 (Retrieval Agent, Orchestrator,
Synthesis, Crop Impact, FastAPI/Docker) are not started.

### Results, as measured (2020–2024 held-out test set)

Averaged over t+1/t+2/t+3. Identical architecture, hyperparameters, features and
target for both regions — no per-region tuning, so these compare directly.

| Region | RMSE | MAE | R² | Skill vs. climatology |
|--------|------|-----|-----|----------------------|
| Rajasthan (Jaipur centroid) | 1.068 | 0.861 | −0.207 | +0.0011 |
| Barmer (Thar Desert) | 1.416 | 1.086 | −0.013 | −0.0053 |

Per-horizon numbers: [`models/metrics_rajasthan.json`](./models/metrics_rajasthan.json),
[`models/metrics_barmer.json`](./models/metrics_barmer.json), side by side in
[`models/region_comparison.md`](./models/region_comparison.md).

**No model has meaningful skill over climatology.** The deployed LSTM predicts an
essentially constant value (Jaipur's prediction std ≈ 0.005) while actual SPI-3
swings ±2.6 at Jaipur and −4.75…+3.2 at Barmer — see the `test_forecast_plot_*.png`
files. The pipeline is correct and leak-free; the predictors carry no usable 3-month
signal at either location. Four independent angles agree: the target definition
(real SPI-3), an exogenous predictor (ONI, which correlates with local SPI-3 at only
+0.09…+0.11 across leads), the site (a far more drought-variable district gave the
same flat line), and the architecture (below).

### What does work: a 1-month-ahead linear model

Phase 1.3's ablation found one positive number (Ridge, t+1 only), and Phase 1.4
tested whether it replicated across four independent historical windows. It did —
so a dedicated t+1 model was built, with its lookback chosen from {12, 24, 60}
months by validation skill across all four windows, never by test performance.

| Region | lookback | A (2020–24) | B (2015–19) | C (2010–14) | D (2005–09) | mean |
|---|---|---|---|---|---|---|
| Rajasthan | 12mo | +0.159 | +0.368 | +0.240 | +0.281 | **+0.262** |
| Barmer | 12mo | +0.213 | +0.227 | +0.155 | +0.227 | **+0.205** |

Positive in 4 of 4 windows for both regions, comfortably above the +0.1 threshold
set in advance. The 12-month lookback beat 60 months decisively, so the project's
original 60-month window had been diluting the signal it did have.

Read it narrowly, as measured: this is a **linear** model, at **one month** lead,
on a target that is **partly already observed** — SPI-3 is a 3-month accumulation,
so SPI-3 at t+1 shares two of its three months with SPI-3 at t. That overlap is
measured rather than assumed: naive persistence earns only +0.102 (Rajasthan) and
+0.011 (Barmer), and the model beats persistence by +0.180/+0.189 in 4 of 4
windows. The overlap also explains the horizon cliff exactly — t+1 overlaps the
target by 2 months, t+2 by 1, t+3 by 0, which is precisely where skill vanishes.

### What each horizon is actually worth

Phase 1.5 tested chaining the t+1 model forward against training a dedicated model
per horizon. **Direct won all four cells**; recursion was neutral at t+2 and
actively harmful at t+3 (Rajasthan −0.216 vs −0.015) as reconstruction error
compounded. The tool now reports each horizon's measured skill rather than a
hardcoded confidence:

| Region | t+1 | t+2 | t+3 |
|---|---|---|---|
| Rajasthan | **+0.262 validated** | +0.077 weak/directional | −0.015 no skill |
| Barmer | **+0.205 validated** | +0.044 weak/directional | −0.049 no skill |

`forecast_drought_risk()` returns these alongside the predictions, and labels t+3
*"no skill — shown for context only, do not rely on this figure."* Anything
consuming this tool should honour that.

Full breakdown in [`models/region_comparison.md`](./models/region_comparison.md),
[`models/rolling_window_check.json`](./models/rolling_window_check.json) and
[`models/metrics_t1_ridge.json`](./models/metrics_t1_ridge.json). These numbers are
reported as measured, never as targets.

## Project layout

```
forecasting/        # Phase 1 — the forecasting agent
  config.py           # region registry, feature list, split windows, hyperparams
  fetch_data.py       # Open-Meteo historical pull, cached to data/raw/
  enso.py             # NOAA ONI pull + parsers, cached to models/oni_series.parquet
  baseline_ridge.py   # Phase 1.3 ablation — linear baseline
  lstm_small.py       # Phase 1.3 ablation — small, slow, patient LSTM
  rolling_check.py    # Phase 1.4 — does t+1 skill replicate across 4 windows?
  t1_model.py         # Phase 1.4 — dedicated 1-month-ahead Ridge model
  recursive.py        # Phase 1.5 — recursive vs direct for t+2/t+3, horizon manifest
  clean.py            # leak-free cleaning + feature engineering
  split.py            # chronological split, train-only SPI-3 gamma fit, scaling, windowing
  train.py            # model build + train + save
  evaluate.py         # test-set metrics + forecast sanity-check plot
  tool.py             # forecast_drought_risk() — the callable tool
data/raw/           # fetched API responses, {region}_raw.parquet (gitignored)
data/processed/     # cleaned + feature-engineered, {region}_clean.parquet (gitignored)
models/             # per-region model/scaler/spi_params + ONI cache (gitignored);
                    # metrics_*.json, test_forecast_plot_*.png,
                    # training_history_*.json, region_comparison.md (tracked)
tests/              # leakage, SPI-3 gamma-fit, ENSO-parse and multi-region tests
```

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements-phase1.txt
```

Python 3.11 specifically — see [`CLAUDE.md`](./CLAUDE.md) §1 for why.

## Running the pipeline

```bash
python -m forecasting.fetch_data        # every region; cached after the first run
python -m forecasting.enso              # cached; NOAA ONI series, shared by all regions
python -m forecasting.train rajasthan   # one region at a time
python -m forecasting.train barmer
python -m forecasting.evaluate          # all regions + region_comparison.md
python -m forecasting.baseline_ridge    # Phase 1.3 ablation
python -m forecasting.lstm_small        # Phase 1.3 ablation
python -m forecasting.rolling_check     # Phase 1.4 robustness check
python -m forecasting.t1_model          # Phase 1.4 dedicated t+1 model
python -m forecasting.recursive         # Phase 1.5 horizon comparison + manifest
python -m forecasting.tool              # a live 3-month forecast per region
pytest tests/ -v
```

## Development workflow

See [`CLAUDE.md`](./CLAUDE.md) for the rules Claude Code (or any contributor) follows
in this repo: plan before multi-file changes, tests before numeric code, one pipeline
stage per session, and no metric or resume claim that isn't backed by a real
measured eval run.
