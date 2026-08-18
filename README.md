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

**Forecasting (Drought) Agent closed at Phase 1.5; Heat Stress Agent first pass
complete.** A leak-free LSTM predicts
SPI-3 at t+1/t+2/t+3 and is wrapped as a standalone LangChain tool. Phase 1.1
replaced the Phase-1 z-score SPI proxy with real gamma-fit SPI-3 (McKee et al. 1993)
and added NOAA's Oceanic Niño Index as an exogenous predictor; Phase 1.2 added
Barmer as an independently-trained second region; Phase 1.3 ablated the architecture
against a Ridge baseline and a small LSTM; Phase 1.4 validated the one positive
result across four historical windows and built a dedicated 1-month-ahead model;
Phase 1.5 settled t+2/t+3 (direct beat recursive everywhere) and made the tool
report measured per-horizon confidence; Phase 1.6 tested the Indian Ocean Dipole
as a second exogenous predictor and rejected it on the measurements. **Phase 2
(Retrieval Agent) and Phase 3 (Orchestrator + Synthesis) are complete** — a
citable RAG corpus over IMD/NDMA/ICAR references plus this project's own evidence,
and a LangGraph orchestrator whose every reported number is mechanically verified
against source data. **Phase 4 (Crop Impact Agent) is complete** — a hybrid agent
whose risk-dominance decision and yield-impact lookup are deterministic, with one
LLM call used only to explain results it is handed. Phases 5–6 (Evaluation Suite,
FastAPI/Docker) are not started.

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

### Tested and rejected: the Indian Ocean Dipole

IOD is a documented driver of Indian monsoon variability, independent of ENSO, and
was the one predictor never tried. Phase 1.6 added it (`iod`, `iod_lag1`) and
re-ran the whole selection process across the same four windows and both regions.
It correlates with ONI at r = +0.386, so it is related but not redundant.

**Result: rejected.** Five of six region/horizon cells got *worse*; the single
nominal gain (+0.0011) held in only 2 of 4 windows against the project's standing
3-of-4 bar. Two extra features on ~350 training sequences cost more in variance
than they return. The feature is not in the model, the default pipeline is
byte-identical to what Phase 1.5 measured, and
[`models/iod_comparison.json`](./models/iod_comparison.json) keeps the numbers.

That closes the drought feature set: five angles tested — target definition,
architecture, site, horizon method, and exogenous predictors.

## Retrieval Agent (RAG)

The two forecasting tools state numbers; neither could explain itself or cite
authority for its thresholds. `retrieval/` builds the piece that does — a bounded,
curated corpus (**9 documents, 224 chunks**) with a `retrieve_context()` tool that
returns citations, not just text.

- **Type A — domain reference (181 chunks):** IMD Heat Wave FAQ, NDMA Heat Wave
  page, NDMA Drought Management Guidelines, NIH Roorkee SPI methodology, IMD GKMS
  Agromet SOP, ICAR-ATARI Jodhpur Agro-Advisory. Every PDF was extraction-checked
  before inclusion; `https://ndma.gov.in/Natural-Hazards/Drought` is a verified
  404 and was replaced by the NIDM-hosted official guidelines.
- **Type B — this project's own evidence (43 chunks):** `PROJECT_LOG.md` and both
  region-comparison files. This is the more important half. It is what lets the
  system answer "how reliable is the 2-month forecast" with the measured
  **+0.0766, weak/directional** instead of a number a language model made up.
- **Type C — live IMD outlooks:** never indexed, fetched fresh at report time,
  quoted whole and attributed to IMD by name, with an explicit "unavailable"
  state rather than a silent omission or an undated stale copy.

Chunking differs by type on purpose: header-aware for Type B, so a skill score
stays in the same chunk as the caveat that qualifies it; paragraph packing with
overlap for Type A prose. Embeddings are `gemini-embedding-001` at 768-dim with
`RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` used on the correct sides.

**Evaluation: precision@5 = 1.00** across 12 hand-authored queries (7 domain
reference, 5 project evidence), MRR 1.00, run without the document-type filter.
Read skeptically — nine documents with barely-overlapping vocabularies make this
an easy test, so it confirms the plumbing rather than proving robust retrieval.
A stricter check confirms the retrieved *text* contains the answers themselves
(`+0.0766`, `47`/`6.4`, "gamma"), which is what the Synthesis agent will depend
on. Full RAGAS-style evaluation is Phase 5.

```bash
python -m retrieval.build       # verify sources, chunk, embed, populate Chroma
python -m retrieval.evaluate    # the 12-query precision check
python -m retrieval.tool "how reliable is the 2-month drought forecast?"
```

Requires a Gemini API key in `GEMINI_API_KEY` (or a gitignored `.env`).

## Orchestrator + Synthesis

`orchestrator/` turns the three tools into something a person can query. A
LangGraph state graph
(`parse_request → call_tools → fetch_type_c → synthesize → verify_grounding →
finalise`) routes a natural-language request using Gemini 3.6 Flash function
calling, always fetches IMD's live outlooks, and writes the report. The prompt is
checked in at [`orchestrator/prompts/synthesis.md`](./orchestrator/prompts/synthesis.md)
so it is reviewable rather than buried in a string.

```bash
python -m orchestrator.graph "drought and heat risk for Barmer, with reliability"
```

### The grounding checker

Every number in the report is verified mechanically — **not by another LLM**, since
an LLM checking an LLM shares the failure mode being checked. `check_grounding()`
extracts each figure and requires it to appear in the tool outputs or the retrieved
chunks. A report may round a source value to its own precision (`+0.26` against
`+0.2622` passes); anything materially different is flagged, logged, and triggers
one regeneration. If it still fails, the report ships with an explicit
unverified-figures banner rather than looking clean.

**It caught a real fabrication in testing.** Asked for a 4-month forecast that does
not exist, the model correctly declined it — then explained SPI by reciting the
standard classification bands (`-1.0 to -1.49 moderately dry`) and cited them to a
retrieved document. Those numbers are in **no document in the corpus**; they came
from the model's training data, and they are *correct in the real world*, which is
what makes that class of error dangerous. Preserved in
[`orchestrator/grounding_caught_sample.json`](./orchestrator/grounding_caught_sample.json),
and the prompt now forbids reciting reference tables that are not in the sources.

Live end-to-end tests are opt-in (`RUN_LIVE_ORCHESTRATOR=1`) because
the free tier allows 20 generate_content calls per day (5 RPM) and each
scenario costs two. The 11 offline checker tests — including corrupted-report and
number-fragment attacks — always run.

## Crop Impact Agent

`crop_impact/` turns the risk signals into a crop-yield assessment — and it is
deliberately a **hybrid**, not an LLM asked for a percentage:

1. **`dominant_risk()`** decides which risk actually binds yield for a crop,
   region and month. Plain Python, unit-tested, no model call (a test asserts
   this). The rule is written out in
   [`crop_impact/dominance_rule.md`](./crop_impact/dominance_rule.md).
2. **`lookup_yield_impact()`** reads a sourced coefficient from
   [`crop_impact/yield_impact_table.json`](./crop_impact/yield_impact_table.json).
   A table read — it never computes, interpolates or scales.
3. **One Gemini call** explains those two results in plain language. It is never
   asked to produce a number.
4. **The Phase-3 grounding checker**, reused unmodified, verifies every figure in
   that explanation.

```bash
python -m crop_impact.tool rajasthan wheat 2006-02   # steps 1-2 only, no LLM call
```

### A no-skill forecast can never drive an assessment

The most important property here: a drought horizon labelled *no skill* declares
nothing, however alarming its value. A t+3 SPI-3 of −2.5 is severe on this
project's own thresholds and is still refused, because five phases of evidence
say that horizon is worthless. `weak/directional` may corroborate but never
decide; only `validated` decides. Heat is likewise never treated as a forecast —
this system has none — so a request about a future month can never return heat as
dominant, and the reasoning says so in words.

### What the table actually contains, and what it does not

**One** sourced coefficient survived verification: **wheat × heat = 5.6% yield
loss**, from the Union Minister of State for Agriculture's statement to the Rajya
Sabha (4 April 2025) on the 2021–22 North Western Plain Zone wheat season at
+5.5 °C. It is stored with its verbatim quote and an explicit caveat that it is a
*single anchor point, not a slope* — and the lookup enforces the exposure band it
was measured in, so a +3.68 °C March returns **no estimate** rather than a scaled
one.

Three combinations are recorded as gaps: bajra × drought, bajra × heat, wheat ×
drought. Four candidate sources — including two already in this project's own RAG
corpus — were checked and rejected, each with its reason recorded in the table.
The ICAR-ATARI document is the instructive one: it is full of pearl millet and
wheat percentages, and every one is a gain from *adopting agro-advisories*, not a
climate-driven loss. Citing it would have looked authoritative and been wrong.

So the agent's most common honest answer is **"no sourced yield-impact estimate
available for this crop and risk combination"**, with the reason attached. That is
the same discipline as the Heat Agent's `forecast_available: False`: the system
declines rather than inventing a plausible number.

A real captured run, grounded and verified, is checked in at
[`crop_impact/narrative_sample.json`](./crop_impact/narrative_sample.json).

## Heat Stress — the second risk type

Same regions, same source, same discipline; no shared features with drought. The
target (`heat_anomaly`, the standardised monthly Tmax anomaly) was
distribution-checked *before* modelling this time — skew −0.05/−0.26, excess
kurtosis +0.63/+0.76, tails within 0.19 of a standard normal's, reaching both
tails — so the Phase-1 mistake of shipping a broken target could not repeat.

**Result: no skill at any horizon, either region.**

| Region | Lookback | t+1 | t+2 | t+3 |
|---|---|---|---|---|
| Rajasthan | 24mo | −0.053 | −0.161 | −0.163 |
| Barmer | 60mo | −0.021 | −0.017 | −0.012 |

Ridge does beat a persistence baseline (+0.08 to +0.35) — but persistence is much
*worse* than climatology here (−0.14 to −0.61), because monthly heat anomalies
barely persist, so beating it means little. Climatology is the baseline that
counts, and the model loses to it. Full picture in
[`models/heat_region_comparison.md`](./models/heat_region_comparison.md).

Phase 1.1 then tested three further hypotheses — an extremes-focused target
(hottest day of the month), the heat wave day count as a forecast target, and
antecedent dryness (SPI-3, read from the Drought Agent) as a feature. **None
cleared the bar**; the best of 36 measured cells reached +0.038. SPI-3 helps only
in the sense of moving clearly-worse-than-climatology models back to
equal-to-climatology — regularisation, not signal. The clearest statement of the
failure: for May 2024 the count model predicted **0.7–0.9 heat wave days** where
**8 and 7** actually occurred.

Worth recording because it nearly went the other way: the first Phase 1.1 run
showed the extremes target at +0.44/+0.63 skill. That was a bug, not a discovery —
the baseline functions were scoring the new target against the *old* target's
climatology. Fixed, it collapsed to −0.28/+0.02. A wrong baseline is the most
flattering bug available in forecasting work.

What the heat agent *does* deliver reliably is the operational indicator, which is
observation rather than forecast: monthly heat wave day counts under IMD's plains
criteria (Tmax ≥ 45 °C, or ≥ 40 °C with a ≥ 4.5 °C departure from a ±7-day
day-of-year normal; severe at 47 °C / 6.4 °C; spells need ≥ 2 consecutive days),
adapted to a single grid point — IMD's real rule requires two stations in a
subdivision, which one grid cell cannot satisfy. It correctly picks out the
19 May 2016 national-record event at both locations.

`forecast_heat_stress_risk(region, month=None)` serves exactly that and nothing
else. It is a **reporting function, not a forecaster**: the no-skill prediction
fields were removed outright in Phase 1.2 rather than shipped behind a warning
label, and the response carries `forecast_available: False` plus a note pointing
at the evidence, so a caller can tell observation from forecast programmatically.
The training and evaluation code that established the negative result stays in the
tree with its evidence files — the record is worth keeping, the dead interface was
not.

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
  iod.py              # Phase 1.6 — NOAA Dipole Mode Index fetch + parsers
  iod_check.py        # Phase 1.6 — before/after IOD comparison (rejected)
  clean.py            # leak-free cleaning + feature engineering
  split.py            # chronological split, train-only SPI-3 gamma fit, scaling, windowing
  train.py            # model build + train + save
  evaluate.py         # test-set metrics + forecast sanity-check plot
  tool.py             # forecast_drought_risk() — the callable tool
heat/               # Heat Stress agent — the second risk type
  target.py           # train-only Tmax climatology, distribution check, IMD heat wave rules
  dataset.py          # causal features + leak-free split, reusing the window builder
  model.py            # Ridge first (per the drought ablation), climatology + persistence
  tool.py             # forecast_heat_stress_risk() — observed heat wave reporting
  phase11.py          # Phase 1.1 — extremes/count targets + SPI-3 cross-feature
data/raw/           # fetched API responses, {region}_raw/_daily.parquet (gitignored)
data/processed/     # cleaned + feature-engineered, {region}_clean.parquet (gitignored)
models/             # per-region model/scaler/spi_params + ONI cache (gitignored);
                    # metrics_*.json, test_forecast_plot_*.png,
                    # training_history_*.json, region_comparison.md (tracked)
orchestrator/       # Phase 3 — Orchestrator + Synthesis
  graph.py            # the LangGraph state graph and tool routing
  grounding.py        # mechanical number verification (no LLM)
  prompts/synthesis.md  # the synthesis prompt, checked in
retrieval/          # Phase 2 — the Retrieval Agent (RAG)
  config.py           # the corpus definition: Type A/B/C sources
  sources.py          # defensive fetch + PDF/HTML extraction verification
  chunk.py            # header-aware (Type B) and overlap (Type A) chunking
  embed.py            # gemini-embedding-001, batched, backed off, resumable
  store.py            # ChromaDB build/query
  outlooks.py         # Type C live IMD outlooks, never indexed
  tool.py             # retrieve_context() — the callable tool
  evaluate.py         # the 12-query precision check
crop_impact/        # Phase 4 — the Crop Impact Agent
  config.py           # crops, sensitive windows, dominance thresholds
  dominance.py        # dominant_risk() — deterministic, no LLM
  dominance_rule.md   # the decision rule, checked in and reviewable
  yield_impact.py     # sourced-coefficient lookup — a table read, not a formula
  yield_impact_table.json   # coefficients with citations, and recorded gaps
  prompts/narrative.md      # the one prompt, reviewable rather than inline
  tool.py             # assess_crop_impact() — the Phase-4 deliverable
  narrative_sample.json     # a real grounded run, kept as evidence
tests/              # leakage, SPI-3 gamma-fit, ENSO-parse, multi-region and
                    # retrieval-corpus tests
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
python -m forecasting.iod_check         # Phase 1.6 IOD feature test
python -m forecasting.tool              # a live 3-month forecast per region
python -m heat.model                    # Heat Stress: fit, evaluate, compare
python -m heat.phase11                  # Heat Phase 1.1 target/feature grid
python -m heat.tool                     # observed heat wave counts per region
python -m crop_impact.tool rajasthan wheat 2006-02   # crop impact, no LLM call
pytest tests/ -v
```

## Development workflow

See [`CLAUDE.md`](./CLAUDE.md) for the rules Claude Code (or any contributor) follows
in this repo: plan before multi-file changes, tests before numeric code, one pipeline
stage per session, and no metric or resume claim that isn't backed by a real
measured eval run.
