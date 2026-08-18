# Evaluation — how good is this system, honestly

The canonical scorecard. Every number here was measured in an earlier phase and is
cited to the file it was measured in — this document **consolidates, it does not
re-derive**. `tests/test_evaluation.py` asserts that every number quoted below
actually appears in its cited source, using the same number-matching primitives
`orchestrator/grounding.py` applies to the system's own generated reports. The
scorecard is held to the standard the system is held to.

Two measurements are new in Phase 5 and appear nowhere earlier: the grounding
checker's own precision and recall (§5), and sentence-level faithfulness of real
reports (§6).

*Source: `evaluation/checker_eval_results.json`, `PROJECT_LOG.md`.*

**One-line summary.** One genuinely validated forecasting result (drought, 1-month
lead, linear), one thoroughly established null (heat stress), retrieval plumbing
that works on an admittedly easy test, a mechanical fabrication checker that
caught a real fabrication in testing and measures at precision 1.00 / recall 0.93
— with one real defect this phase found in it — and a crop-impact agent whose most
common honest answer is "no sourced estimate available".

---

## 1. Drought forecasting

*Source: `PROJECT_LOG.md` Phases 1–1.6, `models/horizon_manifest.json`,
`models/metrics_t1_ridge.json`, `models/region_comparison.md`.*

Skill score is measured against a climatology baseline, so 0 means "no better than
predicting the seasonal normal". The project's bar, fixed in Phase 1.3 before the
results were in, is +0.1.

| Region | t+1 | t+2 | t+3 |
|---|---|---|---|
| rajasthan | **+0.2622** validated | +0.0766 weak/directional | -0.0145 no skill |
| barmer | **+0.2053** validated | +0.0438 weak/directional | -0.0489 no skill |

The t+1 result replicates in **4 of 4** independent historical windows at both
regions, and beats naive persistence as well as climatology — persistence alone
earns only +0.1024 at rajasthan and +0.0105 at barmer, and the model beats it in
4 of 4 windows at both sites.

**What was tested and rejected**, five angles, each closing off an explanation for
the failure at longer horizons: target definition (the Phase-1 SPI proxy was a
naive z-score and could not reach the severe band; replaced with gamma-fit SPI-3),
architecture (LSTM 128→64 vs a small LSTM vs Ridge — the linear model won),
site choice (Barmer is genuinely harsher and behaves the same), horizon method
(direct beat recursive in all four cells), and exogenous predictors (ENSO kept,
the Indian Ocean Dipole tested and rejected — it made 5 of 6 cells worse).

**Honest framing.** The validated result is narrow and specific: a *linear* model,
at *one month* lead, with a *12-month* lookback, on a target that partly overlaps
already-observed data. It is not the 3-month deep-learning forecast the original
spec asked for — that one has no skill, and four phases of evidence say so.

## 2. Heat stress

*Source: `PROJECT_LOG.md` Heat Phases 1, 1.1 and 1.2; `models/heat_phase11.json`.*

**A thoroughly established null.** Three target definitions (monthly mean Tmax
anomaly, hottest-day anomaly, heat wave day count) × two feature sets (heat-only,
heat + drought cross-feature) × two regions × three horizons = **36 measured
cells**. Best cell anywhere: **+0.0378**, against the +0.1 bar.

The negative is unusually well-supported because a false positive was caught on
the way: an early run showed skill of +0.44 and +0.63, which turned out to be a
hardcoded baseline predicting the wrong quantity. A wrong baseline is the most
flattering bug available in forecasting work.

**What is actually served.** `forecast_heat_stress_risk()` returns observations
only — `forecast_available: False`, no prediction fields at all. The IMD heat wave
day counter it does serve is reliable: it identifies the 19 May 2016 national
record date and May 2024's spells at both sites.

## 3. Retrieval

*Source: `PROJECT_LOG.md` Phase 2, `retrieval/eval_results.json`,
`retrieval/corpus_manifest.json`.*

Corpus: **9 documents, 224 chunks** — 181 domain-reference (Type A) and 43
project-evidence (Type B).

| Metric | Value |
|---|---|
| precision@5, overall | **1.0** |
| precision@5, Type A | 1.0 |
| precision@5, Type B | 1.0 |
| MRR | 1.0 |

**Read that skeptically — a perfect score usually means an easy test, and it does
here.** This caveat is carried over verbatim from Phase 2 rather than softened.
Nine documents with barely-overlapping vocabularies make "which document" a soft
problem. What this establishes is that the plumbing is correct — task_type
handling, index, metadata filtering, citations — not that retrieval is robust on
ambiguous queries.

A real gap was found and closed during that phase: the first build had **zero**
drought sources on the domain-reference side while the eval set happened to
contain zero drought queries — a metric that would have reported perfect
precision while never testing half the domain the project is about.

## 4. Orchestrator grounding — the fabrication that was caught

*Source: `PROJECT_LOG.md` Phase 3, `orchestrator/grounding_caught_sample.json`.*

**This is the most important line in this scorecard, and it is not a success
story about everything working.**

Asked for a 4-month drought forecast that does not exist, the model correctly
declined the forecast — then explained SPI by reciting the standard McKee
classification bands and citing them to a real corpus document that does not
contain them. Those numbers appear nowhere in the corpus. They came from training
data, they were attributed to a source that does not carry them, and **they are
correct in the real world**, which is what makes that class of error dangerous. A
careful human reviewer would very likely have accepted it.

The pipeline behaved as designed: attempt 1 flagged the figures, the retry named
them, attempt 2 removed one and kept the other, and having exhausted its single
regeneration the graph attached an unverified-figures banner rather than returning
a clean-looking report.

Three further report-quality defects surfaced only by running it: skill scores
cited to the project log when they came from the tool output (grounded but
mis-attributed), a false claim that IMD's outlook was unavailable when both
fetches had succeeded, and "no skill" glossed as "no better than random chance"
when the measured meaning is no better than **climatology** — a much stronger
baseline. Overstating a failure is as inaccurate as understating it.

## 5. The grounding checker's own accuracy — new in Phase 5

*Source: `evaluation/checker_eval_results.json`. Fully offline, no LLM involved.*

The checker had been exercised by hand-written tests but never **measured**. On a
hand-labelled adversarial set of **14 reports and 40 labelled numbers**:

| Metric | Value |
|---|---|
| Precision | **1.0** |
| Recall | **0.9286** |
| F1 | **0.963** |
| Cases fully correct | 13 / 14 |

Precision of 1.0 means it never flagged a legitimate figure across this set —
important, because a checker that cries wolf gets ignored. Recall of 0.9286 means
it missed one fabrication out of fourteen.

**The miss is real and is reported rather than patched.** An integer percentage is
compared against its fraction form at the report token's own precision, so `12%`
becomes `0.12`, rounds to zero decimal places as `0.0`, and matches any source
containing a zero — and tool outputs are full of legitimate zeros. **Any invented
integer percentage below 50% is currently accepted** when the sources contain a
zero. That is precisely the fabrication the Crop Impact Agent exists to prevent:
"drought typically reduces yields by around 12%" passes the checker today.

It is not fixed in this phase, deliberately: Phase 5's stated non-goals forbid
tuning a measured component, and fixing it here would mean these numbers no longer
describe the code that was measured. The failing case is kept in the set, and the
fix belongs in its own change — after which this section must be re-measured.

## 6. Faithfulness and answer relevance — new in Phase 5

*Source: `evaluation/faithfulness_results.json`, `evaluation/eval_requests.json`.*

`check_grounding()` verifies that every **number** traces to a source. It says
nothing about whether a **claim** is entailed by the sources. This section
measures that gap.

**Set size: 3 items, and why.** Sizing was done against the measured free-tier
budget — 5 RPM / 20 RPD per key — *before* any test was written. One item costs 2
calls for the report, a third if synthesis retries, a fourth if it routes to the
crop tool, plus 1 judge call: 3–5 calls each. A 10–15 item set is 30–75 calls, two
to four days of quota on one key. Three items fits a single day with margin, and
this run was a **single-day, single-key run** with no results discarded.

| Metric | Value |
|---|---|
| Mean faithfulness | **0.9867** |
| Mean answer relevance | **1.0** |
| Unsupported claims, all items | 1 |
| Mechanical grounding clean on every item | yes |
| Routing correct on every item | yes |

| Item | Faithfulness | Relevance |
|---|---|---|
| `drought_reliability` | 1.0 | 1.0 |
| `crop_impact_wheat` | 1.0 | 1.0 |
| `impossible_request` | 0.96 | 1.0 |

**The one unsupported claim is not a hallucination, and calling it one would be
wrong.** The judge flagged the report for stating that skill score is
`1 - RMSE_model/RMSE_climatology`. That formula is not in any retrieved chunk — so
the judge is technically correct — but it comes from
`orchestrator/prompts/synthesis.md`, which *instructs* the model to define the
labels precisely, a rule added in Phase 3 after "no skill" was glossed as "no
better than random chance". The claim originates in a checked-in, reviewable
prompt that the judge was never shown.

That is a limitation of this evaluation, not of the report: **the judge sees tool
outputs and retrieved chunks but not the system prompt**, so definitions the
prompt legitimately supplies score as unsupported. A future version should either
show the judge the prompt or exclude prompt-sourced definitions from the claim
set. Reported here rather than corrected into a cleaner-looking 1.0.

**Trust level.** This is the only place in the whole project where an LLM judges
an LLM's output. The exception is deliberate: faithfulness is a soft entailment
judgement with no mechanical form, unlike the discrete fact-checking
`check_grounding()` does. The two are complementary, and this one is explicitly
lower-trust — a judge sharing training data with the writer can agree with it for
the wrong reasons, and this setup cannot detect that. **Where the two disagree,
the mechanical checker wins.**

**Coverage gap, stated:** with three slots, heat-only is not covered as a
standalone request. It is exercised indirectly through the crop-impact item.

## 7. Crop impact

*Source: `PROJECT_LOG.md` Phase 4, `crop_impact/yield_impact_table.json`,
`crop_impact/narrative_sample.json`.*

| Crop × risk | Sourced coefficient? |
|---|---|
| wheat × heat | **yes — 5.6% yield loss**, cited to a Rajya Sabha statement of 4 April 2025 on the 2021-22 NWPZ wheat season |
| bajra × drought | no |
| bajra × heat | no |
| wheat × drought | no |

Four candidate sources were checked and three rejected, each with its reason
recorded in the table. The instructive one: the ICAR-ATARI Jodhpur document,
already in this project's own RAG corpus, is full of pearl millet and wheat
percentages — and every one is a gain from *adopting agro-advisories*, not a
climate-driven loss.

**The awkward shape of this result is stated rather than smoothed:** drought is
the risk type with the validated forecast, and it is the one with no sourced
coefficient.

**A real design mismatch was found by measurement.** Judging heat by IMD heat wave
days returned "no heat" for every warm wheat season on record, because IMD's plains
criteria gate on Tmax ≥ 40 °C, which February and March almost never reach. The
warmest February at Jaipur in the whole record — 2006, **+5.35 °C** mean departure
— records **zero** IMD heat wave days. A metric that passes because it cannot see
what it is measuring.

## 8. Test suite

*Source: `PROJECT_LOG.md` Phases 3, 3.1 and 4.*

The offline suite runs without an API key and costs nothing. Live orchestrator and
crop-impact scenarios are opt-in behind `RUN_LIVE_ORCHESTRATOR=1`, because the
free tier allows 20 `generate_content` requests per day.

---

## What this scorecard does not claim

*Source: `PROJECT_LOG.md`, `evaluation/faithfulness_results.json`.*

- That the 3-month drought forecast works. It does not, and five angles of
  evidence say so.
- That heat stress can be forecast with this data. Thirty-six measured cells say
  it cannot.
- That retrieval is robust. It was measured on an easy test and the caveat stands.
- That the grounding checker is complete. It has a measured, documented
  false-negative class, unfixed as of this scorecard.
- That the faithfulness score is authoritative. It is one LLM's judgement of
  another's, on three items, and is labelled lower-trust than the mechanical
  checker throughout.

## Inconsistency found while building this document

*Source: `PROJECT_LOG.md`, `models/metrics_t1_ridge.json`.*

Consolidating the record surfaced one genuine discrepancy, which is the kind of
finding this phase was meant to produce.

`PROJECT_LOG.md`'s Phase 1.4 per-window table reports the dedicated t+1 model at
rajasthan windows C **+0.2303** and D **+0.3092**, and barmer C **+0.1754** and D
**+0.2890**. Those are **pre-leak-fix** numbers. Phase 1.5 found and fixed a
leakage bug affecting exactly windows C and D, re-ran, and recorded the corrected
*means* (+0.2667 → +0.2622 and +0.2304 → +0.2053) — but the per-window figures in
the Phase 1.4 table were never restated. The current values in
`models/metrics_t1_ridge.json` are rajasthan C **+0.2403** / D **+0.2809** and
barmer C **+0.1546** / D **+0.2269**.

The Phase 1.4 entry is not wrong as a historical record — it documents what that
run produced, and Phase 1.5 says plainly that the numbers moved. But a reader
skimming the table could quote +0.3092 as a current result when the shipped model
measures +0.2809. **This scorecard's §1 uses the post-fix values throughout.**
