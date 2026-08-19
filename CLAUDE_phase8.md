# CLAUDE.md — Phase 8: English UI, Plain-Language Reports, Live Data Refresh + 3 New Cited Sources

**Where this sits:** Phase 7 (audit fixes + frontend) is done, uncommitted, tested against a real live report. Four issues/requests surfaced from actually using it:

1. The frontend's Hindi labels, while well-intentioned, weren't what the user wanted — switch to English.
2. The synthesis report is accurate but too technical for a non-technical reader — terms like "skill score", "validated", "t+1" mean nothing without context.
3. The forecast's underlying weather data stops at `2024-12-31` (`forecasting/config.py: FETCH_END`), so a query for "the next 3 months" from today (Aug 2026+) falls ~20 months past the data the model was given, and the crop-impact tool correctly reports "insufficient data" rather than guessing. **The fix is to give the project's own already-working model fresh input data, not to substitute in another source's forecast.** The model's t+1 skill score (~+0.21 to +0.26) is real, measured, and good — it just needs current inputs to produce a current answer.
4. The user wants **more external data sources woven in, cited explicitly** — the same pattern the project already uses for IMD's outlook: fetched live, attributed by name, and never blended into this project's own measured figures. Think of it the way a research paper cites and builds on prior work, never claims someone else's finding as its own. Three sources were chosen (Part 4 below) for being free, reliable, and genuinely complementary to what the project already has.

**Explicit non-goal, stated for the record:** do not fetch a forecast/number from any external source (existing or new) and present it as this project's own computed result. Every external source — IMD and the three new ones in Part 4 — must be labelled by name and kept separate from the project's own measured figures, exactly as `synthesis.md` rule 3 already requires for IMD. This is a hard constraint, not a style preference — flag it explicitly in your final report if any instruction anywhere seems to conflict with it, rather than resolving the conflict silently.

---

## Part 1 — Frontend: English, not Hindi

- Change `frontend/index.html` and `frontend/app.js`'s static Hindi labels (headings, field labels, dropdown option text, buttons, banner text, chip prompts) to English. Keep the underlying region/crop/risk-type *values* sent to the API unchanged (`"rajasthan"`, `"bajra"`, etc.) — only the visible text changes.
- The design system CSS (`frontend/nocturne.css`) and layout stay as-is — this is a text/copy change, not a redesign.
- Update `tests/test_frontend.py` if it asserts on any specific Hindi string.

## Part 2 — Reports readable by someone with no technical background

Rewrite `orchestrator/prompts/synthesis.md`'s structure and tone so a reader who has never heard "skill score" or "SPI-3" can follow the report, **without weakening any of the existing honesty rules (1-10) — those stay exactly as strict.** This is a presentation change, not a content change: every number, label and citation rule already in the prompt must still hold.

Concretely:

- **Add a plain-language glossary layer.** Instead of only showing raw labels (`validated`, `weak/directional`, `no skill`), pair each with a one-line plain-English explanation inline the first time it's used in a report, e.g.: *"Month 1: reliable — this project's own testing shows this figure is meaningfully better than just guessing the seasonal average."* / *"Month 3: not reliable — shown for completeness, but this project's own testing found no real predictive value at this range, so don't act on it."* The underlying label (`validated`/`no skill`/etc.) must still appear as structured data in the API response (the frontend's tag badges depend on it) — this is about how the *prose* explains it, not about removing the label.
- **Rename "t+1/t+2/t+3" in reader-facing prose** to "Month 1 / Month 2 / Month 3" (or the actual calendar month if known, e.g. "September 2026"). Keep `t+1` etc. as internal field names in the API/data — only the human-readable text changes.
- **Cut jargon**, or define it in the same sentence it's used: "SPI-3 (a standard way of measuring how much drier or wetter than usual the last 3 months of rainfall have been)" rather than assuming the reader knows it.
- **Move the existing "How to read this" section to the top**, right after the Summary, rewritten in plain language — so a reader gets the "can I trust this number" framing before the details, not after.
- Keep the existing structure otherwise (Summary / Drought / Heat / Crop impact / IMD outlook / How to read this) — just simplify the language throughout.

Add a test that renders a report through the real prompt structure (or checks the prompt file itself) and confirms the plain-language pairing survives for at least one `no skill` case — the honesty content must not get lost in the simplification.

## Part 3 — Live data refresh, so the model's own forecast is actually current

**Do not touch the existing fixed historical dataset or re-run the fixed-window evaluation** (`TEST = slice("2020", "2024")` and the four labelled windows in `forecasting/config.py`) — those splits are what every published skill score in `PROJECT_LOG.md`/`EVALUATION.md` was measured against, and changing that data would silently invalidate all of them.

Instead, add a **separate, additive refresh path**:

1. A new function (e.g. `forecasting/fetch_data.refresh_recent(region)`) that fetches only new daily data since the cached file's last date up to today (small, fast pull — reuse the existing chunking/backoff logic in `fetch_data.py`, don't duplicate it), and appends it to a **separate rolling cache** (e.g. `data/raw/{region}_recent.parquet`), leaving `{region}_raw.parquet` (the fixed 1980-2024 archive used for evaluation) untouched.
2. `forecast_drought_risk()` (and the heat tool, for its observation-only reporting) should use the union of the fixed archive + the rolling recent cache when computing the input features (12-month lookback, SPI-3, etc.) for a *live* request — so "the next 3 months" is computed relative to the actual current date, using the already-fitted Ridge model (no retraining needed — same model, current inputs).
3. Wire a refresh step into `scripts/setup.py` (or a new small `scripts/refresh.py`) so a person can run one command before a demo session to bring the data current, and document it in `SETUP_FROM_CLEAN.md`.
4. `/health` should report how stale the rolling cache is (e.g. `data_current_through: "2026-08-15"`) so the frontend/API never silently serves a forecast anchored to a date far in the past without saying so.
5. **This must not change any existing published skill score.** Add a check (same discipline as the audit's skill-score numeric diff) confirming the fixed-window evaluation numbers in `PROJECT_LOG.md`/`EVALUATION.md` are unchanged after this work.

## Part 4 — Three new cited external sources

All three follow the exact IMD pattern already in the codebase (`retrieval`'s Type C live-fetch, kept out of the indexed corpus, attributed by name in the report, never merged into this project's own tool outputs). Pick reliable, genuinely complementary, zero-cost sources — these three, chosen for that reason:

1. **IMD (India Meteorological Department) — expand the existing integration.** Already fetched live for the seasonal/extended-range outlook. Check whether IMD publishes anything else freely accessible and relevant (e.g. district-level rainfall bulletins) that would strengthen the existing "IMD's current outlook" section without duplicating what the project's own SPI-3 model already measures. If nothing further is realistically fetchable for free, say so plainly and move on — don't force an addition that doesn't exist.

2. **NASA POWER (Prediction of Worldwide Energy Resources).** Free, no API key, no registration — a public REST API (`power.larc.nasa.gov`). Provides agriculturally-relevant meteorological data including reference evapotranspiration and solar radiation, which the crop-impact tool currently cannot speak to (recall the "irrigation demand estimate isn't available" gap flagged in earlier reports). Fetch live per request (like IMD), for the requested region/month, and surface it as its own clearly-attributed subsection — "NASA POWER reports..." — usable to add an irrigation-demand or evapotranspiration data point to the crop-impact section *only when it's actually relevant to the request*, never forced in.

3. **data.gov.in (India's open government data portal).** Free, requires a simple free API key (document how to obtain one in `SETUP_FROM_CLEAN.md`, and handle its absence the same way the Gemini key's absence is already handled — degrade gracefully, don't crash). Look for official district-level crop production/yield or rainfall datasets that could either (a) strengthen `crop_impact/yield_impact_table.json`'s sourced-coefficient citations, or (b) be cited live in the crop-impact section the same way IMD is cited for weather. If a genuinely useful, stable dataset isn't findable in the time this phase allows, report exactly what was tried and why it didn't pan out — this is real research, not guaranteed to succeed, and a documented dead end is a legitimate outcome, not a failure to hide.

**For all three:** every fetched value must appear in the report attributed by source name, must be checkable by the mechanical grounding checker exactly like IMD's outlook already is, and must never be described as a measurement "this project" made. If a fetch fails (network, missing key, source down), report that plainly — same discipline as the existing IMD-unavailable handling — never substitute a plausible-sounding number.

## Definition of Done

- [ ] Frontend labels in English, values unchanged, tests updated
- [ ] `synthesis.md` rewritten for a non-technical reader, honesty rules 1-10 untouched, a test confirms plain-language pairing survives for a `no skill` case
- [ ] Rolling recent-data cache added, additive only — fixed evaluation archive and its published numbers untouched (explicitly checked, not assumed)
- [ ] A live query for "the next 3 months" from today's actual date produces a real forecast (not "insufficient data" purely because of stale input), still carrying its honest skill labels
- [ ] `/health` reports data currency
- [ ] IMD integration reviewed/expanded where genuinely possible, or explicitly reported as already complete
- [ ] NASA POWER wired in, live-fetched, clearly attributed, degrades gracefully on failure
- [ ] data.gov.in explored; either a genuinely useful dataset wired in (attributed, graceful on missing key/failure) or a documented, honest dead end
- [ ] `PROJECT_LOG.md` updated
- [ ] No external forecast/number from any source (IMD, NASA POWER, data.gov.in, or anything else) is ever presented as this project's own computed result, anywhere in the report or the frontend — call this out explicitly as checked, not just assumed compliant

## When done

Report a real example: the same "Barmer, next 3 months" query as before and after this phase, showing the forecast now targets real upcoming months, still with accurate skill labels (including if a horizon still comes back "no skill" — that must stay honestly reported, this phase makes the *data* current, it does not manufacture skill that measurement doesn't support). Also show one report where at least one of the three new sources appears, clearly attributed, alongside the project's own figures — and confirm which of the three actually panned out vs. which (if any) hit a real dead end.
