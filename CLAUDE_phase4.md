# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 4: Crop Impact Agent

**Where this sits:** three tools exist and are wired into a working, grounded orchestrator — `forecast_drought_risk()` (SPI-3, validated at t+1 only, weak at t+2, no skill at t+3), `forecast_heat_stress_risk()` (observed heatwave-day counts only — `forecast_available: False`, no predictive skill was ever found), and `retrieve_context()` (a citable RAG corpus). This phase adds a fourth tool that turns those risk signals into a crop-yield-impact assessment.

**The standing design principle for this phase (set at the Aug 17 scope decision, do not relitigate it):** this agent must be **generic across risk types**, not hardcoded to assume drought is always the dominant factor. Given whichever risk signals exist in state at call time (today: drought + heat; potentially flood later), it must determine **per crop/region/time which risk factor actually dominates** before estimating yield impact.

**The single most important constraint in this phase, carried over from Phase 3 and non-negotiable:** every number in the crop-impact output must be traceable to either (a) a tool output, (b) a retrieved/cited agronomic source, or (c) a small set of deterministic rules that are themselves sourced and tested. This project has now caught four separate instances of an unverified or wrongly-attributed number looking right and being wrong (Phase 1 SPI proxy, Heat 1.1 hardcoded baseline, Phase 1.4 window leak, Phase 3's SPI classification-band fabrication). A fifth surface — an LLM asked to translate climate risk into "% yield loss" — is exactly the kind of question a language model will confidently answer from general knowledge whether or not it has real grounding for *this* project's numbers. Treat that as the default failure mode to design against, not an edge case.

---

## 0. Architecture decision for this phase — hybrid, not pure LLM

Two extremes were considered and rejected in favour of a hybrid:

- **Pure deterministic** (no LLM call): safest and cheapest, but produces a dry structured output, not a report a person can read, and doesn't match the "agentic" framing of the rest of the project.
- **Pure LLM** (hand the model the raw risk numbers and ask for a yield-impact narrative): this is the fabrication risk described above — asking a model to invent a %-yield-loss figure it was never trained or grounded on for these specific crops/regions/data.

**This phase uses a hybrid, in this order:**

1. A **deterministic dominant-risk-factor function**, plain Python, fully unit-tested, no LLM — given the current tool outputs for a region/month, decide which risk type (drought, heat, or "none dominant") is the binding constraint on yield for a given crop, and why. This is the piece the project's own design principle is actually asking for, and it must not be delegated to an LLM's judgement call.
2. A **sourced, deterministic yield-impact lookup/formula** — not invented, not asked of the LLM. Find real published crop-loss coefficients for the crops and risk types in scope (ICAR-ATARI Jodhpur's Agro-Advisory is already in the RAG corpus and is a legitimate first place to look; IMD/ICAR drought and heat-stress agronomy literature is the next place). If no genuinely sourced coefficient exists for a crop/risk combination, the tool must say **"no sourced yield-impact estimate available for this crop/risk combination"** rather than let the LLM extrapolate one — this is the same discipline as Heat Stress's `forecast_available: False`, applied here.
3. **One Gemini call** that takes the deterministic dominant-risk-factor decision and the sourced yield-impact number (both already computed, both already grounded) and writes the **narrative explanation** — plain-language reasoning about why this risk dominates, what it means for the named crop, and how confident the underlying forecast is (reusing the drought/heat honesty labels verbatim, exactly as Phase 3's synthesis prompt already requires). The LLM is not asked to invent a number at any point — only to explain numbers it is handed.
4. The existing `orchestrator/grounding.py` (unmodified, reused as-is) checks this new report text exactly as it checks the Phase 3 synthesis report, against tool outputs + the deterministic yield-impact result + retrieved chunks.

This keeps the one call that actually needs judgement (the plain-language explanation) LLM-driven, while keeping the two decisions that must be trustworthy (which risk dominates, what the yield impact is) deterministic and testable.

---

## 1. Crop and region scope

Keep this bounded and honest about what it covers, the same way the project scoped down from 5 risk types to 2.

- **Regions:** the two already in the system — Rajasthan (Jaipur centroid) and Barmer. No new regions this phase.
- **Crops:** pick 2, chosen for real agronomic relevance to these two regions and for having genuinely sourceable drought/heat sensitivity data — the standard candidates are **Bajra (pearl millet)**, a drought-tolerant Kharif staple grown in both regions, and **Wheat**, a heat-sensitive Rabi crop grown in Rajasthan. If sourced yield-loss data turns out to only exist for one of the two, ship with one crop and say so explicitly in `PROJECT_LOG.md` rather than force a second crop on invented numbers.
- Do **not** attempt Flood or Coastal risk types — they are not built, and the standing scope decision already documents this as future work. If drought or heat data is unavailable for a region/month, the agent must decline that combination, not guess.

## 2. `assess_crop_impact()` — the callable tool

```python
def assess_crop_impact(region: str, crop: str, month: str | None = None) -> dict:
    """
    Deterministic core (no LLM):
      1. Pull forecast_drought_risk(region) and forecast_heat_stress_risk(region, month).
      2. Determine the dominant risk factor for this crop's current growth stage
         (see dominant_risk() below) — or "none dominant" if neither risk signal
         is severe enough to matter, or "insufficient data" if a required signal
         is unavailable (e.g. drought t+3, which carries no measured skill).
      3. Look up the sourced yield-impact coefficient for (crop, dominant_risk,
         severity) from a small, cited table — or return
         "no sourced yield-impact estimate available" if no real coefficient exists.

    One Gemini call (grounded, checked exactly like Phase 3's synthesis):
      4. Write a short plain-language explanation of the above — never asked to
         produce a number itself.

    Returns: {region, crop, month, dominant_risk, risk_reasoning,
              yield_impact_pct: float | None, yield_impact_source: str | None,
              confidence_label: str,   # reuses the underlying forecast's own label
              narrative: str, grounding: dict}
    """
```

### `dominant_risk()` — the deterministic decision, unit-tested directly

- Must be a pure function taking the tool outputs (or a fixture standing in for them) and returning a risk type + a one-line reason, with **no Gemini call inside it** — mirror the discipline of `check_grounding()` in Phase 3 (a test should assert this function's source contains no LLM call, same pattern as `test_checker_is_not_an_llm`).
- Must respect the honesty labels already established: if the only drought signal available is t+3 ("no skill"), it cannot be treated as equally trustworthy as t+1 ("validated") when deciding dominance. Weight or gate by the horizon's measured confidence, don't just compare raw numbers.
- Heat has **no forecast at all** — only observed heatwave-day counts for past/current months. For a future month, heat can only be judged as "unknown" or based on climatological typicality, never as if it were a forecast. Be explicit about this distinction in the returned reasoning — do not silently treat an observed count as if it were a prediction.
- Write down, in `crop_impact/dominance_rule.md` (checked in, reviewable, same pattern as `orchestrator/prompts/synthesis.md`), the actual rule being used and why — e.g. "drought dominates if t+1 SPI-3 < X and skill label is validated; heat dominates if the most recent observed month had ≥N heatwave days; otherwise none dominant." Do not bury this logic only in code comments.

### Yield-impact lookup — sourced, not invented

- Before writing any coefficient, verify the source the same way Phase 2 verified every RAG document — check it actually says what you're about to cite, record the citation, and if a plausible source turns out not to have usable numbers for these specific crops, say so and either find a better source or leave that combination as "not available."
- Store the table as a small checked-in JSON/YAML (`crop_impact/yield_impact_table.json`) with a citation field per entry, not hardcoded numeric literals scattered through code — this is what makes it inspectable and what the grounding checker's mechanical string-matching depends on.

## 3. Grounding — reuse, don't reinvent

- `orchestrator/grounding.py`'s `check_grounding()` is already generic (report text + tool outputs + retrieved chunks) — extend the "tool outputs" dict passed to it to include the deterministic `dominant_risk`/`yield_impact` result, and it should work unmodified. If it genuinely cannot handle something about this new output shape, that is worth flagging and fixing narrowly, not a reason to write a second grounding checker.
- Same retry-once-then-banner behaviour as Phase 3's `synthesize`/`verify_grounding`/`finalise` nodes — reuse those nodes' pattern rather than duplicating the retry logic.
- Add `assess_crop_impact` to the orchestrator's routable tools (`orchestrator/graph.py`'s `TOOLS` list) so a request like "what's the impact on wheat in Rajasthan" routes here through the same function-calling mechanism as the other three tools.

## 4. Quota discipline — read before writing any test

This project's real free-tier budget, corrected in Phase 3.1, is **5 RPM / 20 RPD per API key**. This phase adds a Gemini call to the pipeline: a report that touches crop impact now costs up to 3 calls (routing + crop-impact narrative + main synthesis) in the best case, up to 5 in the worst case (one retry each on crop-impact and main synthesis). Follow the exact pattern already established:

- Offline tests (deterministic `dominant_risk()`, yield-impact lookup, and `check_grounding()` extension) must run always, cost nothing, and be the majority of this phase's test coverage.
- Live end-to-end tests (real Gemini calls) go behind the same `RUN_LIVE_ORCHESTRATOR=1` gate already in `tests/test_orchestrator.py` — do not invent a second env var. Cache each scenario once per test session with the same `_run()` pattern, don't call live per-assertion.
- State the actual quota cost of this phase's live tests in a comment, the same way `tests/test_orchestrator.py`'s `_live_enabled()` docstring does, updated for the corrected 3-5-calls-per-scenario cost.

## 5. Tests — adversarial first, happy path second

- Unit tests for `dominant_risk()` covering: drought clearly dominant (validated t+1, severe SPI-3), heat clearly dominant (multiple recent heatwave days, mild drought), neither dominant (both mild), and the "insufficient data" case (only t+3 drought available, no recent heat observation).
- A test that a t+3 "no skill" drought signal is **never** allowed to independently declare drought dominant — this is the single most important correctness property of this phase, parallel to Phase 3's "declines a horizon it cannot forecast" test.
- A test that an unsourced crop/risk combination returns the explicit "not available" state rather than a fabricated coefficient — mirrors Heat 1.2's "no forecast fields can reappear" test.
- A live (gated) test that deliberately asks for a crop/region combination with no sourced yield data and asserts the system declines rather than invents a plausible-sounding percentage — this is this phase's version of Phase 3's "ask it something it can't know" test, and is just as important here.
- A live (gated) test asserting the narrative's percentage figures are traceable to `yield_impact_table.json`'s cited values, not paraphrased into a different number.

## Stopping rule

This is a first working version, same posture as every prior phase. If the deterministic yield-impact table can only be sourced for one crop, or one region, ship with that narrower scope and say so plainly in `PROJECT_LOG.md` — do not fill gaps with invented coefficients to make the table look more complete than the sourcing supports.

## Definition of Done

- [ ] `dominant_risk()` — deterministic, unit-tested, no LLM call, respects horizon confidence labels and heat's observation-not-forecast nature
- [ ] `crop_impact/dominance_rule.md` — the actual decision rule, checked in and reviewable
- [ ] `crop_impact/yield_impact_table.json` — sourced coefficients with citations, or explicit gaps where no real source exists
- [ ] `assess_crop_impact()` tool, wired into `orchestrator/graph.py`'s routable tools
- [ ] One Gemini call for the narrative only, never for a number; existing `check_grounding()` reused and passing on this output shape
- [ ] Offline tests for the deterministic pieces (majority of coverage), live tests gated behind `RUN_LIVE_ORCHESTRATOR=1`
- [ ] The "t+3 cannot declare dominance" test and the "unsourced combination declines rather than invents" test both present and passing
- [ ] `PROJECT_LOG.md` updated with what was actually sourced, what wasn't, and the honest final state — including quota cost per report now that a fourth call type exists

## When done

Report which crop/region/risk combinations have real sourced yield-impact data and which don't, what `dominant_risk()` actually decided in a couple of concrete example cases, whether the grounding checker needed any changes to handle the new output shape, and the updated per-report quota cost. If a sourced coefficient search came up short for a crop, say so — that's a legitimate result for this phase, not a gap to paper over.
