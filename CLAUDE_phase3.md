# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 3: Orchestrator + Synthesis Agent

**Where this sits:** three callable tools exist — `forecast_drought_risk()`, `forecast_heat_stress_risk()`, `retrieve_context()` — plus two live Type C fetchers (IMD seasonal + extended-range outlooks). Nothing yet decides which tools to call for a given request, or turns their raw outputs into a report a person can read. This phase builds that: a LangGraph orchestrator that routes a request to the right tools, and a synthesis step that writes the final report — grounded in what the tools actually returned, not in what the LLM thinks is plausible.

**The single most important constraint in this phase, stated up front:** every number in the final report must be traceable to a tool output or a retrieved chunk. This project has caught three separate instances of an unverified number looking right and being wrong (the Phase 1.1 SPI proxy, the Heat 1.1 hardcoded-baseline false positive, near-misses elsewhere) — an LLM synthesis step is a new, larger surface for exactly that failure mode, and it must be caught mechanically, not just by careful prompting.

---

## 1. Orchestrator (LangGraph)

- Input: a natural-language request (e.g. "What's the drought and heat risk for Barmer next month?") plus optionally explicit `region`/`risk_types`/`month` parameters for programmatic callers who don't want to go through NL parsing.
- Use Gemini 2.5 Flash's function-calling to let the model decide which of the three tools to invoke based on the request — do not hand-write a keyword-matching router; function-calling is what it's for, and it generalizes better than regex over region/risk-type names. Bind `forecast_drought_risk`, `forecast_heat_stress_risk`, and `retrieve_context` as callable tools.
- Always fetch Type C (seasonal + extended-range) for any region the request touches — these are cheap (no embedding cost, direct fetch) and materially improve the report, per the Phase 2 design.
- If the request doesn't specify a horizon, default to reporting all three (t+1/t+2/t+3) for drought, since the honest per-horizon labels are exactly the point — do not let the LLM quietly pick just the flattering one.
- One LangGraph state graph: `parse_request → call_tools (parallel where possible) → fetch_type_c → synthesize → verify_grounding → return`. Keep it linear and inspectable — no speculative multi-agent debate/reflection loops in this first pass.

## 2. Synthesis

- One Gemini 2.5 Flash call, given: the raw tool outputs (JSON), the retrieved Type A/B chunks with their citations, and the Type C excerpts with IMD attribution.
- System prompt must explicitly instruct, and the prompt text should be checked into `orchestrator/prompts/synthesis.md` so it's reviewable, not buried in a Python string:
  - State every skill score/confidence label exactly as given by the tool — including "no skill" labels, verbatim, not softened ("might still be informative" is exactly the kind of hedge that undoes an honest label).
  - Never state a number that isn't present in the tool outputs or retrieved chunks.
  - Attribute IMD's Type C outlook separately and by name ("IMD's current seasonal outlook states...") — never merge it into the project's own measured result.
  - If `forecast_available: False` (Heat Stress), say so plainly rather than working around it with hedge language.
  - Cite retrieved Type A/B chunks inline (e.g. "(source: region_comparison.md)" or a footnote-style reference) so a reader can verify.

## 3. The grounding checker — mechanical, not another LLM call

This is the load-bearing safety net for this phase, and it must not itself be an LLM (an LLM checking an LLM has the same failure mode as the LLM being checked).

```python
def check_grounding(report_text: str, tool_outputs: dict, retrieved_chunks: list[dict]) -> dict:
    """
    Extracts every numeric figure from report_text (regex: signed decimals,
    percentages, integers plausibly referring to a measured value — skill
    scores, day counts, temperatures) and verifies each appears, verbatim or
    within reasonable float-formatting tolerance (e.g. +0.2622 vs +0.26),
    in the tool_outputs JSON or the retrieved_chunks text.
    Returns: {"grounded": bool, "unverified_numbers": [...], "total_checked": int}
    """
```

- If `unverified_numbers` is non-empty, the graph should not silently return the report — either regenerate once with the flagged numbers listed in the retry prompt ("these figures could not be verified against source data, remove or correct them"), or if it fails a second time, return the report with an explicit warning banner rather than pretending it's clean. Log every grounding failure to `orchestrator/grounding_failures.jsonl` — even after this phase, these are the canary for the exact failure mode this whole project has been designed to avoid.
- This check is deliberately conservative/mechanical (string matching, not semantic understanding) — false positives (flagging a number that's actually fine, e.g. a date or a count of documents) are an acceptable cost for not missing a real fabrication. Tune the regex to skip obvious non-claims (page numbers, years) but don't try to make it perfect — err toward over-flagging.

## 4. Tests — adversarial, not just happy-path

- At least 3 scenarios end-to-end (e.g. "Rajasthan, drought only", "Barmer, both risk types", "a region/month with no heat wave days") verifying: correct tools were called, every number in the final report traces to a real source, Type C content is separately attributed, "no skill" labels survive verbatim into the report text.
- **Deliberately try to make the LLM fabricate something** — e.g. ask a question whose honest answer is "we don't know" (a 4-month-ahead drought forecast, which doesn't exist) and verify the system says so rather than inventing a plausible-sounding number by extrapolation. This is the single most important test in this phase — a synthesis step that gracefully declines beyond its data is worth more than one that always produces a confident-sounding paragraph.
- Test the grounding checker itself with a deliberately corrupted report (inject a fake number) and confirm it's caught — don't just trust that it works because it looks right in the happy-path tests.

## Stopping rule

This is a first working version of the Orchestrator/Synthesis pipeline, not a final one. If the grounding checker catches real fabrications during testing, fix the prompt/retry logic and report what was caught and how — that's a legitimate part of this phase's story, not a reason to iterate indefinitely. If everything passes cleanly on the first honest run, say so plainly, the same way Phase 2's clean eval result was reported with appropriate skepticism rather than as an unqualified win.

## Definition of Done

- [ ] LangGraph orchestrator with function-calling tool routing over all three existing tools
- [ ] Type C (seasonal + extended-range) fetched and passed to synthesis for any touched region
- [ ] `orchestrator/prompts/synthesis.md` — the actual prompt, checked in and reviewable
- [ ] `check_grounding()` implemented, mechanical (no LLM), with a documented tolerance rule for float formatting
- [ ] Grounding failures logged to `orchestrator/grounding_failures.jsonl`, not silently discarded
- [ ] 3+ end-to-end scenario tests, plus the deliberate-fabrication-attempt test, plus a corrupted-report test for the checker itself
- [ ] `PROJECT_LOG.md` updated with what the grounding checker caught (if anything) during testing, and the final honest state of this phase
- [ ] `.gitignore` extended for `grounding_failures.jsonl` (runtime log, not a build artifact to track) but keep a small sample or summary in evidence if a fabrication was genuinely caught during testing — that's a result worth keeping, the same way the Phase 1.4 leak and Heat 1.1 baseline bug are kept in the log

## When done

Report which tools the orchestrator correctly routes to for each test scenario, what (if anything) the grounding checker caught during testing and how it was handled, and the result of the deliberate "ask it something it can't know" test. This is the piece that makes the two existing agents and the retrieval layer into something a person can actually query.
