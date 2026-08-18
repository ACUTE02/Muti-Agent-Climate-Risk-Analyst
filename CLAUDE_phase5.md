# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 5: Evaluation Suite

**Where this sits:** four tools exist, wired into a grounded orchestrator, all four phases closed with honestly-reported results (drought: validated only at t+1; heat: no forecast skill anywhere, observation-only; retrieval: precision@5 = 1.00 flagged as an easy test; crop impact: one sourced coefficient, three gaps, one real measurement mismatch caught and fixed). What does **not** yet exist is a single place that pulls all of that together and answers, in one document, "how good is this system, honestly, as of today" — and a genuinely *new* measurement this project hasn't done yet: whether the synthesis reports are faithful to their sources at the sentence level, not just number-by-number.

**What this phase is not:** it is not a chance to re-measure things already measured. Drought and heat skill scores, retrieval precision, and the Phase-3 fabrication catch are established results — Phase 5 **consolidates and cites them**, it does not re-run the forecasting or retrieval pipelines. The one place this phase adds real new measurement is faithfulness/groundedness at the sentence level (RAGAS-style), which is different from what `check_grounding()` already checks (see §2).

**The quota reality, unchanged from Phase 3/4 and worth restating up front:** 5 RPM / 20 RPD per key. This phase must be designed so the overwhelming majority of its value comes from **deterministic consolidation of results this project already has**, with a small, explicitly bounded number of new live calls for the one genuinely new measurement. Do not design an evaluation suite that needs to burn quota to be useful — that would make this project's own evaluation phase a victim of the constraint the rest of the project has been honest about.

---

## 1. `EVALUATION.md` — the consolidated, honest scorecard (no LLM cost)

A single checked-in document, written from the numbers already in `PROJECT_LOG.md`, `README.md`, and the `models/`/`retrieval/`/`crop_impact/` evidence files — not re-derived, **cited to where each number was actually measured** (file + phase), the same discipline `orchestrator/grounding.py` enforces on the LLM's own reports.

Must include, each stated at the confidence level it was actually measured at:
- Drought: per-horizon skill scores and labels, both regions, with the one-line honest summary ("validated at t+1 only, five angles tested").
- Heat: the null result across 36 cells, and what the operational (non-forecast) tool actually delivers.
- Retrieval: precision@5/MRR, with the "easy test" caveat carried over verbatim, not softened.
- Orchestrator grounding: the one real fabrication caught in Phase 3, how it was caught and fixed — this is arguably the single most important line in the whole scorecard, and it must not read as "and everything else worked perfectly."
- Crop Impact: which crop/risk combinations have a sourced coefficient (one) and which don't (three), and the real design-mismatch bug found and fixed.

A test should assert every number quoted in `EVALUATION.md` actually appears in its cited source file — reuse `orchestrator/grounding.py`'s number-extraction/matching primitives rather than writing a second regex from scratch, since the problem ("does this number appear in that source") is identical.

## 2. Faithfulness evaluation — the genuinely new measurement

`check_grounding()` verifies that every **number** in a report traces to a source. It does not check whether a **claim** (a sentence with no number in it — "drought is the binding risk because...") is actually supported by what the sources say, or whether the report contains something a source doesn't support at all. That is a different, real gap, and it is what RAGAS-style faithfulness measures.

- Build a small (10-15 item) held-out evaluation set of realistic requests spanning all four tools (drought-only, heat-only, both, crop impact, and at least one adversarial "asks for something the system can't know" case reused conceptually from Phase 3's `impossible` scenario) — written before results exist, the same discipline Phase 2's 12 queries used.
- For each, capture the full report **once**, cached (same `_run()`-per-scenario pattern as `tests/test_orchestrator.py`), and score two things against the retrieved chunks + tool outputs:
  - **Faithfulness**: is every claim in the report entailed by the source material, not just every number? This needs one additional Gemini call per report to judge sentence-level entailment — state plainly in this phase's log that this is the one place in the whole project where an LLM checks an LLM's output, and why that's an acceptable exception here specifically (it is scoring *faithfulness*, a soft judgement, not verifying discrete facts the way `check_grounding()` mechanically does — the two are complementary, not redundant, and this one is explicitly logged as lower-trust than the mechanical checker).
  - **Answer relevance**: does the report actually answer what was asked, or does it wander? Can be scored with a second small prompt, or folded into the same judging call if that keeps the quota cost down — the phase should default to folding it in unless there's a clear reason not to.
- **Quota-bound this explicitly before writing a single test**: 10-15 items × (1 orchestrator report generation, up to 5 calls worst case + 1-2 judge calls) is easily 60-100+ calls — far past a day's budget on one key. Either shrink the held-out set to something that fits one key's daily quota with margin (e.g. 3-4 items), or state plainly in `PROJECT_LOG.md` that this evaluation was run across multiple days / multiple keys and say so, the same honesty this project has applied to every other quota constraint. Do not silently run it in a way that assumes unlimited quota.

## 3. Grounding-checker reliability — its own false-positive/negative rate

The mechanical grounding checker is the project's single most safety-critical piece, and it has never itself been evaluated for accuracy (only exercised by hand-written test cases). This section is fully offline — no LLM cost.

- Build a labelled adversarial set: reports with known-fabricated numbers injected (following Phase 3's existing corrupted-report tests as a starting pattern, expanded), reports that are entirely clean, and edge cases the checker has previously had trouble with (the Phase 3.1 content-block bug, the Phase 4 number-fragment class of bug) — reconstructed as fixtures, not live calls.
- Report the checker's precision and recall on this labelled set directly in `EVALUATION.md` — a checker that is itself unverified is exactly the kind of unexamined-metric failure this project has caught three times already (Heat-1.1 baseline, Phase-1.4 leak, Phase-3.1 content-block bug); evaluating it closes that loop rather than assuming the existing test suite already proves it.

## 4. Explicit non-goals for this phase

- No re-running of the LSTM/Ridge forecasting pipelines — those results are closed and cited, not reproduced.
- No new crop/region coverage — that is Phase 4's scope, already closed.
- No attempt to improve any measured number (retrieval precision, drought skill, etc.) — this phase measures and reports, it does not tune.

---

## Tests

- Offline (must run always, no quota cost): the `EVALUATION.md` citation-check test, and the full grounding-checker precision/recall suite from §3.
- Live (gated behind the existing `RUN_LIVE_ORCHESTRATOR=1`, cost stated explicitly in the test file's docstring the same way Phase 3/4 did): the faithfulness/relevance judging calls from §2, sized to whatever held-out-set size the quota-bound analysis in §2 actually lands on.

## Stopping rule

If the held-out faithfulness set has to be shrunk to fit the daily quota, ship the smaller set and say so plainly — a 3-item honestly-reported faithfulness check is worth more than a 15-item one silently run across several days without saying so, or one that was never actually completed. If the grounding checker's measured precision/recall turns out imperfect, report the actual numbers; do not tune the checker's regex to pass this phase's own test set, since that would be optimizing for the test rather than for real reports.

## Definition of Done

- [ ] `EVALUATION.md` — consolidated scorecard, every number cited to its actual source phase/file, test asserting every quoted number traces back
- [ ] Faithfulness/relevance held-out set built, sized against a stated quota budget (not assumed unlimited), run, and reported with the same honesty as every other phase's results
- [ ] Grounding-checker precision/recall measured on a labelled adversarial set, fully offline
- [ ] `PROJECT_LOG.md` updated with the actual faithfulness/relevance numbers and the checker's measured precision/recall — including if either result is not great
- [ ] `README.md`'s status section updated to point at `EVALUATION.md` as the canonical scorecard

## When done

Report the checker's measured precision/recall, the faithfulness/relevance scores from the held-out set (and how large that set actually was, and why), and whether building `EVALUATION.md` surfaced any number in the project's own record that turned out to be stale, mis-cited, or inconsistent across files — that would be a legitimate and useful finding for this phase, not a failure.
