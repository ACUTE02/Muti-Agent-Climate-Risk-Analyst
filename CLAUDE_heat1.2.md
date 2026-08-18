# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Heat Stress Agent — Phase 1.2: trim the tool to what actually works

**Why this phase exists:** Phase 1.1 closed the question honestly — 36 measured cells, best skill +0.0378 against a +0.1 bar, no target/feature combination forecasts heat stress with this data. `forecast_heat_stress_risk()` currently still returns three separate no-skill prediction fields (`heat_anomaly`, `heat_extreme`, `heatwave_day_count` forecasts), each labelled "no skill — do not rely on this figure." Serving three flavors of a result that doesn't work is dead weight on the live interface, even though the work that proved it doesn't work is worth keeping.

**This is a cleanup phase, not a new experiment.** No new modeling, no new hypotheses. One rule: keep everything that produced evidence, trim only what the live tool serves.

---

## 1. What stays exactly as-is (do not touch)

- `heat/target.py`, `heat/model.py`, and all training/evaluation code for `heat_anomaly`, `heat_extreme`, `heatwave_day_count`, including the Phase 1.1 SPI-3 cross-agent feature test.
- All evidence files: `models/heat_metrics_*.json`, `models/heat_region_comparison.md`, `models/heat_manifest.json`, everything Phase 1 and 1.1 produced.
- `PROJECT_LOG.md`'s existing Phase 1 and 1.1 sections — this is the resume-facing record of a genuine, rigorous negative result (3 targets × 2 feature sets × 2 regions × 3 horizons, honestly tested and reported). Nothing about the historical record changes.
- Their existing tests (`tests/test_heat_target.py`, etc.) — these test that the *training/evaluation* code behaves correctly, which is still true and still worth guarding even though the models it evaluates aren't shipped.

## 2. What changes: `forecast_heat_stress_risk()` only

Simplify the tool's live contract to serve only what is actually reliable — the operational IMD heat wave indicator. Remove the three no-skill forecast fields (`predicted_heat_anomaly`, `predicted_heat_extreme`, `predicted_heatwave_days`, and their `horizon_confidence` entries) from the return value entirely — do not keep them present-but-labeled-unreliable, remove them.

New contract:

```python
def forecast_heat_stress_risk(region: str, month: str | None = None) -> dict:
    """
    Reports OBSERVED heat wave activity for the given month (defaults to the
    most recent month with data). This function does not forecast future
    heat stress — Phase 1/1.1 established there is no usable predictive
    skill for this risk type with the available zero-cost data (see
    PROJECT_LOG.md, Heat Stress Agent Phase 1 and 1.1). It reports what
    happened, using IMD's heat wave criteria, adapted for a single grid
    point (see heat/target.py for the adaptation).
    """
    return {
        "region": region,
        "month": <the resolved month>,
        "heatwave_days": <int, observed count for this month>,
        "severe_heatwave_days": <int>,
        "had_heatwave_spell": <bool, >=2 consecutive qualifying days>,
        "max_tmax_c": <float, hottest day this month>,
        "forecast_available": False,
        "note": "Heat Stress forecasting was tested (mean anomaly, extreme-day anomaly, and heat wave day count, with and without a drought cross-feature) and found to have no usable skill at any horizon. This function reports observed conditions only. See PROJECT_LOG.md for the full record.",
    }
```

The `forecast_available: False` and the `note` field matter — a caller (including the future Orchestrator/Crop Impact Agent) should be able to tell programmatically that this is observation, not forecast, without parsing prose.

## 3. Update call sites and docs

- Anywhere in the codebase or `README.md` that describes `forecast_heat_stress_risk()` as producing a forecast/prediction — correct it to describe an observation/reporting function.
- If the Orchestrator or any other in-progress code already assumes a `horizon_confidence`-shaped response from this function, flag it — don't silently break a caller.

## Definition of Done

- [ ] `forecast_heat_stress_risk()` returns only observed heat wave data — no predicted fields
- [ ] `forecast_available: False` and an honest `note` field present
- [ ] Training/evaluation code, tests, and all evidence files untouched
- [ ] `tests/test_heat_tool.py` updated to test the new, smaller contract (not deleted — rewritten to match)
- [ ] `README.md` and any other docs describing this tool corrected
- [ ] `PROJECT_LOG.md` gets one short entry noting the interface was trimmed post-Phase-1.1, with a pointer back to the full evidence — not a new result, just a housekeeping note

## When done

Confirm the new tool contract, that evidence/training code is untouched, and that nothing else broke. This is the last step before the Phases 1.6 + Heat 1 + Heat 1.1 + Heat 1.2 commit.
