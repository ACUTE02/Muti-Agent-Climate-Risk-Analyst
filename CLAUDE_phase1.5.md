# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 1.5: Recursive (chained) forecasting + honest per-horizon confidence — final round

**This is the last round of iteration on the Forecasting Agent's horizon question.** After this phase, commit regardless of outcome and move to Heat Stress. Do not open a Phase 1.6.

**The idea being tested:** instead of predicting t+2 and t+3 directly from the original input window (as Phase 1.3/1.4 did), use the already-validated t+1 model recursively — predict t+1, feed that prediction back in as if it were an observed value, predict t+2 from the shifted window, then use both predictions to predict t+3. Compare this against direct prediction honestly, on real measured numbers, not assumed error arithmetic. Whichever wins for each horizon gets used; every horizon's reported confidence must come from actual test results, not a hand-set label.

---

## 1. Recursive predictor

Reuse the Phase 1.4 model exactly (Ridge, 12-month lookback, region-specific alpha) as the single building block — do not retrain a new model per step.

```python
def recursive_forecast(model, window_12mo_features, horizon=3):
    """
    window_12mo_features: the most recent 12 months of FEATURES, shape (12, n_features)
    Returns [pred_t1, pred_t2, pred_t3]
    """
    preds = []
    window = window_12mo_features.copy()
    for step in range(horizon):
        pred = model.predict(flatten(window))  # predict next month's spi3
        preds.append(pred)
        # shift window forward by one month, appending the prediction as the newest "known" value
        new_row = build_next_feature_row(window, predicted_spi3=pred)
        # month_sin/month_cos: computable exactly (calendar is known in advance)
        # lag/rolling features: shift naturally from the now-longer series (predictions included)
        # oni: use the most recent actually-known ONI value (do not fabricate a future ONI)
        window = append_and_drop_oldest(window, new_row)
    return preds
```

Be explicit in code comments about which features are exactly known in advance (calendar-based: month_sin/month_cos) versus which are being approximated (ONI held at last known value — note this as a real limitation, not hidden).

## 2. Direct predictors for t+2 and t+3 (fair comparison baseline)

Phase 1.4 only built a dedicated model for t+1. For a fair comparison, build the analogous dedicated direct models for t+2 and t+3 — same process as Phase 1.4 Section 3 (lookback chosen from {12, 24, 60} by validation skill, per region, per horizon).

## 3. Compare recursive vs. direct, on the same 4 windows as Phase 1.4

For t+2 and t+3, evaluate both approaches across windows A/B/C/D (same window definitions as Phase 1.4), both regions:

| Horizon | Region | Approach | A | B | C | D | mean skill |
|---|---|---|---|---|---|---|---|
| t+2 | rajasthan | recursive | | | | | |
| t+2 | rajasthan | direct | | | | | |
| t+2 | barmer | recursive | | | | | |
| t+2 | barmer | direct | | | | | |
| t+3 | rajasthan | recursive | | | | | |
| t+3 | rajasthan | direct | | | | | |
| t+3 | barmer | recursive | | | | | |
| t+3 | barmer | direct | | | | | |

Whichever approach has the higher mean skill for a given (horizon, region) is the one used going forward for that cell. It's fine if recursive wins for one region/horizon and direct wins for another — report it exactly as it comes out, don't average away the difference.

## 4. Update the tool to report real, measured per-horizon confidence

`forecast_drought_risk()` must report each horizon's actual measured skill score from this phase's testing, not a hardcoded label:

```python
return {
    "region": region,
    "predicted_values": [pred_t1, pred_t2, pred_t3],
    "horizon_confidence": [
        {"horizon": "t+1", "skill_score": <measured>, "method": "direct", "label": <derived from measured value, see below>},
        {"horizon": "t+2", "skill_score": <measured>, "method": "recursive" or "direct", "label": ...},
        {"horizon": "t+3", "skill_score": <measured>, "method": "recursive" or "direct", "label": ...},
    ],
    "risk_score": float,
    "risk_flags": [...],
}
```

Label derivation — thresholds set from what this project has already established, not arbitrary: `skill_score >= 0.1` → `"validated"` (matches the bar used throughout Phase 1.3/1.4), `0 < skill_score < 0.1` → `"weak/directional"`, `skill_score <= 0` → `"no skill — shown for context only, do not rely on this figure"`.

## Stopping rule

Report the comparison table and the final per-horizon labels honestly, whatever they turn out to be. This is the last iteration — after this, update `region_comparison.md`, `PROJECT_LOG.md`, and README with final numbers, and the phase is done regardless of whether recursion beat direct prediction or not.

## Definition of Done

- [ ] Recursive predictor implemented, reusing the Phase 1.4 model (no new model trained for the recursive step itself)
- [ ] Direct dedicated models for t+2 and t+3 built (same process as Phase 1.4 §3)
- [ ] Both compared across all 4 windows, both regions, both remaining horizons — full table reported, not just the winner
- [ ] `forecast_drought_risk()` returns per-horizon measured skill scores and derived labels, not hardcoded confidence
- [ ] `region_comparison.md` and `PROJECT_LOG.md` updated with the final, complete picture: what's validated, what's weak, what's not — for all 3 horizons, both regions
- [ ] `.gitignore` extended for any new evidence files

## When done

Report back the full comparison table and the final per-horizon labels for both regions. This closes out the Forecasting Agent for real — commit Phases 1 through 1.5 after this, then Heat Stress.
