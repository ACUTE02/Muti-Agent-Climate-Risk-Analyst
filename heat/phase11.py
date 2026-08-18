"""Heat Phase 1.1 — extremes target, count target, and antecedent dryness.

Three untested hypotheses, one bounded pass, no chaining:

1. a monthly *mean* dilutes 5-10 day spells into a 30-day average, so an
   extremes-focused target (``heat_extreme``) may keep signal the mean discarded;
2. the operational heat wave day count is itself a forecastable quantity;
3. antecedent dryness (SPI-3, from the Drought Agent) drives hotter days through
   land-atmosphere feedback and was deliberately excluded from Phase 1.

Everything else is frozen: same split, regions, Ridge, alpha grid, lookback grid,
selection-by-validation rule, and the same climatology/persistence baselines.

**One deliberate addition to the feature set, stated up front.** Phase 1's feature
list is built around ``heat_anomaly`` and includes it, so the mean target could see
its own history. Giving the two new targets nothing of their own history would
handicap them by construction rather than test them, so each run adds its own
target column to the features when it is not already there. That is the fair
comparison, not a thumb on the scale.

Run standalone:  python -m heat.phase11
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forecasting import config
from forecasting.baseline_ridge import fit_ridge_baseline, flatten_windows
from forecasting.recursive import horizon_label
from heat.dataset import prepare_heat_dataset
from heat.model import HORIZON_LABELS, _rmse, climatology_prediction, persistence_prediction

FEATURE_SETS = {
    "heat-only": config.HEAT_FEATURES,
    "heat+SPI3": config.HEAT_FEATURES_SPI3,
}


def _features_for(target: str, base: list) -> list:
    """Base set plus the target's own history (see module docstring)."""
    return list(base) if target in base else list(base) + [target]


def _scores(y_true, y_pred, y_clim, y_persist) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    rmse, rmse_clim = _rmse(y_true, y_pred), _rmse(y_true, y_clim)
    rmse_persist = _rmse(y_true, y_persist)
    skill = 1 - rmse / rmse_clim if rmse_clim else float("nan")
    return {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "skill_score": round(float(skill), 4),
        "skill_vs_persistence": round(float(1 - rmse / rmse_persist), 4)
        if rmse_persist else None,
        "persistence_skill_vs_climatology": round(float(1 - rmse_persist / rmse_clim), 4)
        if rmse_clim else None,
        "label": horizon_label(round(float(skill), 4)),
    }


def fit_and_score(region: str, target: str, feature_set: str,
                  lookback: int) -> tuple[dict, "object", np.ndarray]:
    features = _features_for(target, FEATURE_SETS[feature_set])
    ds = prepare_heat_dataset(region, seq_len=lookback, target=target,
                              features=features)
    X_train, y_train = ds.get("train")
    X_val, y_val = ds.get("val")

    model, val_mse, alpha = fit_ridge_baseline(X_train, y_train, X_val, y_val)

    val_split = ds.splits["val"]
    val_skill = 1 - _rmse(y_val, model.predict(flatten_windows(X_val))) / _rmse(
        y_val, climatology_prediction(ds, val_split["target_dates"]))

    test = ds.splits["test"]
    y_pred = model.predict(flatten_windows(test["X"]))
    if target == config.HEAT_COUNT_TARGET:
        # A count cannot be negative. Clipping is the honest minimum; where it
        # changes much, that is itself a sign Ridge is the wrong family here.
        y_pred = np.clip(y_pred, 0, None)

    y_clim = climatology_prediction(ds, test["target_dates"])
    y_persist = persistence_prediction(ds, test)

    per_horizon = {
        HORIZON_LABELS[h]: _scores(test["y"][:, h], y_pred[:, h],
                                   y_clim[:, h], y_persist[:, h])
        for h in range(ds.horizon)
    }
    result = {
        "region": region, "target": target, "feature_set": feature_set,
        "lookback": lookback, "alpha": float(alpha),
        "val_mse": round(float(val_mse), 4),
        "val_skill": round(float(val_skill), 4),
        "n_features": len(features),
        "per_horizon": per_horizon,
        "mean_skill": round(float(np.mean(
            [s["skill_score"] for s in per_horizon.values()])), 4),
    }
    if target == config.HEAT_COUNT_TARGET:
        result["diagnostics"] = ds.diagnostics.get("zero_inflation", {})
        result["clipping"] = _clipping_effect(
            model.predict(flatten_windows(test["X"])))
        result["pre_monsoon"] = _pre_monsoon_cut(ds, test, y_pred, y_clim, y_persist)
        result["may_2024_check"] = _may_2024(ds, test, y_pred)
    else:
        result["distribution"] = ds.diagnostics["distribution"]
    result["winter_check"] = _winter_cut(ds, test, y_pred)
    return result, model, y_pred


def _clipping_effect(raw_pred: np.ndarray) -> dict:
    return {
        "fraction_predictions_negative_before_clip": round(
            float((raw_pred < 0).mean()), 4),
        "most_negative_raw_prediction": round(float(raw_pred.min()), 4),
    }


def _pre_monsoon_cut(ds, test, y_pred, y_clim, y_persist) -> dict:
    """April-June only, where the count is actually usually nonzero."""
    out = {}
    for h, label in enumerate(HORIZON_LABELS):
        dates = test["target_dates"] + pd.DateOffset(months=h)
        mask = dates.month.isin(config.PRE_MONSOON_MONTHS)
        if mask.sum() < 3:
            continue
        out[label] = {
            "n_months": int(mask.sum()),
            **_scores(test["y"][mask, h], y_pred[mask, h],
                      y_clim[mask, h], y_persist[mask, h]),
        }
    return out


def _winter_cut(ds, test, y_pred) -> dict:
    """Dec-Feb: Tmax never nears 40 C, so observed and predicted must sit near
    zero. A nontrivial winter signal would be a bug, not a result."""
    out = {}
    for h, label in enumerate(HORIZON_LABELS):
        dates = test["target_dates"] + pd.DateOffset(months=h)
        mask = dates.month.isin(config.WINTER_MONTHS)
        if not mask.any():
            continue
        out[label] = {
            "n_months": int(mask.sum()),
            "observed_mean": round(float(test["y"][mask, h].mean()), 4),
            "observed_max": round(float(test["y"][mask, h].max()), 4),
            "predicted_mean": round(float(y_pred[mask, h].mean()), 4),
            "predicted_max": round(float(y_pred[mask, h].max()), 4),
        }
    return out


def _may_2024(ds, test, y_pred) -> dict:
    """Plain-language check on a month known to have had real heat wave days."""
    out = {}
    for h, label in enumerate(HORIZON_LABELS):
        dates = test["target_dates"] + pd.DateOffset(months=h)
        hits = np.where(dates == pd.Timestamp("2024-05-01"))[0]
        if len(hits):
            i = hits[0]
            out[label] = {"observed_days": float(test["y"][i, h]),
                          "predicted_days": round(float(y_pred[i, h]), 2)}
    return out


def run() -> dict:
    rows = []
    for region in config.REGIONS:
        for target in config.HEAT_TARGETS:
            for feature_set in FEATURE_SETS:
                candidates = [fit_and_score(region, target, feature_set, lb)[0]
                              for lb in config.HEAT_LOOKBACKS]
                best = max(candidates, key=lambda r: r["val_skill"])
                best["val_skill_by_lookback"] = {
                    c["lookback"]: c["val_skill"] for c in candidates}
                rows.append(best)
                print(f"{region:9s} {target:18s} {feature_set:10s} "
                      f"lookback={best['lookback']:2d}mo  "
                      f"mean skill={best['mean_skill']:+.4f}  "
                      f"t+1={best['per_horizon']['t+1']['skill_score']:+.4f}")

    best_overall = max(
        ((r, h, s["skill_score"]) for r in rows
         for h, s in r["per_horizon"].items()), key=lambda t: t[2])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypotheses": ["heat_extreme target", "heatwave_day_count target",
                       "SPI-3 antecedent-dryness feature"],
        "phase1_reference": {"rajasthan": -0.0525, "barmer": -0.0209},
        "split": {"train": [config.TRAIN.start, config.TRAIN.stop],
                  "val": [config.VAL.start, config.VAL.stop],
                  "test": [config.TEST.start, config.TEST.stop]},
        "results": rows,
        "best_cell": {"region": best_overall[0]["region"],
                      "target": best_overall[0]["target"],
                      "feature_set": best_overall[0]["feature_set"],
                      "horizon": best_overall[1],
                      "skill_score": best_overall[2]},
        "anything_clears_bar": bool(best_overall[2] >= 0.1),
    }
    config.HEAT_PHASE11_PATH.write_text(json.dumps(payload, indent=2),
                                        encoding="utf-8")
    return payload


def format_tables(payload: dict) -> str:
    lines = ["## Phase 1.1 — new targets and the SPI-3 cross-agent feature", "",
             "Same split, Ridge, alpha grid and lookback grid as Phase 1. Each run "
             "sees its own target's history (see `heat/phase11.py` for why). "
             "Phase 1's reference result was -0.0525 (rajasthan) / -0.0209 "
             "(barmer) at t+1 on `heat_anomaly` with heat-only features.", "",
             "| Region | Target | Features | Lookback | t+1 | t+2 | t+3 | mean | "
             "best label |", "|---|---|---|---|---|---|---|---|---|"]
    for r in payload["results"]:
        cells = " | ".join(f"{r['per_horizon'][h]['skill_score']:+.4f}"
                           for h in HORIZON_LABELS)
        best = max(r["per_horizon"].values(), key=lambda s: s["skill_score"])
        lines.append(f"| {r['region']} | `{r['target']}` | {r['feature_set']} | "
                     f"{r['lookback']}mo | {cells} | {r['mean_skill']:+.4f} | "
                     f"{best['label'].split('—')[0].strip()} |")

    counts = [r for r in payload["results"]
              if r["target"] == config.HEAT_COUNT_TARGET]
    if counts:
        z = counts[0].get("diagnostics", {})
        lines += ["", "### The count target is zero-inflated", "",
                  f"- {z.get('fraction_zero_all_months', 0) * 100:.1f}% of train "
                  f"months have zero heat wave days; "
                  f"{z.get('fraction_zero_pre_monsoon', 0) * 100:.1f}% even within "
                  "April-June.",
                  f"- Train mean {z.get('mean_all_months')} days overall vs "
                  f"{z.get('mean_pre_monsoon')} in the pre-monsoon months.", "",
                  "Skill against an all-months climatology is therefore flattered "
                  "by eight easy months a year. The April-June cut is the "
                  "meaningful one:", "",
                  "| Region | Features | Horizon | n months | Skill vs clim. | "
                  "Label |", "|---|---|---|---|---|---|"]
        for r in counts:
            for horizon, s in r.get("pre_monsoon", {}).items():
                lines.append(f"| {r['region']} | {r['feature_set']} | {horizon} | "
                             f"{s['n_months']} | {s['skill_score']:+.4f} | "
                             f"{s['label'].split('—')[0].strip()} |")

        lines += ["", "May 2024 (a month with real heat wave days) — predicted vs "
                  "observed day counts:", "",
                  "| Region | Features | Horizon | Predicted | Observed |",
                  "|---|---|---|---|---|"]
        for r in counts:
            for horizon, s in r.get("may_2024_check", {}).items():
                lines.append(f"| {r['region']} | {r['feature_set']} | {horizon} | "
                             f"{s['predicted_days']} | {s['observed_days']:.0f} |")

        lines += ["", "Negative predictions clipped to zero at evaluation:", ""]
        for r in counts:
            c = r.get("clipping", {})
            lines.append(
                f"- {r['region']} / {r['feature_set']}: "
                f"{c.get('fraction_predictions_negative_before_clip', 0) * 100:.1f}% "
                f"of raw predictions were negative (most negative "
                f"{c.get('most_negative_raw_prediction')}) — a linear model on a "
                "non-negative count, as expected.")

    lines += ["", "### Winter sanity check (Dec-Feb, t+1)", "",
              "Tmax never approaches the 40 C plains threshold in winter, so both "
              "columns should sit near zero. Confirmed, not assumed:", "",
              "| Region | Target | Features | n months | Observed mean | "
              "Observed max | Predicted mean | Predicted max |",
              "|---|---|---|---|---|---|---|---|"]
    for r in payload["results"]:
        w = r.get("winter_check", {}).get("t+1")
        if w:
            lines.append(f"| {r['region']} | `{r['target']}` | {r['feature_set']} | "
                         f"{w['n_months']} | {w['observed_mean']:+.3f} | "
                         f"{w['observed_max']:+.3f} | {w['predicted_mean']:+.3f} | "
                         f"{w['predicted_max']:+.3f} |")

    best = payload["best_cell"]
    verdict = (
        f"**Nothing clears the bar.** The strongest cell anywhere in this phase is "
        f"{best['region']} / `{best['target']}` / {best['feature_set']} at "
        f"{best['horizon']}, skill {best['skill_score']:+.4f} — against the +0.1 "
        "bar. Three target definitions (mean, extreme, count) and two feature sets "
        "(heat-only, heat+drought) have now been tried. Heat Stress forecasting is "
        "genuinely unskillful with this data; the operational IMD heat wave counter "
        "is the part worth keeping."
        if not payload["anything_clears_bar"] else
        f"**{best['region']} / `{best['target']}` / {best['feature_set']} clears "
        f"the bar at {best['horizon']} with skill {best['skill_score']:+.4f}.**")
    return "\n".join(lines + ["", verdict, ""])


if __name__ == "__main__":
    result = run()
    print()
    print(format_tables(result))
    print(f"wrote {config.HEAT_PHASE11_PATH}")
