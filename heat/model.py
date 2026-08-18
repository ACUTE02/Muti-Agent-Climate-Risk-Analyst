"""Heat Stress modelling — Ridge first, per Phase 1.3's finding.

The Drought Agent's ablation showed a linear model matching or beating both LSTM
variants on this data volume (358 training sequences), so this starts at Ridge
rather than working down to it. An LSTM gets built here only if Ridge shows real
signal that a nonlinear model might extend — not speculatively.

Both baselines are computed up front this time: climatology (the project-standard
skill denominator) *and* persistence, which the Drought Agent only added after the
fact in Phase 1.4 and found mattered.

Run standalone:  python -m heat.model
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from forecasting import config
from forecasting.baseline_ridge import fit_ridge_baseline, flatten_windows
from forecasting.recursive import horizon_label
from forecasting.split import DEFAULT_WINDOWS, SplitWindows
from heat.dataset import HeatDataset, prepare_heat_dataset

HORIZON_LABELS = [f"t+{h}" for h in range(1, config.HORIZON + 1)]


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def climatology_prediction(ds: HeatDataset,
                           target_dates: pd.DatetimeIndex) -> np.ndarray:
    """Train-period mean of *this run's* target for each target's calendar month.

    Must follow ``ds.target``, not the default: scoring a different target against
    heat_anomaly's climatology compares against a baseline that is not even
    predicting the right quantity, which inflates skill enormously.
    """
    train_vals = ds.frame.loc[ds.windows.train, ds.target]
    monthly_mean = train_vals.groupby(train_vals.index.month).mean()

    preds = np.empty((len(target_dates), ds.horizon), dtype="float32")
    for h in range(ds.horizon):
        months = (target_dates + pd.DateOffset(months=h)).month
        preds[:, h] = pd.Index(months).map(monthly_mean).to_numpy(dtype=float)
    return preds


def persistence_prediction(ds: HeatDataset, split: dict) -> np.ndarray:
    """"Next months look like this month" — the naive forecast to beat."""
    origin_values = ds.frame.loc[split["window_end"], ds.target].to_numpy()
    return np.repeat(origin_values[:, None], ds.horizon, axis=1)


def _scores(y_true, y_pred, y_clim, y_persist) -> dict:
    rmse = _rmse(y_true, y_pred)
    rmse_clim = _rmse(y_true, y_clim)
    rmse_persist = _rmse(y_true, y_persist)
    return {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse_climatology": round(rmse_clim, 4),
        "rmse_persistence": round(rmse_persist, 4),
        "skill_score": round(1 - rmse / rmse_clim, 4),
        "skill_vs_persistence": round(1 - rmse / rmse_persist, 4),
        "persistence_skill_vs_climatology": round(1 - rmse_persist / rmse_clim, 4),
        "label": horizon_label(round(1 - rmse / rmse_clim, 4)),
    }


def fit_and_score(region: str, lookback: int,
                  windows: SplitWindows = DEFAULT_WINDOWS) -> tuple:
    """One joint 3-horizon Ridge for one region and lookback."""
    ds = prepare_heat_dataset(region, windows=windows, seq_len=lookback)
    X_train, y_train = ds.get("train")
    X_val, y_val = ds.get("val")

    model, val_mse, alpha = fit_ridge_baseline(X_train, y_train, X_val, y_val)

    val_split = ds.splits["val"]
    val_skill = 1 - _rmse(y_val, model.predict(flatten_windows(X_val))) / _rmse(
        y_val, climatology_prediction(ds, val_split["target_dates"]))

    test = ds.splits["test"]
    y_pred = model.predict(flatten_windows(test["X"]))
    y_clim = climatology_prediction(ds, test["target_dates"])
    y_persist = persistence_prediction(ds, test)

    per_horizon = {
        HORIZON_LABELS[h]: _scores(test["y"][:, h], y_pred[:, h],
                                   y_clim[:, h], y_persist[:, h])
        for h in range(ds.horizon)
    }
    return model, ds, {
        "lookback": lookback,
        "alpha": float(alpha),
        "val_mse": round(float(val_mse), 4),
        "val_skill": round(float(val_skill), 4),
        "n_train_windows": int(len(X_train)),
        "n_test_windows": int(len(test["X"])),
        "per_horizon": per_horizon,
        "averaged": _scores(test["y"].ravel(), y_pred.ravel(),
                            y_clim.ravel(), y_persist.ravel()),
    }


def run_region(region: str) -> dict:
    """Select the lookback on validation, then report the test set honestly."""
    candidates = {}
    for lookback in config.HEAT_LOOKBACKS:
        model, ds, scores = fit_and_score(region, lookback)
        candidates[lookback] = (model, ds, scores)
        print(f"[{region}] lookback={lookback:2d}mo alpha={scores['alpha']:g} "
              f"val_skill={scores['val_skill']:+.4f}")

    chosen = max(candidates, key=lambda lb: candidates[lb][2]["val_skill"])
    model, ds, scores = candidates[chosen]

    joblib.dump(model, config.heat_model_path(region, 0))   # joint 3-horizon model
    joblib.dump(ds.climatology, config.heat_climatology_path(region))
    joblib.dump(ds.scaler, config.heat_scaler_path(region))

    metrics = {
        "region": region,
        "region_label": config.REGIONS[region]["label"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "ridge_joint_3horizon",
        "target": ds.target,
        "features": config.HEAT_FEATURES,
        "split": {"train": [config.TRAIN.start, config.TRAIN.stop],
                  "val": [config.VAL.start, config.VAL.stop],
                  "test": [config.TEST.start, config.TEST.stop]},
        "chosen_lookback": chosen,
        "chosen_by": "validation skill vs climatology",
        "val_skill_by_lookback": {lb: c[2]["val_skill"]
                                  for lb, c in candidates.items()},
        "target_distribution_check": ds.distribution,
        **{k: v for k, v in scores.items() if k != "lookback"},
    }
    config.heat_metrics_path(region).write_text(json.dumps(metrics, indent=2),
                                                encoding="utf-8")
    return metrics


def write_comparison(all_metrics: dict) -> str:
    lines = [
        "# Heat Stress — model / region comparison",
        "",
        f"Target `{config.HEAT_TARGET}`: standardised monthly Tmax anomaly, "
        "climatology fit on train rows only. Split is the project standard — "
        f"train {config.TRAIN.start}-{config.TRAIN.stop}, "
        f"val {config.VAL.start}-{config.VAL.stop}, "
        f"test {config.TEST.start}-{config.TEST.stop}. Ridge on flattened lookback "
        "windows, alpha from {0.1, 1, 10, 100, 1000} by validation MSE, lookback "
        "from {12, 24, 60} months by validation skill. Labels use the same bar as "
        "the Drought Agent: >= +0.1 validated, 0 to 0.1 weak/directional, "
        "<= 0 no skill.",
        "",
        "## Target distribution check (train partition)",
        "",
        "| Region | mean | std | skew | excess kurtosis | p5 (vs -1.645) | "
        "p95 (vs +1.645) | approx. normal |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for region, m in all_metrics.items():
        d = m["target_distribution_check"]
        lines.append(
            f"| {region} | {d['mean']:+.3f} | {d['std']:.3f} | {d['skew']:+.3f} | "
            f"{d['excess_kurtosis']:+.3f} | {d['percentiles_empirical']['p5']:+.3f} | "
            f"{d['percentiles_empirical']['p95']:+.3f} | "
            f"{'yes' if d['approximately_normal'] else 'NO'} |")

    lines += [
        "", "## Skill by horizon — vs climatology and vs persistence", "",
        "| Region | Lookback | Horizon | RMSE | R² | Skill vs clim. | "
        "Skill vs persistence | Persistence vs clim. | Label |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for region, m in all_metrics.items():
        for horizon, s in m["per_horizon"].items():
            lines.append(
                f"| {region} | {m['chosen_lookback']}mo | {horizon} | "
                f"{s['rmse']:.3f} | {s['r2']:+.3f} | {s['skill_score']:+.4f} | "
                f"{s['skill_vs_persistence']:+.4f} | "
                f"{s['persistence_skill_vs_climatology']:+.4f} | {s['label']} |")

    lines += ["", "## Lookback selection (validation skill)", "",
              "| Region | " + " | ".join(f"{lb}mo" for lb in config.HEAT_LOOKBACKS)
              + " | chosen |",
              "|---|" + "---|" * (len(config.HEAT_LOOKBACKS) + 1)]
    for region, m in all_metrics.items():
        cells = " | ".join(f"{m['val_skill_by_lookback'][lb]:+.4f}"
                           for lb in config.HEAT_LOOKBACKS)
        lines.append(f"| {region} | {cells} | **{m['chosen_lookback']}mo** |")

    lines += ["", _verdict(all_metrics)]
    text = "\n".join(lines) + "\n"
    config.HEAT_COMPARISON_PATH.write_text(text, encoding="utf-8")
    return text


def _verdict(all_metrics: dict) -> str:
    best = max(
        ((region, horizon, s["skill_score"])
         for region, m in all_metrics.items()
         for horizon, s in m["per_horizon"].items()),
        key=lambda t: t[2])
    region, horizon, skill = best

    if skill >= 0.1:
        return (f"**Validated signal: {region} at {horizon} reaches skill "
                f"{skill:+.4f}** against climatology. Whether it replicates on "
                "other historical windows is a separate question and would be a "
                "follow-up phase, not this one — same sequencing the Drought "
                "Agent used.")
    if skill > 0:
        return (f"**Weak/directional at best.** The strongest cell ({region}, "
                f"{horizon}) reaches {skill:+.4f} — above zero but below the "
                "+0.1 bar this project treats as validated. Not worth building "
                "on without more signal.")
    return ("**No skill at any horizon, either region.** Every skill score sits "
            "at or below zero, meaning nothing beats predicting the seasonal "
            "normal. Unlike the Drought Agent's first pass, this is not a target "
            "artifact: the target was distribution-checked before modelling and "
            "is near-normal. Reported and stopped here, per the phase's stopping "
            "rule.")


def write_manifest(all_metrics: dict) -> dict:
    manifest = {"model": "ridge_joint_3horizon", "regions": {}}
    for region, m in all_metrics.items():
        manifest["regions"][region] = {
            "lookback": m["chosen_lookback"],
            "horizons": [
                {"horizon": horizon, "skill_score": s["skill_score"],
                 "method": "ridge", "label": s["label"]}
                for horizon, s in m["per_horizon"].items()
            ],
        }
    config.HEAT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2),
                                         encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import sys

    results = {name: run_region(name)
               for name in (sys.argv[1:] or list(config.REGIONS))}
    write_manifest(results)
    print()
    print(write_comparison(results))
