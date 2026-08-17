"""Phase 1.5 — recursive (chained) forecasting vs. direct prediction.

Phase 1.4 validated a 1-month-ahead Ridge model. This asks whether t+2 and t+3 are
better served by *chaining* that model — predict t+1, feed the prediction back as
if observed, predict t+2, and so on — than by predicting them directly from the
original window. Measured both ways on the same four windows, no assumed error
arithmetic.

**What is exactly known vs. approximated when stepping forward**

* ``month_sin`` / ``month_cos`` — exactly known. The calendar is not a forecast.
* ``rainfall_mm`` and everything derived from it (lags, ``roll3_mean``,
  ``roll12_sum``, ``anomaly``) — reconstructed from the predicted SPI-3 by
  inverting the gamma transform: a predicted SPI-3 implies a 3-month accumulation,
  and subtracting the two known preceding months gives the implied new month.
  Approximate, and errors compound with each step. That is the honest cost of
  recursion and part of what this comparison measures.
* ``oni`` — **held at the last observed value.** A real limitation, not hidden:
  ENSO does evolve over a 3-month horizon, and using a stale value understates
  what a live system with a real ENSO forecast could do. Fabricating a future ONI
  would be worse, so the flat hold is stated rather than dressed up.

Run standalone:  python -m forecasting.recursive
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from forecasting import config
from forecasting.baseline_ridge import fit_ridge_baseline, flatten_windows
from forecasting.clean import clip_baseline
from forecasting.evaluate import _scores, climatology_baseline
from forecasting.split import SplitWindows, inverse_spi3, prepare_dataset
from forecasting.t1_model import LOOKBACKS, _skill

PRODUCTION_WINDOW = "A"          # the standing 1980-2015 / 2016-2019 / 2020-2024 split
LABEL_VALIDATED = 0.1            # the bar used throughout Phase 1.3/1.4


def horizon_label(skill: float) -> str:
    if skill >= LABEL_VALIDATED:
        return "validated"
    if skill > 0:
        return "weak/directional"
    return "no skill — shown for context only, do not rely on this figure"


# --------------------------------------------------------------------------- #
# Recursive forecasting
# --------------------------------------------------------------------------- #
def _baseline_by_month(ds) -> pd.Series:
    """The same anomaly climatology clean() used, for rebuilding anomaly rows."""
    baseline = clip_baseline(config.BASELINE, ds.windows.train)
    vals = ds.frame.loc[baseline, config.SPI_SOURCE]
    return vals.groupby(vals.index.month).mean()


def _next_feature_row(date: pd.Timestamp, spi3: float, rain: list[float],
                      last_oni: float, baseline_by_month: pd.Series) -> np.ndarray:
    """One synthesised month, in exactly ``config.FEATURES`` order."""
    values = {
        "rainfall_mm": rain[-1],
        "spi3": spi3,
        "anomaly": rain[-1] - float(baseline_by_month[date.month]),
        "month_sin": np.sin(2 * np.pi * date.month / 12),   # exactly known
        "month_cos": np.cos(2 * np.pi * date.month / 12),   # exactly known
        "rainfall_mm_lag1": rain[-2],
        "rainfall_mm_lag3": rain[-4],
        "rainfall_mm_lag6": rain[-7],
        "rainfall_mm_lag12": rain[-13],
        "roll3_mean": float(np.mean(rain[-3:])),
        "roll12_sum": float(np.sum(rain[-12:])),
        "oni": last_oni,                                     # held, see module docstring
    }
    return np.array([values[f] for f in config.FEATURES], dtype=float)


def recursive_forecast(model, ds, origin: pd.Timestamp, lookback: int,
                       horizon: int = config.HORIZON,
                       baseline_by_month: pd.Series | None = None) -> list[float]:
    """Chain the 1-month model forward ``horizon`` steps from ``origin``.

    ``origin`` is the last observed month; the window fed to the model is the
    ``lookback`` months ending there.
    """
    frame = ds.frame
    baseline_by_month = (_baseline_by_month(ds) if baseline_by_month is None
                         else baseline_by_month)

    end = frame.index.get_loc(origin) + 1
    window = frame[config.FEATURES].to_numpy(dtype=float)[end - lookback:end].copy()
    rain = list(frame[config.SPI_SOURCE].to_numpy(dtype=float)[max(0, end - 30):end])
    last_oni = float(frame["oni"].to_numpy()[end - 1])

    preds, date = [], origin
    for _ in range(horizon):
        z = float(model.predict(
            ds.scaler.transform(window).reshape(1, -1)).ravel()[0])
        preds.append(z)

        date = date + pd.DateOffset(months=1)
        # Invert the prediction back to implied rainfall so the next row's lag and
        # rolling features can be built. Two of the three months in the implied
        # accumulation are already known.
        accum = inverse_spi3(z, date.month, ds.spi_params)
        rain.append(max(0.0, accum - rain[-1] - rain[-2]))
        window = np.vstack([window[1:],
                            _next_feature_row(date, z, rain, last_oni,
                                              baseline_by_month)])
    return preds


# --------------------------------------------------------------------------- #
# Direct dedicated models per horizon (same process as Phase 1.4 §3)
# --------------------------------------------------------------------------- #
def fit_direct(region: str, windows: SplitWindows, horizon: int, lookback: int):
    """Ridge trained to predict horizon ``h`` alone from a ``lookback``-month window."""
    ds = prepare_dataset(region, save=False, windows=windows,
                         seq_len=lookback, horizon=horizon)
    col = horizon - 1
    X_train, y_train = ds.get("train")
    X_val, y_val = ds.get("val")

    model, val_mse, alpha = fit_ridge_baseline(
        X_train, y_train[:, [col]], X_val, y_val[:, [col]])

    val_split = ds.splits["val"]
    val_base = climatology_baseline(ds, val_split["target_dates"])[:, col]
    val_skill = _skill(y_val[:, col],
                       model.predict(flatten_windows(X_val)).ravel(),
                       val_base)
    return model, alpha, val_skill, ds


def select_direct_lookback(region: str, horizon: int) -> tuple[int, dict]:
    """Lookback by mean VALIDATION skill across all four windows — never test."""
    means = {}
    for lookback in LOOKBACKS:
        skills = [fit_direct(region, SplitWindows(*w), horizon, lookback)[2]
                  for w in config.ROLLING_WINDOWS.values()]
        means[lookback] = round(float(np.mean(skills)), 4)
    return max(means, key=means.get), means


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #
def compare(region: str, horizon: int, direct_lookback: int) -> list[dict]:
    """Recursive vs direct for one (region, horizon), on all four windows.

    Both are scored on the *same* test origins and targets — the horizon-3 window
    set — so the comparison is like for like.
    """
    rows = []
    for name, w in config.ROLLING_WINDOWS.items():
        windows = SplitWindows(*w)
        col = horizon - 1

        # --- recursive: chain the Phase-1.4 t+1 model (12mo) forward ----------
        ds1 = prepare_dataset(region, save=False, windows=windows,
                              seq_len=12, horizon=1)
        t1_model, _, _ = fit_ridge_baseline(*ds1.get("train"), *ds1.get("val"))

        ds_eval = prepare_dataset(region, save=False, windows=windows,
                                  seq_len=12, horizon=config.HORIZON)
        test = ds_eval.splits["test"]
        base = climatology_baseline(ds_eval, test["target_dates"])[:, col]
        baseline_by_month = _baseline_by_month(ds_eval)

        rec = np.array([
            recursive_forecast(t1_model, ds_eval, origin, 12,
                               baseline_by_month=baseline_by_month)
            for origin in test["window_end"]
        ])
        rows.append({"region": region, "horizon": f"t+{horizon}", "window": name,
                     "approach": "recursive", "lookback": 12,
                     "n_test_windows": int(len(rec)),
                     **_scores(test["y"][:, col], rec[:, col], base)})

        # --- direct: dedicated model for this horizon ------------------------
        model, alpha, _, _ = fit_direct(region, windows, horizon, direct_lookback)
        ds_direct = prepare_dataset(region, save=False, windows=windows,
                                    seq_len=direct_lookback, horizon=config.HORIZON)
        d_test = ds_direct.splits["test"]
        d_base = climatology_baseline(ds_direct, d_test["target_dates"])[:, col]
        d_pred = model.predict(flatten_windows(d_test["X"])).ravel()
        rows.append({"region": region, "horizon": f"t+{horizon}", "window": name,
                     "approach": "direct", "lookback": direct_lookback,
                     "alpha": float(alpha), "n_test_windows": int(len(d_pred)),
                     **_scores(d_test["y"][:, col], d_pred, d_base)})
    return rows


def run() -> dict:
    t1 = json.loads(config.T1_METRICS_PATH.read_text(encoding="utf-8"))

    rows, lookback_choices, winners = [], {}, {}
    for region in config.REGIONS:
        # t+1 is settled by Phase 1.4: direct, 12-month lookback.
        window_a = next(r for r in t1["all_runs"] if r["region"] == region
                        and r["window"] == PRODUCTION_WINDOW and r["lookback"] == 12)
        winners[(region, 1)] = {
            "method": "direct", "lookback": 12,
            "skill_score": t1["summary"][region]["mean_test_skill"],
            "rmse_window_a": window_a["rmse"],
        }
        for horizon in (2, 3):
            chosen, means = select_direct_lookback(region, horizon)
            lookback_choices[f"{region}_t+{horizon}"] = {
                "chosen": chosen, "mean_val_skill_by_lookback": means}

            horizon_rows = compare(region, horizon, chosen)
            rows += horizon_rows

            means_by_approach = {
                approach: round(float(np.mean(
                    [r["skill_score"] for r in horizon_rows
                     if r["approach"] == approach])), 4)
                for approach in ("recursive", "direct")
            }
            best = max(means_by_approach, key=means_by_approach.get)
            window_a = next(r for r in horizon_rows
                            if r["approach"] == best and r["window"] == PRODUCTION_WINDOW)
            winners[(region, horizon)] = {
                "method": best,
                "lookback": 12 if best == "recursive" else chosen,
                "skill_score": means_by_approach[best],
                "rmse_window_a": window_a["rmse"],
                "mean_skill_by_approach": means_by_approach,
            }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": ("For t+2 and t+3, does chaining the validated t+1 model beat "
                     "predicting those horizons directly?"),
        "oni_limitation": ("recursive steps hold ONI at its last observed value; "
                           "a live system with a real ENSO forecast could do better"),
        "direct_lookback_selection": lookback_choices,
        "results": rows,
        "winners": {f"{r}_t+{h}": v for (r, h), v in winners.items()},
    }
    config.HORIZON_COMPARISON_PATH.write_text(json.dumps(payload, indent=2),
                                              encoding="utf-8")
    _write_manifest(winners)
    return payload


def _write_manifest(winners: dict) -> dict:
    """Persist the winning method per (region, horizon) plus production models.

    Production models are fit on window A — the standing split — so the tool
    serves forecasts from the same split every other artifact reports on.
    """
    windows = SplitWindows(*config.ROLLING_WINDOWS[PRODUCTION_WINDOW])
    manifest = {"production_window": PRODUCTION_WINDOW, "regions": {}}

    for region in config.REGIONS:
        entries = []
        for horizon in range(1, config.HORIZON + 1):
            w = winners[(region, horizon)]
            if w["method"] == "direct":
                if horizon == 1:
                    ds = prepare_dataset(region, save=False, windows=windows,
                                         seq_len=w["lookback"], horizon=1)
                    model, _, _ = fit_ridge_baseline(*ds.get("train"), *ds.get("val"))
                else:
                    model, _, _, _ = fit_direct(region, windows, horizon,
                                                w["lookback"])
                joblib.dump(model, config.horizon_model_path(region, horizon))
            entries.append({
                "horizon": f"t+{horizon}",
                "method": w["method"],
                "lookback": w["lookback"],
                "skill_score": w["skill_score"],
                "rmse_window_a": w["rmse_window_a"],
                "label": horizon_label(w["skill_score"]),
            })
        manifest["regions"][region] = entries

    config.HORIZON_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")
    return manifest


def format_table(payload: dict) -> str:
    names = list(config.ROLLING_WINDOWS)
    lines = [f"| Horizon | Region | Approach | {' | '.join(names)} | mean skill |",
             "|---|---|---|" + "---|" * (len(names) + 1)]
    for horizon in (2, 3):
        for region in config.REGIONS:
            for approach in ("recursive", "direct"):
                sel = [r for r in payload["results"]
                       if r["region"] == region and r["horizon"] == f"t+{horizon}"
                       and r["approach"] == approach]
                cells = " | ".join(f"{r['skill_score']:+.4f}" for r in sel)
                mean = np.mean([r["skill_score"] for r in sel])
                win = payload["winners"][f"{region}_t+{horizon}"]["method"] == approach
                mark = " **←**" if win else ""
                lines.append(f"| t+{horizon} | {region} | {approach} | {cells} | "
                             f"{mean:+.4f}{mark} |")
    return "\n".join(lines)


if __name__ == "__main__":
    result = run()
    print(format_table(result))
    print()
    manifest = json.loads(config.HORIZON_MANIFEST_PATH.read_text(encoding="utf-8"))
    for region, entries in manifest["regions"].items():
        print(f"{region}:")
        for e in entries:
            print(f"  {e['horizon']}  {e['method']:9s} lookback={e['lookback']:2d}mo "
                  f"skill={e['skill_score']:+.4f}  {e['label']}")
    print(f"\nwrote {config.HORIZON_COMPARISON_PATH}")
    print(f"wrote {config.HORIZON_MANIFEST_PATH}")
