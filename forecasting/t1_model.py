"""Phase 1.4 §3 — a dedicated 1-month-ahead Ridge model.

Only reached because the Section-1 check confirmed the signal replicates: Barmer
positive in 4 of 4 windows, Rajasthan 3 of 4. Two changes from the joint model,
both pre-specified:

* trained on **t+1 alone**, rather than splitting capacity across three horizons
  where two show no skill;
* lookback chosen from {12, 24, 60} months **by validation performance averaged
  across all four windows** — not by test performance, and not on window A alone.

Validation skill (``1 - RMSE_val/RMSE_val_climatology``) is the selection metric
rather than raw validation MSE, because the four windows cover periods with
different SPI-3 variance and raw MSE is not comparable across them.

Run standalone:  python -m forecasting.t1_model
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forecasting import config
from forecasting.baseline_ridge import fit_ridge_baseline, flatten_windows
from forecasting.evaluate import _scores, climatology_baseline
from forecasting.metrics import skill_from_predictions
from forecasting.split import SplitWindows, prepare_dataset

LOOKBACKS = (12, 24, 60)
HORIZON = 1                     # t+1 only — the whole point of this model


def _skill(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> float:
    return skill_from_predictions(y_true, y_pred, y_base)


def fit_one(region: str, window_name: str, windows: SplitWindows,
            lookback: int, features: list | None = None) -> dict:
    """One region × window × lookback: fit on train, select alpha on val, and
    record validation *and* test skill separately."""
    ds = prepare_dataset(region, save=False, windows=windows,
                         seq_len=lookback, horizon=HORIZON,
                         features=features)
    X_train, y_train = ds.get("train")
    X_val, y_val = ds.get("val")

    model, val_mse, alpha = fit_ridge_baseline(X_train, y_train, X_val, y_val)

    val_split = ds.splits["val"]
    val_base = climatology_baseline(ds, val_split["target_dates"])
    val_skill = _skill(y_val.ravel(),
                       model.predict(flatten_windows(X_val)).ravel(),
                       val_base.ravel())

    test = ds.splits["test"]
    test_pred = model.predict(flatten_windows(test["X"]))
    test_base = climatology_baseline(ds, test["target_dates"])
    scores = _scores(test["y"].ravel(), test_pred.ravel(), test_base.ravel())

    # Validity check, not a competitor: SPI-3 is a 3-month accumulation, so
    # SPI3(t+1) shares two of its three months with SPI3(t). Persistence —
    # "next month's SPI-3 equals this month's" — exploits that overlap with no
    # modelling at all. If Ridge cannot beat it, the skill above is mostly
    # mechanical, not meteorological.
    persistence = ds.frame[config.TARGET].reindex(
        test["target_dates"] - pd.DateOffset(months=1)).to_numpy(dtype=float)
    scores["persistence_skill_vs_climatology"] = round(
        _skill(test["y"].ravel(), persistence, test_base.ravel()), 4)
    scores["ridge_skill_vs_persistence"] = round(
        _skill(test["y"].ravel(), test_pred.ravel(), persistence), 4)

    return {
        "region": region, "window": window_name, "lookback": lookback,
        "alpha": float(alpha), "val_mse": round(float(val_mse), 4),
        "val_skill": round(val_skill, 4),
        "n_train_windows": int(len(X_train)),
        "n_test_windows": int(len(test["X"])),
        **scores,
    }


def select_lookback(rows: list[dict], region: str) -> tuple[int, dict]:
    """Pick by mean VALIDATION skill across the four windows. Test is not consulted."""
    by_lookback = {
        lb: round(float(np.mean([r["val_skill"] for r in rows
                                 if r["region"] == region and r["lookback"] == lb])), 4)
        for lb in LOOKBACKS
    }
    return max(by_lookback, key=by_lookback.get), by_lookback


def run() -> dict:
    rows = [
        fit_one(region, name, SplitWindows(*w), lookback)
        for region in config.REGIONS
        for name, w in config.ROLLING_WINDOWS.items()
        for lookback in LOOKBACKS
    ]

    summary = {}
    for region in config.REGIONS:
        chosen, val_means = select_lookback(rows, region)
        selected = [r for r in rows if r["region"] == region
                    and r["lookback"] == chosen]
        skills = [r["skill_score"] for r in selected]
        summary[region] = {
            "chosen_lookback": chosen,
            "chosen_by": "mean validation skill across all four windows",
            "mean_val_skill_by_lookback": val_means,
            "test_skill_by_window": {r["window"]: r["skill_score"] for r in selected},
            "test_r2_by_window": {r["window"]: r["r2"] for r in selected},
            "n_positive_windows": sum(1 for s in skills if s > 0),
            "n_windows": len(skills),
            "mean_test_skill": round(float(np.mean(skills)), 4),
            "worst_test_skill": round(float(min(skills)), 4),
            "best_test_skill": round(float(max(skills)), 4),
            "mean_persistence_skill_vs_climatology": round(float(np.mean(
                [r["persistence_skill_vs_climatology"] for r in selected])), 4),
            "mean_ridge_skill_vs_persistence": round(float(np.mean(
                [r["ridge_skill_vs_persistence"] for r in selected])), 4),
            "ridge_beats_persistence_in_windows": sum(
                1 for r in selected if r["ridge_skill_vs_persistence"] > 0),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "ridge_t1_dedicated",
        "horizon": "t+1",
        "lookbacks_searched": list(LOOKBACKS),
        "alphas_searched": list(config.RIDGE_ALPHAS),
        "selection_rule": ("lookback chosen by mean validation skill across all "
                           "four windows; test sets never consulted for selection"),
        "summary": summary,
        "all_runs": rows,
    }
    config.T1_METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _lookup(mapping: dict, key):
    """JSON round-trips dict keys to strings; in-memory they are ints."""
    return mapping[key] if key in mapping else mapping[str(key)]


def format_tables(payload: dict) -> str:
    names = list(config.ROLLING_WINDOWS)
    out = ["Mean VALIDATION skill by lookback (selection metric):",
           f"| Region | {' | '.join(f'{lb}mo' for lb in LOOKBACKS)} | chosen |",
           "|---|" + "---|" * (len(LOOKBACKS) + 1)]
    for region, s in payload["summary"].items():
        cells = " | ".join(f"{_lookup(s['mean_val_skill_by_lookback'], lb):+.4f}"
                           for lb in LOOKBACKS)
        out.append(f"| {region} | {cells} | **{s['chosen_lookback']}mo** |")

    out += ["", "TEST skill of the selected model, per window:",
            f"| Region | lookback | {' | '.join(names)} | positive | mean |",
            "|---|---|" + "---|" * (len(names) + 2)]
    for region, s in payload["summary"].items():
        cells = " | ".join(f"{s['test_skill_by_window'][n]:+.4f}" for n in names)
        out.append(f"| {region} | {s['chosen_lookback']}mo | {cells} | "
                   f"{s['n_positive_windows']}/{s['n_windows']} | "
                   f"{s['mean_test_skill']:+.4f} |")

    out += ["", "Validity check — how much of that is the SPI-3 overlap?",
            "| Region | Ridge vs climatology | Persistence vs climatology | "
            "Ridge vs persistence | beats persistence |",
            "|---|---|---|---|---|"]
    for region, s in payload["summary"].items():
        out.append(f"| {region} | {s['mean_test_skill']:+.4f} | "
                   f"{s['mean_persistence_skill_vs_climatology']:+.4f} | "
                   f"{s['mean_ridge_skill_vs_persistence']:+.4f} | "
                   f"{s['ridge_beats_persistence_in_windows']}/{s['n_windows']} |")
    return "\n".join(out)


if __name__ == "__main__":
    result = run()
    print(format_tables(result))
    print(f"\nwrote {config.T1_METRICS_PATH}")
