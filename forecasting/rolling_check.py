"""Phase 1.4 — does the Barmer t+1 Ridge signal replicate on other periods?

Phase 1.3 found one positive number: Ridge, Barmer, t+1 only, skill +0.1314 on the
2020-2024 test set. One result from one fixed period. This re-runs the *same*
pipeline — same features, same target, same alpha grid, nothing new — across four
independent historical windows and reads the count plainly.

Validation, not tuning. Nothing here changes the model.

Each window refits its own ``month_stats``, ``spi_params`` and scaler on its own
train partition (Rule A), so window B/C/D are as leak-free as A. ONI needs no
refitting — it is an external global series, merely sliced to the dates in play.

Run standalone:  python -m forecasting.rolling_check
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from forecasting import config
from forecasting.baseline_ridge import fit_ridge_baseline, flatten_windows
from forecasting.evaluate import _scores, climatology_baseline
from forecasting.split import SplitWindows, prepare_dataset

HORIZON_INDEX = 0        # t+1 — the horizon in question
HORIZON_LABEL = "t+1"


def run_one(region: str, window_name: str, windows: SplitWindows) -> dict:
    """Fit Ridge on one region/window and score t+1 on that window's test set."""
    ds = prepare_dataset(region, save=False, windows=windows)
    X_train, y_train = ds.get("train")
    X_val, y_val = ds.get("val")

    model, val_mse, alpha = fit_ridge_baseline(X_train, y_train, X_val, y_val)

    split = ds.splits["test"]
    y_pred = model.predict(flatten_windows(split["X"]))
    y_base = climatology_baseline(ds, split["target_dates"])

    scores = _scores(split["y"][:, HORIZON_INDEX],
                     y_pred[:, HORIZON_INDEX],
                     y_base[:, HORIZON_INDEX])

    return {
        "region": region,
        "window": window_name,
        "train": [windows.train.start, windows.train.stop],
        "val": [windows.val.start, windows.val.stop],
        "test": [windows.test.start, windows.test.stop],
        "horizon": HORIZON_LABEL,
        "alpha": float(alpha),
        "val_mse": round(float(val_mse), 4),
        "n_train_windows": int(len(X_train)),
        "n_test_windows": int(len(split["X"])),
        **scores,
    }


def verdict(results: list[dict], region: str, threshold: int = 3) -> dict:
    """Strict count. 'Positive in 2 of 4' is not 'mostly confirmed'."""
    rows = [r for r in results if r["region"] == region]
    positive = [r for r in rows if r["skill_score"] > 0]
    confirmed = len(positive) >= threshold
    return {
        "region": region,
        "n_windows": len(rows),
        "n_positive": len(positive),
        "positive_windows": [r["window"] for r in positive],
        "threshold": f"{threshold} of {len(rows)}",
        "skill_by_window": {r["window"]: r["skill_score"] for r in rows},
        "mean_skill": round(float(np.mean([r["skill_score"] for r in rows])), 4),
        "confirmed": confirmed,
    }


def run(threshold: int = 3) -> dict:
    results = [
        run_one(region, name, SplitWindows(*w))
        for region in config.REGIONS
        for name, w in config.ROLLING_WINDOWS.items()
    ]
    verdicts = {r: verdict(results, r, threshold) for r in config.REGIONS}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": ("Does the Phase-1.3 Ridge t+1 skill replicate across "
                     "independent historical windows, or was 2020-2024 a "
                     "favourable slice?"),
        "model": "ridge",
        "horizon": HORIZON_LABEL,
        "alphas_searched": list(config.RIDGE_ALPHAS),
        "windows": {k: {"train": [v[0].start, v[0].stop],
                        "val": [v[1].start, v[1].stop],
                        "test": [v[2].start, v[2].stop]}
                    for k, v in config.ROLLING_WINDOWS.items()},
        "results": results,
        "verdicts": verdicts,
    }
    config.ROLLING_CHECK_PATH.write_text(json.dumps(payload, indent=2),
                                         encoding="utf-8")
    return payload


def format_table(payload: dict) -> str:
    names = list(config.ROLLING_WINDOWS)
    lines = [
        f"| Region | {' | '.join(names)} | positive | mean |",
        "|---|" + "---|" * (len(names) + 2),
    ]
    for region, v in payload["verdicts"].items():
        cells = " | ".join(f"{v['skill_by_window'][n]:+.4f}" for n in names)
        lines.append(f"| {region} | {cells} | {v['n_positive']}/{v['n_windows']} | "
                     f"{v['mean_skill']:+.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    result = run()
    print(format_table(result))
    print()
    for region, v in result["verdicts"].items():
        state = "CONFIRMED" if v["confirmed"] else "NOT CONFIRMED"
        print(f"{region}: {state} — positive in {v['n_positive']} of "
              f"{v['n_windows']} windows {v['positive_windows']} "
              f"(needs {v['threshold']})")
    print(f"\nwrote {config.ROLLING_CHECK_PATH}")
