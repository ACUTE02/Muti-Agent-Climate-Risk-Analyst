"""Phase 1.6 — does the Indian Ocean Dipole add anything beyond ENSO?

One bounded, single-hypothesis test on an otherwise frozen pipeline. Nothing
changes except the feature set: same target, same model type, same regions, same
windows, same lookback grid, same alpha grid, same selection rule.

Both variants are re-run here rather than compared against numbers on disk, so the
"without IOD" column is produced by exactly the code that produces the "with IOD"
column — the only honest way to attribute a difference to the feature.

Adoption bar, matching every prior confirmation in this project: a horizon needs a
higher mean test skill *and* an improvement holding in at least 3 of the 4
historical windows.

Run standalone:  python -m forecasting.iod_check
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from forecasting import config
from forecasting.baseline_ridge import flatten_windows
from forecasting.enso import fetch_oni
from forecasting.evaluate import _scores, climatology_baseline
from forecasting.iod import fetch_iod
from forecasting.recursive import fit_direct
from forecasting.split import SplitWindows
from forecasting.t1_model import LOOKBACKS, fit_one

WINDOW_NAMES = list(config.ROLLING_WINDOWS)
MIN_IMPROVED_WINDOWS = 3


def iod_oni_correlation() -> dict:
    """Context for whichever way the result lands: is IOD even new information?"""
    oni, iod = fetch_oni(), fetch_iod()
    window = slice(config.FETCH_START, config.FETCH_END)
    a, b = oni.loc[window], iod.loc[window]
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    train = (common >= f"{config.TRAIN.start}-01-01") & \
            (common <= f"{config.TRAIN.stop}-12-31")
    return {
        "n_months": int(len(common)),
        "pearson_full_period": round(float(a.corr(b)), 4),
        "pearson_train_only": round(float(a[train].corr(b[train])), 4),
        "spearman_full_period": round(float(a.corr(b, method="spearman")), 4),
        "oni_std": round(float(a.std()), 4),
        "iod_std": round(float(b.std()), 4),
    }


def _score_direct(region: str, windows: SplitWindows, horizon: int,
                  lookback: int, features: list | None) -> dict:
    """Test scores for a dedicated direct model at horizon >= 2."""
    model, alpha, val_skill, ds = fit_direct(region, windows, horizon, lookback,
                                             features=features)
    col = horizon - 1
    test = ds.splits["test"]
    pred = model.predict(flatten_windows(test["X"])).ravel()
    base = climatology_baseline(ds, test["target_dates"])[:, col]
    return {"val_skill": round(float(val_skill), 4), "alpha": float(alpha),
            **_scores(test["y"][:, col], pred, base)}


def _score(region: str, window_name: str, horizon: int, lookback: int,
           features: list | None) -> dict:
    windows = SplitWindows(*config.ROLLING_WINDOWS[window_name])
    if horizon == 1:
        return fit_one(region, window_name, windows, lookback, features=features)
    return _score_direct(region, windows, horizon, lookback, features)


def run_variant(features: list | None, label: str) -> dict:
    """Full selection + evaluation for one feature set, both regions, 3 horizons."""
    out: dict = {}
    for region in config.REGIONS:
        out[region] = {}
        for horizon in range(1, config.HORIZON + 1):
            rows = {
                (lookback, name): _score(region, name, horizon, lookback, features)
                for lookback in LOOKBACKS for name in WINDOW_NAMES
            }
            # lookback by mean VALIDATION skill across all four windows
            val_means = {
                lookback: round(float(np.mean(
                    [rows[(lookback, n)]["val_skill"] for n in WINDOW_NAMES])), 4)
                for lookback in LOOKBACKS
            }
            chosen = max(val_means, key=val_means.get)
            by_window = {n: rows[(chosen, n)]["skill_score"] for n in WINDOW_NAMES}
            entry = {
                "chosen_lookback": chosen,
                "mean_val_skill_by_lookback": val_means,
                "test_skill_by_window": by_window,
                "mean_test_skill": round(float(np.mean(list(by_window.values()))), 4),
            }
            out[region][f"t+{horizon}"] = entry
            print(f"[{label}] {region:9s} t+{horizon}  lookback={chosen:2d}mo  "
                  f"mean test skill={entry['mean_test_skill']:+.4f}")
    return out


def decide(without: dict, with_iod: dict) -> dict:
    """Per horizon: better on the mean AND improved in >= 3 of 4 windows?"""
    verdicts = {}
    for region in config.REGIONS:
        for horizon in range(1, config.HORIZON + 1):
            key = f"t+{horizon}"
            a, b = without[region][key], with_iod[region][key]
            improved = [n for n in WINDOW_NAMES
                        if b["test_skill_by_window"][n] > a["test_skill_by_window"][n]]
            delta = round(b["mean_test_skill"] - a["mean_test_skill"], 4)
            verdicts[f"{region}|{key}"] = {
                "skill_without_iod": a["mean_test_skill"],
                "skill_with_iod": b["mean_test_skill"],
                "change": delta,
                "windows_improved": len(improved),
                "improved_in": improved,
                "adopt": bool(delta > 0 and len(improved) >= MIN_IMPROVED_WINDOWS),
            }
    return verdicts


def run() -> dict:
    correlation = iod_oni_correlation()
    print(f"IOD-ONI correlation: pearson(full)={correlation['pearson_full_period']:+.4f} "
          f"pearson(train)={correlation['pearson_train_only']:+.4f} "
          f"spearman={correlation['spearman_full_period']:+.4f}\n")

    without = run_variant(None, "no IOD")
    print()
    with_iod = run_variant(config.FEATURES_IOD, "with IOD")
    verdicts = decide(without, with_iod)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": "Does the IOD add forecast skill beyond ENSO for these sites?",
        "features_without_iod": list(config.FEATURES),
        "features_with_iod": list(config.FEATURES_IOD),
        "adoption_rule": (f"higher mean test skill AND improvement in at least "
                          f"{MIN_IMPROVED_WINDOWS} of {len(WINDOW_NAMES)} windows"),
        "iod_oni_correlation": correlation,
        "without_iod": without,
        "with_iod": with_iod,
        "verdicts": verdicts,
        "adopted_any": any(v["adopt"] for v in verdicts.values()),
    }
    config.IOD_COMPARISON_PATH.write_text(json.dumps(payload, indent=2),
                                          encoding="utf-8")
    return payload


def format_table(payload: dict) -> str:
    lines = ["| Region | Horizon | Skill without IOD | Skill with IOD | Change | "
             "Windows improved | Adopt |",
             "|---|---|---|---|---|---|---|"]
    for key, v in payload["verdicts"].items():
        region, horizon = key.split("|")
        lines.append(
            f"| {region} | {horizon} | {v['skill_without_iod']:+.4f} | "
            f"{v['skill_with_iod']:+.4f} | {v['change']:+.4f} | "
            f"{v['windows_improved']}/{len(WINDOW_NAMES)} | "
            f"{'YES' if v['adopt'] else 'no'} |")
    return "\n".join(lines)


if __name__ == "__main__":
    result = run()
    print()
    print(format_table(result))
    print()
    print("ADOPTED" if result["adopted_any"] else
          "NOT ADOPTED — IOD does not clear the bar at any horizon")
    print(f"wrote {config.IOD_COMPARISON_PATH}")
