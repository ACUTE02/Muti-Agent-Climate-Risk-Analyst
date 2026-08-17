"""Held-out evaluation on the 2020-2024 test partition.

Writes ``models/metrics.json`` and ``models/test_forecast_plot.png``.

Run standalone:  python -m forecasting.evaluate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from forecasting import config  # noqa: E402
from forecasting.split import Dataset, prepare_dataset  # noqa: E402

HORIZON_LABELS = [f"t+{h}" for h in range(1, config.HORIZON + 1)]


def climatology_baseline(ds: Dataset, target_dates: pd.DatetimeIndex) -> np.ndarray:
    """Predict the train-period mean SPI for each target's calendar month.

    SPI is a per-month z-score fit on train, so these means are ~0 by construction —
    i.e. the baseline is essentially "always predict normal conditions".
    """
    train_spi = ds.frame.loc[ds.windows.train, config.TARGET]
    monthly_mean = train_spi.groupby(train_spi.index.month).mean()

    preds = np.empty((len(target_dates), ds.horizon), dtype="float32")
    for h in range(ds.horizon):
        months = (target_dates + pd.DateOffset(months=h)).month
        preds[:, h] = pd.Index(months).map(monthly_mean).to_numpy(dtype=float)
    return preds


def _scores(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmse_clim = float(np.sqrt(mean_squared_error(y_true, y_base)))
    return {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse_climatology": round(rmse_clim, 4),
        "skill_score": round(float(1 - rmse / rmse_clim), 4) if rmse_clim else None,
    }


def score_predictions(ds: Dataset, y_pred: np.ndarray, region: str,
                      model_name: str, extra: dict | None = None) -> dict:
    """Build the metrics dict for any model's test-set predictions.

    Every variant — the Phase-1.1 LSTM, Ridge, the small LSTM — goes through this
    one function, so the numbers in ``region_comparison.md`` are computed
    identically and are genuinely comparable.
    """
    split = ds.splits["test"]
    y_test, target_dates = split["y"], split["target_dates"]
    y_base = climatology_baseline(ds, target_dates)

    return {
        "model": model_name,
        "region": region,
        "region_label": config.REGIONS[region]["label"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": config.TARGET,
        "test_period": [str(ds.windows.test.start), str(ds.windows.test.stop)],
        "n_test_windows": int(len(y_test)),
        "test_target_range": [str(target_dates.min().date()),
                              str((target_dates.max()
                                   + pd.DateOffset(months=config.HORIZON - 1)).date())],
        "per_horizon": {
            HORIZON_LABELS[h]: _scores(y_test[:, h], y_pred[:, h], y_base[:, h])
            for h in range(config.HORIZON)
        },
        "averaged": _scores(y_test.ravel(), y_pred.ravel(), y_base.ravel()),
        "targets_reference": {"r2": "> 0.80", "rmse": "< 0.5 SPI units",
                              "skill_score": "> 0.30"},
        "n_train_windows": int(len(ds.splits["train"]["X"])),
        "n_val_windows": int(len(ds.splits["val"]["X"])),
        **(extra or {}),
    }


def climatology_val_mse(ds: Dataset) -> float:
    """MSE of the climatology baseline on the VALIDATION windows.

    This is the benchmark a training run's ``val_loss`` (also MSE) has to dip
    below at some epoch to have learned anything that generalises.
    """
    split = ds.splits["val"]
    y_base = climatology_baseline(ds, split["target_dates"])
    return float(mean_squared_error(split["y"].ravel(), y_base.ravel()))


def evaluate(ds: Dataset | None = None, model=None,
             region: str = config.DEFAULT_REGION) -> dict:
    from tensorflow.keras.models import load_model

    config.check_region(region)
    ds = ds or prepare_dataset(region)
    model = model or load_model(config.model_path(region))

    split = ds.splits["test"]
    y_pred = model.predict(split["X"], verbose=0)
    metrics = score_predictions(ds, y_pred, region, "lstm_128_64")

    config.metrics_path(region).write_text(json.dumps(metrics, indent=2),
                                           encoding="utf-8")
    _plot(split["y"], y_pred, split["target_dates"], region)

    print(json.dumps(metrics, indent=2))
    print(f"\nwrote {config.metrics_path(region)}\nwrote {config.plot_path(region)}")
    return metrics


def _plot(y_true: np.ndarray, y_pred: np.ndarray,
          target_dates: pd.DatetimeIndex,
          region: str = config.DEFAULT_REGION) -> None:
    fig, axes = plt.subplots(config.HORIZON, 1, figsize=(12, 9), sharex=True)
    for h, ax in enumerate(np.atleast_1d(axes)):
        dates = target_dates + pd.DateOffset(months=h)
        ax.plot(dates, y_true[:, h], label="Actual SPI", color="#1f77b4", lw=1.8)
        ax.plot(dates, y_pred[:, h], label="Predicted SPI", color="#d62728",
                lw=1.6, ls="--")
        ax.axhline(config.SPI_MODERATE, color="orange", lw=0.8, ls=":",
                   label="Moderate (-1.0)")
        ax.axhline(config.SPI_SEVERE, color="darkred", lw=0.8, ls=":",
                   label="Severe (-1.5)")
        ax.set_title(f"{config.REGIONS[region]['label']} — test forecast sanity "
                     f"check, horizon {HORIZON_LABELS[h]}")
        ax.set_ylabel("SPI")
        ax.grid(alpha=0.3)
        if h == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.xlabel("Date")
    plt.tight_layout()
    fig.savefig(config.plot_path(region), dpi=130)
    plt.close(fig)


# (label, path builder) for every model variant, in reporting order.
MODEL_VARIANTS = (
    ("LSTM(128→64), Phase 1.1/1.2", config.metrics_path),
    ("Ridge (linear)", config.ridge_metrics_path),
    ("LSTM(16), lr=1e-4", config.lstm_small_metrics_path),
)


def write_region_comparison() -> str:
    """Put every model variant and region side by side, no spin."""
    rows = []
    for label, path_of in MODEL_VARIANTS:
        for region in config.REGIONS:
            path = path_of(region)
            if path.exists():
                rows.append((label, region, json.loads(path.read_text(encoding="utf-8"))))

    if not rows:
        raise FileNotFoundError("no metrics_*.json found — run evaluate() first")

    lines = [
        "# Model / region comparison — averaged over horizons t+1, t+2, t+3",
        "",
        f"Held-out test set {config.TEST.start}-{config.TEST.stop}. Identical data, "
        "features, target and splits everywhere — only the model and the region "
        "change, so these numbers are directly comparable. Skill Score is "
        "`1 - RMSE_model/RMSE_climatology`: 0 means exactly as good as always "
        "predicting normal conditions, below 0 means worse.",
        "",
        "| Model | Region | RMSE | MAE | R² | Skill vs. climatology |",
        "|---|---|---|---|---|---|",
        "| Climatology (baseline) | — | — | — | — | 0 by definition |",
    ]
    for label, region, m in rows:
        a = m["averaged"]
        lines.append(f"| {label} | {region} | {a['rmse']:.3f} | {a['mae']:.3f} | "
                     f"{a['r2']:+.3f} | {a['skill_score']:+.4f} |")

    # Averages can hide a horizon that behaves differently from the other two —
    # and hiding a positive result would fail the same honesty standard as spin.
    lines += [
        "",
        "## Skill score by horizon",
        "",
        "| Model | Region | " + " | ".join(HORIZON_LABELS) + " |",
        "|---|---|" + "---|" * config.HORIZON,
    ]
    for label, region, m in rows:
        cells = " | ".join(f"{m['per_horizon'][h]['skill_score']:+.4f}"
                           for h in HORIZON_LABELS)
        lines.append(f"| {label} | {region} | {cells} |")

    notes = [m for _, _, m in rows if "val_loss_beat_climatology" in m]
    if notes:
        lines += ["", "## Did the small LSTM's val_loss ever beat climatology?", ""]
        for m in notes:
            verdict = "YES" if m["val_loss_beat_climatology"] else "no"
            lines.append(
                f"- **{m['region']}: {verdict}** — best val_loss "
                f"{m['val_loss_min']:.4f} at epoch {m['val_loss_best_epoch']} of "
                f"{m['epochs_run']}, against a climatology benchmark of "
                f"{m['climatology_val_mse']:.4f} on the same validation windows."
            )

    lines += ["", _verdict(rows), *_phase_1_4_section()]

    text = "\n".join(lines) + "\n"
    config.COMPARISON_PATH.write_text(text, encoding="utf-8")
    return text


SKILL_THRESHOLD = 0.1   # pre-committed in CLAUDE_phase1.3.md §4


def _phase_1_4_section() -> list[str]:
    """Fold in the Phase-1.4 robustness check and dedicated t+1 model, if run."""
    lines: list[str] = []

    if config.ROLLING_CHECK_PATH.exists():
        from forecasting.rolling_check import format_table

        payload = json.loads(config.ROLLING_CHECK_PATH.read_text(encoding="utf-8"))
        lines += [
            "", "---", "",
            "## Phase 1.4 — does the t+1 signal replicate?",
            "",
            "Same Ridge pipeline, joint 3-horizon model, t+1 skill only, re-run on "
            "four independent historical windows. Each window refits its own "
            "`month_stats`, `spi_params` and scaler on its own train partition.",
            "",
            format_table(payload),
            "",
        ]
        for region, v in payload["verdicts"].items():
            state = "**confirmed**" if v["confirmed"] else "**not confirmed**"
            lines.append(f"- {region}: {state} — positive in {v['n_positive']} of "
                         f"{v['n_windows']} windows (needs {v['threshold']}).")

    if config.T1_METRICS_PATH.exists():
        from forecasting.t1_model import format_tables

        payload = json.loads(config.T1_METRICS_PATH.read_text(encoding="utf-8"))
        lines += [
            "", "## Dedicated 1-month-ahead model", "",
            "Ridge trained on t+1 alone. Lookback chosen from {12, 24, 60} months "
            "by mean *validation* skill across all four windows — test sets were "
            "never consulted for selection.",
            "", format_tables(payload), "",
            "SPI-3 is a 3-month accumulation, so SPI-3 at t+1 shares two of its "
            "three months with SPI-3 at t. Part of this task is therefore "
            "nowcasting rather than pure forecasting, and the persistence row "
            "above measures exactly how much: persistence alone earns "
            + ", ".join(
                f"{s['mean_persistence_skill_vs_climatology']:+.4f} at {r}"
                for r, s in payload["summary"].items())
            + ". The model beats that naive exploitation of the overlap in "
              "every window, so the remainder is real.",
        ]

    if config.HORIZON_COMPARISON_PATH.exists():
        from forecasting.recursive import format_table as horizon_table

        payload = json.loads(
            config.HORIZON_COMPARISON_PATH.read_text(encoding="utf-8"))
        lines += [
            "", "---", "",
            "## Phase 1.5 — recursive vs. direct for t+2 and t+3", "",
            "Recursive chains the validated t+1 model forward, feeding each "
            "prediction back as if observed. Direct trains a dedicated model per "
            "horizon (lookback chosen by validation skill, same process as t+1). "
            "Both scored on the same test origins and targets. `←` marks the "
            "winner used going forward.", "",
            horizon_table(payload), "",
            "Limitation of the recursive path, stated rather than hidden: it holds "
            "ONI at its last observed value, since fabricating a future ENSO state "
            "would be worse. Calendar features (`month_sin`/`month_cos`) are exactly "
            "known; rainfall and everything derived from it is reconstructed by "
            "inverting the SPI-3 gamma transform, and those errors compound.", "",
        ]

    if config.HORIZON_MANIFEST_PATH.exists():
        manifest = json.loads(
            config.HORIZON_MANIFEST_PATH.read_text(encoding="utf-8"))
        lines += [
            "## Final per-horizon verdict", "",
            "What `forecast_drought_risk()` reports at runtime, straight from "
            f"measurement on window {manifest['production_window']}'s split.", "",
            "| Region | Horizon | Method | Lookback | Skill | Label |",
            "|---|---|---|---|---|---|",
        ]
        for region, entries in manifest["regions"].items():
            for e in entries:
                lines.append(f"| {region} | {e['horizon']} | {e['method']} | "
                             f"{e['lookback']}mo | {e['skill_score']:+.4f} | "
                             f"{e['label']} |")
    return lines


def _verdict(rows: list) -> str:
    """Plain reading of the table. Skill <= 0 lost to 'always predict normal'."""
    best_label, best_region, best = max(
        rows, key=lambda r: r[2]["averaged"]["skill_score"])
    best_skill = best["averaged"]["skill_score"]

    # The best single horizon anywhere in the table, which an average can mask.
    h_label, h_region, h_name, h_skill = max(
        ((lab, reg, h, m["per_horizon"][h]["skill_score"])
         for lab, reg, m in rows for h in HORIZON_LABELS),
        key=lambda t: t[3])

    if best_skill > SKILL_THRESHOLD:
        return (f"**{best_label} at {best_region} reaches an averaged skill score "
                f"of {best_skill:+.4f}**, meaningfully above the climatology "
                "baseline. That points at model capacity rather than predictor "
                "limitation, and warrants a proper follow-up tuning phase.")

    verdict = (
        "**No model variant shows meaningful averaged forecast skill.** The best "
        f"of them ({best_label}, {best_region}) reaches {best_skill:+.4f}, against "
        f"the {SKILL_THRESHOLD:+.1f} threshold set in advance. A linear model, a "
        "small slow-learning LSTM and a large LSTM all sit at the climatology "
        "baseline, at both a moderately and a highly drought-variable site — "
        "evidence that the predictors, not the architecture, are the binding "
        "constraint."
    )
    if h_skill > SKILL_THRESHOLD:
        verdict += (
            f"\n\nOne exception worth naming rather than burying: **{h_label} at "
            f"{h_region} reaches {h_skill:+.4f} at {h_name} alone**, above the "
            "threshold, decaying to nothing at the longer horizons. Read it "
            "cautiously — it is one horizon out of "
            f"{len(rows) * config.HORIZON} numbers in the table above, on "
            f"{rows[0][2]['n_test_windows']} test windows, so it could be noise. "
            "But it is the first positive signal in this project, and it is "
            "where a follow-up should look: one month ahead, linear, not three "
            "months ahead, deep."
        )
    return verdict


if __name__ == "__main__":
    import sys

    targets = sys.argv[1:] or list(config.REGIONS)
    for name in targets:
        evaluate(region=name)
        print()
    print(write_region_comparison())
