"""Phase 1.3 ablation — a plain linear baseline.

The question this answers: does *any* exploitable signal exist in these features?
A deep model can fail for reasons of its own — capacity, optimization, too many
parameters for 358 training sequences. A Ridge regression on the same windows has
none of those failure modes. If it also cannot beat climatology, "there is no
signal here" is a much better supported conclusion than a deep model failing alone.

Run standalone:  python -m forecasting.baseline_ridge [region ...]
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from forecasting import config
from forecasting.evaluate import score_predictions
from forecasting.split import Dataset, prepare_dataset


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """(n, 60, n_features) -> (n, 60 * n_features), one row per window."""
    return X.reshape(X.shape[0], -1)


def fit_ridge_baseline(X_train, y_train, X_val, y_val,
                       alphas=config.RIDGE_ALPHAS):
    """Multi-output Ridge (sklearn supports (n, 3) targets natively).

    Alpha is selected by validation MSE — the only tuning in this phase, and a
    single small pre-committed grid over one hyperparameter, not an open search.
    """
    flat_train, flat_val = flatten_windows(X_train), flatten_windows(X_val)
    best = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(flat_train, y_train)
        val_mse = mean_squared_error(y_val, model.predict(flat_val))
        if best is None or val_mse < best[1]:
            best = (model, val_mse, alpha)
    return best      # (fitted model, val_mse, chosen alpha)


def run(region: str = config.DEFAULT_REGION, ds: Dataset | None = None) -> dict:
    ds = ds or prepare_dataset(region, save=False)
    X_train, y_train = ds.get("train")
    X_val, y_val = ds.get("val")

    model, val_mse, alpha = fit_ridge_baseline(X_train, y_train, X_val, y_val)
    print(f"[{region}] ridge alpha={alpha} val_mse={val_mse:.4f} "
          f"(searched {list(config.RIDGE_ALPHAS)})")

    y_pred = model.predict(flatten_windows(ds.splits["test"]["X"]))
    metrics = score_predictions(ds, y_pred, region, "ridge", extra={
        "alpha": float(alpha),
        "alphas_searched": list(config.RIDGE_ALPHAS),
        "val_mse": float(val_mse),
        "n_flattened_features": int(flatten_windows(X_train).shape[1]),
    })

    config.ridge_metrics_path(region).write_text(json.dumps(metrics, indent=2),
                                                 encoding="utf-8")
    a = metrics["averaged"]
    print(f"[{region}] ridge test: RMSE {a['rmse']:.3f}  R2 {a['r2']:+.3f}  "
          f"skill {a['skill_score']:+.4f}")
    print(f"wrote {config.ridge_metrics_path(region)}")
    return metrics


if __name__ == "__main__":
    import sys

    for name in (sys.argv[1:] or list(config.REGIONS)):
        run(name)
