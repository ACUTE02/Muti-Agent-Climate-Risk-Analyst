"""Chronological split, train-only statistics, scaling, and sequence framing.

Nothing here shuffles. The order is deliberate and enforces Rule A:

1. split the cleaned frame by date;
2. fit the SPI-3 gamma parameters on **train rows only**, then use those same
   parameters to compute SPI-3 for train *and* val *and* test;
3. fit ``MinMaxScaler`` on **train rows only**, then transform everything;
4. build sliding windows.

Sliding-window note (deviation from CLAUDE.md §6, deliberate and flagged):
the spec fixes a 60-month input window but gives VAL only 48 months (2016-2019)
and TEST only 60 (2020-2024). A window that must lie *entirely* inside its own
partition would yield **zero** val and test sequences, which makes
``EarlyStopping(monitor="val_loss")`` and the 2020-2024 test metrics impossible.
So: every window's *targets* lie strictly inside one partition, train windows lie
entirely inside train, and val/test windows may reach back into earlier months for
their 60-month history. That history is strictly in the past relative to the
prediction — the same information a live forecast would have — so it is not
look-ahead leakage. The statistics that *would* leak (scaler, month_stats,
anomaly baseline, IQR bounds) remain train-only.

Run standalone:  python -m forecasting.split
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from forecasting import config
from forecasting.clean import clean_india_climate, clip_baseline
from forecasting.enso import attach_oni
from forecasting.iod import attach_iod
from forecasting.fetch_data import load_or_fetch_region


class SplitWindows(NamedTuple):
    """The three date ranges a run is built on. Phase 1.4 walks these backwards
    through the record; every window refits its own train-only statistics."""

    train: slice = config.TRAIN
    val: slice = config.VAL
    test: slice = config.TEST


DEFAULT_WINDOWS = SplitWindows()


@dataclass
class Dataset:
    """Everything downstream stages need, plus the dates for leakage assertions."""

    frame: pd.DataFrame                  # cleaned features + spi3 + oni, full period
    scaler: MinMaxScaler
    month_stats: pd.DataFrame
    spi_params: dict = field(default_factory=dict)
    region: str = config.DEFAULT_REGION
    windows: SplitWindows = DEFAULT_WINDOWS
    seq_len: int = config.SEQ_LEN
    horizon: int = config.HORIZON
    features: list = field(default_factory=lambda: list(config.FEATURES))
    splits: dict = field(default_factory=dict)   # name -> dict(X, y, target_dates,
    #                                              window_start, window_end)

    def get(self, name: str):
        s = self.splits[name]
        return s["X"], s["y"]


def split_by_date(df: pd.DataFrame, windows: SplitWindows = DEFAULT_WINDOWS
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / val / test partitions — no shuffling, ever."""
    return df.loc[windows.train], df.loc[windows.val], df.loc[windows.test]


def fit_month_stats(train_df: pd.DataFrame,
                    target: str = config.SPI_SOURCE) -> pd.DataFrame:
    """Per-calendar-month mean/std of ``target``, fit on TRAIN ROWS ONLY (Rule A).

    Computing this on the whole frame — as the original tech spec's reference code
    did — would leak val/test values into the statistics applied to val/test rows.

    Phase 1.1 retired the naive z-score SPI this fed, so nothing in the model
    consumes it now; it is still fit and saved as the Phase-1 ``month_stats.joblib``
    artifact and as the reference implementation the SPI-3 leakage test mirrors.
    """
    grouped = train_df.groupby(train_df.index.month)[target]
    agg = grouped.agg(["mean", "std"])
    agg.index.name = "month"
    return agg


def fit_spi3_params(train_df: pd.DataFrame,
                    accum: str = config.SPI_ACCUM) -> dict[int, dict]:
    """Fit the SPI-3 gamma parameters per calendar month, TRAIN ROWS ONLY (Rule A).

    Real SPI (McKee et al. 1993) is not a z-score of rainfall: a gamma is fit to
    the accumulated rainfall of each calendar month, and the fitted CDF is pushed
    through the inverse normal CDF. That is what gives SPI usable tails in *both*
    directions — the Phase-1 z-score proxy could not reach -1.5 at all.

    ``q`` carries the point mass at zero rainfall, which a gamma cannot represent.
    """
    params: dict[int, dict] = {}
    for month in range(1, 13):
        vals = train_df.loc[train_df.index.month == month, accum].dropna()
        if vals.empty:
            raise ValueError(f"no train rows for calendar month {month}")

        q = float((vals == 0).mean())          # P(dry 3-month window)
        nonzero = vals[vals > 0]
        alpha, _, beta = stats.gamma.fit(nonzero, floc=0)   # MLE, location fixed at 0
        params[month] = {"q": q, "alpha": float(alpha), "beta": float(beta),
                         "n_train": int(len(vals))}
    return params


def to_spi3(x: float, month: int, params: dict[int, dict]) -> float:
    """Transform one accumulated-rainfall value into SPI-3 with train-fit params."""
    if not np.isfinite(x):
        return np.nan
    p = params[month]
    if x <= 0:
        h = p["q"]
    else:
        h = p["q"] + (1 - p["q"]) * stats.gamma.cdf(x, a=p["alpha"], scale=p["beta"])
    h = min(max(h, 1e-6), 1 - 1e-6)            # clip before ppf to avoid +/-inf
    return float(stats.norm.ppf(h))


def inverse_spi3(z: float, month: int, params: dict[int, dict]) -> float:
    """SPI-3 back to an accumulated-rainfall total — the inverse of :func:`to_spi3`.

    Needed by the recursive forecaster: a predicted SPI-3 has to be turned back
    into implied rainfall before the next month's lag and rolling features can be
    built from it.
    """
    p = params[month]
    h = float(stats.norm.cdf(z))
    if h <= p["q"]:                       # inside the dry-month point mass
        return 0.0
    scaled = (h - p["q"]) / (1 - p["q"])
    scaled = min(max(scaled, 1e-9), 1 - 1e-9)
    return float(stats.gamma.ppf(scaled, a=p["alpha"], scale=p["beta"]))


def compute_spi3(df: pd.DataFrame, params: dict[int, dict],
                 accum: str = config.SPI_ACCUM) -> pd.Series:
    """Apply already-fit (train-only) gamma params to compute SPI-3 for any subset."""
    values = [to_spi3(x, m, params)
              for x, m in zip(df[accum].to_numpy(dtype=float), df.index.month)]
    return pd.Series(values, index=df.index, name=config.TARGET)


def fit_scaler(frame: pd.DataFrame,
               windows: SplitWindows = DEFAULT_WINDOWS,
               features: list | None = None) -> MinMaxScaler:
    """Fit the feature scaler on TRAIN ROWS ONLY (Rule A)."""
    scaler = MinMaxScaler()
    scaler.fit(frame.loc[windows.train, features or config.FEATURES])
    return scaler


def build_windows(X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex,
                  in_partition: np.ndarray, strict_history: bool,
                  seq_len: int = config.SEQ_LEN,
                  horizon: int = config.HORIZON) -> dict:
    """Slide a ``seq_len``-month window over the series.

    A window is kept when all ``horizon`` targets fall inside the partition. With
    ``strict_history=True`` the 60 input months must also be inside it (used for
    train, so no train window ever touches val/test).
    """
    Xs, Ys, target_dates, win_start, win_end = [], [], [], [], []
    n = len(dates)
    for i in range(n - seq_len - horizon + 1):
        t_idx = range(i + seq_len, i + seq_len + horizon)
        if not in_partition[list(t_idx)].all():
            continue
        if strict_history and not in_partition[i:i + seq_len].all():
            continue
        Xs.append(X[i:i + seq_len])
        Ys.append(y[list(t_idx)])
        target_dates.append(dates[i + seq_len])
        win_start.append(dates[i])
        win_end.append(dates[i + seq_len - 1])

    return {
        "X": np.asarray(Xs, dtype="float32"),
        "y": np.asarray(Ys, dtype="float32"),
        "target_dates": pd.DatetimeIndex(target_dates),
        "window_start": pd.DatetimeIndex(win_start),
        "window_end": pd.DatetimeIndex(win_end),
    }


def prepare_dataset(region: str = config.DEFAULT_REGION,
                    save: bool = True,
                    windows: SplitWindows = DEFAULT_WINDOWS,
                    seq_len: int = config.SEQ_LEN,
                    horizon: int = config.HORIZON,
                    features: list | None = None) -> Dataset:
    """Full leak-free pipeline: raw -> cleaned -> split -> SPI -> scale -> windows.

    ``windows`` selects the train/val/test date ranges; every statistic below is
    refit on whichever train partition it names, so alternative windows stay as
    leak-free as the standing one.
    """
    config.check_region(region)
    features = list(features or config.FEATURES)
    raw = load_or_fetch_region(region)
    # Both statistics computed inside clean() — the IQR outlier bounds and the
    # anomaly baseline — must come from THIS window's train partition (Rule A/C).
    cleaned = attach_oni(clean_india_climate(
        raw,
        train_end=windows.train.stop,
        baseline=clip_baseline(config.BASELINE, windows.train),
    ))
    # Phase 1.6 tested the IOD and rejected it (models/iod_comparison.json), so
    # the DMI is merged only when a caller explicitly asks for those columns.
    # The default path stays exactly what Phase 1.5 measured and committed.
    if any(f.startswith("iod") for f in features):
        cleaned = attach_iod(cleaned)

    # --- 1. split first -----------------------------------------------------
    train_df, val_df, test_df = split_by_date(cleaned, windows)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("A partition came out empty — check the fetched date range.")

    # --- 2. train-only statistics -> SPI-3 everywhere -----------------------
    month_stats = fit_month_stats(train_df)          # retained Phase-1 artifact
    spi_params = fit_spi3_params(train_df)
    frame = cleaned.copy()
    frame[config.TARGET] = compute_spi3(frame, spi_params)
    if save:
        frame.to_parquet(config.processed_path(region))

    # --- 3. train-only scaler ----------------------------------------------
    scaler = fit_scaler(frame, windows, features)
    X_all = scaler.transform(frame[features]).astype("float32")
    y_all = frame[config.TARGET].to_numpy(dtype="float32")   # raw SPI units
    dates = pd.DatetimeIndex(frame.index)

    if save:
        joblib.dump(scaler, config.scaler_path(region))
        joblib.dump(month_stats, config.month_stats_path(region))
        joblib.dump(spi_params, config.spi_params_path(region))

    # --- 4. windows ---------------------------------------------------------
    masks = {
        "train": dates.isin(frame.loc[windows.train].index),
        "val": dates.isin(frame.loc[windows.val].index),
        "test": dates.isin(frame.loc[windows.test].index),
    }
    splits = {
        name: build_windows(X_all, y_all, dates, mask,
                            strict_history=(name == "train"),
                            seq_len=seq_len, horizon=horizon)
        for name, mask in masks.items()
    }
    return Dataset(frame=frame, scaler=scaler, month_stats=month_stats,
                   spi_params=spi_params, splits=splits, region=region,
                   windows=windows, seq_len=seq_len, horizon=horizon,
                   features=features)


def latest_window(frame: pd.DataFrame, scaler: MinMaxScaler,
                  seq_len: int = config.SEQ_LEN,
                  features: list | None = None) -> tuple[np.ndarray, pd.Timestamp]:
    """Most recent ``seq_len`` months of scaled features, for inference."""
    tail = frame[features or config.FEATURES].iloc[-seq_len:]
    if len(tail) < seq_len:
        raise ValueError(f"Need {seq_len} months of history, have {len(tail)}")
    X = scaler.transform(tail).astype("float32")[None, ...]
    return X, tail.index[-1]


if __name__ == "__main__":
    import sys

    ds = prepare_dataset(sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_REGION)
    for name, s in ds.splits.items():
        print(f"{name:5s} X={s['X'].shape} y={s['y'].shape} "
              f"targets {s['target_dates'].min().date()} -> "
              f"{s['target_dates'].max().date()}")
    spi3 = ds.frame[config.TARGET]
    print(f"\nspi3 over the full record: mean={spi3.mean():+.3f} std={spi3.std():.3f} "
          f"min={spi3.min():+.2f} max={spi3.max():+.2f}")
    print(f"months below -1.5 (Severe): {(spi3 < config.SPI_SEVERE).sum()}, "
          f"below -1.0 (Moderate): {(spi3 < config.SPI_MODERATE).sum()}")
    print("\nspi3 gamma params (train-only):")
    for month, p in ds.spi_params.items():
        print(f"  {month:2d}  q={p['q']:.3f} alpha={p['alpha']:8.4f} "
              f"beta={p['beta']:9.3f} n={p['n_train']}")
