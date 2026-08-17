"""Leak-free cleaning and feature engineering.

Three rules govern this module (see CLAUDE.md §4):

* **Rule A — split before you fit anything.** Any statistic used to *transform* a
  column is fit on the train partition only. The IQR outlier bounds below are
  therefore derived from train rows alone, and SPI is not computed here at all —
  it needs train-only per-month stats, so it lives in :mod:`forecasting.split`.
* **Rule B — no two-sided smoothing.** Every rolling window in this file is causal
  (``center=False``, the pandas default). No ``seasonal_decompose`` anywhere.
* **Rule C — the anomaly baseline is a fixed historical window** (1981-2010),
  which sits inside the train range, not "the whole series".

Run standalone:  python -m forecasting.clean
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting import config

LAGS = (1, 3, 6, 12)
OUTLIER_IQR_K = 3.0        # replace points beyond Q1/Q3 -/+ 3*IQR
OUTLIER_MEDIAN_WINDOW = 12  # months of *past* values used for the replacement median
INTERPOLATE_LIMIT = 2       # fill gaps shorter than 3 months


def _monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.asfreq("MS")


def _replace_outliers_causal(series: pd.Series, train_end: str) -> pd.Series:
    """Clip IQR outliers, replacing them with a strictly backward-looking median.

    Bounds come from the train partition only (Rule A); the replacement value is
    the median of the previous ``OUTLIER_MEDIAN_WINDOW`` months, shifted by one so
    the outlier itself never contributes (Rule B).
    """
    train_vals = series.loc[:train_end].dropna()
    q1, q3 = train_vals.quantile(0.25), train_vals.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - OUTLIER_IQR_K * iqr, q3 + OUTLIER_IQR_K * iqr

    past_median = (series.shift(1)
                         .rolling(OUTLIER_MEDIAN_WINDOW, min_periods=1)
                         .median())
    is_outlier = (series < lo) | (series > hi)
    return series.mask(is_outlier & past_median.notna(), past_median)


def clip_baseline(baseline: slice, train: slice) -> slice:
    """Keep the anomaly baseline strictly inside the train partition (Rule C).

    The standing 1981-2010 baseline sits inside the standing 1980-2015 train
    range, but Phase 1.4's alternative windows train on shorter spans — window D
    ends in 2000 — and an unclipped baseline would reach past its train partition
    into val/test. Clipping keeps every window as leak-free as the standing one.
    """
    start = max(str(baseline.start), str(train.start))
    stop = min(str(baseline.stop), str(train.stop))
    if start > stop:
        raise ValueError(f"baseline {baseline} does not overlap train {train}")
    return slice(start, stop)


def clean_india_climate(df: pd.DataFrame,
                        target: str = config.SPI_SOURCE,
                        train_end: str | None = None,
                        baseline: slice | None = None) -> pd.DataFrame:
    """Interpolate gaps<3mo -> IQR outlier replace (causal rolling median) ->
    anomaly (1981-2010 baseline) -> cyclical month encoding -> lag features
    (1,3,6,12) -> causal rolling features (roll3_mean, roll12_sum) -> the 3-month
    causal accumulation SPI-3 is fit to (rainfall_roll3_sum).

    Does NOT compute SPI-3 — that requires train-only gamma parameters, so it
    happens in split.py after the split (Rule A).
    """
    train_end = train_end or config.TRAIN.stop
    baseline = baseline or config.BASELINE
    out = _monthly_index(df)

    # 1. gaps shorter than 3 months -> time interpolation; longer gaps stay NaN
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].interpolate(method="time",
                                            limit=INTERPOLATE_LIMIT,
                                            limit_area="inside")

    # 2. outliers on the target, bounds fit on train rows only
    out[target] = _replace_outliers_causal(out[target], train_end)

    # 3. anomaly vs a fixed monthly climatology from a historical window that
    #    lies entirely inside the train partition (Rule C)
    baseline_vals = out.loc[baseline, target]
    baseline_by_month = baseline_vals.groupby(baseline_vals.index.month).mean()
    out["anomaly"] = out[target] - out.index.month.map(baseline_by_month)

    # 4. cyclical month encoding
    month = out.index.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    # 5. lag features (strictly past values)
    for lag in LAGS:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)

    # 6. causal rolling features — center=False is the pandas default (Rule B)
    out["roll3_mean"] = out[target].rolling(3, min_periods=1).mean()
    out["roll12_sum"] = out[target].rolling(12, min_periods=1).sum()

    # 6b. the SPI-3 accumulation: a full 3-month total ending at each month.
    # Distinct from roll3_mean — min_periods=3 so a partial window is never
    # passed to the gamma fit as if it were a complete 3-month accumulation.
    out[config.SPI_ACCUM] = out[target].rolling(config.SPI_ACCUM_MONTHS,
                                                min_periods=config.SPI_ACCUM_MONTHS).sum()

    # 7. drop the warm-up rows the lags left NaN, plus any un-interpolated gap
    return out.dropna()


if __name__ == "__main__":
    from forecasting.fetch_data import load_or_fetch_region

    raw = load_or_fetch_region(config.DEFAULT_REGION)
    cleaned = clean_india_climate(raw)
    cleaned.to_parquet(config.processed_path(config.DEFAULT_REGION))
    print(cleaned.shape, cleaned.index.min().date(), "->", cleaned.index.max().date())
    print(cleaned.tail())
    print("NaNs:", int(cleaned.isna().sum().sum()))
