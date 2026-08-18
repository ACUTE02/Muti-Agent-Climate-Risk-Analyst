"""Heat Stress dataset assembly — same leak-free discipline as the Drought Agent.

Order is deliberate and identical in spirit to ``forecasting/split.py``:

1. split by date first;
2. fit the Tmax climatology on **train rows only**, then apply it everywhere;
3. build causal features (lags and trailing rolling means — nothing centred);
4. fit the scaler on **train rows only**;
5. build sliding windows, reusing the Drought Agent's window builder so the
   causality guarantees are literally the same code.

Run standalone:  python -m heat.dataset [region]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from forecasting import config
from forecasting.enso import fetch_oni
from forecasting.fetch_data import load_or_fetch_daily
from forecasting.split import DEFAULT_WINDOWS, SplitWindows, build_windows
from heat.target import (check_distribution, compute_heat_anomaly,
                         compute_heat_extreme, fit_daily_normals,
                         fit_daily_tmax_climatology, fit_heat_climatology,
                         flag_heat_wave_days, monthly_heat_wave_counts,
                         monthly_max_tmax, monthly_tmax, zero_inflation)


@dataclass
class HeatDataset:
    frame: pd.DataFrame                  # features + target + operational counts
    scaler: MinMaxScaler
    climatology: pd.DataFrame
    normals: pd.Series
    distribution: dict
    region: str = config.DEFAULT_REGION
    target: str = config.HEAT_TARGET
    features: list = field(default_factory=lambda: list(config.HEAT_FEATURES))
    diagnostics: dict = field(default_factory=dict)
    windows: SplitWindows = DEFAULT_WINDOWS
    seq_len: int = 12
    horizon: int = config.HORIZON
    splits: dict = field(default_factory=dict)

    def get(self, name: str):
        return self.splits[name]["X"], self.splits[name]["y"]


def load_drought_spi3(region: str, windows: SplitWindows) -> pd.Series:
    """Cross-agent dependency: SPI-3 comes from the Drought Agent's pipeline.

    Motivation is land-atmosphere feedback — dry soil means less evaporative
    cooling, so antecedent dryness plausibly precedes hotter days. Rebuilt through
    the Drought Agent's own ``prepare_dataset`` for *this* window, so the gamma
    parameters behind SPI-3 stay fit on this window's train partition rather than
    being inherited from a differently-split saved artifact.
    """
    from forecasting.split import prepare_dataset as drought_dataset

    ds = drought_dataset(region, save=False, windows=windows)
    return ds.frame[config.TARGET].rename("spi3")


def attach_spi3(frame: pd.DataFrame, spi3: pd.Series,
                min_overlap: float = 0.9) -> pd.DataFrame:
    """Merge SPI-3 and its lags, distinguishing expected truncation from drift.

    The Drought Agent's frame starts a year later than the heat frame — it drops
    12 months to warm up its lag-12 features — so a short leading truncation is
    expected and handled by trimming. Anything else is refused loudly: an
    *interior* gap or a small overlap means the two agents' date ranges have
    genuinely diverged, and training on silently-NaN columns would hide it.
    """
    overlap = frame.index.intersection(spi3.dropna().index)
    if len(overlap) < min_overlap * len(frame):
        raise ValueError(
            f"SPI-3 covers only {len(overlap)} of {len(frame)} heat months "
            f"({len(overlap) / max(len(frame), 1):.0%}) — the Drought and Heat "
            "agents' date ranges have drifted apart, refusing to proceed.")

    trimmed = frame.loc[overlap.min():overlap.max()]
    aligned = spi3.reindex(trimmed.index)
    if aligned.isna().any():
        gaps = trimmed.index[aligned.isna()]
        raise ValueError(
            f"SPI-3 has {len(gaps)} interior gap(s) inside the shared range, "
            f"e.g. {gaps[:3].tolist()} — refusing to train on silent NaNs.")

    out = trimmed.copy()
    out["spi3"] = aligned.to_numpy(dtype=float)
    for lag in (1, 3):
        out[f"spi3_lag{lag}"] = aligned.shift(lag).to_numpy(dtype=float)
    return out


def build_features(anomaly: pd.Series, oni: pd.Series) -> pd.DataFrame:
    """Causal features only — trailing windows, no centring, no future rows."""
    out = pd.DataFrame({config.HEAT_TARGET: anomaly})

    for lag in config.HEAT_LAGS:
        out[f"{config.HEAT_TARGET}_lag{lag}"] = anomaly.shift(lag)
    for window in config.HEAT_ROLL_WINDOWS:
        out[f"heat_roll{window}_mean"] = anomaly.rolling(window, min_periods=1).mean()

    month = out.index.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)   # exactly known in advance
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["oni"] = oni.reindex(out.index).to_numpy(dtype=float)

    if out["oni"].isna().any():
        raise ValueError("ONI missing for some months of the heat series")
    return out


def prepare_heat_dataset(region: str = config.DEFAULT_REGION,
                         windows: SplitWindows = DEFAULT_WINDOWS,
                         seq_len: int = 12,
                         horizon: int = config.HORIZON,
                         target: str = config.HEAT_TARGET,
                         features: list | None = None,
                         save: bool = False) -> HeatDataset:
    """Build every candidate target and feature, then select the requested pair.

    All three Phase-1.1 targets are computed here so the comparison grid runs off
    one consistent frame: ``heat_anomaly`` (monthly mean), ``heat_extreme``
    (hottest single day) and ``heatwave_day_count`` (IMD-criteria days).
    """
    config.check_region(region)
    features = list(features or config.HEAT_FEATURES)
    daily = load_or_fetch_daily(region)

    # --- targets, all from train-only climatologies ---------------------------
    tmax = monthly_tmax(daily)
    climatology = fit_heat_climatology(tmax, windows.train)
    anomaly = compute_heat_anomaly(tmax, climatology)

    daily_climatology = fit_daily_tmax_climatology(daily, windows.train)
    extreme = compute_heat_extreme(monthly_max_tmax(daily), daily_climatology)

    # --- operational indicator: the tool's flags, and now also a target -------
    normals = fit_daily_normals(daily, windows.train)
    counts = monthly_heat_wave_counts(flag_heat_wave_days(daily, normals))

    # --- features -------------------------------------------------------------
    frame = build_features(anomaly, fetch_oni()).join(counts).join(tmax)
    frame[config.HEAT_EXTREME_TARGET] = extreme.reindex(frame.index)
    frame[config.HEAT_COUNT_TARGET] = frame["heat_wave_days"].astype(float)

    if any(f.startswith("spi3") for f in features):
        frame = attach_spi3(frame, load_drought_spi3(region, windows))

    frame = frame.dropna(subset=features + [target])

    scaler = MinMaxScaler()
    scaler.fit(frame.loc[windows.train, features])

    X_all = scaler.transform(frame[features]).astype("float32")
    y_all = frame[target].to_numpy(dtype="float32")
    dates = pd.DatetimeIndex(frame.index)

    masks = {name: dates.isin(frame.loc[part].index)
             for name, part in (("train", windows.train), ("val", windows.val),
                                ("test", windows.test))}
    splits = {
        name: build_windows(X_all, y_all, dates, mask,
                            strict_history=(name == "train"),
                            seq_len=seq_len, horizon=horizon)
        for name, mask in masks.items()
    }

    diagnostics = {"distribution": check_distribution(frame[target], windows.train)}
    if target == config.HEAT_COUNT_TARGET:
        diagnostics["zero_inflation"] = zero_inflation(frame[target], windows.train)

    if save:
        joblib.dump(climatology, config.heat_climatology_path(region))
        joblib.dump(scaler, config.heat_scaler_path(region))

    return HeatDataset(frame=frame, scaler=scaler, climatology=climatology,
                       normals=normals, distribution=diagnostics["distribution"],
                       region=region, target=target, features=features,
                       diagnostics=diagnostics, windows=windows, seq_len=seq_len,
                       horizon=horizon, splits=splits)


if __name__ == "__main__":
    import sys

    for name in (sys.argv[1:] or list(config.REGIONS)):
        ds = prepare_heat_dataset(name)
        print(f"=== {name} ===")
        for split, s in ds.splits.items():
            print(f"  {split:5s} X={s['X'].shape} y={s['y'].shape}")
        d = ds.distribution
        print(f"  target: mean={d['mean']:+.3f} std={d['std']:.3f} "
              f"skew={d['skew']:+.3f} excess_kurtosis={d['excess_kurtosis']:+.3f} "
              f"approx_normal={d['approximately_normal']}")
        print(f"  tails: empirical {d['percentiles_empirical']} vs normal "
              f"{d['percentiles_standard_normal']}")
        recent = ds.frame.loc["2024", ["heat_wave_days", "severe_heat_wave_days",
                                       "had_heat_wave_spell", "max_tmax_c"]]
        print(f"  2024 heat wave days by month:\n{recent.to_string()}")
