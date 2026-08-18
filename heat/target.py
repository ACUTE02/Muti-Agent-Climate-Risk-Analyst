"""Heat Stress target construction — built properly before any modelling.

The Drought Agent lost two phases to a naive z-score target that silently could
not represent its own extremes (see PROJECT_LOG.md Phase 1/1.1). So the target
here is checked against a standard normal *before* anything is trained on it, and
the operational heat wave indicator is kept strictly separate from the regression
target rather than conflated with it.

Two distinct things live in this module:

1. ``heat_anomaly`` — the continuous regression target: the standardised monthly
   Tmax anomaly, using a per-calendar-month climatology fit on TRAIN ROWS ONLY.
2. Heat wave **day counts** — an operational indicator for the tool's risk flags,
   computed from daily Tmax against IMD's real criteria. Never a model target.

IMD heat wave criteria (plains; both regions here are plains, neither is a hill
station), from https://ndma.gov.in/Natural-Hazards/Heat-Wave and IMD's FAQ at
https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf:

* heat wave day: Tmax >= 45 C, OR (Tmax >= 40 C AND Tmax - normal >= 4.5 C)
* severe heat wave day: Tmax >= 47 C, OR (Tmax >= 40 C AND Tmax - normal >= 6.4 C)
* an actual heat wave needs >= 2 consecutive qualifying days

**Documented adaptation:** IMD declares a heat wave when the criteria are met at
at least two stations in a meteorological subdivision. This project models a
single Open-Meteo grid point per region, so the two-station rule cannot apply and
the criteria are evaluated at that one point. That makes these counts an
indicative single-point adaptation of IMD's definition, not an official IMD
declaration — stated here rather than glossed over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from forecasting import config


# --------------------------------------------------------------------------- #
# 1. The regression target
# --------------------------------------------------------------------------- #
def monthly_tmax(daily: pd.DataFrame) -> pd.Series:
    """Monthly mean of daily Tmax, indexed by month start."""
    out = daily[config.HEAT_SOURCE].resample("MS").mean()
    out.index.name = "date"
    return out.rename(config.HEAT_SOURCE)


def fit_heat_climatology(monthly: pd.Series, train: slice) -> pd.DataFrame:
    """Per-calendar-month mean/std of monthly Tmax, fit on TRAIN ROWS ONLY.

    **Deviation worth stating:** the phase spec describes the climatology as the
    mean and std of *daily* Tmax per calendar month. Standardising a monthly mean
    by a daily standard deviation would divide by roughly 3-5x too much spread —
    the resulting "z-score" would have a standard deviation near 0.3, and the
    distribution check against a standard normal (which the same section asks
    for) could not pass by construction. The climatology is therefore fit on the
    monthly-mean series, which is what makes the target a genuine z-score and
    mirrors how ``month_stats`` worked for drought.
    """
    train_vals = monthly.loc[train]
    grouped = train_vals.groupby(train_vals.index.month)
    out = grouped.agg(["mean", "std"])
    out.index.name = "month"
    return out


def compute_heat_anomaly(monthly: pd.Series,
                         climatology: pd.DataFrame) -> pd.Series:
    """Apply already-fit (train-only) climatology to any subset."""
    months = pd.Index(monthly.index.month)
    mean = months.map(climatology["mean"]).to_numpy(dtype=float)
    std = months.map(climatology["std"]).to_numpy(dtype=float)
    std = np.where((std == 0) | ~np.isfinite(std), np.nan, std)
    values = (monthly.to_numpy(dtype=float) - mean) / std
    return pd.Series(values, index=monthly.index, name=config.HEAT_TARGET)


def check_distribution(anomaly: pd.Series, train: slice) -> dict:
    """Is the target actually standard-normal? Measured, not assumed.

    Phase 1's SPI proxy looked reasonable and was not — it was one-sided and
    could never reach its own "Severe" threshold. This reports the numbers that
    would have caught that: skew, kurtosis, and empirical tails against theory.
    """
    train_vals = anomaly.loc[train].dropna()
    theoretical = {q: float(stats.norm.ppf(q)) for q in (0.05, 0.25, 0.75, 0.95)}
    empirical = {q: float(train_vals.quantile(q)) for q in theoretical}

    skew = float(stats.skew(train_vals))
    kurtosis = float(stats.kurtosis(train_vals))       # excess (normal = 0)
    return {
        "n_train_months": int(len(train_vals)),
        "mean": round(float(train_vals.mean()), 4),
        "std": round(float(train_vals.std()), 4),
        "min": round(float(train_vals.min()), 4),
        "max": round(float(train_vals.max()), 4),
        "skew": round(skew, 4),
        "excess_kurtosis": round(kurtosis, 4),
        "percentiles_empirical": {f"p{int(q * 100)}": round(v, 4)
                                  for q, v in empirical.items()},
        "percentiles_standard_normal": {f"p{int(q * 100)}": round(v, 4)
                                        for q, v in theoretical.items()},
        "max_percentile_gap": round(max(abs(empirical[q] - theoretical[q])
                                        for q in theoretical), 4),
        # |skew| < 0.5 and |excess kurtosis| < 1 is the usual "near enough to
        # normal to standardise without a transform" rule of thumb. Note this
        # tests SHAPE only.
        "approximately_normal": bool(abs(skew) < 0.5 and abs(kurtosis) < 1.0),
        # ...and shape alone is not enough. heat_extreme is normal-shaped but
        # centred near +1.4 with std ~0.6, because the hottest day of a month
        # sits well above that month's daily mean by construction. Location and
        # scale are therefore checked separately rather than assumed from shape.
        "approximately_standard_normal": bool(
            abs(skew) < 0.5 and abs(kurtosis) < 1.0
            and abs(float(train_vals.mean())) < 0.25
            and 0.75 < float(train_vals.std()) < 1.33),
    }


def monthly_max_tmax(daily: pd.DataFrame) -> pd.Series:
    """Hottest single day of each month — the basis of the extremes target."""
    out = daily[config.HEAT_SOURCE].resample("MS").max()
    out.index.name = "date"
    return out.rename("tmax_max_c")


def fit_daily_tmax_climatology(daily: pd.DataFrame, train: slice) -> pd.DataFrame:
    """Per-calendar-month mean/std of **daily** Tmax, TRAIN ROWS ONLY.

    Deliberately the daily climatology, unlike ``fit_heat_climatology``: the
    extremes target is itself a daily-scale quantity (one day's Tmax), so the
    daily spread is the right yardstick for it. Standardising a daily value by
    the monthly-mean spread would inflate it several-fold.
    """
    train_vals = daily.loc[train, config.HEAT_SOURCE]
    out = train_vals.groupby(train_vals.index.month).agg(["mean", "std"])
    out.index.name = "month"
    return out


def compute_heat_extreme(monthly_max: pd.Series,
                         daily_climatology: pd.DataFrame) -> pd.Series:
    """Standardised hottest-day anomaly, using the train-only daily climatology."""
    months = pd.Index(monthly_max.index.month)
    mean = months.map(daily_climatology["mean"]).to_numpy(dtype=float)
    std = months.map(daily_climatology["std"]).to_numpy(dtype=float)
    std = np.where((std == 0) | ~np.isfinite(std), np.nan, std)
    values = (monthly_max.to_numpy(dtype=float) - mean) / std
    return pd.Series(values, index=monthly_max.index,
                     name=config.HEAT_EXTREME_TARGET)


def zero_inflation(series: pd.Series, train: slice) -> dict:
    """How zero-heavy is the count target? Reported, not hand-waved."""
    train_vals = series.loc[train]
    pre = train_vals[train_vals.index.month.isin(config.PRE_MONSOON_MONTHS)]
    return {
        "n_train_months": int(len(train_vals)),
        "fraction_zero_all_months": round(float((train_vals == 0).mean()), 4),
        "fraction_zero_pre_monsoon": round(float((pre == 0).mean()), 4),
        "mean_all_months": round(float(train_vals.mean()), 4),
        "mean_pre_monsoon": round(float(pre.mean()), 4),
        "max": int(train_vals.max()),
    }


# --------------------------------------------------------------------------- #
# 2. The operational indicator (never a model target)
# --------------------------------------------------------------------------- #
def fit_daily_normals(daily: pd.DataFrame, train: slice,
                      baseline: slice = config.BASELINE) -> pd.Series:
    """Day-of-year normal Tmax, pooled over a +/- 7-day window.

    Choice stated as the spec asks: a bare day-of-year mean over the 30 baseline
    years has only ~30 samples per day (standard error ~0.6 C, large next to the
    4.5 C departure threshold), while a flat monthly normal misassigns days at
    month edges by more than 2 C during the steep April-May warming. Pooling all
    days within +/- 7 calendar days across the baseline years gives ~450 samples
    per day-of-year — smooth and stable — and follows the usual WMO practice of a
    windowed daily climatology.

    The baseline is clipped to the train partition, per the Phase 1.4 lesson that
    a fixed baseline period silently reaches past a shorter train window.
    """
    from forecasting.clean import clip_baseline

    period = clip_baseline(baseline, train)
    vals = daily.loc[period, config.HEAT_SOURCE]
    doy = vals.index.dayofyear.to_numpy()

    window = config.NORMAL_WINDOW_DAYS
    values = vals.to_numpy(dtype=float)
    normals = np.empty(366, dtype=float)
    for target_doy in range(1, 367):
        # circular distance so late December pools with early January
        delta = np.abs(doy - target_doy)
        delta = np.minimum(delta, 366 - delta)
        normals[target_doy - 1] = values[delta <= window].mean()

    return pd.Series(normals, index=pd.RangeIndex(1, 367, name="dayofyear"),
                     name="normal_tmax_c")


def flag_heat_wave_days(daily: pd.DataFrame, normals: pd.Series) -> pd.DataFrame:
    """Per-day heat wave / severe heat wave flags per the adapted IMD criteria."""
    out = daily.copy()
    out["normal_tmax_c"] = pd.Index(out.index.dayofyear).map(normals).to_numpy()
    out["tmax_departure_c"] = out[config.HEAT_SOURCE] - out["normal_tmax_c"]

    hot_enough = out[config.HEAT_SOURCE] >= config.HEAT_WAVE_MIN_TMAX
    out["heat_wave_day"] = (
        (out[config.HEAT_SOURCE] >= config.HEAT_WAVE_ABS_C)
        | (hot_enough & (out["tmax_departure_c"] >= config.HEAT_WAVE_DEP_C))
    )
    out["severe_heat_wave_day"] = (
        (out[config.HEAT_SOURCE] >= config.SEVERE_HEAT_WAVE_ABS_C)
        | (hot_enough & (out["tmax_departure_c"] >= config.SEVERE_HEAT_WAVE_DEP_C))
    )
    # IMD calls a heat wave only from >= 2 consecutive qualifying days; a lone hot
    # day is a hot day, not a heat wave.
    out["in_heat_wave_spell"] = _mark_spells(out["heat_wave_day"],
                                             config.HEAT_WAVE_SPELL_DAYS)
    return out


def _mark_spells(flags: pd.Series, min_days: int) -> pd.Series:
    """True for days inside a run of >= ``min_days`` consecutive True values."""
    groups = (flags != flags.shift()).cumsum()
    run_length = flags.groupby(groups).transform("size")
    return flags & (run_length >= min_days)


def monthly_heat_wave_counts(flagged: pd.DataFrame) -> pd.DataFrame:
    """Monthly roll-up of the operational indicator."""
    monthly = pd.DataFrame({
        "heat_wave_days": flagged["heat_wave_day"].resample("MS").sum(),
        "severe_heat_wave_days": flagged["severe_heat_wave_day"].resample("MS").sum(),
        "heat_wave_spell_days": flagged["in_heat_wave_spell"].resample("MS").sum(),
        "max_tmax_c": flagged[config.HEAT_SOURCE].resample("MS").max(),
    })
    monthly["had_heat_wave_spell"] = monthly["heat_wave_spell_days"] > 0
    monthly.index.name = "date"
    return monthly
