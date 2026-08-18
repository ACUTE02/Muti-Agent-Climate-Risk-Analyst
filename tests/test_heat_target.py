"""Heat Stress target: leak-free construction, and heat wave logic pinned to a
known historical event.

The Drought Agent shipped a broken target for two phases because nothing checked
it. These are the checks that would have caught that, applied up front this time.
"""

import numpy as np
import pandas as pd
import pytest

from forecasting import config
from forecasting.fetch_data import load_or_fetch_daily
from forecasting.split import SplitWindows
from heat.dataset import attach_spi3, load_drought_spi3, prepare_heat_dataset
from heat.target import (check_distribution, compute_heat_anomaly,
                         compute_heat_extreme, fit_daily_normals,
                         fit_daily_tmax_climatology, fit_heat_climatology,
                         flag_heat_wave_days, monthly_heat_wave_counts,
                         monthly_max_tmax, monthly_tmax, zero_inflation,
                         _mark_spells)

pytestmark = pytest.mark.skipif(
    not config.daily_path(config.DEFAULT_REGION).exists(),
    reason="daily parquet not fetched — run `python -m forecasting.fetch_data`",
)

_DAILY: dict[str, pd.DataFrame] = {}
_DS: dict[str, object] = {}


@pytest.fixture(params=list(config.REGIONS))
def region(request) -> str:
    if not config.daily_path(request.param).exists():
        pytest.skip(f"{request.param} daily data not fetched")
    return request.param


@pytest.fixture
def daily(region) -> pd.DataFrame:
    if region not in _DAILY:
        _DAILY[region] = load_or_fetch_daily(region)
    return _DAILY[region]


@pytest.fixture
def ds(region):
    if region not in _DS:
        _DS[region] = prepare_heat_dataset(region)
    return _DS[region]


# --------------------------------------------------------------------------- #
# Leak-free target construction
# --------------------------------------------------------------------------- #
def test_climatology_fit_on_train_rows_only(ds, daily):
    fresh = fit_heat_climatology(monthly_tmax(daily), ds.windows.train)
    pd.testing.assert_frame_equal(ds.climatology, fresh)
    assert list(ds.climatology.index) == list(range(1, 13))


def test_climatology_differs_from_a_full_series_fit(ds, daily):
    """Sanity: the leakage check is not vacuous — val/test rows would move it."""
    tmax = monthly_tmax(daily)
    leaky = fit_heat_climatology(tmax, slice(None))
    assert not np.allclose(ds.climatology["mean"].to_numpy(),
                           leaky["mean"].to_numpy())


def test_scaler_fit_on_train_rows_only(ds):
    train_features = ds.frame.loc[ds.windows.train, config.HEAT_FEATURES]
    assert np.allclose(ds.scaler.data_min_, train_features.min().to_numpy())
    assert np.allclose(ds.scaler.data_max_, train_features.max().to_numpy())


def test_daily_normals_never_use_post_train_data(region, daily):
    """Rule C, carried over: the normal is a fixed baseline clipped to train."""
    short = SplitWindows(slice("1980", "2000"), slice("2001", "2004"),
                         slice("2005", "2009"))
    full_history = fit_daily_normals(daily, short.train)
    truncated = fit_daily_normals(daily.loc[:"2000-12-31"], short.train)
    pd.testing.assert_series_equal(full_history, truncated)


def test_windows_stay_causal(ds):
    for name in ("train", "val", "test"):
        split = ds.splits[name]
        assert len(split["X"]) > 0
        assert (split["window_end"] < split["target_dates"]).all()


# --------------------------------------------------------------------------- #
# The distribution check — the step Phase 1 skipped
# --------------------------------------------------------------------------- #
def test_target_is_standardised_on_train(ds):
    train_vals = ds.frame.loc[ds.windows.train, config.HEAT_TARGET]
    assert train_vals.mean() == pytest.approx(0.0, abs=0.05)
    assert train_vals.std() == pytest.approx(1.0, abs=0.1)


def test_distribution_check_reports_the_diagnostics(ds):
    d = ds.distribution
    assert {"skew", "excess_kurtosis", "percentiles_empirical",
            "percentiles_standard_normal", "approximately_normal"} <= set(d)
    assert abs(d["skew"]) < 0.5
    assert abs(d["excess_kurtosis"]) < 1.0
    assert d["approximately_normal"] is True


def test_target_reaches_both_tails(ds):
    """The Phase-1 SPI proxy could not reach its own Severe threshold. This can
    reach both tails — the failure mode is checked, not assumed away."""
    values = ds.frame[config.HEAT_TARGET]
    assert values.min() < -2.0
    assert values.max() > 2.0


# --------------------------------------------------------------------------- #
# IMD heat wave criteria
# --------------------------------------------------------------------------- #
def test_may_2016_extreme_heat_registers(region, daily):
    """May 2016 was India's national-record heat event (Phalodi, Rajasthan,
    19 May 2016). Both grid points sit near it, so it must show up."""
    flagged = flag_heat_wave_days(daily, fit_daily_normals(daily, config.TRAIN))
    counts = monthly_heat_wave_counts(flagged)

    may = counts.loc["2016-05-01"]
    assert may["heat_wave_days"] >= 1
    assert bool(may["had_heat_wave_spell"]) is True
    assert may["max_tmax_c"] > 45.0
    # the peak lands on the record day itself
    assert flagged.loc["2016-05", "tmax_c"].idxmax() == pd.Timestamp("2016-05-19")


def test_heat_wave_rules_match_the_imd_criteria():
    """Each rule isolated: absolute, departure, the 40 C floor, and both severes."""
    normals = pd.Series(40.0, index=pd.RangeIndex(1, 367, name="dayofyear"))
    normals.loc[123] = 35.0        # 2 May 2020, to exercise the 40 C floor alone

    idx = pd.date_range("2020-05-01", periods=6, freq="D")
    daily = pd.DataFrame({"tmax_c": [45.5, 39.9, 44.5, 47.2, 46.5, 34.0]},
                         index=idx)
    out = flag_heat_wave_days(daily, normals)

    # 45.5 >= 45 absolute -> heat wave; departure +5.5 < 6.4 and < 47 -> not severe
    assert out["heat_wave_day"].iloc[0] and not out["severe_heat_wave_day"].iloc[0]
    # 39.9 against a 35 C normal is a +4.9 departure, but sits below the 40 C
    # floor the departure rule requires -> not a heat wave day
    assert out["tmax_departure_c"].iloc[1] == pytest.approx(4.9)
    assert not out["heat_wave_day"].iloc[1]
    # 44.5 is under the 45 absolute, but +4.5 departure exactly meets the rule
    assert out["heat_wave_day"].iloc[2] and not out["severe_heat_wave_day"].iloc[2]
    # 47.2 >= 47 absolute -> severe
    assert out["severe_heat_wave_day"].iloc[3]
    # 46.5 is under the 47 absolute, but +6.5 departure clears the severe rule
    assert out["severe_heat_wave_day"].iloc[4]
    # comfortably normal day
    assert not out["heat_wave_day"].iloc[5]


def test_spell_rule_needs_two_consecutive_days():
    """IMD calls a heat wave only from >= 2 consecutive qualifying days."""
    flags = pd.Series([True, False, True, True, False, True, True, True])
    spells = _mark_spells(flags, config.HEAT_WAVE_SPELL_DAYS)
    assert list(spells) == [False, False, True, True, False, True, True, True]


def test_isolated_hot_day_is_not_a_spell():
    normals = pd.Series(35.0, index=pd.RangeIndex(1, 367, name="dayofyear"))
    idx = pd.date_range("2020-05-01", periods=3, freq="D")
    daily = pd.DataFrame({"tmax_c": [30.0, 46.0, 30.0]}, index=idx)

    counts = monthly_heat_wave_counts(flag_heat_wave_days(daily, normals))
    assert counts["heat_wave_days"].iloc[0] == 1
    assert bool(counts["had_heat_wave_spell"].iloc[0]) is False


def test_operational_counts_are_not_the_model_target(ds):
    """The count is an indicator, deliberately kept out of the feature set."""
    assert "heat_wave_days" in ds.frame.columns
    assert "heat_wave_days" not in config.HEAT_FEATURES
    assert config.HEAT_TARGET not in ("heat_wave_days", "had_heat_wave_spell")


def test_compute_heat_anomaly_applies_given_climatology(ds, daily):
    recomputed = compute_heat_anomaly(monthly_tmax(daily), ds.climatology)
    aligned = recomputed.reindex(ds.frame.index)
    assert np.allclose(aligned, ds.frame[config.HEAT_TARGET])


def test_check_distribution_flags_a_skewed_target():
    """The check must actually be capable of failing — feed it a skewed series."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("1980-01-01", periods=432, freq="MS")
    skewed = pd.Series(rng.exponential(1.0, len(idx)), index=idx)
    result = check_distribution(skewed, config.TRAIN)
    assert result["approximately_normal"] is False
    assert result["skew"] > 0.5


# --------------------------------------------------------------------------- #
# Phase 1.1 — new targets and the cross-agent SPI-3 dependency
# --------------------------------------------------------------------------- #
def test_heat_extreme_uses_the_daily_climatology(region, daily):
    """The extremes target is a daily-scale quantity, so it is standardised by
    the daily spread — not the monthly-mean spread heat_anomaly uses."""
    daily_clim = fit_daily_tmax_climatology(daily, config.TRAIN)
    monthly_clim = fit_heat_climatology(monthly_tmax(daily), config.TRAIN)
    # daily spread is materially wider than the spread of monthly means
    assert (daily_clim["std"] > monthly_clim["std"]).all()

    extreme = compute_heat_extreme(monthly_max_tmax(daily), daily_clim)
    assert extreme.notna().all()
    assert extreme.name == config.HEAT_EXTREME_TARGET


def test_heat_extreme_is_not_a_standard_normal(region, daily):
    """Measured, not assumed: the hottest day of a month sits well above that
    month's daily mean by construction, so this target is centred near +1.4.
    Shape is normal-ish; location and scale are not."""
    daily_clim = fit_daily_tmax_climatology(daily, config.TRAIN)
    extreme = compute_heat_extreme(monthly_max_tmax(daily), daily_clim)
    check = check_distribution(extreme, config.TRAIN)

    assert check["mean"] > 1.0
    assert check["approximately_standard_normal"] is False
    # ...while heat_anomaly is the real thing
    anomaly_check = check_distribution(
        compute_heat_anomaly(monthly_tmax(daily),
                             fit_heat_climatology(monthly_tmax(daily), config.TRAIN)),
        config.TRAIN)
    assert anomaly_check["approximately_standard_normal"] is True


def test_count_target_zero_inflation_is_reported(region, daily):
    normals = fit_daily_normals(daily, config.TRAIN)
    counts = monthly_heat_wave_counts(flag_heat_wave_days(daily, normals))
    stats = zero_inflation(counts["heat_wave_days"].astype(float), config.TRAIN)

    assert stats["fraction_zero_all_months"] > 0.8          # heavily zero-inflated
    assert stats["fraction_zero_pre_monsoon"] < stats["fraction_zero_all_months"]
    assert stats["mean_pre_monsoon"] > stats["mean_all_months"]


def test_spi3_comes_from_the_drought_agent_and_aligns(region):
    """The cross-agent dependency, exercised end to end."""
    spi3 = load_drought_spi3(region, SplitWindows())
    assert spi3.name == "spi3"
    assert spi3.notna().all()
    assert len(spi3) > 400

    frame = pd.DataFrame(index=pd.date_range("1985-01-01", "2020-12-01", freq="MS"))
    out = attach_spi3(frame, spi3)
    assert {"spi3", "spi3_lag1", "spi3_lag3"} <= set(out.columns)
    assert out["spi3_lag1"].iloc[5] == pytest.approx(out["spi3"].iloc[4])
    assert out["spi3_lag3"].iloc[5] == pytest.approx(out["spi3"].iloc[2])


def test_spi3_merge_fails_loudly_on_drift(region):
    """A genuinely divergent date range must raise, not produce silent NaNs."""
    spi3 = load_drought_spi3(region, SplitWindows())
    frame = pd.DataFrame(index=pd.date_range("2050-01-01", "2055-12-01", freq="MS"))
    with pytest.raises(ValueError, match="drifted apart"):
        attach_spi3(frame, spi3)


def test_spi3_merge_fails_loudly_on_interior_gaps(region):
    spi3 = load_drought_spi3(region, SplitWindows()).copy()
    spi3.loc["2000-06-01"] = float("nan")
    frame = pd.DataFrame(index=pd.date_range("1990-01-01", "2010-12-01", freq="MS"))
    with pytest.raises(ValueError, match="interior gap"):
        attach_spi3(frame, spi3)


def test_leading_truncation_is_tolerated_not_raised(region):
    """The Drought Agent starts a year later (lag-12 warm-up) — expected, so the
    guard trims rather than refusing."""
    spi3 = load_drought_spi3(region, SplitWindows())
    frame = pd.DataFrame(index=pd.date_range("1980-01-01", "2020-12-01", freq="MS"))
    out = attach_spi3(frame, spi3)
    assert out.index.min() >= spi3.index.min()
    assert out["spi3"].notna().all()
