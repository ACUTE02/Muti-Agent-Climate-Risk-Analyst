"""The assertions that make this phase different from the prototype.

Every transform statistic must come from the train partition alone, and no window
may ever see the future. These run against the real cached pull in ``data/raw``.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from forecasting import config
from forecasting.clean import clean_india_climate, clip_baseline
from forecasting.split import (SplitWindows, fit_month_stats, fit_scaler,
                               fit_spi3_params, prepare_dataset)

REGIONS = list(config.REGIONS)

# Datasets are rebuilt once per region and shared across this module's tests.
_DS_CACHE: dict[str, object] = {}
_CLEANED_CACHE: dict[str, pd.DataFrame] = {}


@pytest.fixture(params=REGIONS)
def region(request) -> str:
    if not config.raw_path(request.param).exists():
        pytest.skip(f"{request.param} not fetched — run `python -m forecasting.fetch_data`")
    return request.param


@pytest.fixture
def ds(region):
    if region not in _DS_CACHE:
        _DS_CACHE[region] = prepare_dataset(region, save=False)
    return _DS_CACHE[region]


@pytest.fixture
def cleaned(region):
    if region not in _CLEANED_CACHE:
        _CLEANED_CACHE[region] = clean_india_climate(
            pd.read_parquet(config.raw_path(region)))
    return _CLEANED_CACHE[region]


# --------------------------------------------------------------------------- #
# Scaler
# --------------------------------------------------------------------------- #
def test_scaler_fit_on_train_partition_only(ds):
    train_features = ds.frame.loc[config.TRAIN, config.FEATURES]
    assert np.allclose(ds.scaler.data_min_, train_features.min().to_numpy())
    assert np.allclose(ds.scaler.data_max_, train_features.max().to_numpy())


def test_scaler_ignores_extremes_that_only_exist_in_test(ds):
    """An implausible spike planted in the test partition must not move the scaler.

    On the real Rajasthan series the train partition happens to contain the global
    min *and* max of every feature, so comparing the fitted range against the full
    series would pass even for a leaky fit. Planting the extreme makes the check
    bite: if the scaler ever saw val/test rows, ``data_max_`` would jump.
    """
    tampered = ds.frame.copy()
    tampered.loc[tampered.index[-1], "rainfall_mm"] = 99_999.0
    scaler = fit_scaler(tampered)

    idx = config.FEATURES.index("rainfall_mm")
    train_max = ds.frame.loc[config.TRAIN, "rainfall_mm"].max()
    assert scaler.data_max_[idx] == pytest.approx(train_max)
    assert scaler.data_max_[idx] < 99_999.0


def test_saved_scaler_matches_train_only_fit(ds, region):
    if not config.scaler_path(region).exists():
        pytest.skip(f"{region} not trained yet")
    saved = joblib.load(config.scaler_path(region))
    train_features = ds.frame.loc[config.TRAIN, config.FEATURES]
    assert np.allclose(saved.data_min_, train_features.min().to_numpy())
    assert np.allclose(saved.data_max_, train_features.max().to_numpy())


# --------------------------------------------------------------------------- #
# month_stats (the SPI leak the original tech spec's reference code had)
# --------------------------------------------------------------------------- #
def test_month_stats_computed_from_train_rows_only(ds, cleaned):
    fresh = fit_month_stats(cleaned.loc[config.TRAIN])
    pd.testing.assert_frame_equal(ds.month_stats, fresh)
    assert list(ds.month_stats.index) == list(range(1, 13))


def test_month_stats_differ_from_full_series_groupby(ds, cleaned):
    leaky = fit_month_stats(cleaned)   # what the naive whole-frame groupby produces
    assert not np.allclose(ds.month_stats["mean"].to_numpy(),
                           leaky["mean"].to_numpy()), \
        "train-only and full-series month stats are identical — check the split"


def test_saved_month_stats_match_train_only_fit(cleaned, region):
    if not config.month_stats_path(region).exists():
        pytest.skip(f"{region} not trained yet")
    saved = joblib.load(config.month_stats_path(region))
    pd.testing.assert_frame_equal(saved, fit_month_stats(cleaned.loc[config.TRAIN]))


# --------------------------------------------------------------------------- #
# Sliding windows
# --------------------------------------------------------------------------- #
def _bounds(part: slice):
    return pd.Timestamp(f"{part.start}-01-01"), pd.Timestamp(f"{part.stop}-12-31")


def test_every_partition_has_windows(ds):
    for name in ("train", "val", "test"):
        assert len(ds.splits[name]["X"]) > 0, f"{name} produced no sequences"


def test_train_windows_never_touch_val_or_test(ds):
    """Strict containment: train inputs *and* targets stay inside 1980-2015."""
    _, train_end = _bounds(config.TRAIN)
    s = ds.splits["train"]
    assert s["window_end"].max() <= train_end
    last_target = s["target_dates"].max() + pd.DateOffset(months=config.HORIZON - 1)
    assert last_target <= train_end


def test_val_and_test_targets_stay_inside_their_partition(ds):
    for name in ("val", "test"):
        lo, hi = _bounds(getattr(config, name.upper()))
        s = ds.splits[name]
        assert s["target_dates"].min() >= lo
        last_target = s["target_dates"].max() + pd.DateOffset(months=config.HORIZON - 1)
        assert last_target <= hi


def test_no_window_contains_future_information(ds):
    """Causality: every input month strictly precedes the first predicted month."""
    for name in ("train", "val", "test"):
        s = ds.splits[name]
        assert (s["window_end"] < s["target_dates"]).all()
        expected_span = pd.DateOffset(months=config.SEQ_LEN - 1)
        assert ((s["window_start"] + expected_span) == s["window_end"]).all()


def test_val_targets_absent_from_train_targets(ds):
    train_targets = set(ds.splits["train"]["target_dates"])
    for name in ("val", "test"):
        assert not train_targets & set(ds.splits[name]["target_dates"])


# --------------------------------------------------------------------------- #
# Phase 1.4 alternative windows — the same discipline must survive re-slicing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window_name", list(config.ROLLING_WINDOWS))
def test_alternative_windows_refit_everything_on_their_own_train(region, window_name):
    """Windows B/C/D must be as leak-free as the standing split: each refits its
    own scaler and SPI-3 gamma params on its own train partition."""
    windows = SplitWindows(*config.ROLLING_WINDOWS[window_name])
    ds = prepare_dataset(region, save=False, windows=windows,
                         seq_len=12, horizon=1)

    train_features = ds.frame.loc[windows.train, config.FEATURES]
    assert np.allclose(ds.scaler.data_min_, train_features.min().to_numpy())
    assert np.allclose(ds.scaler.data_max_, train_features.max().to_numpy())

    fresh = fit_spi3_params(ds.frame.loc[windows.train])
    for month in range(1, 13):
        for key in ("q", "alpha", "beta"):
            assert ds.spi_params[month][key] == pytest.approx(fresh[month][key])


@pytest.mark.parametrize("window_name", list(config.ROLLING_WINDOWS))
def test_no_window_sees_data_after_its_test_period(region, window_name):
    """The catch-all: delete everything after a window's test period and every
    feature value inside it must be unchanged. Fails if any statistic — the
    anomaly baseline, the IQR bounds, the gamma fit, the scaler — reaches
    forward. This is what caught the baseline leak in the Phase-1.4 windows.
    """
    windows = SplitWindows(*config.ROLLING_WINDOWS[window_name])
    test_end = f"{windows.test.stop}-12-31"

    full = prepare_dataset(region, save=False, windows=windows,
                           seq_len=12, horizon=1)
    raw = pd.read_parquet(config.raw_path(region)).loc[:test_end]
    truncated = clean_india_climate(
        raw,
        train_end=windows.train.stop,
        baseline=clip_baseline(config.BASELINE, windows.train),
    )

    overlap = truncated.index
    pd.testing.assert_frame_equal(
        truncated, full.frame.loc[overlap, truncated.columns])


@pytest.mark.parametrize("window_name", list(config.ROLLING_WINDOWS))
def test_anomaly_baseline_stays_inside_train(window_name):
    """Rule C: the baseline is a fixed historical window, but it must still sit
    inside the train partition of whichever split is in play."""
    windows = SplitWindows(*config.ROLLING_WINDOWS[window_name])
    clipped = clip_baseline(config.BASELINE, windows.train)
    assert clipped.start >= str(windows.train.start)
    assert clipped.stop <= str(windows.train.stop)


@pytest.mark.parametrize("window_name", list(config.ROLLING_WINDOWS))
def test_alternative_windows_stay_causal(region, window_name):
    """A shorter lookback and a single horizon must not loosen causality."""
    windows = SplitWindows(*config.ROLLING_WINDOWS[window_name])
    ds = prepare_dataset(region, save=False, windows=windows,
                         seq_len=12, horizon=1)

    for name, part in (("train", windows.train), ("val", windows.val),
                       ("test", windows.test)):
        s = ds.splits[name]
        assert len(s["X"]) > 0, f"{name}/{window_name} produced no sequences"
        assert (s["window_end"] < s["target_dates"]).all()
        assert s["X"].shape[1] == 12
        assert s["y"].shape[1] == 1

        lo = pd.Timestamp(f"{part.start}-01-01")
        hi = pd.Timestamp(f"{part.stop}-12-31")
        assert s["target_dates"].min() >= lo
        assert s["target_dates"].max() <= hi


def test_no_seasonal_decompose_or_centered_rolling_in_codebase():
    """Definition of Done: the two prototype bugs must not exist anywhere.

    Matches call/import syntax rather than the bare words, so the docstrings that
    warn against these constructs don't trip the check.
    """
    banned = ("seasonal_decompose(", "import statsmodels", "center=True")
    offenders = [
        f"{path.name}: {token}"
        for path in (config.REPO_ROOT / "forecasting").rglob("*.py")
        for token in banned
        if token in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders
