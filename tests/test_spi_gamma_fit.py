"""SPI-3 must be real SPI: train-only gamma parameters, normalised, two-tailed.

The third test is the one that would have caught the Phase-1 bug — under the old
z-score-of-rainfall proxy the "Severe" flag was mathematically unreachable.
"""

import joblib
import pandas as pd
import pytest

from forecasting import config
from forecasting.clean import clean_india_climate
from forecasting.enso import attach_oni
from forecasting.split import compute_spi3, fit_spi3_params, prepare_dataset

_DS_CACHE: dict[str, object] = {}
_CLEANED_CACHE: dict[str, pd.DataFrame] = {}


@pytest.fixture(params=list(config.REGIONS))
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
        _CLEANED_CACHE[region] = attach_oni(clean_india_climate(
            pd.read_parquet(config.raw_path(region))))
    return _CLEANED_CACHE[region]


def _same_params(a: dict, b: dict) -> bool:
    return all(a[m][k] == pytest.approx(b[m][k])
               for m in range(1, 13) for k in ("q", "alpha", "beta"))


def test_spi_params_fit_on_train_rows_only(ds, cleaned):
    """Same Rule A discipline as month_stats: train partition, nothing else."""
    fresh = fit_spi3_params(cleaned.loc[config.TRAIN])
    assert _same_params(ds.spi_params, fresh)
    assert sorted(ds.spi_params) == list(range(1, 13))

    for month, p in ds.spi_params.items():
        n_train = int((cleaned.loc[config.TRAIN].index.month == month).sum())
        assert p["n_train"] == n_train, f"month {month} used {p['n_train']} rows"


def test_spi_params_differ_from_a_full_series_fit(ds, cleaned):
    """Sanity: the leakage test is not vacuous — val/test rows do move the fit."""
    leaky = fit_spi3_params(cleaned)
    assert not _same_params(ds.spi_params, leaky)


def test_saved_spi_params_match_train_only_fit(cleaned, region):
    if not config.spi_params_path(region).exists():
        pytest.skip(f"{region} not trained yet")
    saved = joblib.load(config.spi_params_path(region))
    assert _same_params(saved, fit_spi3_params(cleaned.loc[config.TRAIN]))


def test_train_spi3_is_approximately_standard_normal(ds):
    """The gamma -> normal-CDF transform must actually normalise the distribution."""
    train_spi3 = ds.frame.loc[config.TRAIN, config.TARGET]
    assert train_spi3.mean() == pytest.approx(0.0, abs=0.3)
    assert train_spi3.std() == pytest.approx(1.0, abs=0.3)


def test_severe_threshold_is_reachable(ds):
    """spi3 dips below -1.5 somewhere in the record — impossible before Phase 1.1."""
    spi3 = ds.frame[config.TARGET]
    severe = spi3[spi3 < config.SPI_SEVERE]
    assert len(severe) > 0, "no month reaches Severe — the SPI fix did not take"
    assert spi3.min() < config.SPI_SEVERE


def test_spi3_transform_is_reproducible_from_saved_params(ds):
    """Recomputing with the returned params reproduces the frame's spi3 exactly."""
    recomputed = compute_spi3(ds.frame, ds.spi_params)
    pd.testing.assert_series_equal(recomputed, ds.frame[config.TARGET])
