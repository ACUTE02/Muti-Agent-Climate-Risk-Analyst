"""Cleaning / feature-engineering contract. Runs offline on a synthetic series."""

import numpy as np
import pandas as pd
import pytest

from forecasting import config
from forecasting.clean import clean_india_climate


@pytest.fixture(scope="module")
def synthetic_raw() -> pd.DataFrame:
    """Monsoon-shaped monthly rainfall 1980-2024, with a gap and a wild outlier."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("1980-01-01", "2024-12-01", freq="MS")
    seasonal = 120 * np.exp(-((idx.month - 7.5) ** 2) / 4.0)
    df = pd.DataFrame(
        {
            "temp_c": 25 + 8 * np.sin(2 * np.pi * (idx.month - 4) / 12) + rng.normal(0, 1, len(idx)),
            "rainfall_mm": np.clip(seasonal + rng.normal(0, 12, len(idx)), 0, None),
            "soil_moisture": rng.uniform(0.05, 0.35, len(idx)),
        },
        index=idx,
    )
    df.index.name = "date"
    df.loc["1995-04-01":"1995-05-01", :] = np.nan       # 2-month gap (< 3mo)
    df.loc["2003-02-01", "rainfall_mm"] = 5000.0        # absurd outlier
    return df


@pytest.fixture(scope="module")
def cleaned_synthetic(synthetic_raw) -> pd.DataFrame:
    """For the assertions that check values planted in the synthetic series."""
    return clean_india_climate(synthetic_raw)


@pytest.fixture(params=["synthetic", *config.REGIONS])
def cleaned(request, synthetic_raw) -> pd.DataFrame:
    """General pipeline properties are checked on the synthetic series *and* on
    every region's real data — the properties are not region-specific."""
    if request.param == "synthetic":
        return clean_india_climate(synthetic_raw)

    raw_path = config.raw_path(request.param)
    if not raw_path.exists():
        pytest.skip(f"{request.param} not fetched — run `python -m forecasting.fetch_data`")
    return clean_india_climate(pd.read_parquet(raw_path))


def test_no_nans(cleaned):
    assert cleaned.isna().sum().sum() == 0


def test_cyclical_month_encoding_bounded(cleaned):
    for col in ("month_sin", "month_cos"):
        assert cleaned[col].between(-1, 1).all()
    # sin^2 + cos^2 == 1 for a genuine cyclical encoding
    assert np.allclose(cleaned["month_sin"] ** 2 + cleaned["month_cos"] ** 2, 1.0)


def test_all_engineered_features_present(cleaned):
    # spi3 needs train-only gamma params, oni comes from NOAA — both added later
    expected = [f for f in config.FEATURES if f not in (config.TARGET, "oni")]
    assert set(expected).issubset(cleaned.columns)


def test_spi_not_computed_in_clean(cleaned):
    """Rule A: SPI-3 needs train-only gamma params, so clean() must not produce it."""
    assert config.TARGET not in cleaned.columns
    assert "spi" not in cleaned.columns   # the Phase-1 z-score proxy is gone for good


def test_spi_accumulation_is_a_full_causal_three_month_total(cleaned):
    """SPI-3 is fit to a 3-month accumulation — never a partial window."""
    accum = cleaned[config.SPI_ACCUM]
    manual = cleaned["rainfall_mm"].rolling(3, min_periods=3).sum()
    assert np.allclose(accum.iloc[2:], manual.iloc[2:])
    assert accum.notna().all()
    assert (accum >= 0).all()


def test_outlier_replaced_with_past_values_only(cleaned_synthetic, synthetic_raw):
    """The 5000mm spike is gone, and its replacement came from earlier months."""
    value = cleaned_synthetic.loc["2003-02-01", "rainfall_mm"]
    assert value < 500
    past = synthetic_raw.loc[:"2003-01-01", "rainfall_mm"].tail(12).median()
    assert value == pytest.approx(past, rel=1e-6)


def test_gap_shorter_than_three_months_interpolated(cleaned_synthetic):
    assert pd.Timestamp("1995-04-01") in cleaned_synthetic.index
    assert pd.Timestamp("1995-05-01") in cleaned_synthetic.index


def test_rolling_features_are_backward_looking(cleaned):
    """roll3_mean at t equals the mean of t-2..t — never touching t+1.

    The first rows are skipped because the rolling windows are computed before the
    12-month lag warm-up is dropped, so they legitimately carry pre-1981 history.
    """
    manual3 = cleaned["rainfall_mm"].rolling(3, min_periods=1).mean()
    assert np.allclose(cleaned["roll3_mean"].iloc[2:], manual3.iloc[2:])

    manual12 = cleaned["rainfall_mm"].rolling(12, min_periods=1).sum()
    assert np.allclose(cleaned["roll12_sum"].iloc[11:], manual12.iloc[11:])

    shifted_future = cleaned["rainfall_mm"].shift(-1).rolling(3, min_periods=1).mean()
    assert not np.allclose(cleaned["roll3_mean"].iloc[2:-1], shifted_future.iloc[2:-1])


def test_future_rows_cannot_change_past_features(synthetic_raw, cleaned_synthetic):
    """The strongest causality check: truncate the series and nothing before the
    cut moves. A two-sided window or a full-series statistic would break this."""
    truncated = clean_india_climate(synthetic_raw.loc[:"2019-12-01"])
    overlap = truncated.index
    pd.testing.assert_frame_equal(truncated, cleaned_synthetic.loc[overlap])


def test_lag_features_match_shift(cleaned):
    for lag in (1, 3, 6, 12):
        col = f"rainfall_mm_lag{lag}"
        assert np.allclose(cleaned[col].iloc[lag:],
                           cleaned["rainfall_mm"].iloc[:-lag])


