"""The Heat Stress tool's contract: observations only, and honestly labelled.

Phase 1.2 trimmed this tool to what the measurements support. These tests pin the
smaller contract — including that no forecast field sneaks back in.
"""

import pandas as pd
import pytest

from forecasting import config
from heat.tool import (NO_FORECAST_NOTE, forecast_heat_stress_risk,
                       observed_heat_wave_months)

pytestmark = pytest.mark.skipif(
    not config.daily_path(config.DEFAULT_REGION).exists(),
    reason="daily parquet not fetched — run `python -m forecasting.fetch_data`",
)

EXPECTED_KEYS = {"region", "month", "heatwave_days", "severe_heatwave_days",
                 "had_heatwave_spell", "max_tmax_c", "forecast_available", "note"}


@pytest.fixture(params=list(config.REGIONS))
def region(request) -> str:
    if not config.daily_path(request.param).exists():
        pytest.skip(f"{request.param} daily data not fetched")
    return request.param


@pytest.fixture
def result(region):
    return forecast_heat_stress_risk.invoke({"region": region})


def test_returns_exactly_the_trimmed_schema(result):
    assert set(result) == EXPECTED_KEYS


def test_no_forecast_fields_remain(result):
    """Phase 1.2 removed the no-skill predictions rather than labelling them."""
    forbidden = {"predicted_heat_anomaly", "predicted_heat_extreme",
                 "predicted_heatwave_days", "horizon_confidence", "risk_flags"}
    assert not (forbidden & set(result))


def test_declares_itself_observation_not_forecast(result):
    """A caller must be able to tell programmatically, without parsing prose."""
    assert result["forecast_available"] is False
    assert result["note"] == NO_FORECAST_NOTE


def test_observed_values_are_plain_types(result):
    assert isinstance(result["heatwave_days"], int)
    assert isinstance(result["severe_heatwave_days"], int)
    assert isinstance(result["had_heatwave_spell"], bool)
    assert isinstance(result["max_tmax_c"], float)
    assert 0 <= result["heatwave_days"] <= 31
    assert result["severe_heatwave_days"] <= result["heatwave_days"]


def test_defaults_to_the_latest_month_with_data(region, result):
    counts = observed_heat_wave_months(region)
    assert result["month"] == f"{counts.index.max():%Y-%m}"


@pytest.mark.parametrize("month_arg", ["2024-05", "2024-05-01"])
def test_a_specific_month_can_be_requested(region, month_arg):
    out = forecast_heat_stress_risk.invoke({"region": region, "month": month_arg})
    assert out["month"] == "2024-05"
    # May 2024 was a real heat event at both sites (see PROJECT_LOG.md)
    assert out["heatwave_days"] >= 1
    assert out["had_heatwave_spell"] is True
    assert out["max_tmax_c"] > 45


def test_winter_month_reports_no_heat_wave(region):
    """Tmax never nears the 40 C plains threshold in January."""
    out = forecast_heat_stress_risk.invoke({"region": region, "month": "2024-01"})
    assert out["heatwave_days"] == 0
    assert out["severe_heatwave_days"] == 0
    assert out["had_heatwave_spell"] is False


def test_unknown_month_fails_loudly(region):
    with pytest.raises(ValueError, match="No observations for"):
        forecast_heat_stress_risk.invoke({"region": region, "month": "1970-05"})


def test_unparseable_month_fails_loudly(region):
    with pytest.raises(ValueError, match="Could not parse month"):
        forecast_heat_stress_risk.invoke({"region": region, "month": "not-a-month"})


def test_unsupported_region_raises():
    with pytest.raises(ValueError, match="Unsupported region"):
        forecast_heat_stress_risk.invoke({"region": "nowhere"})


def test_counts_match_the_underlying_observations(region, result):
    counts = observed_heat_wave_months(region)
    row = counts.loc[pd.Timestamp(result["month"] + "-01")]
    assert result["heatwave_days"] == int(row["heat_wave_days"])
    assert result["max_tmax_c"] == pytest.approx(float(row["max_tmax_c"]), abs=0.05)


def test_both_risk_types_expose_a_tool():
    """Two risk types: drought forecasts, heat reports. Both callable."""
    from forecasting.tool import forecast_drought_risk

    assert forecast_drought_risk.name == "forecast_drought_risk"
    assert forecast_heat_stress_risk.name == "forecast_heat_stress_risk"
