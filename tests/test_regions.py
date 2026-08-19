"""Multi-region wiring: each region resolves to its own artifacts, and an
unsupported region fails loudly instead of silently answering with another
region's model."""

import pytest

from forecasting import config
from forecasting.recursive import horizon_label
from forecasting.tool import forecast_drought_risk

PATH_BUILDERS = (config.raw_path, config.processed_path, config.model_path,
                 config.scaler_path, config.month_stats_path,
                 config.spi_params_path, config.metrics_path, config.plot_path,
                 config.history_path, config.lstm_small_model_path,
                 config.ridge_metrics_path)


def test_more_than_one_region_is_registered():
    assert len(config.REGIONS) >= 2
    assert config.DEFAULT_REGION in config.REGIONS


@pytest.mark.parametrize("build", PATH_BUILDERS, ids=lambda b: b.__name__)
def test_artifact_paths_are_unique_per_region(build):
    paths = {region: build(region) for region in config.REGIONS}
    assert len(set(paths.values())) == len(paths), f"collision: {paths}"
    for region, path in paths.items():
        assert region in path.name


def test_oni_is_shared_not_per_region():
    """ENSO is a global index — one cached copy serves every region."""
    assert config.ONI_PATH.name == "oni_series.parquet"
    assert not any(r in config.ONI_PATH.name for r in config.REGIONS)


@pytest.mark.parametrize("bad", ["nowhere", "Rajasthan", "", "jaisalmer"])
def test_unsupported_region_raises_a_clear_error(bad):
    with pytest.raises(ValueError, match="Unsupported region"):
        config.check_region(bad)

    with pytest.raises(ValueError, match="Unsupported region"):
        forecast_drought_risk.invoke({"region": bad})


@pytest.mark.parametrize("region", list(config.REGIONS))
def test_forecast_returns_measured_per_horizon_confidence(region):
    """Phase 1.5 schema: every horizon carries its own measured skill and a label
    derived from it — never a hardcoded confidence."""
    if not config.HORIZON_MANIFEST_PATH.exists():
        pytest.skip("horizon manifest missing — run `python -m forecasting.recursive`")

    result = forecast_drought_risk.invoke({"region": region})
    # Phase 8 added the three provenance fields: which months this forecast is
    # actually about, and how current its inputs are. Kept as an exact-set
    # assertion so a future field cannot slip in unreviewed.
    assert set(result) == {"region", "predicted_values", "horizon_confidence",
                           "risk_score", "risk_flags", "model_rmse_test",
                           "forecast_anchor_month", "forecast_months",
                           "data_currency"}
    assert result["region"] == region
    assert len(result["predicted_values"]) == config.HORIZON
    assert len(result["risk_flags"]) == config.HORIZON
    assert all(f in {"Normal", "Moderate", "Severe"} for f in result["risk_flags"])
    assert 0 <= result["risk_score"] <= 10

    confidence = result["horizon_confidence"]
    assert [c["horizon"] for c in confidence] == [
        f"t+{h}" for h in range(1, config.HORIZON + 1)]
    for c in confidence:
        assert c["method"] in {"direct", "recursive"}
        assert isinstance(c["skill_score"], float)
        assert c["label"] == horizon_label(c["skill_score"])


@pytest.mark.parametrize("skill,expected", [
    (0.2622, "validated"), (0.1, "validated"),
    (0.0766, "weak/directional"), (0.0001, "weak/directional"),
    (0.0, "no skill"), (-0.0489, "no skill"),
])
def test_confidence_labels_follow_the_preset_thresholds(skill, expected):
    """0.1 is the bar used throughout Phase 1.3/1.4 — not invented here."""
    assert horizon_label(skill).startswith(expected)


@pytest.mark.parametrize("region", list(config.REGIONS))
def test_supported_regions_have_coordinates_and_a_label(region):
    meta = config.REGIONS[region]
    assert set(meta) == {"lat", "lon", "label"}
    assert 6 < meta["lat"] < 38 and 68 < meta["lon"] < 98    # inside India
    assert meta["label"]
