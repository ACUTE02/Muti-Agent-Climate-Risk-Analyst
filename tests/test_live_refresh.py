"""Phase 8 Part 3 — the rolling recent cache is additive and changes no result.

The whole risk of this feature is one mistake: letting fresh data reach the
evaluation path. If it did, ``TEST = slice("2020","2024")`` would cover different
rows and every skill score published in PROJECT_LOG.md and EVALUATION.md would be
silently wrong. These tests exist to make that mistake loud.

The heavier check (refitting Ridge and diffing against the committed metrics) is
skipped when the recent cache is absent, because without it the test proves
nothing — there would be no fresh data that *could* have leaked.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from forecasting import config
from forecasting.fetch_data import (data_currency, load_live_region,
                                    load_or_fetch_region, load_recent_region,
                                    _month_is_complete)

REGIONS = list(config.REGIONS)


def _has_recent(region: str) -> bool:
    return not load_recent_region(region).empty


needs_recent = pytest.mark.skipif(
    not any(_has_recent(r) for r in REGIONS),
    reason="no rolling cache on disk — run `python -m scripts.refresh` first")


# --------------------------------------------------------------------------- #
# The two caches stay separate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("region", REGIONS)
def test_refresh_never_touches_the_fixed_archive(region):
    """The archive must still end at FETCH_END no matter how often we refresh."""
    archive = load_or_fetch_region(region)
    assert str(archive.index.max().date()) == "2024-12-01", (
        "the fixed archive moved — every published skill score is now suspect")


@pytest.mark.parametrize("region", REGIONS)
@needs_recent
def test_live_union_is_longer_than_the_archive(region):
    archive, live = load_or_fetch_region(region), load_live_region(region)
    assert len(live) > len(archive)
    assert live.index.max() > archive.index.max()
    # The union must not have altered any archive row.
    pd.testing.assert_frame_equal(live.loc[archive.index], archive)


@pytest.mark.parametrize("region", REGIONS)
def test_evaluation_path_reads_the_archive_only(region):
    """`prepare_dataset` is the evaluation path. It must not see fresh data."""
    from forecasting.split import prepare_dataset

    ds = prepare_dataset(region, save=False)
    assert str(ds.frame.index.max().date()) == "2024-12-01", (
        f"prepare_dataset reached {ds.frame.index.max().date()} — it is reading "
        "the rolling cache, which invalidates the published windows")


# --------------------------------------------------------------------------- #
# The published numbers still reproduce, with fresh data sitting on disk
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("region", REGIONS)
@needs_recent
def test_published_ridge_metrics_are_unchanged(region):
    """Refit the real evaluation Ridge and diff against the committed JSON."""
    from forecasting.baseline_ridge import fit_ridge_baseline, flatten_windows
    from forecasting.evaluate import _scores, climatology_baseline
    from forecasting.split import prepare_dataset

    path = config.ridge_metrics_path(region)
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    published = json.loads(path.read_text(encoding="utf-8"))

    ds = prepare_dataset(region, save=False)
    model, _, _ = fit_ridge_baseline(*ds.get("train"), *ds.get("val"))

    test = ds.splits["test"]
    y_pred = model.predict(flatten_windows(test["X"]))
    y_base = climatology_baseline(ds, test["target_dates"])

    for h in range(ds.horizon):
        got = _scores(test["y"][:, h], y_pred[:, h], y_base[:, h])
        assert got == published["per_horizon"][f"t+{h + 1}"], (
            f"{region} t+{h + 1} moved after the refresh")

    averaged = _scores(test["y"].ravel(), y_pred.ravel(), y_base.ravel())
    assert averaged == published["averaged"], f"{region} averaged metrics moved"


# --------------------------------------------------------------------------- #
# Partial months must never enter the monthly frame
# --------------------------------------------------------------------------- #
def test_incomplete_month_is_rejected():
    """A half-finished month's rainfall *sum* looks like a drought. Drop it."""
    full = pd.DataFrame(
        {"rainfall_mm": [1.0] * 31},
        index=pd.date_range("2026-01-01", periods=31, freq="D"))
    partial = full.iloc[:19]

    assert _month_is_complete(full, pd.Timestamp("2026-01-01")) is True
    assert _month_is_complete(partial, pd.Timestamp("2026-01-01")) is False


@pytest.mark.parametrize("region", REGIONS)
@needs_recent
def test_recent_cache_holds_only_complete_months(region):
    """Every cached month must be one the daily record fully covers."""
    from forecasting.fetch_data import load_live_daily

    daily = load_live_daily(region)
    for month in load_recent_region(region).index:
        assert _month_is_complete(daily, month), f"{month.date()} is incomplete"


# --------------------------------------------------------------------------- #
# Currency is reported, and reported honestly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("region", REGIONS)
def test_data_currency_reports_the_binding_limit(region):
    info = data_currency(region)
    assert info["archive_through"] == "2024-12-01"
    assert info["data_current_through"] <= info["weather_through"], (
        "the anchor cannot be more current than the weather data behind it")
    if info["oni_through"]:
        assert info["data_current_through"] == min(info["weather_through"],
                                                   info["oni_through"])
    assert info["months_behind_today"] >= 0


def test_health_reports_data_currency():
    from fastapi.testclient import TestClient
    from api.app import app

    body = TestClient(app).get("/health").json()
    assert "data_currency" in body
    for region in REGIONS:
        assert "data_current_through" in body["data_currency"][region]


# --------------------------------------------------------------------------- #
# The forecast says which real months it is about
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("region", REGIONS)
def test_forecast_names_the_months_it_predicts(region):
    from forecasting.tool import forecast_drought_risk

    result = forecast_drought_risk.invoke({"region": region})
    anchor = pd.Timestamp(result["forecast_anchor_month"])

    assert len(result["forecast_months"]) == 3
    for offset, month in enumerate(result["forecast_months"], start=1):
        assert pd.Timestamp(month) == anchor + pd.DateOffset(months=offset)

    # Each horizon carries its own month, and its own measured label — the
    # refresh moves the dates, never the skill.
    for offset, entry in enumerate(result["horizon_confidence"], start=1):
        assert entry["month"] == result["forecast_months"][offset - 1]
        assert entry["label"], "a horizon must always carry its measured label"


@pytest.mark.parametrize("region", REGIONS)
@needs_recent
def test_refreshed_forecast_targets_months_after_the_fixed_archive(region):
    """The point of Part 3: a live query is no longer stuck in early 2025."""
    from forecasting.tool import forecast_drought_risk

    result = forecast_drought_risk.invoke({"region": region})
    assert pd.Timestamp(result["forecast_months"][0]) > pd.Timestamp("2025-01-01"), (
        "the forecast is still anchored to the fixed archive")


# --------------------------------------------------------------------------- #
# The crop tool and the forecast must agree about which month is which
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("region", REGIONS)
def test_crop_horizon_maps_to_the_same_month_the_forecast_names(region):
    """A horizon offset by one would put t+1's SPI-3 under the wrong month.

    Caught for real during Phase 8: the crop tool anchored on the newest
    *weather* month while the forecast anchored on the newest month with every
    feature present (ONI trails weather), so the two disagreed by one month.
    Both now derive from data_currency, and this pins them together.
    """
    from crop_impact.tool import latest_data_month, resolve_target_month
    from forecasting.tool import forecast_drought_risk

    forecast = forecast_drought_risk.invoke({"region": region})
    assert str(latest_data_month(region).date()) == forecast["forecast_anchor_month"]

    for crop in ("bajra", "wheat"):
        try:
            target, latest, horizon = resolve_target_month(region, crop, None)
        except Exception:
            continue                      # crop not in scope for this region
        assert str(latest.date()) == forecast["forecast_anchor_month"]
        if 1 <= horizon <= 3:
            assert str(target.date()) == forecast["forecast_months"][horizon - 1], (
                f"{crop}: horizon t+{horizon} points at {target.date()}, but the "
                f"forecast calls that month {forecast['forecast_months'][horizon - 1]}")


@needs_recent
def test_a_forward_looking_crop_question_now_gets_a_live_drought_signal():
    """Part 3's Definition of Done, at the crop layer.

    Before the refresh, bajra's sensitive window (Aug-Sep) was always in the
    past relative to a 2024-12 archive, so the drought signal was unavailable
    for the reason "that month has already happened" — stale input, not a real
    limitation. It should now resolve forward and carry a real signal.
    """
    from crop_impact.tool import collect_signals

    signals = collect_signals("barmer", "bajra", None)
    if signals["horizon"] < 1:
        pytest.skip("bajra's window is genuinely behind the current anchor")
    assert signals["drought"] is not None, signals["drought_reason"]
    assert signals["drought"]["horizon_confidence"], "no per-horizon labels"
