"""Phase-1 deliverable: the forecasting agent's callable tool.

Standalone for now — Phase 3 wires it into the LangGraph orchestrator.

Run standalone:  python -m forecasting.tool
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd
from langchain_core.tools import tool

from forecasting import config
from forecasting.clean import clean_india_climate
from forecasting.enso import attach_oni, fetch_oni
from forecasting.fetch_data import data_currency, load_live_region
from forecasting.split import compute_spi3, latest_window


def spi_to_risk_score(avg_predicted_spi: float) -> float:
    """Map mean forecast SPI to a 0-10 risk score.

    Tunable placeholder: Phase 5 will calibrate this mapping against real NDMA
    drought records rather than the linear rule used here.
    """
    return round(min(10, max(0, -avg_predicted_spi * 4 + 2)), 1)


def spi_to_flag(spi: float) -> str:
    if spi < config.SPI_SEVERE:
        return "Severe"
    if spi < config.SPI_MODERATE:
        return "Moderate"
    return "Normal"


@lru_cache(maxsize=None)
def _artifacts(region: str):
    """Load one region's scaler, SPI-3 params and per-horizon models.

    Since Phase 1.5 the forecast is served by the per-horizon Ridge models the
    measurements actually favoured, not by the LSTM — which has no skill at any
    horizon (see models/region_comparison.md). Which model answers which horizon
    is read from the manifest, not hardcoded here.
    """
    config.check_region(region)
    paths = [config.scaler_path(region), config.spi_params_path(region),
             config.HORIZON_MANIFEST_PATH]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {region} artifacts {missing} — run "
            f"`python -m forecasting.t1_model` and "
            f"`python -m forecasting.recursive` first."
        )

    scaler = joblib.load(config.scaler_path(region))
    spi_params = joblib.load(config.spi_params_path(region))
    manifest = json.loads(
        config.HORIZON_MANIFEST_PATH.read_text(encoding="utf-8"))["regions"][region]

    models = {}
    for entry in manifest:
        horizon = int(entry["horizon"].removeprefix("t+"))
        if entry["method"] == "direct":
            path = config.horizon_model_path(region, horizon)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path.name} missing — run `python -m forecasting.recursive`")
            models[horizon] = joblib.load(path)
    return scaler, spi_params, manifest, models


def _feature_frame(region: str, spi_params: dict) -> pd.DataFrame:
    """Latest cleaned features for a region, with SPI-3 from train-only gamma params.

    Trimmed to the last month for which *every* feature exists. In practice the
    binding constraint is ONI: NOAA publishes it as a 3-month running mean, so it
    trails the weather data by a month or two. Forward-filling the gap would mean
    feeding the model a made-up ENSO value, so the frame stops where the real data
    stops and the forecast is anchored there — which the tool then states
    explicitly rather than letting the reader assume it starts from today.
    """
    # load_live_region, not load_or_fetch_region: a *live* forecast reads the
    # fixed archive plus the rolling recent cache, so "the next three months"
    # means the next three months from now. The evaluation path deliberately
    # still reads the fixed archive alone (forecasting/split.py), which is why
    # every published skill score is unaffected by a refresh.
    cleaned = clean_india_climate(load_live_region(region))
    oni = fetch_oni()
    usable = cleaned.index.intersection(oni.dropna().index)
    frame = attach_oni(cleaned.loc[usable], oni)
    frame[config.TARGET] = compute_spi3(frame, spi_params)
    return frame


@tool
def forecast_drought_risk(region: str = config.DEFAULT_REGION) -> dict:
    """
    Forecasts drought risk (SPI-3, 3-month horizon) for a supported Indian region.
    Loads the region's scaler/SPI-3 parameters and the per-horizon models, pulls
    the latest months of features, and returns a structured forecast in which each
    horizon carries its own measured skill score — t+1 is validated, t+2 is weak,
    t+3 has no skill and is returned for context only.
    """
    config.check_region(region)
    scaler, spi_params, manifest, models = _artifacts(region)
    frame = _feature_frame(region, spi_params)
    anchor = frame.index[-1]        # last month with complete real inputs

    predicted, confidence = [], []
    for entry in manifest:
        horizon = int(entry["horizon"].removeprefix("t+"))
        if entry["method"] == "direct":
            X, _ = latest_window(frame, scaler, seq_len=entry["lookback"])
            value = float(models[horizon].predict(X.reshape(1, -1)).ravel()[0])
        else:
            value = _recursive_value(region, frame, scaler, spi_params,
                                     entry["lookback"], horizon, models)
        predicted.append(value)
        confidence.append({
            "horizon": entry["horizon"],
            "skill_score": entry["skill_score"],
            "method": entry["method"],
            "label": entry["label"],
            # Which real month this horizon actually refers to. "t+1" alone is
            # meaningless to a reader, and silently ambiguous if the inputs are
            # stale — naming the month makes staleness visible instead.
            "month": str((anchor + pd.DateOffset(months=horizon)).date()),
        })

    return {
        "region": region,
        "predicted_values": predicted,
        "horizon_confidence": confidence,
        "risk_score": spi_to_risk_score(sum(predicted) / len(predicted)),
        "risk_flags": [spi_to_flag(v) for v in predicted],
        "model_rmse_test": manifest[0]["rmse_window_a"],
        # Provenance of the *inputs*, so a forecast anchored to old data can
        # never look current. See forecasting.fetch_data.data_currency.
        "forecast_anchor_month": str(anchor.date()),
        "forecast_months": [str((anchor + pd.DateOffset(months=h)).date())
                            for h in (1, 2, 3)],
        "data_currency": data_currency(region),
    }


def _recursive_value(region: str, frame, scaler, spi_params, lookback: int,
                     horizon: int, models: dict) -> float:
    """Chain the t+1 model forward — used only if the manifest picked recursive."""
    from forecasting.recursive import recursive_forecast
    from forecasting.split import Dataset, SplitWindows

    ds = Dataset(frame=frame, scaler=scaler, month_stats=None,
                 spi_params=spi_params, region=region, windows=SplitWindows())
    preds = recursive_forecast(models[1], ds, frame.index[-1], lookback,
                               horizon=horizon)
    return float(preds[-1])


if __name__ == "__main__":
    import sys

    for name in (sys.argv[1:] or list(config.REGIONS)):
        print(json.dumps(forecast_drought_risk.invoke({"region": name}), indent=2))
