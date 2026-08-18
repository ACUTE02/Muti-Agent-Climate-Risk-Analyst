"""Heat Stress agent — the callable tool.

**This does not forecast.** Phases 1 and 1.1 tested heat stress prediction across
three target definitions (monthly mean Tmax anomaly, hottest-day anomaly, and IMD
heat wave day count), two feature sets (heat-only and heat + a drought
cross-feature), two regions and three horizons — 36 measured cells, best skill
+0.0378 against a +0.1 bar. Nothing forecasts heat stress with this data.

So the live interface serves only what is reliable: the *observed* IMD heat wave
indicator. The training and evaluation code that established the negative result
stays in `heat/target.py`, `heat/model.py` and `heat/phase11.py` with its evidence
files — the record is worth keeping; shipping three flavours of a result that does
not work is not.

Run standalone:  python -m heat.tool [region ...]
"""

from __future__ import annotations

import json

import pandas as pd
from langchain_core.tools import tool

from forecasting import config
from forecasting.fetch_data import load_or_fetch_daily
from heat.target import (fit_daily_normals, flag_heat_wave_days,
                         monthly_heat_wave_counts)

NO_FORECAST_NOTE = (
    "Heat Stress forecasting was tested (mean anomaly, extreme-day anomaly, and "
    "heat wave day count, with and without a drought cross-feature) and found to "
    "have no usable skill at any horizon. This function reports observed "
    "conditions only. See PROJECT_LOG.md for the full record."
)


def observed_heat_wave_months(region: str) -> pd.DataFrame:
    """Monthly IMD heat wave counts for a region, from daily observations.

    The day-of-year normals behind the departure rule are fit on the train
    partition only, the same discipline as everywhere else in this project — even
    though nothing here is a forecast, so leakage could not flatter a result.
    """
    daily = load_or_fetch_daily(config.check_region(region))
    normals = fit_daily_normals(daily, config.TRAIN)
    return monthly_heat_wave_counts(flag_heat_wave_days(daily, normals))


def _resolve_month(counts: pd.DataFrame, month: str | None) -> pd.Timestamp:
    if month is None:
        return counts.index.max()
    try:
        resolved = pd.Timestamp(month).replace(day=1)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not parse month {month!r} — expected something like "
            "'2024-05' or '2024-05-01'.") from exc

    if resolved not in counts.index:
        raise ValueError(
            f"No observations for {resolved.date():%Y-%m}. Available range: "
            f"{counts.index.min():%Y-%m} to {counts.index.max():%Y-%m}.")
    return resolved


@tool
def forecast_heat_stress_risk(region: str = config.DEFAULT_REGION,
                              month: str | None = None) -> dict:
    """
    Reports OBSERVED heat wave activity for the given month (defaults to the most
    recent month with data). This function does not forecast future heat stress —
    Phase 1/1.1 established there is no usable predictive skill for this risk type
    with the available zero-cost data (see PROJECT_LOG.md, Heat Stress Agent Phase
    1 and 1.1). It reports what happened, using IMD's heat wave criteria, adapted
    for a single grid point (see heat/target.py for the adaptation).
    """
    config.check_region(region)
    counts = observed_heat_wave_months(region)
    resolved = _resolve_month(counts, month)
    row = counts.loc[resolved]

    return {
        "region": region,
        "month": f"{resolved:%Y-%m}",
        "heatwave_days": int(row["heat_wave_days"]),
        "severe_heatwave_days": int(row["severe_heat_wave_days"]),
        "had_heatwave_spell": bool(row["had_heat_wave_spell"]),
        "max_tmax_c": round(float(row["max_tmax_c"]), 1),
        "forecast_available": False,
        "note": NO_FORECAST_NOTE,
    }


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    month = args.pop() if args and "-" in args[-1] else None
    for name in (args or list(config.REGIONS)):
        print(json.dumps(
            forecast_heat_stress_risk.invoke({"region": name, "month": month}),
            indent=2))
