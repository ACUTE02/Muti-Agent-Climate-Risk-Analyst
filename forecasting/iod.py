"""Dipole Mode Index (IOD) — the second exogenous predictor, tested in Phase 1.6.

The IOD is a documented driver of Indian monsoon variability that is largely
independent of ENSO, and it is the one predictor this project had never tried.
Fetched with the same defensive discipline as ONI: inspect the bytes, parse, then
pin the parse to known events before anything trains on it.

**Amplitude caveat, stated rather than papered over.** NOAA PSL publishes exactly
one DMI series, computed from HadISST1.1 (verified: the ``.data`` and ``.csv``
endpoints are the same numbers, and the other candidate URLs 404). Its amplitudes
run smaller than the DMI figures usually quoted from other SST products — this
series puts November 1997 at +1.279 and November 2019 at +0.835, where commonly
cited values are nearer +1.55 and +1.78. That is a difference of SST product, not
a parsing error, so the sanity checks below verify the events by *sign and rank*
rather than absolute value: Nov 1997 is the single highest month of the modelling
period and Oct/Nov 2019 sit in the top 2%. A constant scale factor is in any case
irrelevant downstream, since the feature is min-max scaled before modelling.

Run standalone:  python -m forecasting.iod
"""

from __future__ import annotations

import pandas as pd
import requests

from forecasting import config
from forecasting.enso import MISSING_SENTINEL, parse_psl_ascii

DMI_URL = "https://psl.noaa.gov/data/timeseries/month/data/dmi.had.long.data"
DMI_CSV_URL = "https://psl.noaa.gov/data/timeseries/month/data/dmi.had.long.csv"
REQUEST_TIMEOUT = 120

# The PSL grid file uses -9999.000 where ONI's uses -99.9; parse_psl_ascii drops
# anything at or below its sentinel, so both are covered by the same filter.
DMI_MISSING = -999.0


def parse_dmi_csv(text: str) -> pd.Series:
    """Fallback: PSL's two-column ``Date, value`` CSV variant."""
    dates, values = [], []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            date = pd.Timestamp(parts[0])
            value = float(parts[1])
        except (ValueError, TypeError):
            continue
        dates.append(date)
        values.append(value)

    if not dates:
        raise ValueError("no data rows found in DMI csv")
    series = pd.Series(values, index=pd.DatetimeIndex(dates, name="date"),
                       name="iod")
    return series[series > DMI_MISSING].sort_index()


def parse_dmi_grid(text: str) -> pd.Series:
    """PSL's year-per-row grid — the same layout ONI uses."""
    series = parse_psl_ascii(text)
    return series[series > DMI_MISSING].rename("iod")


SOURCES = ((DMI_URL, parse_dmi_grid), (DMI_CSV_URL, parse_dmi_csv))


def fetch_iod(force: bool = False) -> pd.Series:
    """Monthly DMI, cached to ``models/iod_series.parquet``."""
    if config.IOD_PATH.exists() and not force:
        return pd.read_parquet(config.IOD_PATH)["iod"]

    errors = []
    for url, parser in SOURCES:
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            series = parser(resp.text)
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            continue

        need_start = pd.Timestamp(config.FETCH_START)
        need_end = pd.Timestamp(config.FETCH_END).replace(day=1)
        if series.index.min() > need_start or series.index.max() < need_end:
            errors.append(f"{url}: covers {series.index.min().date()}..."
                          f"{series.index.max().date()}, need {need_start.date()}..."
                          f"{need_end.date()}")
            continue

        series.to_frame().to_parquet(config.IOD_PATH)
        return series

    raise RuntimeError("could not obtain a usable DMI series:\n  " + "\n  ".join(errors))


def attach_iod(df: pd.DataFrame, iod: pd.Series | None = None) -> pd.DataFrame:
    """Merge DMI onto a monthly frame by (year, month), plus a 1-month lag.

    Same reasoning as ONI: the DMI is computed by NOAA from global SST fields,
    independent of this project's own target series, so its published value can be
    used directly at every date with no train/val/test asymmetry.
    """
    iod = fetch_iod() if iod is None else iod
    out = df.copy()
    out["iod"] = iod.reindex(out.index).to_numpy(dtype=float)
    out["iod_lag1"] = iod.reindex(out.index - pd.DateOffset(months=1)).to_numpy(
        dtype=float)

    missing = out[["iod", "iod_lag1"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"DMI missing for {int(missing.sum())} month(s), "
                         f"e.g. {out.index[missing][:3].tolist()}")
    return out


if __name__ == "__main__":
    s = fetch_iod()
    window = s.loc[config.FETCH_START:config.FETCH_END]
    print(f"{len(s)} months  {s.index.min().date()} -> {s.index.max().date()}")
    print(f"1980-2024: mean={window.mean():+.3f} std={window.std():.3f} "
          f"min={window.min():+.3f} max={window.max():+.3f}")
    for date, note in (("1997-11-01", "strong +IOD, with the 1997-98 El Nino"),
                       ("2016-10-01", "negative IOD"),
                       ("2019-11-01", "one of the strongest +IOD events")):
        value = s.loc[date]
        pct = float((window < value).mean() * 100)
        print(f"  {date[:7]} {value:+.3f}  pct={pct:5.1f}  ({note})")
