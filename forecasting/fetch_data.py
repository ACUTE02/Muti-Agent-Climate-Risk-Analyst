"""Open-Meteo Historical Weather API pull (ERA5 reanalysis, no API key required).

The raw monthly pull is cached to ``data/raw/{region}_raw.parquet``. Never re-fetch
during development — call :func:`load_or_fetch_region`, which reads the cache.

Run standalone:  python -m forecasting.fetch_data
"""

from __future__ import annotations

import time
from typing import Iterator

import pandas as pd
import requests

from forecasting import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily aggregates exist for temp/precip; soil moisture is only served hourly by the
# archive endpoint, so it is requested hourly and averaged down.
DAILY_VARS = ["temperature_2m_mean", "temperature_2m_max",
              "temperature_2m_min", "precipitation_sum"]
HOURLY_VARS = ["soil_moisture_0_to_7cm"]

# The full 1980-2024 hourly series is ~394k values; request it in decade chunks so a
# single slow response cannot time the whole pull out.
CHUNK_YEARS = 10
REQUEST_TIMEOUT = 180


def _chunks(start: str, end: str) -> Iterator[tuple[str, str]]:
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    while lo <= hi:
        chunk_end = min(lo + pd.DateOffset(years=CHUNK_YEARS) - pd.Timedelta(days=1), hi)
        yield lo.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        lo = chunk_end + pd.Timedelta(days=1)


def _request(lat: float, lon: float, start: str, end: str) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "GMT",
    }
    for attempt in range(4):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            # 429 here is Open-Meteo's *per-minute* quota, so a few seconds of
            # backoff is not enough — wait out the minute.
            time.sleep(65 if resp.status_code == 429 else 5 * (attempt + 1))
            continue
        raise RuntimeError(f"Open-Meteo {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("Open-Meteo request failed after 4 attempts")


def _fetch_raw_frames(lat: float, lon: float, start: str,
                      end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames, hourly_frames = [], []
    for c_start, c_end in _chunks(start, end):
        payload = _request(lat, lon, c_start, c_end)

        d = pd.DataFrame(payload["daily"])
        d["time"] = pd.to_datetime(d["time"])
        daily_frames.append(d.set_index("time"))

        h = pd.DataFrame(payload["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        hourly_frames.append(h.set_index("time"))

    return pd.concat(daily_frames).sort_index(), pd.concat(hourly_frames).sort_index()


def _to_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily frame the Heat Stress agent works from."""
    out = pd.DataFrame({
        "temp_c": daily["temperature_2m_mean"],
        "tmax_c": daily["temperature_2m_max"],
        "tmin_c": daily["temperature_2m_min"],
        "rainfall_mm": daily["precipitation_sum"],
    })
    out.index.name = "date"
    return out


def _to_monthly(daily: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    monthly = pd.DataFrame({
        "temp_c": daily["temperature_2m_mean"].resample("MS").mean(),
        "tmax_c": daily["temperature_2m_max"].resample("MS").mean(),
        "tmin_c": daily["temperature_2m_min"].resample("MS").mean(),
        "rainfall_mm": daily["precipitation_sum"].resample("MS").sum(min_count=1),
        "soil_moisture": hourly["soil_moisture_0_to_7cm"].resample("MS").mean(),
    })
    monthly.index.name = "date"
    return monthly


def fetch_openmeteo_india(lat: float, lon: float,
                          start: str = config.FETCH_START,
                          end: str = config.FETCH_END) -> pd.DataFrame:
    """Daily temp/precip/soil-moisture for one lat/lon, resampled to monthly.

    Returns columns: temp_c, tmax_c, tmin_c, rainfall_mm, soil_moisture — indexed
    by month-start date.
    """
    daily, hourly = _fetch_raw_frames(lat, lon, start, end)
    return _to_monthly(daily, hourly)


def _fetch_and_cache(region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One API pull, two cached frames — monthly for drought, daily for heat."""
    meta = config.REGIONS[config.check_region(region)]
    daily, hourly = _fetch_raw_frames(meta["lat"], meta["lon"],
                                      config.FETCH_START, config.FETCH_END)
    monthly, daily_out = _to_monthly(daily, hourly), _to_daily(daily)
    monthly.to_parquet(config.raw_path(region))
    daily_out.to_parquet(config.daily_path(region))
    return monthly, daily_out


def load_or_fetch_region(region: str = config.DEFAULT_REGION,
                         force: bool = False) -> pd.DataFrame:
    """Return the cached raw monthly frame for ``region``, fetching only if missing."""
    config.check_region(region)
    path = config.raw_path(region)
    if path.exists() and not force:
        return pd.read_parquet(path)
    return _fetch_and_cache(region)[0]


def load_or_fetch_daily(region: str = config.DEFAULT_REGION,
                        force: bool = False) -> pd.DataFrame:
    """Daily temperatures for ``region`` — what the Heat Stress agent needs."""
    config.check_region(region)
    path = config.daily_path(region)
    if path.exists() and not force:
        return pd.read_parquet(path)
    return _fetch_and_cache(region)[1]


if __name__ == "__main__":
    for name in config.REGIONS:
        frame = load_or_fetch_region(name)
        print(f"{name}: {frame.shape} rows {frame.index.min().date()} -> "
              f"{frame.index.max().date()}")
        print(frame.head())
        print(frame.isna().sum())
