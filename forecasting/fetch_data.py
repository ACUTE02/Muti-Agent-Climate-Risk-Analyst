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


# --------------------------------------------------------------------------- #
# Rolling recent cache — Phase 8
# --------------------------------------------------------------------------- #
# The fixed archive stops at FETCH_END (2024-12-31) because that is the data every
# published skill score was measured against. A live question ("the next three
# months") asked in 2026 was therefore being answered from inputs ~20 months old.
#
# This path is strictly *additive*: it writes only to the `_recent` files and
# never touches `{region}_raw.parquet` / `{region}_daily.parquet`. The model is
# not retrained — same fitted Ridge coefficients, current inputs.


def _month_is_complete(daily: pd.DataFrame, month_start: pd.Timestamp) -> bool:
    """Does the daily frame cover every day of this month?

    This matters more than it looks. A monthly rainfall *sum* over a half-finished
    month is a small number, and a small rainfall number is indistinguishable from
    a dry month once it reaches SPI-3 — so including the running month would
    manufacture a drought signal out of nothing but the calendar. Partial months
    are dropped rather than scaled up, because scaling would be an estimate and
    this project does not publish estimates as measurements.
    """
    days_in_month = month_start.days_in_month
    covered = daily.loc[str(month_start.year) + "-" + f"{month_start.month:02d}"]
    return len(covered.dropna(how="all")) >= days_in_month


def refresh_recent(region: str, today: str | None = None) -> dict:
    """Fetch everything after the caches' last date, up to ``today``.

    Returns a summary rather than printing, so callers (setup, /health, tests)
    can report it. Never raises on "nothing to do" — an already-current cache is
    a normal outcome, not an error.
    """
    config.check_region(region)
    meta = config.REGIONS[region]
    end = pd.Timestamp(today) if today else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    archive_daily = pd.read_parquet(config.daily_path(region))
    have_daily = archive_daily
    recent_daily_file = config.recent_daily_path(region)
    if recent_daily_file.exists():
        have_daily = pd.concat([archive_daily, pd.read_parquet(recent_daily_file)])

    last_have = have_daily.index.max()
    start = (last_have + pd.Timedelta(days=1)).normalize()

    summary = {
        "region": region,
        "archive_through": str(archive_daily.index.max().date()),
        "requested_through": str(end.date()),
    }

    if start > end:
        summary.update(fetched_days=0, note="already current")
        return {**summary, **data_currency(region)}

    daily_raw, hourly_raw = _fetch_raw_frames(
        meta["lat"], meta["lon"], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    new_daily = _to_daily(daily_raw).dropna(how="all")
    if recent_daily_file.exists():
        new_daily = pd.concat([pd.read_parquet(recent_daily_file), new_daily])
    new_daily = new_daily[~new_daily.index.duplicated(keep="last")].sort_index()
    new_daily.to_parquet(recent_daily_file)

    # Monthly frame is derived from the *union* of archive + recent daily, so a
    # month straddling the archive boundary is aggregated from all of its days,
    # then filtered to the months the fixed archive does not already own.
    union_daily = pd.concat([archive_daily, new_daily])
    union_daily = union_daily[~union_daily.index.duplicated(keep="last")].sort_index()

    hourly = hourly_raw.sort_index()
    monthly = _to_monthly(
        union_daily.rename(columns={"temp_c": "temperature_2m_mean",
                                    "tmax_c": "temperature_2m_max",
                                    "tmin_c": "temperature_2m_min",
                                    "rainfall_mm": "precipitation_sum"}),
        hourly)

    archive_monthly_end = pd.read_parquet(config.raw_path(region)).index.max()
    monthly = monthly[monthly.index > archive_monthly_end]

    # Drop any month the daily record does not fully cover (see _month_is_complete).
    complete = [m for m in monthly.index if _month_is_complete(union_daily, m)]
    dropped = [str(m.date()) for m in monthly.index if m not in complete]
    monthly = monthly.loc[complete]
    monthly.to_parquet(config.recent_path(region))

    summary.update(
        fetched_days=int(len(new_daily)),
        new_complete_months=int(len(monthly)),
        dropped_incomplete_months=dropped,
    )
    return {**summary, **data_currency(region)}


def load_recent_region(region: str) -> pd.DataFrame:
    """Monthly rows newer than the fixed archive, or an empty frame."""
    path = config.recent_path(region)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_recent_daily(region: str) -> pd.DataFrame:
    """Daily rows newer than the fixed archive, or an empty frame."""
    path = config.recent_daily_path(region)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _union(archive: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    if recent.empty:
        return archive
    joined = pd.concat([archive, recent.reindex(columns=archive.columns)])
    return joined[~joined.index.duplicated(keep="last")].sort_index()


def load_live_region(region: str = config.DEFAULT_REGION) -> pd.DataFrame:
    """Fixed archive + rolling recent, monthly. What a *live* forecast reads.

    The evaluation path deliberately does not use this — it calls
    ``load_or_fetch_region`` so its windows stay pinned to the fixed archive.
    """
    return _union(load_or_fetch_region(region), load_recent_region(region))


def load_live_daily(region: str = config.DEFAULT_REGION) -> pd.DataFrame:
    """Fixed archive + rolling recent, daily. What the heat tool observes from."""
    return _union(load_or_fetch_daily(region), load_recent_daily(region))


def data_currency(region: str) -> dict:
    """How current this region's inputs are — surfaced by /health and the tools.

    Reports two different "through" dates on purpose, because they differ and the
    difference matters:

    * ``weather_through`` — the last complete month of Open-Meteo data.
    * ``data_current_through`` — the last month for which *every* model input
      exists, which is what the forecast is actually anchored to.

    The gap between them is normally ONI: NOAA publishes it as a 3-month running
    mean, so it trails by a month or two. Reporting only the weather date would
    overstate how current the forecast is.
    """
    config.check_region(region)
    archive = pd.read_parquet(config.raw_path(region))
    recent = load_recent_region(region)
    weather_end = recent.index.max() if not recent.empty else archive.index.max()

    try:
        from forecasting.enso import fetch_oni
        oni_end = fetch_oni().dropna().index.max()
    except Exception:                        # ONI unavailable is reportable, not fatal
        oni_end = None

    anchor = min(weather_end, oni_end) if oni_end is not None else weather_end
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    months_behind = ((today.year - anchor.year) * 12 + (today.month - anchor.month))

    return {
        "archive_through": str(archive.index.max().date()),
        "weather_through": str(weather_end.date()),
        "oni_through": str(oni_end.date()) if oni_end is not None else None,
        "data_current_through": str(anchor.date()),
        "months_behind_today": int(months_behind),
        "limiting_input": ("ONI (NOAA publishes it as a 3-month running mean, so it "
                           "trails the weather data)"
                           if oni_end is not None and oni_end < weather_end
                           else "Open-Meteo weather archive"),
        "refresh_command": "python -m scripts.refresh",
    }


if __name__ == "__main__":
    for name in config.REGIONS:
        frame = load_or_fetch_region(name)
        print(f"{name}: {frame.shape} rows {frame.index.min().date()} -> "
              f"{frame.index.max().date()}")
        print(frame.head())
        print(frame.isna().sum())
