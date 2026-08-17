"""Oceanic Niño Index (ONI) — the exogenous ENSO predictor.

ONI is NOAA's 3-month running mean of ERSST SST anomalies in the Niño 3.4 region.
It is computed from global sea-surface temperatures, entirely independent of the
local rainfall series being modelled, so — unlike ``month_stats`` or the SPI-3
gamma parameters, which are statistics *of this project's own target* — its
published value can be used directly at every date with no train/val/test
asymmetry and no leakage risk.

The primary source is NOAA PSL's ASCII grid; the CPC seasonal table is the
fallback. The parsers below were written against the actual downloaded bytes
(inspected first, not assumed), and ``tests/test_enso_sanity.py`` pins the parse
to two well-established historical events before the series is allowed anywhere
near the model.

Phase 6 note: NOAA's most recent 1-2 months of ONI are *provisional* and get
revised as SST analyses are updated. Irrelevant for this 1980-2024 historical
backtest, but live inference will need to tolerate a revised — or entirely
missing — latest month.

Run standalone:  python -m forecasting.enso
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from forecasting import config

PSL_URL = "https://psl.noaa.gov/data/correlation/oni.data"
CPC_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

MISSING_SENTINEL = -99.0    # the PSL file writes -99.90 for months not yet observed
REQUEST_TIMEOUT = 120

# CPC labels each overlapping 3-month season by its trigram; the standard
# convention centres DJF on January, JFM on February, and so on.
SEASON_CENTER_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def _to_series(pairs: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    s = pd.Series(dict(pairs), name="oni").sort_index()
    s.index = pd.DatetimeIndex(s.index, name="date")
    return s[s > MISSING_SENTINEL].dropna()


def parse_psl_ascii(text: str) -> pd.Series:
    """Parse NOAA PSL's ``oni.data``: a start/end-year header, then one row per
    year of 12 monthly values, then a sentinel line and free-text metadata."""
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty ONI response")

    header = lines[0].split()
    try:
        start_year, end_year = int(header[0]), int(header[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unexpected PSL header line: {lines[0]!r}") from exc

    pairs: list[tuple[pd.Timestamp, float]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 13:          # sentinel + metadata lines fall out here
            continue
        try:
            year = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if not start_year <= year <= end_year:
            continue
        pairs.extend((pd.Timestamp(year=year, month=m, day=1), v)
                     for m, v in enumerate(values, start=1))

    if not pairs:
        raise ValueError("no data rows found in PSL ONI file")
    return _to_series(pairs)


def parse_cpc_seasonal(text: str) -> pd.Series:
    """Parse CPC's seasonal table (``SEAS YR TOTAL ANOM`` rows), mapping each
    overlapping 3-month season to its centre month."""
    pairs: list[tuple[pd.Timestamp, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].upper() not in SEASON_CENTER_MONTH:
            continue
        try:
            year, anom = int(parts[1]), float(parts[-1])
        except ValueError:
            continue
        month = SEASON_CENTER_MONTH[parts[0].upper()]
        pairs.append((pd.Timestamp(year=year, month=month, day=1), anom))

    if not pairs:
        raise ValueError("no season rows found in CPC ONI table")
    return _to_series(pairs)


SOURCES = ((PSL_URL, parse_psl_ascii), (CPC_URL, parse_cpc_seasonal))


def fetch_oni(force: bool = False) -> pd.Series:
    """Monthly ONI series, cached to ``models/oni_series.parquet``."""
    if config.ONI_PATH.exists() and not force:
        return pd.read_parquet(config.ONI_PATH)["oni"]

    errors = []
    for url, parser in SOURCES:
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            series = parser(resp.text)
        except Exception as exc:                      # try the next source
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            continue

        _assert_covers_modelling_period(series, url)
        series.to_frame().to_parquet(config.ONI_PATH)
        return series

    raise RuntimeError("could not obtain a usable ONI series:\n  " + "\n  ".join(errors))


def _assert_covers_modelling_period(series: pd.Series, url: str) -> None:
    need_start = pd.Timestamp(config.FETCH_START)
    need_end = pd.Timestamp(config.FETCH_END).replace(day=1)
    if series.index.min() > need_start or series.index.max() < need_end:
        raise ValueError(
            f"{url} returned ONI covering {series.index.min().date()}..."
            f"{series.index.max().date()}, which does not span "
            f"{need_start.date()}...{need_end.date()}"
        )


def attach_oni(df: pd.DataFrame, oni: pd.Series | None = None) -> pd.DataFrame:
    """Merge ONI onto a monthly climate frame by (year, month)."""
    oni = fetch_oni() if oni is None else oni
    out = df.copy()
    out["oni"] = oni.reindex(out.index).to_numpy(dtype=float)
    if out["oni"].isna().any():
        gaps = out.index[out["oni"].isna()]
        raise ValueError(f"ONI missing for {len(gaps)} month(s), e.g. {gaps[:3].tolist()}")
    return out


if __name__ == "__main__":
    s = fetch_oni()
    print(f"{len(s)} months  {s.index.min().date()} -> {s.index.max().date()}")
    print(f"1997-12 = {s.loc['1997-12-01']:+.2f}  (strong El Nino, expect > +2.0)")
    print(f"2010-12 = {s.loc['2010-12-01']:+.2f}  (strong La Nina, expect < -1.0)")
    print(f"range {s.min():+.2f} .. {s.max():+.2f}, mean {np.mean(s):+.3f}")
