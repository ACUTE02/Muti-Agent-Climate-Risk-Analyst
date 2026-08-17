"""The ONI parse is pinned to known history before the model is allowed to use it.

An unvalidated ENSO parse is worse than no ENSO feature — a wrong column offset
would silently inject noise that looks like signal.
"""

import pandas as pd
import pytest

from forecasting import config
from forecasting.enso import fetch_oni, parse_cpc_seasonal, parse_psl_ascii

# Trimmed to the rows under test, byte-for-byte in the real layout: a start/end
# year header, 13-token data rows, the sentinel line, then free-text metadata.
PSL_FIXTURE = """ 1950         2026
 1997  -0.50  -0.36  -0.10   0.28   0.75   1.22   1.60   1.90   2.14   2.33   2.40   2.39
 2010   1.50   1.22   0.84   0.35  -0.17  -0.66  -1.05  -1.35  -1.56  -1.64  -1.64  -1.54
 2026  -0.37  -0.14   0.13   0.51   0.98   1.40 -99.90 -99.90 -99.90 -99.90 -99.90 -99.90
  -99.9
 ONI from CPC
  Provided by NOAA/PSL
"""

CPC_FIXTURE = """ SEAS  YR  TOTAL ANOM
  NDJ 1997  28.34  2.39
  OND 2010  25.72 -1.64
  NDJ 2010  25.85 -1.54
"""


# --------------------------------------------------------------------------- #
# Parsers — offline, no network
# --------------------------------------------------------------------------- #
def test_psl_parser_reads_the_grid_layout():
    s = parse_psl_ascii(PSL_FIXTURE)
    assert s.loc["1997-12-01"] == pytest.approx(2.39)
    assert s.loc["2010-12-01"] == pytest.approx(-1.54)
    assert s.loc["2010-01-01"] == pytest.approx(1.50)


def test_psl_parser_drops_the_missing_value_sentinel():
    s = parse_psl_ascii(PSL_FIXTURE)
    assert s.loc["2026-06-01"] == pytest.approx(1.40)
    assert pd.Timestamp("2026-07-01") not in s.index   # -99.90 must not survive
    assert s.min() > -10


def test_cpc_seasonal_parser_maps_seasons_to_centre_months():
    s = parse_cpc_seasonal(CPC_FIXTURE)
    assert s.loc["1997-12-01"] == pytest.approx(2.39)    # NDJ -> December
    assert s.loc["2010-11-01"] == pytest.approx(-1.64)   # OND -> November


# --------------------------------------------------------------------------- #
# The real series — the two events the spec requires
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def oni_series():
    if not config.ONI_PATH.exists():
        pytest.skip("ONI not cached yet — run `python -m forecasting.enso`")
    return fetch_oni()


def test_peak_of_the_1997_98_el_nino(oni_series):
    assert oni_series.loc["1997-12-01"] > 2.0


def test_strong_2010_11_la_nina(oni_series):
    assert oni_series.loc["2010-12-01"] < -1.0


def test_series_spans_the_modelling_period(oni_series):
    assert oni_series.index.min() <= pd.Timestamp(config.FETCH_START)
    assert oni_series.index.max() >= pd.Timestamp(config.FETCH_END).replace(day=1)
    assert oni_series.notna().all()


def test_values_are_physically_plausible(oni_series):
    """ONI is an SST anomaly in degrees C — historically within roughly +/-3."""
    assert oni_series.abs().max() < 3.5
    assert oni_series.std() == pytest.approx(0.85, abs=0.5)
