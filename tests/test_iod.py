"""IOD (Dipole Mode Index): parse pinned to known events, and the Phase 1.6
rejection kept honest — the feature must stay out of the default pipeline.
"""

import json

import pandas as pd
import pytest

from forecasting import config
from forecasting.iod import attach_iod, fetch_iod, parse_dmi_csv, parse_dmi_grid

# Real layout, trimmed: header years, 13-token rows, -9999.000 sentinel, metadata.
GRID_FIXTURE = """ 1870 2026
1997    -0.110     0.079     0.043     0.054     0.026     0.081     0.447     0.633     0.771     0.874     1.279     0.863
2019     0.387     0.416     0.225     0.258     0.540     0.605     0.597     0.436     0.892     0.964     0.835     0.243
2026     0.123     0.529     0.285     0.279     0.146 -9999.000 -9999.000 -9999.000 -9999.000 -9999.000 -9999.000 -9999.000
-9999
DMI HadISST1.1
Timeseries output created at NOAA PSL
"""

CSV_FIXTURE = """Date, DMI HadISST1.1  missing value -9999
1997-11-01,    1.279
2019-11-01,    0.835
2026-07-01, -9999.000
"""


# --------------------------------------------------------------------------- #
# Parsers — offline
# --------------------------------------------------------------------------- #
def test_grid_parser_reads_the_psl_layout():
    s = parse_dmi_grid(GRID_FIXTURE)
    assert s.loc["1997-11-01"] == pytest.approx(1.279)
    assert s.loc["2019-10-01"] == pytest.approx(0.964)
    assert s.name == "iod"


def test_grid_parser_drops_the_sentinel():
    s = parse_dmi_grid(GRID_FIXTURE)
    assert s.loc["2026-05-01"] == pytest.approx(0.146)
    assert pd.Timestamp("2026-06-01") not in s.index
    assert s.min() > -10


def test_csv_parser_is_a_usable_fallback():
    s = parse_dmi_csv(CSV_FIXTURE)
    assert s.loc["1997-11-01"] == pytest.approx(1.279)
    assert pd.Timestamp("2026-07-01") not in s.index


# --------------------------------------------------------------------------- #
# The real series
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def iod():
    if not config.IOD_PATH.exists():
        pytest.skip("DMI not cached — run `python -m forecasting.iod`")
    return fetch_iod()


@pytest.fixture(scope="module")
def modelling_window(iod):
    return iod.loc[config.FETCH_START:config.FETCH_END]


def test_known_iod_events_rank_correctly(iod, modelling_window):
    """PSL's HadISST1.1 DMI runs at a smaller amplitude than the DMI figures
    usually quoted, so the events are pinned by sign and rank rather than by
    absolute value — the parse is what is under test, not the SST product."""
    nov_1997 = iod.loc["1997-11-01"]
    nov_2019 = iod.loc["2019-11-01"]
    oct_2016 = iod.loc["2016-10-01"]

    # 1997: the single strongest positive month of the modelling period
    assert nov_1997 == pytest.approx(modelling_window.max())
    # 2019: top 5% positive
    assert (modelling_window < nov_2019).mean() > 0.95
    # 2016: negative, in the lower half
    assert oct_2016 < 0
    assert (modelling_window < oct_2016).mean() < 0.5


def test_series_spans_the_modelling_period(iod):
    assert iod.index.min() <= pd.Timestamp(config.FETCH_START)
    assert iod.index.max() >= pd.Timestamp(config.FETCH_END).replace(day=1)
    assert iod.notna().all()


def test_attach_adds_both_columns_without_gaps():
    frame = pd.DataFrame(index=pd.date_range("1990-01-01", "2020-12-01", freq="MS"))
    out = attach_iod(frame)
    assert {"iod", "iod_lag1"} <= set(out.columns)
    assert out[["iod", "iod_lag1"]].notna().all().all()
    # the lag really is the previous month
    assert out["iod_lag1"].iloc[5] == pytest.approx(out["iod"].iloc[4])


# --------------------------------------------------------------------------- #
# The rejection has to stay a rejection
# --------------------------------------------------------------------------- #
def test_iod_is_not_in_the_default_feature_set():
    """Phase 1.6 measured IOD and rejected it. If someone adopts it later, that
    should be a deliberate change with new numbers, not a silent drift."""
    assert not any(f.startswith("iod") for f in config.FEATURES)
    assert all(f.startswith("iod") for f in config.FEATURES_IOD[len(config.FEATURES):])


@pytest.mark.skipif(not config.IOD_COMPARISON_PATH.exists(),
                    reason="run `python -m forecasting.iod_check`")
def test_recorded_verdict_matches_the_adoption_rule():
    """The stored evidence must agree with the rule it claims to apply."""
    payload = json.loads(config.IOD_COMPARISON_PATH.read_text(encoding="utf-8"))
    for key, v in payload["verdicts"].items():
        expected = v["change"] > 0 and v["windows_improved"] >= 3
        assert v["adopt"] is expected, key
    assert payload["adopted_any"] is any(v["adopt"] for v in payload["verdicts"].values())
