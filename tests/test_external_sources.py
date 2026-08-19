"""Phase 8 Part 4 — the new live sources are cited, not absorbed.

The hard constraint this phase was given: no figure fetched from IMD, NASA POWER
or data.gov.in may ever be presented as something this project computed. These
tests check the machinery that keeps that true — attribution fields, separation
from `tool_outputs`, checkability by the grounding checker, and graceful failure
— plus the specific data hazards each source carries (NASA POWER's -999 fill
value, data.gov.in's missing key and flaky gateway).

Network tests are gated behind RUN_LIVE_EXTERNAL=1, the same discipline the rest
of the suite uses for live calls. They cost no Gemini quota — both APIs are free
— but they do depend on someone else's uptime, which is not a reason to fail CI.
"""

from __future__ import annotations

import json
import os

import pytest

from retrieval import external

live_only = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EXTERNAL") != "1",
    reason="set RUN_LIVE_EXTERNAL=1 to hit NASA POWER / data.gov.in for real")

ATTRIBUTION_FIELDS = ("id", "title", "publisher", "citation", "fetched_at")


# --------------------------------------------------------------------------- #
# Attribution is structural, not a matter of the model's good manners
# --------------------------------------------------------------------------- #
def test_every_source_carries_its_publisher_and_citation_even_when_unavailable():
    """An unavailable source still has to say whose it was and why it failed."""
    payload = external.fetch_external_sources(region="rajasthan", crop=None)
    assert payload["sources"], "no external sources were attempted"

    for source in payload["sources"]:
        for field in ATTRIBUTION_FIELDS:
            assert source.get(field), f"{source.get('id')} is missing {field}"
        assert "available" in source
        if not source["available"]:
            assert source.get("reason"), "an unavailable source must say why"


def test_no_source_claims_to_be_this_projects_measurement():
    payload = external.fetch_external_sources(region="rajasthan", crop="bajra")
    for source in payload["sources"]:
        blob = json.dumps(source, default=str).lower()
        for forbidden in ("this project's measured", "this project's forecast",
                          "our forecast", "we predict"):
            assert forbidden not in blob, (
                f"{source['id']} describes itself as this project's own work")


def test_publishers_are_the_real_organisations():
    payload = external.fetch_external_sources(region="barmer", crop="wheat")
    by_id = {s["id"]: s for s in payload["sources"]}
    assert "NASA" in by_id["nasa_power"]["publisher"]
    assert "power.larc.nasa.gov" in by_id["nasa_power"]["citation"]
    assert "Agriculture" in by_id["data_gov_in_mandi"]["publisher"]
    assert "data.gov.in" in by_id["data_gov_in_mandi"]["citation"]


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #
def test_missing_data_gov_key_degrades_instead_of_crashing(monkeypatch):
    """Same contract as a missing Gemini key: reportable, never fatal."""
    monkeypatch.setattr(external, "get_data_gov_key", lambda: None)
    result = external.fetch_mandi_prices("rajasthan", "bajra")

    assert result["available"] is False
    assert "api key" in result["reason"].lower()
    assert "data.gov.in/apis" in result["reason"], "tell the reader how to fix it"


def test_a_crash_inside_a_fetch_becomes_a_reported_reason(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated outage")

    monkeypatch.setattr(external.requests, "get", boom)
    result = external.fetch_nasa_power("rajasthan", "2026-06")
    assert result["available"] is False
    assert "simulated outage" in result["reason"]


def test_unknown_crop_and_absent_crop_are_different_reported_states():
    absent = external.fetch_mandi_prices("rajasthan", None)
    unknown = external.fetch_mandi_prices("rajasthan", "dragonfruit")
    assert absent["available"] is False and unknown["available"] is False
    assert absent["reason"] != unknown["reason"]
    assert "no crop was part of this request" in absent["reason"]


def test_aggregate_never_raises_even_if_both_sources_fail(monkeypatch):
    monkeypatch.setattr(external.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(external, "get_data_gov_key", lambda: None)
    payload = external.fetch_external_sources("rajasthan", "bajra")
    assert payload["any_unavailable"] is True
    assert all(not s["available"] for s in payload["sources"])


# --------------------------------------------------------------------------- #
# NASA POWER's fill value must never become a reading
# --------------------------------------------------------------------------- #
def test_nasa_power_fill_values_are_not_reported_as_measurements(monkeypatch):
    """-999.0 means 'no value'. Reporting it as a temperature would be the worst
    kind of fabrication: a real-looking number that is not data at all."""
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"properties": {"parameter": {
                # Two real days, three fill days.
                "T2M": {"20260601": 33.0, "20260602": 35.0,
                        "20260603": -999.0, "20260604": -999.0,
                        "20260605": -999.0},
                # Entirely missing — must be dropped, not averaged to -999.
                "RH2M": {"20260601": -999.0, "20260602": -999.0},
            }}}

    monkeypatch.setattr(external.requests, "get",
                        lambda *a, **k: FakeResponse())
    result = external.fetch_nasa_power("rajasthan", "2026-06")

    assert result["available"] is True
    assert "RH2M" not in result["values"], "an all-fill parameter must be dropped"

    t2m = result["values"]["T2M"]
    assert t2m["value"] == 34.0, "the mean must use only the two real days"
    assert t2m["days_used"] == 2 and t2m["days_returned"] == 5
    assert "-999" not in result["excerpt"]


def test_nasa_power_reports_no_usable_values_rather_than_inventing_one(monkeypatch):
    class AllFill:
        status_code = 200

        @staticmethod
        def json():
            return {"properties": {"parameter": {"T2M": {"20260601": -999.0}}}}

    monkeypatch.setattr(external.requests, "get", lambda *a, **k: AllFill())
    result = external.fetch_nasa_power("rajasthan", "2026-06")
    assert result["available"] is False
    assert "fill values" in result["reason"]


def test_precipitation_total_is_not_labelled_as_a_daily_rate():
    """Summing mm/day over a month gives mm. Keeping '/day' would overstate the
    quantity by roughly a factor of thirty."""
    long_name, units, how = external.POWER_PARAMS["PRECTOTCORR"]
    assert how == "total"
    assert units == "mm", "a monthly total must not carry per-day units"


def test_evapotranspiration_units_are_carried_through_unconverted():
    """POWER publishes an energy flux. Converting it to mm would be this project
    computing a number and attributing it to NASA."""
    _, units, _ = external.POWER_PARAMS["EVPTRNS"]
    assert units == "MJ/m^2/day"


# --------------------------------------------------------------------------- #
# The orchestrator keeps them separate from this project's own figures
# --------------------------------------------------------------------------- #
def test_external_sources_are_not_merged_into_tool_outputs():
    from api.app import build_response

    state = {
        "report": "x", "tool_outputs": {"forecast_drought_risk": {"region": "barmer"}},
        "grounding": {"grounded": True, "unverified_numbers": [], "total_checked": 0},
        "external": {"sources": [{
            "id": "nasa_power", "title": "NASA POWER", "publisher": "NASA",
            "citation": "https://power.larc.nasa.gov/", "available": True,
            "fetched_at": "2026-08-19T00:00:00+00:00", "excerpt": "mean 33.4 C"}]},
    }
    body = build_response(state, "req")

    assert body["external_sources"], "external sources must be surfaced"
    assert "nasa_power" not in json.dumps(body["tool_outputs"]), (
        "an outside source leaked into this project's own tool outputs")
    assert body["external_sources"][0]["publisher"] == "NASA"


def test_external_source_numbers_are_checkable_by_the_grounding_checker():
    """Their excerpts must reach the checker's source blob, exactly like IMD's —
    otherwise quoting NASA POWER would look like a fabrication."""
    from orchestrator.grounding import check_grounding

    source = {"excerpt": "NASA POWER reports temperature at 2 m: monthly mean "
                         "33.37 C (from 31 NASA POWER daily values)."}
    clean = check_grounding("NASA POWER reports a monthly mean of 33.37 C.",
                            {}, [source])
    assert clean["grounded"], clean["unverified_numbers"]

    invented = check_grounding("NASA POWER reports a monthly mean of 41.9 C.",
                               {}, [source])
    assert not invented["grounded"], "a number NASA did not publish must be flagged"


def test_renderer_labels_external_sources_by_publisher():
    from orchestrator.graph import _render_sources

    state = {"retrieved_chunks": [], "type_c": {"outlooks": []},
             "external": {"sources": [
                 {"id": "nasa_power", "title": "NASA POWER agrometeorology",
                  "publisher": "NASA Langley Research Center", "available": True,
                  "citation": "https://power.larc.nasa.gov/",
                  "fetched_at": "2026-08-19T00:00:00+00:00",
                  "excerpt": "temperature at 2 m: monthly mean 33.37 C"},
                 {"id": "data_gov_in_mandi", "title": "data.gov.in mandi prices",
                  "publisher": "Ministry of Agriculture", "available": False,
                  "citation": "https://api.data.gov.in/", "reason": "no API key"}]}}

    rendered = _render_sources(state)
    assert "EXTERNAL LIVE SOURCE (NASA Langley Research Center)" in rendered
    assert "EXTERNAL LIVE SOURCE UNAVAILABLE" in rendered
    assert "no API key" in rendered


# --------------------------------------------------------------------------- #
# Live (opt-in)
# --------------------------------------------------------------------------- #
@live_only
def test_live_nasa_power_returns_real_values():
    result = external.fetch_nasa_power("barmer")
    assert result["available"], result.get("reason")
    assert result["values"], "no parameters came back"
    for name, entry in result["values"].items():
        assert entry["value"] != external.POWER_FILL
        assert entry["days_used"] > 0


@live_only
def test_live_data_gov_in_returns_prices_or_says_why_not():
    result = external.fetch_mandi_prices("rajasthan", "bajra")
    if not result["available"]:
        pytest.skip(f"data.gov.in unavailable: {result['reason']}")
    assert result["markets"]
    for row in result["markets"]:
        assert row["modal_price_rs_per_quintal"] is not None
        assert row["market"] and row["arrival_date"]
