"""Phase 6 — the local API.

The offline majority mocks `analyse` so the API's own behaviour is tested
without spending quota: response shaping, error handling, and the honesty fields
a client depends on. One live smoke test is gated behind the same
`RUN_LIVE_ORCHESTRATOR=1` as every other live test in this project.

Quota cost of the live test: one `POST /report`, which is 2 generate_content
calls typically and up to 4 if synthesis retries or the crop tool is routed to,
against a measured 20/day budget.
"""

from __future__ import annotations

import inspect
import os

import pytest
from fastapi.testclient import TestClient

from api import quota
from api.app import app

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures standing in for a finished orchestrator run
# --------------------------------------------------------------------------- #
def fake_state(**overrides) -> dict:
    state = {
        "request": "drought risk for barmer",
        "report": "Forecast SPI-3 is -1.62 at t+1.",
        "grounding": {"grounded": True, "unverified_numbers": [],
                      "total_checked": 7, "source_numbers_available": 20},
        "tool_calls": [{"name": "forecast_drought_risk", "args": {}, "id": "1"},
                       {"name": "retrieve_context", "args": {}, "id": "2"}],
        "tool_outputs": {
            "forecast_drought_risk": {
                "region": "barmer",
                "predicted_values": [-1.62, -0.41, 0.08],
                "horizon_confidence": [
                    {"horizon": "t+1", "skill_score": 0.2053,
                     "method": "direct", "label": "validated"},
                    {"horizon": "t+2", "skill_score": 0.0438,
                     "method": "direct", "label": "weak/directional"},
                    {"horizon": "t+3", "skill_score": -0.0489,
                     "method": "direct",
                     "label": "no skill — shown for context only, do not rely "
                              "on this figure"},
                ],
            },
        },
        "retrieved_chunks": [{"source": "NIH Roorkee", "citation": "https://x",
                              "section": "SPI", "score": 0.77}],
        "type_c": {"outlooks": [{"title": "IMD Extended Range",
                                 "available": True, "excerpt": "..."}]},
        "warnings": [],
    }
    state.update(overrides)
    return state


@pytest.fixture
def mock_analyse(monkeypatch):
    """Replace the orchestrator so the API is tested without Gemini calls."""
    calls = {}

    def _install(state: dict):
        def fake(**kwargs):
            calls.update(kwargs)
            return state
        monkeypatch.setattr("orchestrator.graph.analyse", fake)
        return calls

    return _install


# --------------------------------------------------------------------------- #
# /health — must never call an LLM
# --------------------------------------------------------------------------- #
def test_health_makes_no_llm_call():
    """Same guard as test_checker_is_not_an_llm: assert it by reading the source.

    /health is the endpoint a container orchestrator hits every 30 seconds. If it
    ever costs a Gemini call, a 20/day budget is gone in ten minutes.
    """
    from api.app import health

    source = inspect.getsource(health)
    for banned in ("analyse", "ChatGoogleGenerativeAI", "get_chat_model",
                   "invoke_with_backoff", "generate_content", "embed_query"):
        assert banned not in source, f"/health must not reference {banned}"


def test_health_reports_readiness_and_what_is_missing():
    body = client.get("/health").json()

    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["chroma_index_ready"], bool)
    assert isinstance(body["missing_artifacts"], list)
    assert "quota" in body
    if body["status"] == "degraded":
        assert body["missing_artifacts"] or not body["chroma_index_ready"], \
            "degraded must say what is actually missing"


def test_health_and_setup_check_agree():
    """The two readiness checks must not disagree about whether we are ready."""
    from scripts.setup import verify

    ok, missing = verify()
    body = client.get("/health").json()
    assert body["forecast_artifacts_ready"] == (
        not [m for m in missing if "chroma" not in m and "chunks" not in m])


# --------------------------------------------------------------------------- #
# /report — response shaping
# --------------------------------------------------------------------------- #
def test_report_calls_analyse_and_returns_the_report(mock_analyse):
    calls = mock_analyse(fake_state())
    response = client.post("/report", json={"request": "drought risk for barmer",
                                            "region": "barmer"})

    assert response.status_code == 200
    body = response.json()
    assert body["report"].startswith("Forecast SPI-3")
    assert calls["request"] == "drought risk for barmer"
    assert calls["region"] == "barmer"


def test_grounding_status_is_structured_not_inferred_from_prose(mock_analyse):
    mock_analyse(fake_state())
    body = client.post("/report", json={"request": "x"}).json()

    assert body["grounding"]["status"] == "clean"
    assert body["grounding"]["grounded"] is True
    assert body["grounding"]["total_checked"] == 7
    assert body["grounding"]["explanation"]


def test_ungrounded_report_is_flagged_as_a_warning_with_the_numbers(mock_analyse):
    mock_analyse(fake_state(grounding={
        "grounded": False, "unverified_numbers": ["-0.99", "-1.49"],
        "total_checked": 9}))
    body = client.post("/report", json={"request": "x"}).json()

    assert body["grounding"]["status"] == "warning"
    assert body["grounding"]["unverified_numbers"] == ["-0.99", "-1.49"]
    assert "-0.99" in body["grounding"]["explanation"]


def test_missing_report_is_not_reported_as_grounded(mock_analyse):
    """Zero numbers checked is a vacuous pass — the API must not launder it."""
    mock_analyse(fake_state(report="", grounding={
        "grounded": False, "unverified_numbers": [], "total_checked": 0,
        "report_missing": True, "reason": "synthesis produced no report"}))
    body = client.post("/report", json={"request": "x"}).json()

    assert body["grounding"]["status"] == "not_generated"
    assert body["grounding"]["grounded"] is False
    assert "nothing to verify" in body["grounding"]["explanation"]


def test_horizon_labels_stay_distinct_and_are_never_collapsed(mock_analyse):
    """The single most important client-facing property of this API.

    Five phases established that t+1, t+2 and t+3 are worth different amounts.
    Averaging them into one confidence number server-side would undo all of it.
    """
    mock_analyse(fake_state())
    horizons = client.post("/report", json={"request": "x"}).json()[
        "horizon_confidence"]

    assert [h["horizon"] for h in horizons] == ["t+1", "t+2", "t+3"]
    assert [h["reliable"] for h in horizons] == [True, False, False]
    assert horizons[0]["label"] == "validated"
    assert horizons[2]["label"].startswith("no skill")


def test_reliable_follows_the_label_not_a_threshold_on_the_score(mock_analyse):
    """A high skill score with a non-validated label must still read unreliable."""
    state = fake_state()
    state["tool_outputs"]["forecast_drought_risk"]["horizon_confidence"][1][
        "skill_score"] = 0.99
    mock_analyse(state)
    horizons = client.post("/report", json={"request": "x"}).json()[
        "horizon_confidence"]

    assert horizons[1]["skill_score"] == 0.99
    assert horizons[1]["reliable"] is False


def test_no_skill_horizon_appears_as_an_explicit_missing_data_flag(mock_analyse):
    mock_analyse(fake_state())
    flags = client.post("/report", json={"request": "x"}).json()["missing_data"]

    assert any("t+3" in f["what"] and not f["available"] for f in flags)


def test_heat_without_a_forecast_is_stated_not_omitted(mock_analyse):
    state = fake_state()
    state["tool_outputs"]["forecast_heat_stress_risk"] = {
        "region": "barmer", "month": "2024-05", "heatwave_days": 7,
        "forecast_available": False, "note": "no usable skill at any horizon"}
    mock_analyse(state)
    flags = client.post("/report", json={"request": "x"}).json()["missing_data"]

    heat = [f for f in flags if "heat stress forecast" in f["what"]]
    assert heat and heat[0]["available"] is False
    assert "no usable skill" in heat[0]["reason"]


def test_unsourced_yield_impact_is_stated_with_its_reason(mock_analyse):
    state = fake_state()
    state["tool_outputs"]["assess_crop_impact"] = {
        "region": "barmer", "crop": "bajra", "dominant_risk": "drought",
        "yield_impact_pct": None,
        "yield_impact_status": "no sourced yield-impact estimate available",
        "yield_impact_reason": "nothing states a loss tied to an SPI-3 band"}
    mock_analyse(state)
    flags = client.post("/report", json={"request": "x"}).json()["missing_data"]

    crop = [f for f in flags if "yield-impact estimate" in f["what"]]
    assert crop and crop[0]["available"] is False
    assert "SPI-3" in crop[0]["reason"]


def test_unavailable_imd_outlook_is_surfaced(mock_analyse):
    mock_analyse(fake_state(type_c={"outlooks": [
        {"title": "IMD Seasonal", "available": False, "reason": "HTTP 504"}]}))
    flags = client.post("/report", json={"request": "x"}).json()["missing_data"]

    assert any("IMD Seasonal" in f["what"] and "504" in f["reason"] for f in flags)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def test_quota_exhaustion_returns_429_with_a_specific_explanation(mock_analyse):
    """A quota failure must not masquerade as a generic 500 or a 200."""
    mock_analyse(fake_state(
        report="",
        grounding={"grounded": False, "unverified_numbers": [],
                   "total_checked": 0, "report_missing": True},
        warnings=["synthesis failed: ResourceExhausted: 429 RESOURCE_EXHAUSTED"]))
    response = client.post("/report", json={"request": "x"})

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert "quota" in detail.lower()
    assert str(quota.DAILY_CALL_BUDGET) in detail


def test_unknown_region_is_a_400_not_a_500(monkeypatch):
    def raiser(**kwargs):
        raise ValueError("Unknown region 'jaisalmer'.")
    monkeypatch.setattr("orchestrator.graph.analyse", raiser)

    response = client.post("/report", json={"request": "x",
                                            "region": "jaisalmer"})
    assert response.status_code == 400
    assert "jaisalmer" in response.json()["detail"]


def test_empty_request_is_rejected():
    assert client.post("/report", json={"request": ""}).status_code == 422


# --------------------------------------------------------------------------- #
# /examples, /evaluation, /quota
# --------------------------------------------------------------------------- #
def test_examples_are_ready_to_post_and_warn_about_cost():
    body = client.get("/examples").json()

    assert len(body["examples"]) >= 5
    for example in body["examples"]:
        assert example["body"]["request"].strip()
        assert example["shows"].strip()
    assert "barmer" in body["supported_regions"]
    assert "bajra" in body["supported_crops"]
    assert str(quota.DAILY_CALL_BUDGET) in body["cost_warning"]


def test_examples_include_an_impossible_request():
    ids = {e["id"] for e in client.get("/examples").json()["examples"]}
    assert "impossible_request" in ids


def test_evaluation_serves_the_scorecard_and_a_summary():
    body = client.get("/evaluation").json()

    assert "Evaluation" in body["markdown"]
    assert body["summary"]["grounding_checker"]["precision"] == 1.0
    assert "known_defect" in body["summary"]["grounding_checker"]


def test_evaluation_markdown_endpoint_returns_plain_text():
    response = client.get("/evaluation.md")
    assert response.status_code == 200
    assert response.text.startswith("# Evaluation")


def test_quota_endpoint_reports_budget_and_remaining():
    body = client.get("/quota").json()

    assert body["daily_call_budget"] == quota.DAILY_CALL_BUDGET
    assert body["calls_remaining_today"] <= body["daily_call_budget"]
    assert "separate quota" in body["note"]


def test_quota_counts_real_calls_at_the_chokepoint():
    """The tally must come from the chat chokepoint, not be an estimate."""
    from orchestrator.graph import CALL_TALLY, _tally_call

    before = quota.calls_used_today()
    _tally_call()
    assert quota.calls_used_today() == before + 1
    CALL_TALLY["generate_content"] -= 1        # leave the tally as we found it


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #
def test_cors_allows_a_local_frontend_on_another_port():
    response = client.options(
        "/report",
        headers={"Origin": "http://localhost:5173",
                 "Access-Control-Request-Method": "POST"})
    assert response.headers.get("access-control-allow-origin") == \
        "http://localhost:5173"


def test_cors_does_not_blanket_allow_the_public_internet():
    response = client.options(
        "/report",
        headers={"Origin": "https://evil.example.com",
                 "Access-Control-Request-Method": "POST"})
    assert response.headers.get("access-control-allow-origin") != \
        "https://evil.example.com"


# --------------------------------------------------------------------------- #
# Live smoke test
# --------------------------------------------------------------------------- #
def _live_enabled() -> bool:
    try:
        from retrieval.embed import get_api_key
        return bool(os.environ.get("RUN_LIVE_ORCHESTRATOR")) and bool(get_api_key())
    except Exception:
        return False


@pytest.mark.skipif(not _live_enabled(),
                    reason="set RUN_LIVE_ORCHESTRATOR=1 for the live smoke test")
def test_live_report_is_well_formed():
    """One real /report through the whole pipeline. Costs 2-4 Gemini calls."""
    response = client.post("/report", json={
        "request": "What is the drought risk for Barmer, and how reliable is it?",
        "region": "barmer"})

    assert response.status_code in (200, 429)
    body = response.json()
    if response.status_code == 429:
        pytest.skip(f"quota exhausted, which the API reported correctly: "
                    f"{body['detail'][:120]}")

    assert body["report"].strip()
    assert body["grounding"]["status"] in ("clean", "warning")
    assert body["grounding"]["total_checked"] > 0
    assert "forecast_drought_risk" in body["tools_called"]
    assert len(body["horizon_confidence"]) == 3
