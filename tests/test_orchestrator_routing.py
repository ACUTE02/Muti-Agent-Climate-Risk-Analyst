"""Routing and quota behaviour of the orchestrator, fully offline.

The model, the tools and the live fetches are all stubbed, so these tests make
no Gemini call and no network request. They cover two audit findings:

* explicit caller fields (region, crop, month, risk_types) must reach the tools
  even when the model leaves them out, so a forecast is never reported under a
  region it was not computed for;
* a daily-quota failure must fail fast instead of holding the request open for
  minutes of pointless backoff.
"""

from __future__ import annotations

import time

import pytest

import orchestrator.graph as graph


class StubTool:
    def __init__(self, name: str, args: tuple[str, ...], seen: list):
        self.name = name
        self.args = {a: {} for a in args}
        self._seen = seen

    def invoke(self, args):
        self._seen.append((self.name, dict(args)))
        if self.name == "retrieve_context":
            return []
        return {"region": args.get("region", "rajasthan")}


TOOL_ARGS = {
    "forecast_drought_risk": ("region",),
    "forecast_heat_stress_risk": ("region", "month"),
    "assess_crop_impact": ("region", "crop", "month"),
    "retrieve_context": ("query", "k", "doc_type"),
}


@pytest.fixture
def seen(monkeypatch):
    calls: list = []
    monkeypatch.setattr(graph, "TOOLS_BY_NAME",
                        {n: StubTool(n, a, calls) for n, a in TOOL_ARGS.items()})
    return calls


class StubModel:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"tool_calls": self.tool_calls})()


# --------------------------------------------------------------------------- #
# Caller fields win, even when the model omits them
# --------------------------------------------------------------------------- #
def test_forced_fields_reach_tools_the_model_called_without_them(seen):
    state = {"request": "x", "region": "barmer", "crop": "wheat",
             "month": "2006-02", "warnings": [],
             "tool_calls": [
                 {"name": "forecast_drought_risk", "args": {}},
                 {"name": "forecast_heat_stress_risk", "args": {"region": "rajasthan"}},
                 {"name": "assess_crop_impact", "args": {"region": "barmer", "crop": "bajra"}},
                 {"name": "retrieve_context", "args": {"query": "q"}},
             ]}

    out = graph.call_tools(state)

    by_name = dict(seen)
    assert by_name["forecast_drought_risk"] == {"region": "barmer"}
    assert by_name["forecast_heat_stress_risk"] == {"region": "barmer", "month": "2006-02"}
    assert by_name["assess_crop_impact"] == {"region": "barmer", "crop": "wheat",
                                              "month": "2006-02"}
    assert by_name["retrieve_context"] == {"query": "q"}   # accepts none of them
    assert out["region"] == "barmer"


def test_without_caller_fields_the_models_args_are_untouched(seen):
    state = {"request": "x", "warnings": [],
             "tool_calls": [{"name": "forecast_heat_stress_risk",
                             "args": {"region": "rajasthan", "month": "2024-05"}}]}
    graph.call_tools(state)
    assert seen == [("forecast_heat_stress_risk",
                     {"region": "rajasthan", "month": "2024-05"})]


def test_router_is_told_the_caller_constraints(monkeypatch):
    model = StubModel([{"name": "forecast_drought_risk", "args": {}, "id": "1"}])
    monkeypatch.setattr(graph, "get_chat_model", lambda tools=False: model)

    graph.parse_request({"request": "risk?", "region": "barmer", "crop": "wheat",
                         "month": "2006-02", "risk_types": ["heat_stress"],
                         "warnings": []})

    human = model.messages[1][1]
    assert human.startswith("risk?")
    for fragment in ("region=barmer", "crop=wheat", "month=2006-02",
                     "risk_types=heat_stress"):
        assert fragment in human


def test_router_message_is_unchanged_without_constraints(monkeypatch):
    model = StubModel([{"name": "forecast_drought_risk", "args": {}, "id": "1"}])
    monkeypatch.setattr(graph, "get_chat_model", lambda tools=False: model)
    graph.parse_request({"request": "risk?", "warnings": []})
    assert model.messages[1][1] == "risk?"


def test_tools_demanded_by_risk_types_and_crop_are_added_if_skipped(monkeypatch):
    model = StubModel([{"name": "forecast_drought_risk", "args": {}, "id": "1"}])
    monkeypatch.setattr(graph, "get_chat_model", lambda tools=False: model)

    out = graph.parse_request({"request": "x", "region": "barmer", "crop": "wheat",
                               "risk_types": ["drought", "heat_stress"],
                               "warnings": []})

    names = [c["name"] for c in out["tool_calls"]]
    assert names.count("forecast_drought_risk") == 1        # not duplicated
    assert "forecast_heat_stress_risk" in names
    assert "assess_crop_impact" in names


# --------------------------------------------------------------------------- #
# Quota: a daily cap fails fast
# --------------------------------------------------------------------------- #
DAILY = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric generate_content_free_"
         "tier_requests, limit: 20. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
PER_MINUTE = ("429 RESOURCE_EXHAUSTED. quotaId: "
              "GenerateRequestsPerMinutePerProjectPerModel-FreeTier")


class RaisingModel:
    def __init__(self, message):
        self.message = message
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise RuntimeError(self.message)


@pytest.fixture
def sleeps(monkeypatch):
    recorded: list = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def test_daily_quota_error_is_not_retried(sleeps):
    model = RaisingModel(DAILY)
    with pytest.raises(RuntimeError):
        graph.invoke_with_backoff(model, [])
    assert model.calls == 1
    assert sleeps == []


def test_per_minute_quota_error_is_still_backed_off(sleeps):
    model = RaisingModel(PER_MINUTE)
    with pytest.raises(RuntimeError):
        graph.invoke_with_backoff(model, [])
    assert model.calls == 3
    assert sleeps == [30, 60]


def test_non_quota_error_is_raised_immediately(sleeps):
    model = RaisingModel("500 internal error")
    with pytest.raises(RuntimeError):
        graph.invoke_with_backoff(model, [])
    assert model.calls == 1 and sleeps == []


@pytest.mark.parametrize("message,max_calls,max_sleep", [
    (DAILY, 2, 0),              # routing + one synthesis, no waiting at all
    (PER_MINUTE, 6, 180),       # each call backs off once, no synthesis regeneration
])
def test_exhausted_quota_does_not_hold_a_report_open(monkeypatch, seen, sleeps,
                                                     message, max_calls, max_sleep):
    model = RaisingModel(message)
    monkeypatch.setattr(graph, "get_chat_model", lambda tools=False: model)
    monkeypatch.setattr(graph, "fetch_outlooks",
                        lambda: {"outlooks": [], "any_unavailable": False})
    monkeypatch.setattr(graph, "fetch_external_sources",
                        lambda **k: {"sources": [], "any_unavailable": False})

    state = graph.analyse("drought risk for Barmer")

    assert state["grounding"]["report_missing"] is True
    assert model.calls <= max_calls
    assert sum(sleeps) <= max_sleep
    assert any("synthesis failed" in w for w in state["warnings"])


def test_non_quota_synthesis_failure_still_gets_its_one_regeneration(monkeypatch,
                                                                     seen, sleeps):
    model = RaisingModel("503 backend unavailable")
    monkeypatch.setattr(graph, "get_chat_model", lambda tools=False: model)
    monkeypatch.setattr(graph, "fetch_outlooks",
                        lambda: {"outlooks": [], "any_unavailable": False})
    monkeypatch.setattr(graph, "fetch_external_sources",
                        lambda **k: {"sources": [], "any_unavailable": False})

    graph.analyse("drought risk for Barmer")

    # routing (1) + synthesis (1) + its one regeneration (1)
    assert model.calls == 3
