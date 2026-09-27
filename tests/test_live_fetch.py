"""Report-time live fetches run in parallel under a deadline, with short timeouts.

Before this, IMD's two outlooks, NASA POWER and data.gov.in were fetched one
after another, each with 60-120 s timeouts and three attempts, so one slow host
could hold a report open for minutes (about six, measured, with data.gov.in
timing out). Every test here is offline: the fetchers are stubbed with short
real sleeps so concurrency and the deadline can be measured.
"""

from __future__ import annotations

import time

import pytest

import orchestrator.graph as graph
from retrieval import config, external, outlooks, sources
from retrieval.live import gather_with_deadline

NAP = 0.4


def _slow(value, seconds=NAP):
    def call(*args, **kwargs):
        time.sleep(seconds)
        return value(*args, **kwargs) if callable(value) else value
    return call


# --------------------------------------------------------------------------- #
# The primitive
# --------------------------------------------------------------------------- #
def test_calls_run_concurrently_and_keep_their_order():
    started = time.perf_counter()
    results = gather_with_deadline([_slow("a"), _slow("b"), _slow("c")], 5,
                                   lambda i: "timeout")
    assert results == ["a", "b", "c"]
    assert time.perf_counter() - started < NAP * 2


def test_a_call_past_the_deadline_is_replaced_not_waited_for():
    started = time.perf_counter()
    results = gather_with_deadline([_slow("fast", 0.0), _slow("slow", 3.0)], 0.3,
                                   lambda i: f"timeout-{i}")
    assert results == ["fast", "timeout-1"]
    assert time.perf_counter() - started < 1.5


# --------------------------------------------------------------------------- #
# IMD outlooks
# --------------------------------------------------------------------------- #
@pytest.fixture
def no_outlook_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTLOOK_CACHE_PATH", tmp_path / "outlooks.json")


def test_outlooks_are_fetched_in_parallel(monkeypatch, no_outlook_cache):
    monkeypatch.setattr(outlooks, "fetch_outlook",
                        _slow(lambda src, force=True: {"id": src["id"], "available": True}))
    started = time.perf_counter()
    payload = outlooks.fetch_outlooks()
    assert [o["id"] for o in payload["outlooks"]] == [s["id"] for s in config.TYPE_C_SOURCES]
    assert time.perf_counter() - started < NAP * len(config.TYPE_C_SOURCES)


def test_a_slow_outlook_is_reported_unavailable_with_its_attribution(monkeypatch,
                                                                     no_outlook_cache):
    slow_id = config.TYPE_C_SOURCES[0]["id"]

    def fetch(src, force=True):
        time.sleep(3.0 if src["id"] == slow_id else 0.0)
        return {"id": src["id"], "available": True, "excerpt": "x" * 200}

    monkeypatch.setattr(outlooks, "fetch_outlook", fetch)
    payload = outlooks.fetch_outlooks(deadline_s=0.3)

    slow = next(o for o in payload["outlooks"] if o["id"] == slow_id)
    assert slow["available"] is False
    assert "no response within" in slow["reason"]
    assert slow["publisher"] and slow["citation"]            # still attributed
    assert payload["any_unavailable"] is True


def test_outlook_downloads_use_the_short_live_timeout(monkeypatch):
    seen = {}

    def spy(url, cache_name=None, force=False, timeout=None, attempts=3):
        seen.update(timeout=timeout, attempts=attempts)
        raise RuntimeError("stop here")

    monkeypatch.setattr(outlooks, "fetch_bytes", spy)
    outlooks.fetch_outlook(config.TYPE_C_SOURCES[0])
    assert seen == {"timeout": config.LIVE_FETCH_TIMEOUT,
                    "attempts": config.LIVE_FETCH_ATTEMPTS}


def test_fetch_bytes_does_not_sleep_after_its_last_attempt(monkeypatch):
    sleeps = []
    monkeypatch.setattr(sources.time, "sleep", sleeps.append)
    monkeypatch.setattr(sources.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(RuntimeError):
        sources.fetch_bytes("https://example.invalid/x", attempts=2)
    assert sleeps == [5]


# --------------------------------------------------------------------------- #
# NASA POWER + data.gov.in
# --------------------------------------------------------------------------- #
def test_external_sources_are_fetched_in_parallel(monkeypatch):
    monkeypatch.setattr(external, "fetch_nasa_power",
                        _slow(lambda *a: {"id": "nasa_power", "available": True}))
    monkeypatch.setattr(external, "fetch_mandi_prices",
                        _slow(lambda *a: {"id": "data_gov_in_mandi", "available": True}))
    started = time.perf_counter()
    payload = external.fetch_external_sources("barmer", "bajra")
    assert [s["id"] for s in payload["sources"]] == ["nasa_power", "data_gov_in_mandi"]
    assert time.perf_counter() - started < NAP * 2


def test_a_slow_external_source_is_reported_unavailable(monkeypatch):
    monkeypatch.setattr(external, "fetch_nasa_power",
                        lambda *a: {"id": "nasa_power", "available": True})
    monkeypatch.setattr(external, "fetch_mandi_prices",
                        _slow(lambda *a: {"id": "data_gov_in_mandi", "available": True}, 3.0))
    payload = external.fetch_external_sources("barmer", "bajra", deadline_s=0.3)

    mandi = payload["sources"][1]
    assert mandi["id"] == "data_gov_in_mandi"
    assert mandi["available"] is False and "no response within" in mandi["reason"]
    assert "data.gov.in" in mandi["citation"]
    assert payload["any_unavailable"] is True


def test_mandi_uses_the_short_timeout_and_does_not_sleep_after_its_last_try(monkeypatch):
    calls, sleeps = [], []

    def refused(url, params=None, timeout=None, headers=None):
        calls.append(timeout)
        raise external.requests.ConnectionError("refused")

    monkeypatch.setattr(external, "get_data_gov_key", lambda: "dummy-key")
    monkeypatch.setattr(external.requests, "get", refused)
    monkeypatch.setattr(external.time, "sleep", sleeps.append)

    result = external.fetch_mandi_prices("barmer", "bajra")

    assert result["available"] is False
    assert calls == [config.LIVE_FETCH_TIMEOUT] * config.LIVE_FETCH_ATTEMPTS
    assert len(sleeps) == config.LIVE_FETCH_ATTEMPTS - 1


def test_live_limits_bound_a_report():
    connect, read = config.LIVE_FETCH_TIMEOUT
    assert connect + read <= 30
    assert config.LIVE_FETCH_ATTEMPTS <= 2
    assert config.LIVE_FETCH_DEADLINE_S <= 60


# --------------------------------------------------------------------------- #
# The orchestrator node
# --------------------------------------------------------------------------- #
def test_type_c_node_runs_imd_and_external_side_by_side(monkeypatch):
    monkeypatch.setattr(graph, "fetch_outlooks",
                        _slow({"outlooks": [], "any_unavailable": False}))
    monkeypatch.setattr(graph, "fetch_external_sources",
                        _slow({"sources": [], "any_unavailable": False}))
    started = time.perf_counter()
    out = graph.fetch_type_c({"request": "x", "warnings": []})
    assert time.perf_counter() - started < NAP * 2
    assert out["type_c"]["outlooks"] == [] and out["external"]["sources"] == []


def test_type_c_node_reports_a_crashed_fetch_instead_of_raising(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(graph, "fetch_outlooks", lambda: {"outlooks": [],
                                                          "any_unavailable": False})
    monkeypatch.setattr(graph, "fetch_external_sources", boom)
    out = graph.fetch_type_c({"request": "x", "warnings": []})
    assert "external source fetch failed entirely" in out["warnings"]
    assert out["external"]["sources"] == []
