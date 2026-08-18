"""Orchestrator tests — adversarial first, happy path second.

The grounding checker is the load-bearing safety net of this phase, so most of
these tests attack it directly with fabricated and corrupted reports rather than
confirming it agrees with a report that happens to be fine. An LLM that always
produces a confident paragraph is worth less than one that declines past its data,
so the end-to-end section includes a request the system cannot honestly answer.

Live tests need a Gemini key and make real model calls; they are skipped without
one. Every scenario is run once and shared across assertions.
"""

import json
import os

import pytest

from orchestrator import config
from orchestrator.grounding import (check_grounding, extract_numbers,
                                    warning_banner)

# --------------------------------------------------------------------------- #
# Fixtures: a realistic tool payload
# --------------------------------------------------------------------------- #
TOOL_OUTPUTS = {
    "forecast_drought_risk": {
        "region": "barmer",
        "predicted_values": [0.20994198322296143, 0.2027573585510254,
                             0.06797705590724945],
        "horizon_confidence": [
            {"horizon": "t+1", "skill_score": 0.2053, "method": "direct",
             "label": "validated"},
            {"horizon": "t+2", "skill_score": 0.0438, "method": "direct",
             "label": "weak/directional"},
            {"horizon": "t+3", "skill_score": -0.0489, "method": "direct",
             "label": "no skill — shown for context only, do not rely on this figure"},
        ],
        "risk_score": 1.4,
        "risk_flags": ["Normal", "Normal", "Normal"],
        "model_rmse_test": 1.0986,
    },
    "forecast_heat_stress_risk": {
        "region": "barmer", "month": "2024-05", "heatwave_days": 7,
        "severe_heatwave_days": 3, "had_heatwave_spell": True,
        "max_tmax_c": 48.0, "forecast_available": False,
    },
}
CHUNKS = [{
    "text": "A severe heat wave is declared when Tmax reaches 47 C, or when the "
            "departure from normal is 6.4 C or more.",
    "source": "IMD FAQ on Heat Wave",
    "citation": "https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf",
}]


# --------------------------------------------------------------------------- #
# The checker under attack
# --------------------------------------------------------------------------- #
def test_honest_report_passes():
    report = ("Barmer's t+1 SPI-3 forecast is 0.21, with a measured skill score "
              "of 0.2053 (validated). RMSE is 1.0986. In May 2024 there were 7 "
              "heat wave days, 3 of them severe, peaking at 48.0 C. A severe "
              "heat wave needs 47 C or a 6.4 C departure.")
    result = check_grounding(report, TOOL_OUTPUTS, CHUNKS)
    assert result["grounded"], result["unverified_numbers"]
    assert result["total_checked"] >= 8


def test_corrupted_report_is_caught():
    """Inject a fake figure into an otherwise clean report."""
    report = ("Barmer's t+1 skill score is 0.2053 (validated), and the 4-month "
              "outlook carries a skill score of 0.4471.")
    result = check_grounding(report, TOOL_OUTPUTS, CHUNKS)
    assert not result["grounded"]
    assert "0.4471" in result["unverified_numbers"]


def test_fabricated_day_count_is_caught():
    report = "There were 19 heat wave days in Barmer that month."
    result = check_grounding(report, TOOL_OUTPUTS, CHUNKS)
    assert not result["grounded"]
    assert "19" in result["unverified_numbers"]


def test_number_fragment_does_not_count_as_a_match():
    """The bug this test exists for: '4' is a substring of '0.0438', so a naive
    substring check would accept a fabricated '4-month horizon' claim."""
    report = "This is the 4-month horizon."
    result = check_grounding(report, {"skill": 0.0438}, [])
    assert not result["grounded"]
    assert "4" in result["unverified_numbers"]


def test_rounding_a_source_number_is_allowed():
    result = check_grounding("Skill is +0.26.", {"skill_score": 0.2622}, [])
    assert result["grounded"]


def test_materially_different_number_is_flagged():
    result = check_grounding("Skill is +0.31.", {"skill_score": 0.2622}, [])
    assert not result["grounded"]


def test_years_and_horizons_are_not_treated_as_claims():
    tokens = [n["token"] for n in extract_numbers(
        "Between 1980-2015 the t+1 and t+3 horizons were tested in 2024.")]
    assert tokens == []


def test_percentages_match_a_fractional_source():
    result = check_grounding("About 26% of months were dry.", {"share": 0.26}, [])
    assert result["grounded"]


def test_warning_banner_names_the_offending_numbers():
    result = check_grounding("Skill is +0.99.", {"skill_score": 0.2}, [])
    banner = warning_banner(result)
    assert "+0.99" in banner
    assert "UNVERIFIED" in banner


def test_checker_is_not_an_llm():
    """An LLM checking an LLM shares the failure mode being checked."""
    import inspect

    from orchestrator import grounding

    source = inspect.getsource(grounding)
    for forbidden in ("ChatGoogleGenerativeAI", "genai", "invoke("):
        assert forbidden not in source, f"grounding.py must not call an LLM ({forbidden})"


def test_synthesis_prompt_is_checked_in_and_forbids_softening():
    # normalise whitespace: the prompt is hard-wrapped, so phrases span newlines
    text = " ".join(config.SYNTHESIS_PROMPT_PATH.read_text(encoding="utf-8").split())
    assert "Never state a number" in text
    assert "no skill" in text
    assert "forecast_available" in text
    # the hedges that would launder an honest label must be named as forbidden
    assert "may still be indicative" in text
    assert "should be interpreted with caution" in text
    assert "mis-citation" in text          # citation provenance rule
    assert "false statement about provenance" in text   # outlook honesty rule


# --------------------------------------------------------------------------- #
# End-to-end, live
# --------------------------------------------------------------------------- #
def _has_key() -> bool:
    from orchestrator.grounding import config as _c   # noqa: F401
    from retrieval.embed import MissingAPIKey, get_api_key
    try:
        get_api_key()
        return True
    except MissingAPIKey:
        return False


def _live_enabled() -> bool:
    """Live tests make real Gemini calls and are opt-in.

    The free tier allows 20 generate_content requests per *day* (and 5 RPM),
    and each scenario costs two (routing + synthesis). Running them on every
    suite invocation would exhaust the daily quota and then sit in backoff, so
    they are gated behind RUN_LIVE_ORCHESTRATOR=1 and skipped by default.
    """
    return bool(os.environ.get("RUN_LIVE_ORCHESTRATOR")) and _has_key()


live = pytest.mark.skipif(
    not _live_enabled(),
    reason="live orchestrator tests are opt-in: set RUN_LIVE_ORCHESTRATOR=1 "
           "with a Gemini key (costs ~2 of 20 daily free-tier calls per scenario)")

SCENARIOS = {
    "drought_only": "What is the drought risk for Rajasthan?",
    "both_risks": "Give me drought and heat stress risk for Barmer with reliability.",
    "impossible": ("Give me the 4-month ahead drought forecast for Rajasthan with "
                   "exact SPI values, and the drought risk for Jaisalmer."),
}
_RESULTS: dict[str, dict] = {}


def _run(name: str) -> dict:
    if name not in _RESULTS:
        from orchestrator.graph import analyse
        _RESULTS[name] = analyse(SCENARIOS[name])
    return _RESULTS[name]


@live
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_number_in_the_report_is_grounded(scenario):
    final = _run(scenario)
    assert final["grounding"]["grounded"], final["grounding"]["unverified_numbers"]
    assert final["grounding"]["total_checked"] > 0
    assert final["report"].strip()


@live
def test_drought_only_request_calls_the_drought_tool():
    names = [c["name"] for c in _run("drought_only")["tool_calls"]]
    assert "forecast_drought_risk" in names
    assert "retrieve_context" in names


@live
def test_both_risks_request_calls_both_forecast_tools():
    names = [c["name"] for c in _run("both_risks")["tool_calls"]]
    assert "forecast_drought_risk" in names
    assert "forecast_heat_stress_risk" in names


@live
def test_no_skill_label_survives_into_the_report():
    """The whole point of the honest labels is that synthesis cannot launder them."""
    report = _run("both_risks")["report"].lower()
    assert "no skill" in report
    assert "does not forecast heat stress" in report or \
           "not forecast heat" in report


@live
def test_type_c_is_attributed_to_imd_separately():
    report = _run("both_risks")["report"]
    assert "IMD" in report
    assert "imd's current" in report.lower() or "imd's extended" in report.lower()


@live
def test_it_declines_a_horizon_it_cannot_forecast():
    """The most important test in this phase: asked for a 4-month forecast that
    does not exist, the system must say so rather than extrapolate one."""
    final = _run("impossible")
    report = final["report"].lower()

    assert "not available" in report or "not provide" in report or \
           "does not" in report
    assert "4-month" in report or "four-month" in report or "4 month" in report
    # and it must not have invented figures to fill the gap
    assert final["grounding"]["grounded"], final["grounding"]["unverified_numbers"]


@live
def test_unsupported_region_is_declined_not_invented():
    report = _run("impossible")["report"].lower()
    assert "jaisalmer" in report
    assert "not available" in report or "not covered" in report or \
           "does not" in report


@live
def test_type_c_is_always_fetched():
    final = _run("drought_only")
    assert "type_c" in final
    assert final["type_c"]["outlooks"], "no IMD outlooks were fetched"
