"""Phase 4 — Crop Impact Agent.

Adversarial first, happy path second. The majority of this file is offline and
free: the deterministic pieces are where the correctness properties live, so they
are where the coverage is.

Live tests make real Gemini calls and sit behind the same
``RUN_LIVE_ORCHESTRATOR=1`` gate as ``tests/test_orchestrator.py`` — deliberately
not a second env var.
"""

from __future__ import annotations

import inspect
import json
import os

import pytest

from crop_impact import config, dominance, yield_impact
from crop_impact.dominance import dominant_risk, heat_signal
from crop_impact.tool import assess_crop_impact_core, resolve_target_month
from crop_impact.yield_impact import NOT_AVAILABLE, lookup_yield_impact
from forecasting import config as fconfig

VALIDATED = "validated"
WEAK = "weak/directional"
NO_SKILL = "no skill — shown for context only, do not rely on this figure"


# --------------------------------------------------------------------------- #
# Fixtures standing in for the two tool outputs
# --------------------------------------------------------------------------- #
def drought_out(values, labels=(VALIDATED, WEAK, NO_SKILL)) -> dict:
    return {
        "region": "rajasthan",
        "predicted_values": list(values),
        "horizon_confidence": [
            {"horizon": f"t+{i}", "skill_score": s, "method": "direct",
             "label": lab}
            for i, (s, lab) in enumerate(zip((0.2622, 0.0766, -0.0145), labels), 1)
        ],
        "risk_score": 5.0,
        "risk_flags": ["Severe", "Normal", "Normal"],
        "model_rmse_test": 0.8918,
    }


def heat_out(days=0, severe=0, month="2024-03", departure=None) -> dict:
    out = {
        "region": "rajasthan", "month": month,
        "heatwave_days": days, "severe_heatwave_days": severe,
        "had_heatwave_spell": days >= 2, "max_tmax_c": 41.2,
        "forecast_available": False, "note": "observations only",
    }
    if departure is not None:
        out["mean_tmax_departure_c"] = departure
    return out


# --------------------------------------------------------------------------- #
# The property that matters most: a no-skill horizon decides nothing
# --------------------------------------------------------------------------- #
def test_t3_no_skill_can_never_declare_drought_dominant():
    """The single most important correctness property of this phase.

    A catastrophic-looking SPI-3 on the t+3 horizon must not drive a crop
    assessment: five phases of evidence say that horizon has no measured skill,
    and letting it decide would launder a worthless number into a finding.
    """
    decision = dominant_risk(
        drought=drought_out([0.1, 0.0, -2.5]),
        heat=heat_out(days=0, month="2024-08", departure=0.2),
        crop="bajra", horizon=3, target_month="2024-08", month_number=8)

    assert decision["dominant_risk"] == "none dominant"
    assert decision["signals"]["drought"]["no_skill"] is True
    assert decision["signals"]["drought"]["severity"] == "severe"   # it IS severe
    assert "not allowed to declare dominance" in decision["reason"]


def test_weak_directional_horizon_cannot_decide_either():
    decision = dominant_risk(
        drought=drought_out([0.0, -2.0, 0.0]), heat=None,
        crop="bajra", horizon=2, month_number=8)

    assert decision["dominant_risk"] == "none dominant"
    assert decision["signals"]["drought"]["corroborating"] is True
    assert "may corroborate but may not decide" in decision["reason"]


def test_validated_horizon_may_declare_drought_dominant():
    decision = dominant_risk(
        drought=drought_out([-1.8, 0.0, 0.0]), heat=None,
        crop="bajra", horizon=1, month_number=8)

    assert decision["dominant_risk"] == "drought"
    assert decision["severity"] == "severe"
    assert decision["confidence_label"] == VALIDATED


# --------------------------------------------------------------------------- #
# dominant_risk() — the four cases the phase spec asks for
# --------------------------------------------------------------------------- #
def test_heat_clearly_dominant_over_mild_drought():
    decision = dominant_risk(
        drought=drought_out([-0.3, 0.0, 0.0]),
        heat=heat_out(days=6, severe=2, month="2024-03"),
        crop="wheat", horizon=1, target_month="2024-03", month_number=3)

    assert decision["dominant_risk"] == "heat"
    assert decision["severity"] == "severe"
    assert "observation, not a forecast" in decision["reason"]


def test_neither_dominant_when_both_signals_are_mild():
    decision = dominant_risk(
        drought=drought_out([-0.2, 0.0, 0.0]),
        heat=heat_out(days=0, month="2024-03", departure=0.4),
        crop="wheat", horizon=1, target_month="2024-03", month_number=3)

    assert decision["dominant_risk"] == "none dominant"
    assert decision["severity"] is None


def test_insufficient_data_when_only_a_no_skill_horizon_and_no_heat():
    decision = dominant_risk(
        drought=drought_out([0.0, 0.0, -1.9]), heat=None,
        crop="bajra", horizon=3, month_number=8)

    assert decision["dominant_risk"] == "insufficient data"
    assert "does not allow to drive a decision" in decision["reason"]


def test_tie_breaks_toward_the_observed_signal():
    """Equal severity: the observation wins, and the reasoning says so."""
    decision = dominant_risk(
        drought=drought_out([-1.2, 0.0, 0.0]),                 # moderate
        heat=heat_out(days=4, severe=0, month="2024-03"),      # moderate
        crop="wheat", horizon=1, target_month="2024-03", month_number=3)

    assert decision["dominant_risk"] == "heat"
    assert "heat is an observation and the drought figure is a forecast" \
        in decision["reason"]


def test_more_severe_drought_beats_moderate_heat():
    decision = dominant_risk(
        drought=drought_out([-1.9, 0.0, 0.0]),                 # severe
        heat=heat_out(days=4, severe=0, month="2024-03"),      # moderate
        crop="wheat", horizon=1, target_month="2024-03", month_number=3)

    assert decision["dominant_risk"] == "drought"


# --------------------------------------------------------------------------- #
# Heat is an observation, never a forecast
# --------------------------------------------------------------------------- #
def test_heat_observation_for_another_month_is_unknown_not_reused():
    signal = heat_signal(heat_out(days=9, severe=3, month="2024-05"),
                         target_month="2025-02")

    assert signal["available"] is False
    assert "unknown" in signal["reason"]
    assert "no heat forecast" in signal["reason"]


def test_heat_can_never_be_dominant_for_a_future_month():
    """There is no heat forecast, so a future month can never return heat."""
    decision = dominant_risk(
        drought=drought_out([-0.2, 0.0, 0.0]),
        heat=heat_out(days=9, severe=3, month="2024-12"),
        crop="wheat", horizon=2, target_month="2025-02", month_number=2)

    assert decision["dominant_risk"] != "heat"


# --------------------------------------------------------------------------- #
# Season gate
# --------------------------------------------------------------------------- #
def test_severe_drought_outside_the_sensitive_window_binds_nothing():
    decision = dominant_risk(
        drought=drought_out([-2.4, 0.0, 0.0]), heat=None,
        crop="bajra", horizon=1, month_number=1)          # January, no bajra

    assert decision["dominant_risk"] == "none dominant"
    assert "outside" in decision["reason"]


# --------------------------------------------------------------------------- #
# The departure route — the mismatch that measurement exposed
# --------------------------------------------------------------------------- #
def test_warm_february_is_seen_even_with_zero_imd_heat_wave_days():
    """IMD's >=40 C gate cannot see terminal heat in Feb/Mar; the departure can."""
    signal = heat_signal(heat_out(days=0, month="2006-02", departure=5.35),
                         target_month="2006-02")

    assert signal["severity"] == "severe"
    assert signal["severity_route"] == "monthly mean Tmax departure"


def test_departure_below_moderate_threshold_is_not_a_candidate():
    signal = heat_signal(heat_out(days=0, month="2024-03", departure=2.1),
                         target_month="2024-03")
    assert signal["severity"] is None


# --------------------------------------------------------------------------- #
# Yield-impact lookup — never invents, never scales
# --------------------------------------------------------------------------- #
def test_unsourced_combination_declines_rather_than_inventing():
    """bajra/drought has no published coefficient — it must say so."""
    result = lookup_yield_impact("bajra", "drought", "severe",
                                 {"mean_tmax_departure_c": None})

    assert result["available"] is False
    assert result["yield_impact_pct"] is None
    assert result["status"] == NOT_AVAILABLE
    assert "SPI-3" in result["reason"]


def test_sourced_combination_returns_the_table_value_with_its_caveat():
    result = lookup_yield_impact("wheat", "heat", "severe",
                                 {"mean_tmax_departure_c": 5.35})

    assert result["available"] is True
    assert result["yield_impact_pct"] == 5.6
    assert result["citation"].startswith("http")
    assert "must NOT be scaled per degree" in result["caveat"]


def test_coefficient_is_not_applied_outside_its_exposure_band():
    """3.68 C is a real warm anomaly, but below the band the 5.6% was measured in."""
    result = lookup_yield_impact("wheat", "heat", "severe",
                                 {"mean_tmax_departure_c": 3.68})

    assert result["available"] is False
    assert result["yield_impact_pct"] is None
    assert "below the coefficient's band" in result["reason"]


def test_coefficient_is_not_reused_across_severity_bands():
    result = lookup_yield_impact("wheat", "heat", "moderate",
                                 {"mean_tmax_departure_c": 5.0})

    assert result["yield_impact_pct"] is None
    assert "scaled down from a different severity band" in result["reason"]


def test_no_dominant_risk_means_no_lookup_at_all():
    result = lookup_yield_impact("bajra", "none dominant", None, {})
    assert result["yield_impact_pct"] is None


# --------------------------------------------------------------------------- #
# The table itself is inspectable and honest
# --------------------------------------------------------------------------- #
def test_every_coefficient_carries_a_citation_and_a_quote():
    table = yield_impact.load_table()
    assert table["coefficients"], "the table must not be empty"
    for entry in table["coefficients"]:
        assert entry["citation"].startswith("http")
        assert entry["quoted"].strip()
        assert entry["caveat"].strip()
        # the quoted sentence must actually contain the number being claimed
        assert str(entry["yield_impact_pct"]).replace(".", ".") in entry["quoted"] \
            or "5.6 per cent" in entry["quoted"]


def test_rejected_sources_are_recorded_rather_than_silently_dropped():
    """Two in-corpus documents were checked and found unusable — that is evidence."""
    table = yield_impact.load_table()
    rejected = {r["source"] for r in table["sources_checked_and_rejected"]}

    assert any("ICAR-ATARI" in r for r in rejected)
    assert any("NDMA" in r for r in rejected)
    for entry in table["sources_checked_and_rejected"]:
        assert entry["usable"] is False
        assert entry["finding"].strip()


def test_every_gap_says_what_was_searched():
    for gap in yield_impact.load_table()["gaps"]:
        assert gap["status"] == NOT_AVAILABLE
        assert gap["searched"].strip()
        assert gap["why_nothing_qualified"].strip()


# --------------------------------------------------------------------------- #
# The deterministic core contains no model call
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", [dominance, yield_impact])
def test_deterministic_modules_are_not_llms(module):
    """Same guard as Phase 3's test_checker_is_not_an_llm.

    The decisions that must be trustworthy are plain Python. If a model call ever
    appears in these modules, that is the property gone.
    """
    source = inspect.getsource(module)
    for banned in ("ChatGoogleGenerativeAI", "get_chat_model", "invoke_with_backoff",
                   "generate_content", "langchain_google_genai", ".invoke("):
        assert banned not in source, f"{module.__name__} must not call an LLM"


def test_dominance_rule_document_matches_the_code():
    """The checked-in rule and config must not drift apart."""
    text = config.DOMINANCE_RULE_PATH.read_text(encoding="utf-8")

    assert str(fconfig.SPI_MODERATE) in text
    assert str(fconfig.SPI_SEVERE) in text
    assert str(config.HEAT_CANDIDATE_MIN_DAYS) in text
    assert f"+{config.HEAT_DEPARTURE_SEVERE_C}" in text
    assert f"+{config.HEAT_DEPARTURE_MODERATE_C}" in text
    assert config.DECISIVE_FORECAST_LABEL in text


# --------------------------------------------------------------------------- #
# The tool, offline (no narrative -> no Gemini call, so these are free)
# --------------------------------------------------------------------------- #
def test_out_of_scope_crop_region_combination_is_declined():
    result = assess_crop_impact_core("barmer", "wheat", None,
                                     with_narrative=False)

    assert result["dominant_risk"] == "insufficient data"
    assert result["yield_impact_pct"] is None
    assert "not in scope" in result["risk_reasoning"]


def test_past_month_gets_no_drought_forecast_and_says_why():
    result = assess_crop_impact_core("rajasthan", "wheat", "2024-03",
                                     with_narrative=False)

    drought = result["signals"]["drought"]
    assert drought["available"] is False
    assert "already happened" in drought["reason"]


def test_record_warm_february_produces_the_sourced_estimate():
    """End-to-end through the real pipeline, no model call: Jaipur, Feb 2006."""
    result = assess_crop_impact_core("rajasthan", "wheat", "2006-02",
                                     with_narrative=False)

    assert result["dominant_risk"] == "heat"
    assert result["yield_impact_pct"] == 5.6
    assert result["yield_impact_citation"].startswith("http")
    assert result["confidence_label"].startswith("observed")


def test_default_month_lands_in_the_crops_sensitive_window():
    for crop, spec in config.CROPS.items():
        region = spec["regions"][0]
        target, _, _ = resolve_target_month(region, crop, None)
        assert target.month in spec["sensitive_months"]


def test_unknown_crop_is_rejected():
    with pytest.raises(ValueError, match="Unknown crop"):
        config.check_crop("mangoes")


# --------------------------------------------------------------------------- #
# Orchestrator wiring
# --------------------------------------------------------------------------- #
def test_crop_tool_is_routable_from_the_orchestrator():
    from orchestrator.graph import TOOLS_BY_NAME
    assert "assess_crop_impact" in TOOLS_BY_NAME


def test_synthesis_prompt_forbids_inventing_a_yield_percentage():
    from orchestrator import config as oconfig
    text = oconfig.SYNTHESIS_PROMPT_PATH.read_text(encoding="utf-8")
    assert "yield-impact percentage" in text
    assert "yield_impact_pct: null" in text


# --------------------------------------------------------------------------- #
# Live tests — real Gemini calls, opt-in
# --------------------------------------------------------------------------- #
def _live_enabled() -> bool:
    """Live crop-impact tests make real Gemini calls and are opt-in.

    Quota cost, against the corrected free-tier budget of 5 RPM / 20 RPD
    (Phase 3.1): each ``assess_crop_impact`` call costs **one** generate_content
    request (the narrative; there is no retry loop inside the tool). The two live
    tests below therefore cost 2 requests. A full orchestrator report that routes
    to this tool costs up to 3 in the best case — routing, crop narrative, main
    synthesis — and up to 5 if the main synthesis needs its one regeneration.

    Gated behind the same RUN_LIVE_ORCHESTRATOR=1 as tests/test_orchestrator.py,
    deliberately not a second variable.
    """
    from retrieval.embed import get_api_key
    try:
        return bool(os.environ.get("RUN_LIVE_ORCHESTRATOR")) and bool(get_api_key())
    except Exception:
        return False


live = pytest.mark.skipif(not _live_enabled(),
                          reason="set RUN_LIVE_ORCHESTRATOR=1 for live Gemini tests")


@pytest.fixture(scope="session")
def live_unsourced() -> dict:
    """One cached run of a combination with no sourced yield data."""
    return assess_crop_impact_core("rajasthan", "bajra", "2024-08",
                                   with_narrative=True)


@pytest.fixture(scope="session")
def live_sourced() -> dict:
    """One cached run of the combination that does have a sourced coefficient."""
    return assess_crop_impact_core("rajasthan", "wheat", "2006-02",
                                   with_narrative=True)


@live
def test_live_unsourced_combination_is_declined_not_invented(live_unsourced):
    """This phase's version of Phase 3's "ask it something it cannot know".

    The model is handed a null yield impact and must say so rather than produce a
    plausible percentage from general agronomic knowledge.
    """
    text = live_unsourced["narrative"]
    assert text.strip(), "narrative was not generated"
    assert live_unsourced["yield_impact_pct"] is None
    assert live_unsourced["grounding"]["grounded"], \
        f"ungrounded: {live_unsourced['grounding'].get('unverified_numbers')}"
    assert "%" not in text or "no sourced" in text.lower()


@live
def test_live_narrative_percentages_trace_to_the_table(live_sourced):
    text = live_sourced["narrative"]
    assert text.strip(), "narrative was not generated"
    assert "5.6" in text, "the sourced figure should be reported"
    assert live_sourced["grounding"]["grounded"], \
        f"ungrounded: {live_sourced['grounding'].get('unverified_numbers')}"
    assert live_sourced["grounding"]["total_checked"] > 0, \
        "a pass over zero numbers is a vacuous pass"
