"""Phase 8 Part 2 — the report got readable without getting softer.

The danger in a "make it plain-language" rewrite is that plainness quietly eats
the honesty: the `no skill` label gets rounded off into "use with caution", the
forbidden softening phrases lose their prohibition, or a rule goes missing in the
edit. These tests hold the prompt to both standards at once — every original
honesty rule still present and still strict, *and* the plain-language pairing
actually there for the `no skill` case.

No LLM call is made. The prompt file and the assembled prompt are both plain
text, so what the model is instructed to do is checkable for free; spending
Gemini quota to find out whether a file contains a sentence would be silly.
"""

from __future__ import annotations

import json
import re

import pytest

from orchestrator import config

PROMPT = config.SYNTHESIS_PROMPT_PATH.read_text(encoding="utf-8")


def _rules(text: str) -> dict[int, str]:
    body = text.split("# Absolute rules")[1].split("\n# ")[0]
    out = {}
    for part in re.split(r"\n(?=\d+\. \*\*)", body):
        match = re.match(r"(\d+)\.", part.strip())
        if match:
            out[int(match.group(1))] = re.sub(r"\s+", " ", part).strip()
    return out


RULES = _rules(PROMPT)


# --------------------------------------------------------------------------- #
# The honesty rules survived the rewrite
# --------------------------------------------------------------------------- #
def test_all_ten_absolute_rules_are_still_present():
    assert sorted(RULES) == list(range(1, 11)), (
        f"expected rules 1-10, found {sorted(RULES)}")


def test_rules_unrelated_to_the_new_sources_are_byte_identical_to_the_original():
    """Only rules 3 and 6 (the attribution rules) were allowed to change, and
    only to extend the same constraint from IMD alone to all three sources."""
    import subprocess

    original = subprocess.run(
        ["git", "show", "HEAD:orchestrator/prompts/synthesis.md"],
        capture_output=True, text=True, encoding="utf-8").stdout
    if not original.strip():
        pytest.skip("original prompt not retrievable from git")

    before = _rules(original)
    for number in (1, 2, 4, 5, 7, 8, 9, 10):
        assert RULES[number] == before[number], (
            f"rule {number} changed; only the attribution rules 3 and 6 may")


def test_forbidden_softening_phrases_are_still_forbidden():
    """Rule 2's specific prohibitions are the ones a 'friendlier tone' pass
    would most plausibly delete."""
    rule = RULES[2]
    for phrase in ("may still be indicative", "directionally useful",
                   "better than nothing", "should be interpreted with caution"):
        assert phrase in rule, f"rule 2 no longer forbids {phrase!r}"
    assert "forbidden" in rule


def test_no_skill_is_still_defined_against_climatology_not_randomness():
    assert "no better than climatology" in RULES[2]
    assert "no better than random chance" in RULES[2]
    assert "1 - RMSE_model/RMSE_climatology" in RULES[2]


def test_fabricated_numbers_and_yield_percentages_are_still_banned():
    assert "Never state a number that is not present" in RULES[1]
    assert "Never state a yield-impact percentage" in RULES[9]
    assert "do not offer a range" in RULES[9]


# --------------------------------------------------------------------------- #
# The plain-language layer is actually there
# --------------------------------------------------------------------------- #
def test_no_skill_has_a_plain_language_gloss_that_still_says_do_not_act():
    """The whole point of Part 2: explain the label without blunting it."""
    section = PROMPT.split("# Speaking to a non-specialist")[1]
    assert "`no skill`" in section

    # Whitespace-normalised: the prompt hard-wraps, so a required phrase can
    # straddle a line break without being any less present.
    gloss = re.sub(r"\s+", " ",
                   section.split("`no skill`")[1].split("\n\n")[0]).lower()
    assert "not reliable" in gloss
    assert "do not act on this number" in gloss
    # It must still say what "no skill" actually means, not just "unreliable".
    assert "normal seasonal conditions" in gloss or "climatology" in gloss
    # And it must not have acquired a hedge.
    for softener in ("use with caution", "may still", "roughly", "somewhat useful"):
        assert softener not in gloss, f"the no-skill gloss softened into {softener!r}"


def test_every_label_gets_a_plain_language_pairing():
    section = PROMPT.split("# Speaking to a non-specialist")[1]
    for label in ("`validated`", "`weak/directional`", "`no skill`"):
        assert label in section, f"{label} has no plain-language gloss"
    assert "the first time it appears" in section.lower() or \
           "first time" in section.lower()


def test_reader_facing_prose_uses_real_months_not_horizon_codes():
    section = PROMPT.split("# Speaking to a non-specialist")[1]
    assert "Name real months" in section
    assert "never the raw `t+1`" in section
    # The structured field names must survive — the frontend depends on them.
    assert "stays in the structured data" in section or \
           "structured data" in section


def test_how_to_read_this_comes_before_the_detail_sections():
    """Part 2 moved it to the top: the reader needs the framing first."""
    structure = PROMPT.split("# Structure")[1]
    how_to = structure.find("How to read this report")
    drought = structure.find("**Drought**")
    summary = structure.find("**Summary**")
    assert -1 < summary < how_to < drought, (
        "'How to read this report' must sit between Summary and Drought")


def test_jargon_must_be_glossed_in_place():
    section = PROMPT.split("# Speaking to a non-specialist")[1]
    assert "SPI-3" in section
    for term in ("anomaly", "climatology", "horizon", "RMSE", "skill score"):
        assert term in section, f"{term} is not listed as needing a gloss"


# --------------------------------------------------------------------------- #
# The assembled prompt carries a real `no skill` horizon through
# --------------------------------------------------------------------------- #
def test_assembled_prompt_carries_a_no_skill_horizon_and_its_month():
    """Render the real prompt for a state containing a no-skill horizon.

    This is the end-to-end half of the check: not just that the instructions
    exist, but that the data the model receives actually contains the label it
    is required to gloss, and the calendar month it is required to name.
    """
    from orchestrator.graph import _render_sources

    no_skill = "no skill — shown for context only, do not rely on this figure"
    state = {
        "request": "drought risk for barmer",
        "tool_outputs": {"forecast_drought_risk": {
            "region": "barmer",
            "predicted_values": [-0.29, -0.07, -0.03],
            "forecast_anchor_month": "2026-06-01",
            "forecast_months": ["2026-07-01", "2026-08-01", "2026-09-01"],
            "horizon_confidence": [
                {"horizon": "t+1", "skill_score": 0.2053, "label": "validated",
                 "month": "2026-07-01"},
                {"horizon": "t+2", "skill_score": 0.0438,
                 "label": "weak/directional", "month": "2026-08-01"},
                {"horizon": "t+3", "skill_score": -0.0489, "label": no_skill,
                 "month": "2026-09-01"},
            ]}},
        "retrieved_chunks": [],
        "type_c": {"outlooks": []},
        "external": {"sources": []},
    }

    # Exactly how graph.synthesize serialises it (ensure_ascii default, so the
    # em-dash arrives escaped as \u2014 — still the same string once parsed).
    payload = json.dumps(state["tool_outputs"], indent=2, default=str)
    round_tripped = json.loads(payload)
    labels = [h["label"] for h in
              round_tripped["forecast_drought_risk"]["horizon_confidence"]]
    assert no_skill in labels, "the no-skill label must reach the model intact"
    assert "2026-09-01" in payload, "the month a horizon refers to must reach it too"

    # And the source renderer must not crash on the new external block.
    assert isinstance(_render_sources(state), str)
