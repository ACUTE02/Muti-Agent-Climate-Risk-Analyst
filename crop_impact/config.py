"""Crop Impact Agent configuration — crops, seasons, and decision thresholds.

Single source of truth, the same pattern as forecasting/config.py. Nothing here
is a yield-impact number: those live in yield_impact_table.json with a citation
per entry, because they must be inspectable and mechanically matchable by the
grounding checker.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CROP_IMPACT_DIR = REPO_ROOT / "crop_impact"
PROMPTS_DIR = CROP_IMPACT_DIR / "prompts"
NARRATIVE_PROMPT_PATH = PROMPTS_DIR / "narrative.md"
YIELD_TABLE_PATH = CROP_IMPACT_DIR / "yield_impact_table.json"
DOMINANCE_RULE_PATH = CROP_IMPACT_DIR / "dominance_rule.md"

# --------------------------------------------------------------------------- #
# Crops
# --------------------------------------------------------------------------- #
# `sensitive_months` is the window in which a climate shock actually moves yield
# for this crop — flowering through grain filling, not the whole growing season.
# Outside it the agent reports "none dominant" rather than attributing a loss to
# a month the crop is not vulnerable in.
CROPS = {
    "bajra": {
        "label": "Bajra (pearl millet)",
        "season": "kharif",
        "sown_months": [6, 7],
        "sensitive_months": [8, 9],      # flowering -> grain filling
        "sensitive_stage": "flowering and grain filling",
        "regions": ["rajasthan", "barmer"],
        "primary_risk": "drought",
    },
    "wheat": {
        "label": "Wheat",
        "season": "rabi",
        "sown_months": [11, 12],
        "sensitive_months": [2, 3],      # terminal heat during grain filling
        "sensitive_stage": "grain filling (terminal heat window)",
        # Barmer is arid Thar desert with negligible irrigated wheat — the crop
        # is scoped to Rajasthan (Jaipur centroid) only, rather than pretending
        # a wheat assessment for Barmer means something.
        "regions": ["rajasthan"],
        "primary_risk": "heat",
    },
}
DEFAULT_CROP = "bajra"

# --------------------------------------------------------------------------- #
# Dominance thresholds — see dominance_rule.md for the reasoning
# --------------------------------------------------------------------------- #
# Only a horizon carrying this label may declare drought dominant on its own.
# "weak/directional" can corroborate but never decide; "no skill" is ignored
# entirely. This is the single most important correctness property of the phase.
DECISIVE_FORECAST_LABEL = "validated"
CORROBORATING_FORECAST_LABEL = "weak/directional"

# Observed heat wave days in the month before heat is even a candidate.
HEAT_CANDIDATE_MIN_DAYS = 3

# Second candidacy route: a sustained warm anomaly, in degrees C of monthly mean
# Tmax departure from the 1981-2010 day-of-year normal.
#
# This exists because the IMD heat wave day counter cannot see terminal heat
# stress in wheat. IMD's plains criteria gate on Tmax >= 40 C, which February and
# March at these sites almost never reach — measured directly: the warmest
# February at Jaipur in the whole record (2006, +5.35 C mean departure) records
# **zero** IMD heat wave days. The day counter is a summer indicator; wheat's
# grain-filling window is Feb-Mar. Judging terminal heat by heat wave days would
# report "no heat" for every warm wheat season on record.
#
# SEVERE is set to 4.0 deliberately: it matches the match_band of the one sourced
# coefficient in yield_impact_table.json, so the coefficient is never applied
# outside the exposure range it was measured in. MODERATE at 3.0 is a judgement
# call with no source behind it — flagged here the same way spi_to_risk_score is,
# as a threshold Phase 5 should calibrate rather than a measured quantity.
HEAT_DEPARTURE_MODERATE_C = 3.0
HEAT_DEPARTURE_SEVERE_C = 4.0

# Severity ranking used to break a drought-vs-heat comparison.
SEVERITY_RANK = {"moderate": 1, "severe": 2}


def check_crop(crop: str) -> str:
    if crop not in CROPS:
        raise ValueError(
            f"Unknown crop {crop!r}. Supported: {', '.join(sorted(CROPS))}.")
    return crop
