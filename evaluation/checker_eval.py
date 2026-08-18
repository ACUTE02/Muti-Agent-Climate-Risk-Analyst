"""How accurate is the grounding checker itself?

The mechanical checker is the most safety-critical piece in this project, and
until now it had only ever been *exercised* by hand-written test cases — never
*measured*. An unexamined metric is the exact failure this project has caught
three times already (the Heat-1.1 hardcoded baseline, the Phase-1.4 window leak,
the Phase-3.1 content-block bug). This module closes that loop.

Fully offline: fixtures in, precision and recall out, no API calls, no cost.

**Labelling discipline.** Every case must label *every* number the checker can
extract from its report — each token is either `fabricated` (the checker should
flag it) or `grounded` (it should not). ``score_case`` raises if any extracted
token is unlabelled, so the label set cannot silently drift behind the fixtures
and flatter the result.

**No tuning against this set.** Per the phase's stopping rule, if the measured
numbers are imperfect they are reported as measured. Adjusting the checker's
regex to pass its own evaluation would be optimising for the test rather than
for real reports — which is the thing being guarded against in the first place.

Run standalone:  python -m evaluation.checker_eval
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from evaluation import config
from orchestrator.grounding import check_grounding, extract_numbers

# --------------------------------------------------------------------------- #
# Source material the fixture reports are checked against
# --------------------------------------------------------------------------- #
DROUGHT_OUTPUT = {
    "region": "rajasthan",
    "predicted_values": [-1.62, -0.41, 0.08],
    "horizon_confidence": [
        {"horizon": "t+1", "skill_score": 0.2622, "method": "direct",
         "label": "validated"},
        {"horizon": "t+2", "skill_score": 0.0766, "method": "direct",
         "label": "weak/directional"},
        {"horizon": "t+3", "skill_score": -0.0145, "method": "direct",
         "label": "no skill — shown for context only, do not rely on this figure"},
    ],
    "risk_score": 8.5,
    "risk_flags": ["Severe", "Normal", "Normal"],
    "model_rmse_test": 0.8918,
}

HEAT_OUTPUT = {
    "region": "barmer", "month": "2024-05",
    "heatwave_days": 7, "severe_heatwave_days": 2,
    "had_heatwave_spell": True, "max_tmax_c": 48.3,
    "forecast_available": False,
}

CROP_OUTPUT = {
    "region": "rajasthan", "crop": "wheat", "month": "2006-02",
    "dominant_risk": "heat", "yield_impact_pct": 5.6,
    "confidence_label": "observed — not a forecast",
    "signals": {"heat": {"mean_tmax_departure_c": 5.35, "heatwave_days": 0}},
}

SPI_CHUNK = [{
    "source": "NIH Roorkee — Standardized Precipitation Index methodology",
    "citation": "https://nihroorkee.gov.in/sites/default/files/uploadfiles/SPINov2011.pdf",
    "text": ("The SPI is computed by fitting a gamma distribution to the "
             "precipitation record and transforming it through the normal "
             "cumulative distribution function, following McKee et al. 1993."),
}]


# --------------------------------------------------------------------------- #
# The labelled set
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """One labelled report.

    ``fabricated`` — tokens the checker *should* flag (the positive class).
    ``grounded``   — tokens it should *not* flag.
    Together they must cover every number the checker extracts.
    """
    id: str
    why: str
    report: str
    tool_outputs: dict
    chunks: list = field(default_factory=list)
    fabricated: tuple = ()
    grounded: tuple = ()


CASES: list[Case] = [
    Case(
        id="clean_drought_report",
        why="A wholly grounded report. Any flag here is a false positive.",
        report=(
            "Forecast SPI-3 for rajasthan is -1.62 at t+1, -0.41 at t+2 and 0.08 "
            "at t+3. The t+1 horizon carries a measured skill score of 0.2622 and "
            "is labelled validated; t+2 is 0.0766 and weak/directional; t+3 is "
            "-0.0145 and has no measured skill. Test RMSE is 0.8918 and the "
            "overall risk score is 8.5."),
        tool_outputs=DROUGHT_OUTPUT,
        grounded=("-1.62", "-0.41", "0.08", "0.2622", "0.0766", "-0.0145",
                  "0.8918", "8.5"),
    ),
    Case(
        id="rounded_to_report_precision",
        why=("The documented tolerance rule: a report may round a source value "
             "to its own precision. +0.26 against 0.2622 must pass."),
        report=("The 1-month forecast has a skill score of about 0.26, and the "
                "predicted index is -1.6."),
        tool_outputs=DROUGHT_OUTPUT,
        # "1" comes from the phrase "1-month forecast" — a horizon descriptor,
        # not a measured claim, so it should not be flagged.
        grounded=("1", "0.26", "-1.6"),
    ),
    Case(
        id="fabricated_skill_score",
        why="The commonest fabrication shape: a plausible but invented metric.",
        report=("The 1-month drought forecast achieves a skill score of 0.42, "
                "well above the project threshold. Predicted SPI-3 is -1.62."),
        tool_outputs=DROUGHT_OUTPUT,
        fabricated=("0.42",),
        grounded=("1", "-1.62"),
    ),
    Case(
        id="mckee_band_recitation",
        why=("The real fabrication caught in Phase 3: correct-in-the-world "
             "classification bands recited from training data and cited to a "
             "document that does not contain them."),
        report=(
            "SPI values between -0.99 and 0.99 are near normal, while -1.0 to "
            "-1.49 is moderately dry (source: NIH Roorkee SPI methodology). The "
            "forecast value is -1.62 at t+1."),
        tool_outputs=DROUGHT_OUTPUT,
        chunks=SPI_CHUNK,
        # -1.0 belongs with the other recited band edges: it is part of the same
        # invented table and appears nowhere in this case's sources. It was
        # labelled "grounded" in the first draft of this set, which was a
        # labelling error on my part, not a checker error — corrected here so the
        # measurement reflects ground truth rather than my first guess.
        fabricated=("-0.99", "0.99", "-1.0", "-1.49"),
        grounded=("-1.62",),
    ),
    Case(
        id="subtly_wrong_digit",
        why=("Materially different at the report's own precision. A checker that "
             "let this through would be useless — this is the hard case."),
        report="The measured t+1 skill score is 0.2722.",
        tool_outputs=DROUGHT_OUTPUT,
        fabricated=("0.2722",),
    ),
    Case(
        id="number_fragment_attack",
        why=("A fabricated '4-month horizon' must not be accepted merely because "
             "'4' appears inside the source value 0.0438. This is why the "
             "checker enforces number boundaries rather than substring matching."),
        report="This system can forecast 4 months ahead with a skill of 0.0766.",
        tool_outputs=DROUGHT_OUTPUT,
        fabricated=("4",),
        grounded=("0.0766",),
    ),
    Case(
        id="content_block_leakage",
        why=("The Phase-3.1 bug's signature: a non-text metadata block "
             "stringified into the report body. The checker's behaviour was "
             "correct — those digits are genuinely unverifiable — so every one "
             "of them is a true positive."),
        report=("Forecast SPI-3 is -1.62 at t+1. "
                "{'signature': 'Ct6446h8984z7712', 'thought_tokens': 913}"),
        tool_outputs=DROUGHT_OUTPUT,
        fabricated=("6446", "8984", "7712", "913"),
        grounded=("-1.62",),
    ),
    Case(
        id="clean_heat_observations",
        why="Observation-only heat reporting, fully grounded.",
        report=("Barmer recorded 7 heat wave days in May 2024, of which 2 were "
                "severe, with a maximum temperature of 48.3 C. No forecast is "
                "available for this risk type."),
        tool_outputs=HEAT_OUTPUT,
        grounded=("7", "2", "48.3"),
    ),
    Case(
        id="inflated_heat_count",
        why="A count inflated from 7 to 17 — grounded-looking, materially wrong.",
        report="Barmer recorded 17 heat wave days, with a peak of 48.3 C.",
        tool_outputs=HEAT_OUTPUT,
        fabricated=("17",),
        grounded=("48.3",),
    ),
    Case(
        id="clean_crop_impact",
        why="The sourced coefficient reported exactly as the table holds it.",
        report=("Heat is the binding risk for wheat: the observed mean maximum "
                "temperature ran 5.35 C above normal. The sourced yield impact "
                "is 5.6%."),
        tool_outputs=CROP_OUTPUT,
        grounded=("5.35", "5.6%"),
    ),
    Case(
        id="invented_yield_percentage",
        why=("The Phase-4 failure mode: asked about crop impact, a model "
             "supplies a confident percentage from general agronomic knowledge. "
             "KNOWN FALSE NEGATIVE — see PERCENT_FRACTION_DEFECT below. Kept in "
             "the set as a failing case rather than removed or worked around."),
        report=("Heat is the binding risk for wheat, and drought conditions "
                "typically reduce yields by around 12% in this region."),
        tool_outputs=CROP_OUTPUT,
        fabricated=("12%",),
    ),
    Case(
        id="scaled_coefficient",
        why=("Scaling a single anchor point into a per-degree rate — forbidden "
             "by the yield table's own caveat, and it produces a number that "
             "exists in no source."),
        report=("At 5.35 C above normal the expected loss is 5.45%, derived from "
                "the 5.6% anchor."),
        tool_outputs=CROP_OUTPUT,
        fabricated=("5.45%",),
        grounded=("5.35", "5.6%"),
    ),
    Case(
        id="structural_tokens_only",
        why=("Years, horizon labels and index names carry no measured claim and "
             "must never be flagged — false positives here would make the "
             "checker's output unreadable."),
        report=("Between 1980 and 2024 this project measured SPI-3 at t+1, t+2 "
                "and t+3. See the 2020-2024 held-out test set."),
        tool_outputs=DROUGHT_OUTPUT,
    ),
    Case(
        id="percent_against_fraction",
        why="A source fraction 0.0766 reported as 7.66% is the same quantity.",
        report="The 2-month forecast explains 7.66% of the climatology gap.",
        tool_outputs=DROUGHT_OUTPUT,
        grounded=("2", "7.66%"),
    ),
]


PERCENT_FRACTION_DEFECT = """A real false-negative class this evaluation found in
check_grounding(), reported rather than quietly patched.

An integer percentage is compared against its fraction form at the report token's
own precision. "12%" has zero decimal places, so the fraction 0.12 is rounded to
zero decimals -> 0.0, which matches any source containing 0.0. Tool outputs are
full of legitimate zeros (heatwave_days: 0, severe_heatwave_days: 0), so **any
invented integer percentage below 50% is silently accepted** whenever the sources
contain a zero.

That is exactly the fabrication Phase 4 exists to prevent — "drought typically
reduces yields by around 12%" passes the checker today.

Not fixed in this phase, deliberately. Phase 5's stated non-goals forbid tuning
any measured component, and fixing the checker here would mean the precision and
recall reported in EVALUATION.md no longer describe the code that was measured.
The defect is recorded, the failing case is kept in the set, and the fix belongs
in its own change — after which this evaluation should be re-run and the
scorecard updated.

Likely fix, for whoever picks it up: compare the percent-as-fraction form at the
precision it actually implies (12% -> 0.12 needs two more decimal places than the
token shows), rather than at the token's own zero decimals."""


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_case(case: Case) -> dict:
    """Confusion counts for one labelled report.

    Positive class = "this number is fabricated and should be flagged".
    """
    extracted = {c["token"] for c in extract_numbers(case.report)}
    labelled = set(case.fabricated) | set(case.grounded)
    unlabelled = extracted - labelled
    if unlabelled:
        raise AssertionError(
            f"case {case.id!r} extracts unlabelled tokens {sorted(unlabelled)} — "
            "every extracted number must be labelled fabricated or grounded, so "
            "the measurement cannot drift behind the fixtures")

    result = check_grounding(case.report, case.tool_outputs, case.chunks)
    flagged = set(result["unverified_numbers"])

    true_pos = sorted(flagged & set(case.fabricated))
    false_neg = sorted(set(case.fabricated) - flagged)
    false_pos = sorted(flagged - set(case.fabricated))
    true_neg = sorted(set(case.grounded) - flagged)

    return {
        "id": case.id,
        "why": case.why,
        "tp": len(true_pos), "fp": len(false_pos),
        "fn": len(false_neg), "tn": len(true_neg),
        "missed": false_neg,
        "wrongly_flagged": false_pos,
        "passed": not false_neg and not false_pos,
    }


def evaluate(cases: list[Case] | None = None) -> dict:
    """Precision and recall of the grounding checker over the labelled set."""
    scored = [score_case(c) for c in (cases if cases is not None else CASES)]

    tp = sum(s["tp"] for s in scored)
    fp = sum(s["fp"] for s in scored)
    fn = sum(s["fn"] for s in scored)
    tn = sum(s["tn"] for s in scored)

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)

    return {
        "generated_at": config.timestamp(),
        "what_this_measures": (
            "Precision and recall of orchestrator/grounding.py's check_grounding() "
            "on a hand-labelled adversarial set. Positive class = a number that "
            "is fabricated and should be flagged. Fully offline and "
            "deterministic — no LLM involved in either the checker or this score."),
        "cases": len(scored),
        "numbers_labelled": tp + fp + fn + tn,
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "cases_fully_correct": sum(1 for s in scored if s["passed"]),
        "per_case": scored,
    }


def main() -> dict:
    result = evaluate()
    config.CHECKER_EVAL_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"cases            : {result['cases']}")
    print(f"numbers labelled : {result['numbers_labelled']}")
    print(f"precision        : {result['precision']}")
    print(f"recall           : {result['recall']}")
    print(f"f1               : {result['f1']}")
    print(f"fully correct    : {result['cases_fully_correct']}/{result['cases']}")
    for case in result["per_case"]:
        if not case["passed"]:
            print(f"  ! {case['id']}: missed={case['missed']} "
                  f"wrongly_flagged={case['wrongly_flagged']}")
    print(f"\nwritten to {config.CHECKER_EVAL_PATH}")
    return result


if __name__ == "__main__":
    main()
