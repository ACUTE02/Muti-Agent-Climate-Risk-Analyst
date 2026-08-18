"""Phase 5 — the evaluation suite.

Entirely offline and free. The live faithfulness run is a script
(``python -m evaluation.faithfulness``), not a test, because it costs 3-5 Gemini
calls per item against a 20/day budget; its *results* are asserted here from the
checked-in JSON instead, so the suite verifies the measurement without repeating
its cost.

The centrepiece is ``test_every_number_in_the_scorecard_traces_to_its_source``:
EVALUATION.md is held to exactly the standard the system's own generated reports
are held to, using the same matching primitives.
"""

from __future__ import annotations

import json
import re

import pytest

from evaluation import checker_eval, config
from orchestrator.grounding import (_matches, collect_source_numbers,
                                    extract_numbers)

# --------------------------------------------------------------------------- #
# EVALUATION.md — every quoted number must appear in a cited source
# --------------------------------------------------------------------------- #
SOURCE_LINE = re.compile(r"^\*Source:\s*(.+?)\*\s*$", re.M | re.S)
BACKTICKED = re.compile(r"`([^`]+)`")

# Cross-references, not measured claims: "Phase 5", "section 5", "(§5)", "Phase 1.4".
# Stripped before extraction for the same reason grounding.py strips "t+1" — a
# pointer to where something lives carries no numeric claim to verify.
CROSS_REFS = (
    re.compile(r"\bPhases?\s+[\d.]+(?:\s*(?:-|–|and|,)\s*[\d.]+)*", re.I),
    re.compile(r"§\s*\d+"),
    re.compile(r"^\s*\d{1,2}\.\s"),
)


def _strip_cross_refs(text: str) -> str:
    for pattern in CROSS_REFS:
        text = pattern.sub(" ", text)
    return text


def _sections() -> list[tuple[str, str, list[str]]]:
    """(heading, body, cited files) for each section that quotes numbers."""
    text = config.SCORECARD_PATH.read_text(encoding="utf-8")
    parts = re.split(r"^## ", text, flags=re.M)

    sections = []
    for i, part in enumerate(parts):
        heading = "preamble" if i == 0 else part.splitlines()[0].strip()
        match = SOURCE_LINE.search(part)
        if not match:
            continue
        cited = [c for c in BACKTICKED.findall(match.group(1))
                 if "/" in c or c.endswith(".md")]
        # drop the heading line itself — it is a label, not a claim
        body = part.replace(match.group(0), " ")
        if i:
            _, _, body = body.partition("\n")
        body = _strip_cross_refs(body)
        sections.append((heading, body, cited))
    return sections


def _cited_blob(cited: list[str]) -> tuple[set, str, list[str]]:
    """Concatenated text of the cited files that exist.

    ``models/`` is gitignored, so those files are present locally but absent in a
    fresh clone. Missing files are reported rather than silently treated as
    empty — a check that passes because it read nothing is the vacuous pass this
    project has caught twice.
    """
    blob, missing = "", []
    for name in cited:
        path = config.REPO_ROOT / name
        if path.exists():
            blob += "\n" + path.read_text(encoding="utf-8", errors="ignore")
        else:
            missing.append(name)
    values, source_blob = collect_source_numbers({}, [{"text": blob}])
    return values, source_blob, missing


def test_scorecard_exists_and_every_section_cites_a_source():
    assert config.SCORECARD_PATH.exists(), "EVALUATION.md must exist"
    sections = _sections()
    assert len(sections) >= 8, f"expected the full scorecard, got {len(sections)}"


@pytest.mark.parametrize("heading,body,cited",
                         _sections(),
                         ids=[s[0][:40] for s in _sections()])
def test_every_number_in_the_scorecard_traces_to_its_source(heading, body, cited):
    """The scorecard is held to the standard the system's reports are held to."""
    assert cited, f"section {heading!r} quotes numbers but cites no source file"

    values, blob, missing = _cited_blob(cited)
    present = [c for c in cited if c not in missing]
    assert present, (
        f"section {heading!r} cites only files that do not exist here "
        f"({missing}) — nothing could be verified against")

    unverified = [c["token"] for c in extract_numbers(body)
                  if not _matches(c, values, blob)]
    assert not unverified, (
        f"section {heading!r} quotes {unverified} which appear in none of its "
        f"cited sources {present}"
        + (f" (not checked, absent locally: {missing})" if missing else ""))


# --------------------------------------------------------------------------- #
# Grounding-checker evaluation (offline, deterministic)
# --------------------------------------------------------------------------- #
def test_checker_eval_labels_are_exhaustive():
    """Every extracted number in every case must be labelled.

    score_case() raises otherwise; this asserts the whole set passes that gate,
    so the measurement cannot drift behind the fixtures.
    """
    for case in checker_eval.CASES:
        checker_eval.score_case(case)      # raises on an unlabelled token


def test_checker_eval_is_reproducible_and_matches_the_checked_in_result():
    live = checker_eval.evaluate()
    stored = json.loads(config.CHECKER_EVAL_PATH.read_text(encoding="utf-8"))

    for key in ("precision", "recall", "f1", "true_positives", "false_positives",
                "false_negatives", "true_negatives", "cases"):
        assert live[key] == stored[key], (
            f"{key} drifted: recomputed {live[key]} vs checked-in {stored[key]} "
            "— re-run `python -m evaluation.checker_eval`")


def test_checker_eval_covers_the_failure_modes_that_actually_happened():
    ids = {c.id for c in checker_eval.CASES}
    for required in ("mckee_band_recitation",     # the real Phase-3 fabrication
                     "content_block_leakage",     # the Phase-3.1 bug's signature
                     "number_fragment_attack",    # boundary matching
                     "invented_yield_percentage"):  # the Phase-4 failure mode
        assert required in ids, f"the labelled set must cover {required}"


def test_checker_has_no_false_positives_on_this_set():
    """Precision 1.0 is load-bearing: a checker that cries wolf gets ignored."""
    result = checker_eval.evaluate()
    assert result["false_positives"] == 0
    assert result["precision"] == 1.0


def test_the_known_false_negative_is_recorded_not_hidden():
    """The percent/fraction defect must stay documented and stay failing.

    If someone fixes the checker, this test fails loudly — which is correct: the
    fix must come with a re-measurement and a scorecard update, not silently.
    """
    assert "PERCENT_FRACTION_DEFECT" in dir(checker_eval)
    assert "12%" in checker_eval.PERCENT_FRACTION_DEFECT

    result = checker_eval.evaluate()
    failing = [c for c in result["per_case"] if not c["passed"]]
    assert len(failing) == 1
    assert failing[0]["id"] == "invented_yield_percentage"
    assert failing[0]["missed"] == ["12%"]


def test_scorecard_reports_the_defect_rather_than_only_the_headline():
    text = config.SCORECARD_PATH.read_text(encoding="utf-8")
    assert "false-negative" in text or "12%" in text
    assert "not fixed in this phase" in text.lower()


# --------------------------------------------------------------------------- #
# Faithfulness set and results
# --------------------------------------------------------------------------- #
def test_held_out_set_was_written_before_results_and_states_its_size_rationale():
    spec = checker_eval.json.loads(
        config.REQUESTS_PATH.read_text(encoding="utf-8"))

    assert spec["written_before_results"] is True
    assert spec["size_rationale"].strip()
    assert spec["known_coverage_gap"].strip()
    assert len(spec["requests"]) == config.HELD_OUT_SET_SIZE
    for item in spec["requests"]:
        assert item["targets"], "every item must name the tools it should route to"
        assert item["what_a_faithful_report_must_do"].strip()


def test_held_out_set_includes_an_adversarial_item():
    spec = json.loads(config.REQUESTS_PATH.read_text(encoding="utf-8"))
    assert any("impossible" in i["id"] or "unsupported" in i["why_this_item"].lower()
               for i in spec["requests"])


def test_quota_budget_was_stated_before_the_set_was_sized():
    """The set size must actually fit the budget it claims to fit."""
    worst = config.HELD_OUT_SET_SIZE * config.CALLS_PER_ITEM_WORST
    assert worst <= config.DAILY_REQUEST_BUDGET, (
        f"{config.HELD_OUT_SET_SIZE} items x {config.CALLS_PER_ITEM_WORST} calls "
        f"= {worst} exceeds the {config.DAILY_REQUEST_BUDGET}/day budget")


@pytest.mark.skipif(not config.FAITHFULNESS_PATH.exists(),
                    reason="run `python -m evaluation.faithfulness` first")
def test_faithfulness_results_are_complete_and_labelled_lower_trust():
    result = json.loads(config.FAITHFULNESS_PATH.read_text(encoding="utf-8"))

    assert result["set_size"] == config.HELD_OUT_SET_SIZE
    assert "lower-trust" in result["trust_note"]
    assert "mechanical checker wins" in result["trust_note"]

    summary = result["summary"]
    assert summary["items_scored"] == summary["items_run"], \
        "an item whose judge output could not be parsed must not be silently dropped"
    assert 0.0 <= summary["mean_faithfulness"] <= 1.0
    assert 0.0 <= summary["mean_relevance"] <= 1.0


@pytest.mark.skipif(not config.FAITHFULNESS_PATH.exists(),
                    reason="run `python -m evaluation.faithfulness` first")
def test_mechanical_grounding_was_clean_on_every_judged_report():
    """The high-trust check must pass even where the soft one has quibbles."""
    result = json.loads(config.FAITHFULNESS_PATH.read_text(encoding="utf-8"))
    for item in result["items"]:
        grounding = item["mechanical_grounding"]
        assert grounding.get("grounded") is True, (
            f"{item['id']} was not mechanically grounded: "
            f"{grounding.get('unverified_numbers')}")
        assert grounding.get("total_checked", 0) > 0, \
            f"{item['id']} passed over zero numbers — a vacuous pass"


# --------------------------------------------------------------------------- #
# The judge is the documented exception, and it stays documented
# --------------------------------------------------------------------------- #
def test_the_llm_judge_exception_is_explicit_in_code_and_scorecard():
    import inspect

    from evaluation import faithfulness
    source = inspect.getsource(faithfulness)
    assert "one place in the entire project where an LLM judges an LLM" in source
    assert "lower trust" in source or "lower-trust" in source

    text = config.SCORECARD_PATH.read_text(encoding="utf-8")
    assert "mechanical checker wins" in text


def test_judge_prompt_is_checked_in_and_forbids_authoring():
    text = config.JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    assert "You are a grader, not an author" in text
    assert "unsupported" in text
