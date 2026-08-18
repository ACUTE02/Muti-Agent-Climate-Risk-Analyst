"""Quota as a first-class concern, not an afterthought.

The free tier allows **20 `generate_content` requests per day** (5 RPM, measured
in Phase 3.1). One `/report` costs 2 calls in the best case — routing and
synthesis — a third if synthesis needs its one regeneration, and a fourth if the
request routes to `assess_crop_impact`, which makes its own narrative call.

A demo that silently stops working at request #5 is a bad experience; one that
says exactly why, and how much is left, is fine. The count is a real tally taken
at `orchestrator.graph.invoke_with_backoff` — every chat call in the project goes
through it — not an estimate.

Embedding calls (`retrieve_context`) are **not** counted here: they bill against
a separate embedding quota, and conflating the two would misreport both.
"""

from __future__ import annotations

from datetime import date

DAILY_CALL_BUDGET = 20
TYPICAL_CALLS_PER_REPORT = 2      # routing + synthesis
WORST_CASE_CALLS_PER_REPORT = 4   # + synthesis retry + crop-impact narrative

QUOTA_SIGNATURES = ("429", "resource_exhausted", "quota", "rate limit")

NOTE = ("Counts generate_content calls only, tallied at the single chat "
        "chokepoint. Embedding calls bill against a separate quota and are not "
        "included. The count resets daily in-process; a restart resets it too, "
        "so treat it as a floor, not an audit.")


def calls_used_today() -> int:
    from orchestrator.graph import CALL_TALLY

    if CALL_TALLY["date"] != date.today().isoformat():
        return 0
    return int(CALL_TALLY["generate_content"])


def status() -> dict:
    used = calls_used_today()
    return {
        "daily_call_budget": DAILY_CALL_BUDGET,
        "calls_used_today": used,
        "calls_remaining_today": max(0, DAILY_CALL_BUDGET - used),
        "typical_calls_per_report": TYPICAL_CALLS_PER_REPORT,
        "worst_case_calls_per_report": WORST_CASE_CALLS_PER_REPORT,
        "note": NOTE,
    }


def looks_like_quota_failure(messages: list[str]) -> bool:
    """Does any warning or error text carry a quota/rate-limit signature?"""
    blob = " ".join(messages).lower()
    return any(sig in blob for sig in QUOTA_SIGNATURES)


def exhaustion_detail(messages: list[str]) -> str:
    used = calls_used_today()
    offending = next((m for m in messages
                      if any(s in m.lower() for s in QUOTA_SIGNATURES)), "")
    return (
        f"The Gemini free-tier quota was exhausted, so no report could be "
        f"generated. This project's key allows {DAILY_CALL_BUDGET} "
        f"generate_content requests per day at 5 per minute; {used} have been "
        f"counted in this process today, and one report costs "
        f"{TYPICAL_CALLS_PER_REPORT}-{WORST_CASE_CALLS_PER_REPORT}. "
        f"A per-minute limit clears in about a minute; the daily cap does not "
        f"clear until tomorrow. Underlying error: {offending or 'not reported'}")
