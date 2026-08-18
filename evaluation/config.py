"""Evaluation suite configuration and paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = REPO_ROOT / "evaluation"

SCORECARD_PATH = REPO_ROOT / "EVALUATION.md"
CHECKER_EVAL_PATH = EVALUATION_DIR / "checker_eval_results.json"
REQUESTS_PATH = EVALUATION_DIR / "eval_requests.json"
FAITHFULNESS_PATH = EVALUATION_DIR / "faithfulness_results.json"
JUDGE_PROMPT_PATH = EVALUATION_DIR / "prompts" / "judge.md"

# --------------------------------------------------------------------------- #
# Quota budget — decided before any test was written, per the phase spec
# --------------------------------------------------------------------------- #
# Measured free-tier limits (Phase 3.1): 5 RPM / 250K TPM / 20 RPD per key.
#
# Cost of one faithfulness item:
#   orchestrator report  = 2 calls (routing + synthesis), 3 if synthesis retries,
#                          +1 more if the request routes to assess_crop_impact
#   judge                = 1 call (faithfulness AND relevance folded into one
#                          prompt, per the spec's default)
#   => 3 calls typical, 5 worst case.
#
# A 10-15 item set is therefore 30-75 calls, comfortably past a day's budget on
# one key. The set is sized to what one day actually affords, and the honest
# consequence — a small sample — is reported rather than hidden.
DAILY_REQUEST_BUDGET = 20
CALLS_PER_ITEM_TYPICAL = 3
CALLS_PER_ITEM_WORST = 5
HELD_OUT_SET_SIZE = 3

JUDGE_TEMPERATURE = 0.0


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
