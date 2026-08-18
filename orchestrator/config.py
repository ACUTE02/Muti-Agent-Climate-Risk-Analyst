"""Orchestrator configuration."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_DIR = REPO_ROOT / "orchestrator"
PROMPTS_DIR = ORCHESTRATOR_DIR / "prompts"
SYNTHESIS_PROMPT_PATH = PROMPTS_DIR / "synthesis.md"
GROUNDING_LOG_PATH = ORCHESTRATOR_DIR / "grounding_failures.jsonl"
CAUGHT_SAMPLE_PATH = ORCHESTRATOR_DIR / "grounding_caught_sample.json"

PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# Gemini 3.6 Flash (free tier). 2.5 Flash was deprecated for new API keys —
# 404 NOT_FOUND — see the Phase 3.1 entry in PROJECT_LOG.md.
CHAT_MODEL = "gemini-3.6-flash"
TEMPERATURE = 0.2          # low: this is reporting, not creative writing.
                           # Note: gemini-3.6-flash uses fixed sampling defaults
                           # and ignores this — kept for other/future models.
MAX_SYNTHESIS_ATTEMPTS = 2   # one regeneration after a grounding failure

# Absolute tolerance for float comparison, on top of the precision-rounding rule.
GROUNDING_ABS_TOLERANCE = 5e-5

RETRIEVAL_K = 5
DEFAULT_RISK_TYPES = ("drought", "heat_stress")
