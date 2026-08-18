"""The grounding checker — mechanical, deliberately not another LLM call.

This is the load-bearing safety net of Phase 3. An LLM checking an LLM shares the
failure mode of the LLM being checked, so this is plain string and float matching:
every number in the generated report must appear in the tool outputs or the
retrieved chunks, or it gets flagged.

This project has already caught three cases of an unverified number looking right
and being wrong — the Phase-1 SPI proxy, the Heat-1.1 hardcoded baseline, and the
Phase-1.4 window leak. A synthesis step is a much larger surface for exactly that,
so it is checked mechanically rather than trusted to careful prompting.

**Tolerance rule.** A report number is grounded if some source number matches it
when rounded to the report number's own precision: ``+0.26`` is accepted against a
source ``+0.2622`` because ``round(0.2622, 2) == 0.26``. A small absolute
tolerance (5e-5) additionally absorbs trailing-digit noise, so ``+0.20531``
against a source ``+0.2053`` passes — the value is materially identical and
flagging it would be noise. What is *not* absorbed is a materially different
number: anything off by more than that tolerance at the report's own precision is
flagged.

**Deliberately conservative.** False positives (flagging a figure that is really
fine) are an acceptable price for never missing a fabrication. Obvious non-claims
are skipped — years, horizon labels like t+1, list markers, section numbers — but
no attempt is made to be clever beyond that.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from orchestrator import config

# Signed decimals, integers, and percentages. Kept broad on purpose.
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")

# Structural tokens that carry no measured claim, removed before extraction.
STRUCTURAL_PATTERNS = (
    re.compile(r"\bt\s*\+\s*\d\b", re.I),        # horizon labels: t+1, t+2, t+3
    re.compile(r"\bSPI-?3\b", re.I),             # the index's name, not a value
    re.compile(r"\bgemini-embedding-\d+\b", re.I),
    re.compile(r"\bgemini-[\d.]+-flash\b", re.I),
    re.compile(r"\bMcKee et al\.?,? \d{4}\b", re.I),
    re.compile(r"^\s*\d{1,2}[.)]\s", re.M),      # list markers at line start
    re.compile(r"\bERSST[ .]?v?\d*\b", re.I),
    re.compile(r"\bHadISST[\d.]*\b", re.I),
)

YEAR_MIN, YEAR_MAX = 1900, 2100


def _strip_structural(text: str) -> str:
    for pattern in STRUCTURAL_PATTERNS:
        text = pattern.sub(" ", text)
    # ISO-ish dates and month strings: 2024-05, 2024-05-01, 1980-2015
    text = re.sub(r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b", " ", text)
    return text


def _parse(token: str) -> float | None:
    raw = token.rstrip("%").replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _is_ignorable(token: str, value: float) -> bool:
    """Years and bare small ordinals are not measured claims."""
    if "." not in token and "%" not in token:
        if YEAR_MIN <= abs(value) <= YEAR_MAX and abs(value) == int(abs(value)):
            return True
    return False


def extract_numbers(text: str) -> list[dict]:
    """Every numeric claim in ``text``, with the structural noise removed."""
    cleaned = _strip_structural(text)
    found: list[dict] = []
    for match in NUMBER_RE.finditer(cleaned):
        token = match.group(0)
        value = _parse(token)
        if value is None or _is_ignorable(token, value):
            continue
        decimals = len(token.rstrip("%").split(".")[1]) if "." in token else 0
        found.append({
            "token": token,
            "value": value,
            "decimals": decimals,
            "is_percent": token.endswith("%"),
        })
    return found


def collect_source_numbers(tool_outputs: dict,
                           retrieved_chunks: list[dict]) -> tuple[set, str]:
    """Every number the report is allowed to use, plus the raw source text."""
    blob = json.dumps(tool_outputs, ensure_ascii=False, default=str)
    for chunk in retrieved_chunks or []:
        blob += "\n" + str(chunk.get("text", ""))
        blob += "\n" + str(chunk.get("excerpt", ""))

    values = set()
    for match in NUMBER_RE.finditer(blob):
        value = _parse(match.group(0))
        if value is not None:
            values.add(value)
            if match.group(0).endswith("%"):
                values.add(value / 100.0)
    return values, blob


def _verbatim_in(token: str, source_blob: str) -> bool:
    """Whole-number string match.

    A plain substring test is wrong here: the token "4" occurs inside "0.0438",
    so a fabricated "4-month horizon" would be silently accepted by any source
    containing that skill score. Boundaries are enforced so a number only matches
    a number, never a fragment of one.
    """
    pattern = re.compile(r"(?<![\d.])" + re.escape(token.rstrip("%"))
                         + r"(?![\d])")
    return bool(pattern.search(source_blob))


def _matches(candidate: dict, source_values: set, source_blob: str) -> bool:
    # 1. verbatim hit on a whole number — cheapest and most common
    if _verbatim_in(candidate["token"], source_blob):
        return True

    targets = [candidate["value"]]
    if candidate["is_percent"]:                  # 26% may be sourced as 0.26
        targets.append(candidate["value"] / 100.0)

    for target in targets:
        for source in source_values:
            # the report may round a source value down to its own precision,
            # never the other way around
            if round(source, candidate["decimals"]) == round(target,
                                                             candidate["decimals"]):
                return True
            if abs(source - target) <= config.GROUNDING_ABS_TOLERANCE:
                return True
    return False


def check_grounding(report_text: str, tool_outputs: dict,
                    retrieved_chunks: list[dict]) -> dict:
    """
    Extracts every numeric figure from report_text (regex: signed decimals,
    percentages, integers plausibly referring to a measured value — skill
    scores, day counts, temperatures) and verifies each appears, verbatim or
    within reasonable float-formatting tolerance (e.g. +0.2622 vs +0.26),
    in the tool_outputs JSON or the retrieved_chunks text.
    Returns: {"grounded": bool, "unverified_numbers": [...], "total_checked": int}
    """
    source_values, source_blob = collect_source_numbers(tool_outputs,
                                                        retrieved_chunks)
    candidates = extract_numbers(report_text)

    unverified = [c["token"] for c in candidates
                  if not _matches(c, source_values, source_blob)]
    return {
        "grounded": not unverified,
        "unverified_numbers": unverified,
        "total_checked": len(candidates),
        "source_numbers_available": len(source_values),
    }


def log_failure(request: str, result: dict, report_text: str,
                attempt: int) -> None:
    """Append a grounding failure to the runtime log.

    These are the canary for the failure mode the whole project is built to
    avoid, so they are never silently discarded.
    """
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": request,
        "attempt": attempt,
        "unverified_numbers": result["unverified_numbers"],
        "total_checked": result["total_checked"],
        "report_excerpt": report_text[:1200],
    }
    with config.GROUNDING_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def warning_banner(result: dict) -> str:
    """Shown when a report could not be fully verified — never hidden."""
    numbers = ", ".join(result["unverified_numbers"])
    return (
        "> **UNVERIFIED FIGURES WARNING.** The following numbers in this report "
        f"could not be traced to a tool output or a retrieved source: {numbers}. "
        "They may be fabricated or misquoted. Every other figure was verified "
        "mechanically against source data. Do not rely on the flagged values.\n"
    )
