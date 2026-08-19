"""Phase-4 deliverable: the Crop Impact Agent's callable tool.

Hybrid by design, in this order:

1. ``dominant_risk()`` — deterministic, no LLM, decides which risk binds yield.
2. ``lookup_yield_impact()`` — a sourced table read, no LLM, no arithmetic.
3. **one** Gemini call — writes the plain-language explanation of (1) and (2).
   It is never asked for a number.
4. ``check_grounding()`` — the Phase-3 checker, reused unmodified, verifies every
   number in that explanation against the deterministic results.

The two decisions that must be trustworthy never touch a model. The one step that
genuinely needs judgement — explaining the result to a person — is the only one
that does.

Run standalone:  python -m crop_impact.tool [region] [crop] [month]
"""

from __future__ import annotations

import json

import pandas as pd
from langchain_core.tools import tool

from crop_impact import config
from crop_impact.dominance import dominant_risk
from crop_impact.yield_impact import lookup_yield_impact
from forecasting import config as fconfig
from forecasting.fetch_data import data_currency, load_live_daily
from forecasting.tool import forecast_drought_risk
from heat.target import fit_daily_normals, flag_heat_wave_days
from heat.tool import forecast_heat_stress_risk

MAX_HORIZON = 3


# --------------------------------------------------------------------------- #
# Timing: which signals can describe the month being asked about
# --------------------------------------------------------------------------- #
def latest_data_month(region: str) -> pd.Timestamp:
    """The month the drought forecast is actually anchored to.

    Deliberately *not* the newest weather month. The forecast needs every
    feature, and ONI trails the weather by a month or two, so the weather max can
    be one month ahead of the anchor. Using it here would offset the horizon
    arithmetic by one and quietly attribute t+1's SPI-3 value to the wrong
    calendar month — a wrong number under a right-looking label, which is the
    exact failure this project is built to avoid.
    """
    return pd.Timestamp(data_currency(region)["data_current_through"])


def resolve_target_month(region: str, crop: str,
                         month: str | None) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Return (target month, latest data month, horizon in months).

    ``horizon <= 0`` means the month has already happened, so the drought
    *forecast* cannot describe it; ``1..3`` means it is inside the forecast's
    range.

    With no month given, prefer the crop's next sensitive month that the drought
    forecast can actually reach (horizon 1..MAX_HORIZON), and only fall back to
    the most recent *completed* sensitive window when none is in range.

    Before Phase 8 this only ever looked backwards, because the newest data was
    2024-12 and nothing upcoming was ever reachable — so "what does the drought
    risk mean for my bajra?" resolved to a month already past and answered
    "no drought signal: that month has already happened". Now that the rolling
    cache keeps the inputs current, the forward case is the useful one, and the
    backward fallback is kept for crops whose window is genuinely out of range.
    """
    latest = latest_data_month(region)
    spec = config.CROPS[crop]

    if month is not None:
        try:
            target = pd.Timestamp(month).replace(day=1)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Could not parse month {month!r} — expected something like "
                "'2024-03' or '2024-03-01'.") from exc
    else:
        target = None
        for ahead in range(1, MAX_HORIZON + 1):
            candidate = latest + pd.DateOffset(months=ahead)
            if candidate.month in spec["sensitive_months"]:
                target = pd.Timestamp(candidate)
                break
        if target is None:                     # nothing upcoming is in range
            target = latest
            while target.month not in spec["sensitive_months"]:
                target -= pd.DateOffset(months=1)
            target = pd.Timestamp(target)

    horizon = (target.year - latest.year) * 12 + (target.month - latest.month)
    return target, latest, horizon


def mean_tmax_departure(region: str, target: pd.Timestamp) -> float | None:
    """Observed mean daily Tmax departure from normal, in degrees C, for a month.

    Reuses the Heat Agent's own day-of-year normals (train-only fit, pooled over
    a +/- 7-day window) rather than defining a second climatology. Returns None if
    the month has no observations — a missing measurement must not silently read
    as zero departure.
    """
    daily = load_live_daily(region)
    flagged = flag_heat_wave_days(daily, fit_daily_normals(daily, fconfig.TRAIN))
    monthly = flagged["tmax_departure_c"].resample("MS").mean()
    if target not in monthly.index:
        return None
    value = monthly.loc[target]
    return None if pd.isna(value) else round(float(value), 2)


# --------------------------------------------------------------------------- #
# Signal collection
# --------------------------------------------------------------------------- #
def collect_signals(region: str, crop: str, month: str | None) -> dict:
    """Gather the two risk signals for one region/crop/month, deterministically.

    The two risk types this system has do not describe the same time frame: the
    drought signal is a forecast (t+1..t+3) and the heat signal is an observation
    of a month that has already happened. So which signals are even *applicable*
    depends on when the requested month is, and that is decided here rather than
    left for the model to reason about.
    """
    target, latest, horizon = resolve_target_month(region, crop, month)
    target_str = f"{target:%Y-%m}"

    drought = None
    drought_reason = None
    if 1 <= horizon <= MAX_HORIZON:
        drought = forecast_drought_risk.invoke({"region": region})
    elif horizon <= 0:
        drought_reason = (
            f"{target_str} has already happened, and this system's drought "
            f"signal is a forecast (t+1 to t+3 from {latest:%Y-%m}), so it "
            f"cannot describe a past month")
    else:
        drought_reason = (
            f"{target_str} is {horizon} months after the last month of data "
            f"({latest:%Y-%m}), beyond this system's t+{MAX_HORIZON} forecast "
            f"range")

    # Heat is observations only. Ask for the target month when it exists; if it
    # does not, fall back to the latest observed month so the reasoning can say
    # exactly which month was observed and that it is not the one asked about.
    heat = None
    try:
        heat = forecast_heat_stress_risk.invoke(
            {"region": region, "month": target_str if horizon <= 0 else None})
    except ValueError:
        try:
            heat = forecast_heat_stress_risk.invoke({"region": region})
        except Exception:
            heat = None

    if heat is not None and heat.get("month") == target_str:
        heat["mean_tmax_departure_c"] = mean_tmax_departure(region, target)

    return {"target": target, "target_str": target_str, "latest": latest,
            "horizon": horizon, "drought": drought,
            "drought_reason": drought_reason, "heat": heat}


# --------------------------------------------------------------------------- #
# The narrative — the only LLM call in this module
# --------------------------------------------------------------------------- #
def _narrative_payload(region: str, crop: str, decision: dict, impact: dict,
                       signals: dict) -> dict:
    return {
        "region": region,
        "crop": config.CROPS[crop]["label"],
        "month": signals["target_str"],
        "dominant_risk": decision["dominant_risk"],
        "severity": decision["severity"],
        "risk_reasoning": decision["reason"],
        "confidence_label": decision["confidence_label"],
        "yield_impact_pct": impact.get("yield_impact_pct"),
        "yield_impact_status": impact.get("status"),
        "yield_impact_reason": impact.get("reason"),
        "yield_impact_source": impact.get("source"),
        "yield_impact_citation": impact.get("citation"),
        "yield_impact_quoted": impact.get("quoted"),
        "exposure_definition": impact.get("exposure_definition"),
        "caveat": impact.get("caveat"),
        "signals": decision["signals"],
    }


def write_narrative(payload: dict) -> tuple[str, list[str]]:
    """One Gemini call. Returns (text, warnings) — never raises on a model error."""
    from orchestrator.graph import _extract_text, get_chat_model, invoke_with_backoff

    system_prompt = config.NARRATIVE_PROMPT_PATH.read_text(encoding="utf-8")
    human = ("CROP IMPACT ASSESSMENT (already decided deterministically — explain "
             "it, do not recompute it):\n"
             + json.dumps(payload, indent=2, default=str))
    try:
        response = invoke_with_backoff(get_chat_model(),
                                       [("system", system_prompt),
                                        ("human", human)])
        return _extract_text(response.content), []
    except Exception as exc:
        return "", [f"crop-impact narrative failed: {type(exc).__name__}: {exc}"]


def verify_narrative(text: str, payload: dict) -> dict:
    """Reuse the Phase-3 grounding checker unmodified on this output shape."""
    from orchestrator.grounding import check_grounding

    if not text.strip():
        return {"grounded": False, "unverified_numbers": [], "total_checked": 0,
                "report_missing": True,
                "reason": "the narrative step produced no text"}
    return check_grounding(text, payload, [])


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #
def assess_crop_impact_core(region: str, crop: str, month: str | None,
                            with_narrative: bool) -> dict:
    """Everything except the model call, so tests can exercise it for free."""
    fconfig.check_region(region)
    config.check_crop(crop)

    spec = config.CROPS[crop]
    if region not in spec["regions"]:
        supported = ", ".join(spec["regions"])
        return {
            "region": region, "crop": crop, "month": None,
            "dominant_risk": "insufficient data",
            "risk_reasoning": (
                f"{spec['label']} is not in scope for {region} in this system — "
                f"it is assessed for {supported} only. No assessment is offered "
                f"rather than one that assumes the crop is grown here."),
            "yield_impact_pct": None,
            "yield_impact_source": None,
            "yield_impact_status": "not in scope for this region",
            "confidence_label": "not applicable — crop/region combination out of scope",
            "narrative": "", "grounding": {}, "warnings": [],
        }

    signals = collect_signals(region, crop, month)
    decision = dominant_risk(
        drought=signals["drought"], heat=signals["heat"], crop=crop,
        horizon=max(signals["horizon"], 1), target_month=signals["target_str"],
        month_number=signals["target"].month,
        drought_unavailable_reason=signals["drought_reason"])

    exposure = {"mean_tmax_departure_c":
                (signals["heat"] or {}).get("mean_tmax_departure_c")}
    impact = lookup_yield_impact(crop, decision["dominant_risk"],
                                 decision["severity"], exposure)

    payload = _narrative_payload(region, crop, decision, impact, signals)
    result = {
        "region": region,
        "crop": crop,
        "month": signals["target_str"],
        "dominant_risk": decision["dominant_risk"],
        "risk_reasoning": decision["reason"],
        "yield_impact_pct": impact.get("yield_impact_pct"),
        "yield_impact_source": impact.get("source"),
        "yield_impact_citation": impact.get("citation"),
        "yield_impact_status": impact.get("status"),
        "yield_impact_reason": impact.get("reason"),
        "yield_impact_caveat": impact.get("caveat"),
        "confidence_label": decision["confidence_label"],
        "signals": decision["signals"],
        "narrative": "",
        "grounding": {},
        "warnings": [],
    }

    if with_narrative:
        text, warnings = write_narrative(payload)
        result["narrative"] = text
        result["warnings"] = warnings
        result["grounding"] = verify_narrative(text, payload)
        if not result["grounding"].get("grounded", False) and text.strip():
            from orchestrator.grounding import warning_banner
            result["narrative"] = warning_banner(result["grounding"]) + "\n" + text
    return result


@tool
def assess_crop_impact(region: str = fconfig.DEFAULT_REGION,
                       crop: str = config.DEFAULT_CROP,
                       month: str | None = None) -> dict:
    """
    Assesses the impact of climate risk on a named crop in a supported Indian
    region, for a given month (defaults to the crop's most recent completed
    sensitive growth window).

    Decides deterministically which risk factor — drought or heat — is actually
    binding on yield for that crop, region and month, respecting each signal's
    measured reliability: a drought horizon with no measured skill can never
    declare dominance, and heat is an observation rather than a forecast. Yield
    impact is read from a small table of published, cited coefficients; where no
    real source exists for a crop/risk combination, it reports that no sourced
    estimate is available rather than estimating one.
    """
    return assess_crop_impact_core(region, crop, month, with_narrative=True)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    region = args[0] if len(args) > 0 else fconfig.DEFAULT_REGION
    crop = args[1] if len(args) > 1 else config.DEFAULT_CROP
    month = args[2] if len(args) > 2 else None

    out = assess_crop_impact_core(region, crop, month, with_narrative=False)
    print(json.dumps(out, indent=2, default=str))
