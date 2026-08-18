"""Which risk is actually binding on yield — decided deterministically.

**No LLM call appears in this module, by design.** This is the piece the project's
standing scope decision is actually asking for ("determine which risk factor is
actually dominant before calculating yield loss"), and it is the kind of judgement
a language model will answer confidently from general knowledge whether or not it
has grounds to. Same discipline as ``orchestrator/grounding.py``: the decisions
that must be trustworthy are plain Python, and a test asserts this file contains
no model call.

The rule itself is written up in ``crop_impact/dominance_rule.md`` so it is
reviewable without reading code. Keep the two in sync.

The property that matters most here: **a horizon with no measured skill can never
declare drought dominant.** The Drought Agent's t+3 forecast is labelled "no skill
— shown for context only, do not rely on this figure", and letting it drive a crop
assessment would launder a number this project has spent five phases establishing
is worthless.
"""

from __future__ import annotations

from forecasting import config as fconfig
from crop_impact import config

NO_SKILL_PREFIX = "no skill"


# --------------------------------------------------------------------------- #
# Signal extraction
# --------------------------------------------------------------------------- #
def _horizon_entry(drought: dict, horizon: int) -> dict | None:
    label = f"t+{horizon}"
    for entry in drought.get("horizon_confidence", []):
        if entry.get("horizon") == label:
            return entry
    return None


def _spi_severity(spi: float) -> str | None:
    """Drought severity from predicted SPI-3, using the project's own thresholds."""
    if spi < fconfig.SPI_SEVERE:
        return "severe"
    if spi < fconfig.SPI_MODERATE:
        return "moderate"
    return None


def drought_signal(drought: dict | None, horizon: int,
                   unavailable_reason: str | None = None) -> dict:
    """Normalise the drought tool's output for one horizon.

    ``decisive`` is the gate: only a horizon this project has actually validated
    may declare drought dominant on its own. A "weak/directional" horizon can
    corroborate, and a "no skill" horizon is ignored for dominance entirely — it
    is still reported, so a reader can see it was considered and discarded.

    ``unavailable_reason`` lets a caller that already knows why there is no
    forecast (a month in the past, or beyond t+3) say so in the reasoning,
    instead of the reader getting a bare "not provided".
    """
    if not drought:
        return {"available": False,
                "reason": unavailable_reason or "no drought forecast was provided"}

    entry = _horizon_entry(drought, horizon)
    values = drought.get("predicted_values") or []
    if entry is None or horizon < 1 or horizon > len(values):
        return {"available": False,
                "reason": f"the drought forecast carries no t+{horizon} horizon"}

    label = str(entry.get("label", ""))
    spi = float(values[horizon - 1])
    return {
        "available": True,
        "horizon": f"t+{horizon}",
        "predicted_spi3": spi,
        "label": label,
        "skill_score": entry.get("skill_score"),
        "severity": _spi_severity(spi),
        "decisive": label == config.DECISIVE_FORECAST_LABEL,
        "corroborating": label == config.CORROBORATING_FORECAST_LABEL,
        "no_skill": label.startswith(NO_SKILL_PREFIX),
    }


def heat_signal(heat: dict | None, target_month: str | None) -> dict:
    """Normalise the heat tool's output.

    The Heat Agent does not forecast — Phases 1 and 1.1 established it has no
    skill at any horizon, and its tool returns observations only. So this reads
    as an observation and says so. If the month being asked about is not the
    month that was observed, heat is **unknown** for that month, never inferred.
    """
    if not heat:
        return {"available": False, "observed": False,
                "reason": "no heat observation was provided"}

    observed_month = heat.get("month")
    if target_month and observed_month and target_month != observed_month:
        return {
            "available": False, "observed": False, "month": observed_month,
            "reason": (f"heat is observed only, and the observation is for "
                       f"{observed_month}, not {target_month} — heat stress for "
                       f"{target_month} is unknown, because this system has no "
                       f"heat forecast"),
        }

    days = int(heat.get("heatwave_days", 0))
    severe_days = int(heat.get("severe_heatwave_days", 0))
    departure = heat.get("mean_tmax_departure_c")

    # Two independent routes to candidacy — see config.HEAT_DEPARTURE_MODERATE_C
    # for why the day counter alone is not enough for a Feb/Mar crop.
    severity, route = None, None
    if days >= config.HEAT_CANDIDATE_MIN_DAYS:
        severity = "severe" if severe_days >= 1 else "moderate"
        route = "IMD heat wave day count"
    if departure is not None:
        by_departure = None
        if departure >= config.HEAT_DEPARTURE_SEVERE_C:
            by_departure = "severe"
        elif departure >= config.HEAT_DEPARTURE_MODERATE_C:
            by_departure = "moderate"
        if by_departure and (severity is None
                             or config.SEVERITY_RANK[by_departure]
                             > config.SEVERITY_RANK[severity]):
            severity, route = by_departure, "monthly mean Tmax departure"

    return {
        "available": True,
        "observed": True,                 # never a forecast — see docstring
        "month": observed_month,
        "heatwave_days": days,
        "severe_heatwave_days": severe_days,
        "had_heatwave_spell": bool(heat.get("had_heatwave_spell", False)),
        "max_tmax_c": heat.get("max_tmax_c"),
        "mean_tmax_departure_c": departure,
        "severity": severity,
        "severity_route": route,
    }


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def dominant_risk(drought: dict | None, heat: dict | None, crop: str,
                  horizon: int = 1, target_month: str | None = None,
                  month_number: int | None = None,
                  drought_unavailable_reason: str | None = None) -> dict:
    """Decide which risk binds yield for ``crop``, and say why.

    Pure function: tool output dicts in, a decision dict out. No I/O, no model
    call, no global state — so the unit tests drive it with fixtures rather than
    a live pipeline.

    Returns ``dominant_risk`` as one of ``"drought"``, ``"heat"``,
    ``"none dominant"`` or ``"insufficient data"``.
    """
    config.check_crop(crop)
    spec = config.CROPS[crop]

    drought_sig = drought_signal(drought, horizon, drought_unavailable_reason)
    heat_sig = heat_signal(heat, target_month)
    signals = {"drought": drought_sig, "heat": heat_sig}

    def result(risk: str, reason: str, severity: str | None,
               confidence: str) -> dict:
        return {"dominant_risk": risk, "reason": reason, "severity": severity,
                "confidence_label": confidence, "crop": crop,
                "sensitive_stage": spec["sensitive_stage"], "signals": signals}

    # 1. Season gate. A shock in a month the crop is not vulnerable in does not
    #    bind its yield, however severe the shock is.
    if month_number is not None and month_number not in spec["sensitive_months"]:
        window = "/".join(str(m) for m in spec["sensitive_months"])
        return result(
            "none dominant",
            (f"month {month_number} is outside {spec['label']}'s sensitive "
             f"window (months {window}, {spec['sensitive_stage']}), so no "
             f"climate signal is treated as binding on yield"),
            None, "not applicable — outside the crop's sensitive window")

    # 2. Candidacy. Drought needs a horizon this project actually validated;
    #    heat needs an observation for the month in question.
    drought_ok = bool(drought_sig.get("available")
                      and drought_sig.get("severity")
                      and drought_sig.get("decisive"))
    heat_ok = bool(heat_sig.get("available") and heat_sig.get("severity"))

    # 3. Insufficient data — nothing usable came back from either side.
    usable_drought = drought_sig.get("available") and not drought_sig.get("no_skill")
    if not usable_drought and not heat_sig.get("available"):
        why = []
        if drought_sig.get("no_skill"):
            why.append(f"the only drought signal for this month is "
                       f"{drought_sig['horizon']}, labelled "
                       f"\"{drought_sig['label']}\", which this project does not "
                       f"allow to drive a decision")
        else:
            why.append(drought_sig.get("reason", "no usable drought signal"))
        why.append(heat_sig.get("reason", "no usable heat signal"))
        return result("insufficient data", "; ".join(why), None,
                      "insufficient data — no signal met the bar to be used")

    # 4. Both candidates: rank by severity, and break a tie toward the observed
    #    signal. An observation is a fact; a forecast is an estimate, so when the
    #    two look equally severe the fact wins.
    if drought_ok and heat_ok:
        d_rank = config.SEVERITY_RANK[drought_sig["severity"]]
        h_rank = config.SEVERITY_RANK[heat_sig["severity"]]
        if d_rank > h_rank:
            return _drought_result(result, drought_sig, spec)
        if h_rank > d_rank:
            return _heat_result(result, heat_sig, spec)
        return _heat_result(
            result, heat_sig, spec,
            extra=(f" Drought was equally severe ({drought_sig['severity']}, "
                   f"SPI-3 {drought_sig['predicted_spi3']:.2f} at "
                   f"{drought_sig['horizon']}) but heat is an observation and "
                   f"the drought figure is a forecast, so the observed signal "
                   f"is taken as dominant."))

    if drought_ok:
        return _drought_result(result, drought_sig, spec)
    if heat_ok:
        return _heat_result(result, heat_sig, spec)

    # 5. Signals exist, nothing crosses the bar.
    return result("none dominant", _none_reason(drought_sig, heat_sig), None,
                  "no risk factor reached the threshold to be called dominant")


def _drought_result(result, sig: dict, spec: dict) -> dict:
    reason = (
        f"drought is binding: forecast SPI-3 {sig['predicted_spi3']:.2f} at "
        f"{sig['horizon']} is {sig['severity']} on this project's thresholds "
        f"(moderate below {fconfig.SPI_MODERATE}, severe below "
        f"{fconfig.SPI_SEVERE}), and that horizon is labelled "
        f"\"{sig['label']}\", the only label allowed to decide dominance. "
        f"{spec['label']} is most vulnerable during {spec['sensitive_stage']}.")
    return result("drought", reason, sig["severity"], sig["label"])


def _heat_result(result, sig: dict, spec: dict, extra: str = "") -> dict:
    if sig.get("severity_route") == "monthly mean Tmax departure":
        evidence = (f"observed mean daily maximum temperature in {sig['month']} "
                    f"ran {sig['mean_tmax_departure_c']:+.2f} C against the "
                    f"1981-2010 normal")
    else:
        evidence = (f"{sig['heatwave_days']} observed IMD heat wave day(s) in "
                    f"{sig['month']}"
                    + (f", of which {sig['severe_heatwave_days']} severe"
                       if sig["severe_heatwave_days"] else ""))
    reason = (
        f"heat is binding: {evidence}. This is an observation, not a forecast — "
        f"this system has no heat forecast at all. {spec['label']} is most "
        f"vulnerable during {spec['sensitive_stage']}." + extra)
    return result("heat", reason, sig["severity"],
                  "observed — not a forecast (the Heat Agent has no forecast skill)")


def _none_reason(drought_sig: dict, heat_sig: dict) -> str:
    parts = []
    if drought_sig.get("available"):
        if drought_sig.get("no_skill"):
            parts.append(
                f"the {drought_sig['horizon']} drought forecast is labelled "
                f"\"{drought_sig['label']}\" and is therefore not allowed to "
                f"declare dominance, whatever value it carries "
                f"(SPI-3 {drought_sig['predicted_spi3']:.2f})")
        elif not drought_sig.get("severity"):
            parts.append(
                f"forecast SPI-3 {drought_sig['predicted_spi3']:.2f} at "
                f"{drought_sig['horizon']} is above the moderate-drought "
                f"threshold of {fconfig.SPI_MODERATE}")
        elif not drought_sig.get("decisive"):
            parts.append(
                f"the {drought_sig['horizon']} drought forecast is "
                f"{drought_sig['severity']} (SPI-3 "
                f"{drought_sig['predicted_spi3']:.2f}) but is labelled "
                f"\"{drought_sig['label']}\", which may corroborate but may not "
                f"decide")
    else:
        parts.append(drought_sig.get("reason", "no drought signal"))

    if heat_sig.get("available"):
        departure = heat_sig.get("mean_tmax_departure_c")
        observed = (f"{heat_sig['heatwave_days']} observed heat wave day(s) in "
                    f"{heat_sig['month']}")
        if departure is not None:
            observed += f" and a mean Tmax departure of {departure:+.2f} C"
        parts.append(
            f"{observed} is below both heat candidacy thresholds "
            f"({config.HEAT_CANDIDATE_MIN_DAYS} heat wave days, or "
            f"{config.HEAT_DEPARTURE_MODERATE_C} C departure)")
    else:
        parts.append(heat_sig.get("reason", "no heat signal"))

    return "; ".join(parts)
