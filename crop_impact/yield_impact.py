"""Sourced yield-impact lookup — a table read, never a calculation.

The table lives in ``crop_impact/yield_impact_table.json`` with a citation on
every entry, so a reader can check the number against the source and the
grounding checker can match it as a literal string. This module does not compute,
interpolate or scale anything: if the table has no entry whose exposure band the
observed conditions actually fall into, the answer is "no sourced yield-impact
estimate available", the same posture as the Heat Agent's ``forecast_available:
False``.

Nothing here calls a model either. The narrative step is handed a number that
already exists; it is never asked to produce one.
"""

from __future__ import annotations

import json
from functools import lru_cache

from crop_impact import config

NOT_AVAILABLE = "no sourced yield-impact estimate available"


@lru_cache(maxsize=1)
def load_table() -> dict:
    return json.loads(config.YIELD_TABLE_PATH.read_text(encoding="utf-8"))


def _band_ok(band: dict, exposure: dict) -> tuple[bool, str]:
    """Does the observed exposure fall inside this coefficient's band?

    A coefficient measured under one exposure definition does not transfer to a
    different one for free, so the band is enforced rather than assumed.
    """
    metric = band.get("metric")
    value = (exposure or {}).get(metric)
    if value is None:
        return False, (f"the coefficient is conditioned on {metric}, which was "
                       f"not measured for this request")

    low, high = band.get("min"), band.get("max")
    if low is not None and value < low:
        return False, (f"measured {metric} of {value:.2f} is below the "
                       f"coefficient's band (>= {low}), so no estimate is "
                       f"reported rather than a scaled-down one")
    if high is not None and value > high:
        return False, (f"measured {metric} of {value:.2f} is above the "
                       f"coefficient's band (<= {high})")
    return True, f"measured {metric} of {value:.2f} falls in the coefficient's band"


def find_gap(crop: str, risk: str) -> dict | None:
    for gap in load_table().get("gaps", []):
        if gap["crop"] == crop and gap["risk"] == risk:
            return gap
    return None


def lookup_yield_impact(crop: str, risk: str, severity: str | None,
                        exposure: dict | None = None) -> dict:
    """The sourced coefficient for this crop/risk/severity, or an explicit gap.

    Never returns a number the table does not literally contain.
    """
    config.check_crop(crop)

    if risk in ("none dominant", "insufficient data") or severity is None:
        return {
            "available": False,
            "yield_impact_pct": None,
            "status": NOT_AVAILABLE,
            "reason": ("no single risk factor was found to be binding, so there "
                       "is no crop/risk combination to look up"),
        }

    severities_on_file = []
    for entry in load_table().get("coefficients", []):
        if (entry["crop"], entry["risk"]) != (crop, risk):
            continue
        if entry.get("severity") and entry["severity"] != severity:
            severities_on_file.append(entry["severity"])
            continue

        ok, why = _band_ok(entry.get("match_band", {}), exposure or {})
        if not ok:
            return {
                "available": False,
                "yield_impact_pct": None,
                "status": NOT_AVAILABLE,
                "reason": (f"a sourced coefficient exists for {crop}/{risk} but "
                           f"does not apply here: {why}"),
                "source": entry["source"],
                "citation": entry["citation"],
            }

        return {
            "available": True,
            "yield_impact_pct": entry["yield_impact_pct"],
            "status": "sourced",
            "reason": why,
            "coefficient_id": entry["id"],
            "source": entry["source"],
            "citation": entry["citation"],
            "quoted": entry["quoted"],
            "exposure_definition": entry["exposure_definition"],
            "caveat": entry["caveat"],
            "application": entry["application"],
        }

    if severities_on_file:
        have = ", ".join(sorted(set(severities_on_file)))
        return {
            "available": False,
            "yield_impact_pct": None,
            "status": NOT_AVAILABLE,
            "reason": (f"the table's only sourced coefficient for {crop}/{risk} "
                       f"was measured at {have} severity, and this month is "
                       f"{severity}. No number is reported rather than one "
                       f"scaled down from a different severity band."),
        }

    gap = find_gap(crop, risk)
    if gap:
        return {
            "available": False,
            "yield_impact_pct": None,
            "status": gap["status"],
            "reason": gap["why_nothing_qualified"],
            "searched": gap["searched"],
        }

    return {
        "available": False,
        "yield_impact_pct": None,
        "status": NOT_AVAILABLE,
        "reason": (f"the yield-impact table has neither a coefficient nor a "
                   f"recorded gap for {crop}/{risk} — it is outside the scope "
                   f"this phase sourced"),
    }
