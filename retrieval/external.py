"""Phase 8 Part 4 — live external sources, cited by name and never absorbed.

Same contract as ``retrieval/outlooks.py`` (IMD), and for the same reason: these
are *other people's* measurements. A report may put them beside this project's
own figures, but must always say whose number is whose, and this project must
never present a fetched value as something it computed.

Two sources live here:

* **NASA POWER** (``power.larc.nasa.gov``) — free, no key, no registration.
  Satellite-derived agrometeorology: solar radiation, evapotranspiration energy
  flux, humidity, wind. It speaks to irrigation demand, which this project's own
  tools cannot address at all — the "no irrigation-demand estimate available"
  gap recorded in earlier phases.

* **data.gov.in** — India's Open Government Data platform, Ministry of
  Agriculture & Farmers Welfare mandi (market) prices, updated daily. Free, but
  needs a free API key. Absent key is handled exactly like a missing Gemini key:
  the source reports itself unavailable with a reason, and nothing crashes.

Three rules this module exists to enforce:

1. **Fill values are not measurements.** NASA POWER writes ``-999.0`` for a day
   it has no value for. Reporting that as a temperature would be inventing data
   of the worst kind, so it is filtered before anything else happens.
2. **No derived quantities are passed off as source values.** Where a period
   mean or total is reported, the record says so in words, and the underlying
   units are carried through unchanged (POWER's evapotranspiration is an energy
   flux in MJ/m^2/day — it is *not* silently converted to mm).
3. **Failure is a reportable state, never a substitution.** Every fetch returns
   ``available: False`` with a reason rather than raising or, far worse, quietly
   returning a plausible number.

Run standalone:  python -m retrieval.external [region] [crop]
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import requests

from forecasting import config as fconfig

REQUEST_TIMEOUT = 60
RETRIES = 3

# data.gov.in intermittently returns 502s and connection timeouts even for a
# valid key — observed repeatedly while building this. Hence the retry loop, and
# hence the graceful-degradation path being load-bearing rather than decorative.
USER_AGENT = {"User-Agent": "multi-agent-climate-risk-analyst/0.8 (research)"}

# --------------------------------------------------------------------------- #
# NASA POWER
# --------------------------------------------------------------------------- #
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_FILL = -999.0
# (long name, units after aggregation, how). Summing a per-day rate gives a
# period amount, so precipitation's reported unit is mm, not mm/day — carrying
# "/day" onto a monthly total would misstate the quantity by a factor of ~30.
POWER_PARAMS = {
    "ALLSKY_SFC_SW_DWN": ("all-sky surface shortwave irradiance", "MJ/m^2/day", "mean"),
    "T2M": ("temperature at 2 m", "C", "mean"),
    "PRECTOTCORR": ("corrected precipitation", "mm", "total"),
    "EVPTRNS": ("evapotranspiration energy flux", "MJ/m^2/day", "mean"),
    "RH2M": ("relative humidity at 2 m", "%", "mean"),
    "WS2M": ("wind speed at 2 m", "m/s", "mean"),
}
POWER_CITATION = ("NASA POWER (Prediction of Worldwide Energy Resources), "
                  "https://power.larc.nasa.gov/")


def _base_record(source_id: str, title: str, publisher: str, citation: str,
                 relevant_to: list[str]) -> dict:
    return {
        "id": source_id,
        "title": title,
        "publisher": publisher,
        "citation": citation,
        "relevant_to": relevant_to,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _month_bounds(month: str | None) -> tuple[str, str, str]:
    """(start, end, label) as YYYYMMDD for a 'YYYY-MM' month, defaulting to last
    complete month. POWER lags a few days, so the running month is never used."""
    import pandas as pd

    if month:
        start = pd.Timestamp(month).replace(day=1)
    else:
        start = (pd.Timestamp.now(tz="UTC").tz_localize(None).replace(day=1)
                 - pd.DateOffset(months=1))
    end = start + pd.offsets.MonthEnd(1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), start.strftime("%Y-%m")


def fetch_nasa_power(region: str, month: str | None = None) -> dict:
    """Agrometeorology for one region-month. Never raises."""
    record = _base_record(
        "nasa_power", "NASA POWER agrometeorology (daily, aggregated to the month)",
        "NASA Langley Research Center", POWER_CITATION, ["drought", "crop_impact"])
    try:
        fconfig.check_region(region)
        meta = fconfig.REGIONS[region]
        start, end, label = _month_bounds(month)
        record["period"] = label
        record["location"] = f"{meta['label']} ({meta['lat']}, {meta['lon']})"

        params = {
            "parameters": ",".join(POWER_PARAMS),
            "community": "AG",
            "latitude": meta["lat"], "longitude": meta["lon"],
            "start": start, "end": end, "format": "JSON",
        }
        response = requests.get(POWER_URL, params=params,
                                timeout=REQUEST_TIMEOUT, headers=USER_AGENT)
        if response.status_code != 200:
            return {**record, "available": False,
                    "reason": f"NASA POWER returned HTTP {response.status_code}"}

        block = response.json().get("properties", {}).get("parameter", {})
        values, lines = {}, []
        for name, (long_name, units, how) in POWER_PARAMS.items():
            daily = block.get(name, {})
            # Rule 1: -999.0 is "no value", not a reading.
            usable = [v for v in daily.values() if v != POWER_FILL]
            if not usable:
                continue
            agg = round(sum(usable), 2) if how == "total" else round(
                sum(usable) / len(usable), 2)
            values[name] = {"value": agg, "units": units, "aggregation": how,
                            "days_used": len(usable), "days_returned": len(daily)}
            lines.append(f"{long_name}: monthly {how} {agg} {units} "
                         f"(from {len(usable)} NASA POWER daily values)")

        if not values:
            return {**record, "available": False,
                    "reason": (f"NASA POWER returned no usable values for {label} "
                               "(all days were fill values)")}

        record["values"] = values
        # The excerpt is what the grounding checker reads, so every number the
        # report may quote has to appear here in plain text.
        record["excerpt"] = (
            f"NASA POWER daily agrometeorology for {record['location']}, {label}, "
            f"aggregated over the month: " + "; ".join(lines) + ". "
            "Evapotranspiration is reported by NASA POWER as an energy flux in "
            "MJ/m^2/day and is not converted here."
        )
        record["note"] = ("NASA POWER's own satellite-derived values, summarised "
                          "over the month. Not a measurement by this project, and "
                          "not blended with this project's SPI-3 forecast.")
        return {**record, "available": True}
    except Exception as exc:
        return {**record, "available": False,
                "reason": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# data.gov.in — mandi prices
# --------------------------------------------------------------------------- #
DATA_GOV_KEY_ENV_VARS = ("DATA_GOV_IN_API_KEY", "DATA_GOV_API_KEY")
MANDI_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
MANDI_URL = f"https://api.data.gov.in/resource/{MANDI_RESOURCE_ID}"
MANDI_CITATION = (
    "data.gov.in, 'Current Daily Price of Various Commodities from Various "
    "Markets (Mandi)', Ministry of Agriculture and Farmers Welfare — "
    f"https://api.data.gov.in/resource/{MANDI_RESOURCE_ID}")

# The commodity strings the portal actually uses, which are not the crop names
# this project uses internally. Mapped explicitly rather than guessed at call time.
MANDI_COMMODITY = {
    "bajra": "Bajra(Pearl Millet/Cumbu)",
    "wheat": "Wheat",
}
# Which state a supported region sits in. Both current regions are Rajasthan;
# kept as a map so adding a region does not silently inherit the wrong state.
REGION_STATE = {"rajasthan": "Rajasthan", "barmer": "Rajasthan"}


def get_data_gov_key() -> str | None:
    """The data.gov.in key, or None. Absence is a reportable state, not an error.

    Mirrors retrieval.embed.get_api_key's .env handling so a key in a gitignored
    .env works the same way the Gemini key already does.
    """
    from retrieval.embed import _load_dotenv_if_present

    _load_dotenv_if_present()
    for var in DATA_GOV_KEY_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


def fetch_mandi_prices(region: str, crop: str | None = None) -> dict:
    """Today's official mandi prices for a crop in the region's state."""
    record = _base_record(
        "data_gov_in_mandi", "data.gov.in daily mandi prices",
        "Ministry of Agriculture and Farmers Welfare (via data.gov.in)",
        MANDI_CITATION, ["crop_impact"])

    key = get_data_gov_key()
    if not key:
        return {**record, "available": False,
                "reason": ("no data.gov.in API key is set — set "
                           "DATA_GOV_IN_API_KEY (a free key comes from "
                           "https://data.gov.in/apis). The rest of the report is "
                           "unaffected.")}

    if crop is None:
        return {**record, "available": False,
                "reason": "no crop was part of this request, so no market price "
                          "was looked up"}
    commodity = MANDI_COMMODITY.get(crop)
    if commodity is None:
        return {**record, "available": False,
                "reason": f"no data.gov.in commodity mapping exists for {crop!r}"}

    state = REGION_STATE.get(region, "Rajasthan")
    record.update(crop=crop, commodity=commodity, state=state)
    params = {"api-key": key, "format": "json", "limit": 20,
              "filters[state]": state, "filters[commodity]": commodity}

    last_error = "unknown"
    for attempt in range(RETRIES):
        try:
            response = requests.get(MANDI_URL, params=params,
                                    timeout=REQUEST_TIMEOUT, headers=USER_AGENT)
            if response.status_code == 403:
                return {**record, "available": False,
                        "reason": "data.gov.in rejected the API key (HTTP 403)"}
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                time.sleep(2 * (attempt + 1))
                continue

            records = response.json().get("records", [])
            if not records:
                return {**record, "available": False,
                        "reason": (f"data.gov.in reported no {commodity} prices for "
                                   f"{state} today — not all markets report daily")}

            rows = [{
                "market": r.get("market"), "district": r.get("district"),
                "modal_price_rs_per_quintal": r.get("modal_price"),
                "min_price_rs_per_quintal": r.get("min_price"),
                "max_price_rs_per_quintal": r.get("max_price"),
                "arrival_date": r.get("arrival_date"),
            } for r in records]
            record["markets"] = rows

            listed = "; ".join(
                f"{r['market']} ({r['district']}) modal Rs {r['modal_price_rs_per_quintal']}"
                f"/quintal on {r['arrival_date']}" for r in rows[:6])
            record["excerpt"] = (
                f"data.gov.in daily mandi prices for {commodity} in {state}, "
                f"as published by the Ministry of Agriculture and Farmers Welfare: "
                f"{listed}. {len(rows)} market(s) reported."
            )
            record["note"] = ("Official market prices published by data.gov.in. "
                              "Market prices, not a yield estimate — this project "
                              "does not derive yield impact from them.")
            return {**record, "available": True}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))

    return {**record, "available": False,
            "reason": f"data.gov.in unreachable after {RETRIES} attempts "
                      f"({last_error})"}


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def fetch_external_sources(region: str | None = None, crop: str | None = None,
                           month: str | None = None) -> dict:
    """Every non-IMD live source for one request. Never raises."""
    region = region or fconfig.DEFAULT_REGION
    sources = [fetch_nasa_power(region, month), fetch_mandi_prices(region, crop)]
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
        "any_unavailable": any(not s["available"] for s in sources),
    }


if __name__ == "__main__":
    import json
    import sys

    args = sys.argv[1:]
    region = args[0] if args else fconfig.DEFAULT_REGION
    crop = args[1] if len(args) > 1 else "bajra"
    payload = fetch_external_sources(region, crop)
    for source in payload["sources"]:
        state = "AVAILABLE" if source["available"] else "UNAVAILABLE"
        print(f"=== {source['id']} [{state}]")
        print(f"    {source['citation']}")
        print(f"    {source.get('excerpt', source.get('reason'))}\n")
    print(json.dumps(payload, indent=2)[:1500])
