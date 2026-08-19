"""Bring the live inputs up to date, so a forecast answers about *now*.

The fixed 1980-2024 archive is what every published skill score was measured
against, so it is never touched. This script only ever writes the rolling
`_recent` caches and re-pulls the ONI series, both of which are inputs to live
inference and to nothing that was evaluated.

Nothing is retrained. The Ridge coefficients are the ones the evaluation
measured; only the months they are applied to move forward.

    python -m scripts.refresh              # every region
    python -m scripts.refresh barmer       # just one

Run it before a demo. Costs no API key and no Gemini quota — Open-Meteo and NOAA
are both free and unauthenticated.
"""

from __future__ import annotations

import json
import sys

from forecasting import config
from forecasting.enso import fetch_oni
from forecasting.fetch_data import data_currency, refresh_recent


def refresh_all(regions: list[str] | None = None) -> dict:
    """Refresh weather for each region plus the shared ONI series."""
    regions = regions or list(config.REGIONS)

    # ONI first: it is the input most likely to be the binding constraint on how
    # current a forecast can be, and it is shared across regions.
    oni_result: dict = {}
    try:
        series = fetch_oni(force=True)
        oni_result = {"ok": True, "through": str(series.dropna().index.max().date())}
    except Exception as exc:
        # A failed ONI refresh leaves the cached series in place; the forecast
        # still runs, just anchored further back. Reported, never silent.
        oni_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "note": "kept the previously cached ONI series"}

    per_region = {}
    for region in regions:
        try:
            per_region[region] = refresh_recent(region)
        except Exception as exc:
            per_region[region] = {"ok": False,
                                  "error": f"{type(exc).__name__}: {exc}",
                                  **data_currency(region)}
    return {"oni": oni_result, "regions": per_region}


def main(argv: list[str]) -> int:
    requested = [r for r in argv if not r.startswith("-")]
    for region in requested:
        config.check_region(region)          # fail loudly before any network call

    result = refresh_all(requested or None)
    print(json.dumps(result, indent=2))

    failures = [r for r, v in result["regions"].items() if v.get("ok") is False]
    if not result["oni"].get("ok"):
        print(f"\nONI refresh failed: {result['oni'].get('error')}", file=sys.stderr)
    if failures:
        print(f"\nFailed regions: {', '.join(failures)}", file=sys.stderr)
        return 1

    print("\nForecasts are now anchored to:")
    for region, info in result["regions"].items():
        print(f"  {region:12s} {info['data_current_through']} "
              f"({info['months_behind_today']} month(s) behind today; "
              f"limited by {info['limiting_input']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
