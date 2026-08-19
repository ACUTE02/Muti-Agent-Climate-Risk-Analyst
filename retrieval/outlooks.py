"""Type C — IMD's live outlooks, fetched fresh and never baked into the index.

These change every season, so they cannot live in a Docker-image-time index. They
are fetched at report time, kept whole rather than chunked (a caller wants "here
is what IMD is currently saying", not fragments competing for retrieval ranking),
and attributed to IMD by name.

**Never blended with this project's own numbers.** The project's SPI-3 forecast
and its observed heat wave counts are its own measured results; IMD's outlook is a
separate, separately-attributed statement. A report may carry both, side by side,
labelled.

**Failure is reported, not hidden.** If a fetch fails the caller gets
``available: False`` with a reason, so the report can say "IMD's current outlook
was unavailable" instead of silently omitting context. A cached copy is only ever
returned with its own timestamp attached and ``stale: True``, so nobody mistakes
last season's bulletin for today's.

Run standalone:  python -m retrieval.outlooks
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from retrieval import config
from retrieval.sources import extract_html_text, extract_pdf_text, fetch_bytes

MAX_EXCERPT_CHARS = 1500
STALE_AFTER_DAYS = 21          # ERF is ~2-weekly; older than this is not "current"


def _excerpt_pdf(text: str) -> str:
    """The opening of an IMD bulletin carries the headline statement."""
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:MAX_EXCERPT_CHARS]


def _excerpt_seasonal_html(text: str) -> str:
    """The seasonal page is a navigation hub; keep only its press-release lines.

    Recorded honestly rather than dressed up as a seasonal outlook: what is
    extractable here is IMD's current press-release marquee, not the seasonal
    bulletin itself, which lives in linked PDFs.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    keep = [line for line in lines
            if re.search(r"press release|forecast for the next|dated", line, re.I)
            and len(line) > 40]
    if not keep:
        keep = [line for line in lines if len(line) > 120][:2]
    return " ".join(keep)[:MAX_EXCERPT_CHARS]


def fetch_outlook(source: dict, force: bool = True) -> dict:
    """One live source. Never raises — an unavailable outlook is a reportable state."""
    record = {
        "id": source["id"],
        "title": source["title"],
        "publisher": source["publisher"],
        "citation": source["url"],
        "relevant_to": source["relevant_to"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        raw = fetch_bytes(source["url"], f"outlook_{source['id']}.{source['format']}",
                          force=force)
        if source["format"] == "pdf":
            text, stats = extract_pdf_text(raw, f"outlook_{source['id']}.pdf")
            excerpt = _excerpt_pdf(text)
        else:
            text, stats = extract_html_text(raw)
            excerpt = _excerpt_seasonal_html(text)

        if len(excerpt) < 100:
            return {**record, "available": False,
                    "reason": f"fetched but only {len(excerpt)} chars of usable "
                              "text — page structure may have changed"}
        return {**record, "available": True, "excerpt": excerpt, "stats": stats,
                "note": ("IMD's own current statement, quoted separately from this "
                         "project's measured results — not blended with them.")}
    except Exception as exc:
        return {**record, "available": False,
                "reason": f"{type(exc).__name__}: {exc}"}


def fetch_outlooks(force: bool = True) -> dict:
    """All Type C sources, plus a cached copy written for inspection."""
    outlooks = [fetch_outlook(src, force=force) for src in config.TYPE_C_SOURCES]
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outlooks": outlooks,
        "any_unavailable": any(not o["available"] for o in outlooks),
    }
    config.OUTLOOK_CACHE_PATH.write_text(json.dumps(payload, indent=2,
                                                    ensure_ascii=False),
                                         encoding="utf-8")
    return payload


if __name__ == "__main__":
    payload = fetch_outlooks()
    for outlook in payload["outlooks"]:
        state = "AVAILABLE" if outlook["available"] else "UNAVAILABLE"
        print(f"=== {outlook['id']} [{state}] relevant_to="
              f"{','.join(outlook['relevant_to'])}")
        print(f"    {outlook['citation']}")
        if outlook["available"]:
            print(f"    {outlook['excerpt'][:420]}...")
        else:
            print(f"    reason: {outlook['reason']}")
        print()
