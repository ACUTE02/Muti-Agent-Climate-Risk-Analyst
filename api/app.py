"""The local API — a thin wrapper over the real pipeline.

`POST /report` calls `orchestrator.graph.analyse()`, the exact function the
tests exercise. There is no separate demo path: if this returns a report, it
went through the real router, the real tools and the real grounding checker. The
API layer only reshapes what comes back; it never re-decides anything.

Run locally:  uvicorn api.app:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from api import quota
from api.schemas import (HealthResponse, ReportRequest, ReportResponse)
from crop_impact import config as cconfig
from evaluation import config as econfig
from forecasting import config as fconfig

app = FastAPI(
    title="Climate Risk Analyst — local API",
    version="0.6.0",
    description=(
        "Local backend for the multi-agent climate risk analyst. Wraps the "
        "LangGraph orchestrator directly. Every figure in a returned report has "
        "been checked mechanically against its source data; the grounding "
        "status travels as a structured field, not as prose."),
)

# Permissive CORS for local development only. The frontend is built separately
# with a design tool and will run on some other localhost port; which one is not
# knowable here. This is a localhost-only allowance — a public deployment must
# narrow it, and that decision belongs to the deployment phase, not this one.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Response shaping — promote honesty signals to structured fields
# --------------------------------------------------------------------------- #
def _grounding_status(state: dict) -> dict:
    grounding = state.get("grounding") or {}

    if grounding.get("report_missing"):
        return {
            "status": "not_generated", "grounded": False, "total_checked": 0,
            "unverified_numbers": [], "report_missing": True,
            "explanation": ("No report was produced, so there was nothing to "
                            "verify. Zero numbers checked is not a pass."),
        }

    unverified = list(grounding.get("unverified_numbers", []))
    checked = int(grounding.get("total_checked", 0))
    if grounding.get("grounded"):
        return {
            "status": "clean", "grounded": True, "total_checked": checked,
            "unverified_numbers": [], "report_missing": False,
            "explanation": (f"All {checked} numeric claims in this report were "
                            f"traced to a tool output or a retrieved source."),
        }
    return {
        "status": "warning", "grounded": False, "total_checked": checked,
        "unverified_numbers": unverified, "report_missing": False,
        "explanation": (
            f"{len(unverified)} of {checked} numeric claims could not be traced "
            f"to any source: {', '.join(unverified)}. They may be fabricated or "
            f"misquoted. Do not rely on the flagged values."),
    }


def _horizon_confidence(state: dict) -> list[dict]:
    """Per-horizon labels, kept distinct rather than averaged into a score."""
    drought = (state.get("tool_outputs") or {}).get("forecast_drought_risk") or {}
    rows = []
    for entry in drought.get("horizon_confidence", []):
        rows.append({
            "horizon": entry.get("horizon", ""),
            "skill_score": float(entry.get("skill_score", 0.0)),
            "method": entry.get("method", ""),
            "label": entry.get("label", ""),
            # Deliberately not a threshold on skill_score: the label is the
            # measured verdict and the boolean must follow it, not re-derive it.
            "reliable": entry.get("label") == "validated",
        })
    return rows


def _missing_data(state: dict) -> list[dict]:
    """Everything this system explicitly does not have for this request."""
    outputs = state.get("tool_outputs") or {}
    flags: list[dict] = []

    heat = outputs.get("forecast_heat_stress_risk")
    if isinstance(heat, dict) and heat.get("forecast_available") is False:
        flags.append({
            "what": "heat stress forecast",
            "available": False,
            "reason": heat.get("note", "no forecast is available for heat stress"),
        })

    crop = outputs.get("assess_crop_impact")
    if isinstance(crop, dict):
        if crop.get("yield_impact_pct") is None:
            flags.append({
                "what": f"yield-impact estimate for {crop.get('crop')}",
                "available": False,
                "reason": (crop.get("yield_impact_reason")
                           or crop.get("yield_impact_status")
                           or "no sourced yield-impact estimate available"),
            })
        if crop.get("dominant_risk") in ("insufficient data", "none dominant"):
            flags.append({
                "what": "dominant risk factor",
                "available": False,
                "reason": crop.get("risk_reasoning", ""),
            })

    for horizon in _horizon_confidence(state):
        if horizon["label"].startswith("no skill"):
            flags.append({
                "what": f"reliable drought forecast at {horizon['horizon']}",
                "available": False,
                "reason": horizon["label"],
            })

    for outlook in (state.get("type_c") or {}).get("outlooks", []):
        if not outlook.get("available"):
            flags.append({
                "what": f"IMD outlook: {outlook.get('title', 'unknown')}",
                "available": False,
                "reason": outlook.get("reason", "the fetch did not succeed"),
            })
    return flags


def _retrieved_sources(state: dict) -> list[dict]:
    return [{"source": c.get("source"), "citation": c.get("citation"),
             "section": c.get("section", ""), "score": c.get("score")}
            for c in state.get("retrieved_chunks", [])]


def build_response(state: dict, request_text: str) -> dict:
    return {
        "request": request_text,
        "report": state.get("report", ""),
        "grounding": _grounding_status(state),
        "horizon_confidence": _horizon_confidence(state),
        "missing_data": _missing_data(state),
        "tools_called": [c.get("name") for c in state.get("tool_calls", [])],
        "tool_outputs": state.get("tool_outputs", {}),
        "retrieved_sources": _retrieved_sources(state),
        "warnings": state.get("warnings", []),
        "quota": quota.status(),
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.post("/report", response_model=ReportResponse)
def create_report(body: ReportRequest) -> dict:
    """Answer one request through the real orchestrator pipeline."""
    from orchestrator.graph import analyse

    try:
        state = analyse(request=body.request, region=body.region,
                        risk_types=body.risk_types, month=body.month,
                        crop=body.crop)
    except ValueError as exc:                    # unknown region or crop
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    warnings = list(state.get("warnings", []))
    grounding = state.get("grounding") or {}

    # A quota failure must not masquerade as a generic server error, and must not
    # come back as a 200 with an empty report either.
    if grounding.get("report_missing") and quota.looks_like_quota_failure(warnings):
        raise HTTPException(status_code=429,
                            detail=quota.exhaustion_detail(warnings))

    return build_response(state, body.request)


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    """Cheap liveness check. Makes no LLM call — a test asserts that."""
    chroma_ready, chunks = False, None
    try:
        from retrieval.store import get_collection

        chunks = get_collection().count()
        chroma_ready = chunks > 0
    except Exception:
        chroma_ready, chunks = False, None

    missing = []
    for region in fconfig.REGIONS:
        for path in (fconfig.scaler_path(region), fconfig.spi_params_path(region),
                     *(fconfig.horizon_model_path(region, h) for h in (1, 2, 3))):
            if not path.exists():
                missing.append(path.name)
    if not fconfig.HORIZON_MANIFEST_PATH.exists():
        missing.append(fconfig.HORIZON_MANIFEST_PATH.name)

    from retrieval.config import API_KEY_ENV_VARS
    import os
    key_present = any(os.environ.get(v) for v in API_KEY_ENV_VARS)
    if not key_present:
        env_file = fconfig.REPO_ROOT / ".env"
        key_present = env_file.exists() and "API_KEY" in env_file.read_text(
            encoding="utf-8", errors="ignore")

    ready = chroma_ready and not missing
    return {
        "status": "ok" if ready else "degraded",
        "chroma_index_ready": chroma_ready,
        "chroma_chunks": chunks,
        "forecast_artifacts_ready": not missing,
        "missing_artifacts": sorted(set(missing)),
        "api_key_present": bool(key_present),
        "quota": quota.status(),
        "note": ("'degraded' means the app is up but regenerable artifacts are "
                 "absent — run `python -m scripts.setup`. See "
                 "SETUP_FROM_CLEAN.md."),
    }


@app.get("/evaluation")
def evaluation() -> dict:
    """How good is this system — served without running a new evaluation."""
    if not econfig.SCORECARD_PATH.exists():
        raise HTTPException(status_code=404, detail="EVALUATION.md is not present")

    summary = {}
    if econfig.CHECKER_EVAL_PATH.exists():
        checker = json.loads(econfig.CHECKER_EVAL_PATH.read_text(encoding="utf-8"))
        summary["grounding_checker"] = {
            "precision": checker["precision"], "recall": checker["recall"],
            "f1": checker["f1"], "cases": checker["cases"],
            "known_defect": ("An invented integer percentage below 50% is "
                             "currently accepted when the sources contain a "
                             "zero. Measured, documented, not yet fixed."),
        }
    if econfig.FAITHFULNESS_PATH.exists():
        faith = json.loads(econfig.FAITHFULNESS_PATH.read_text(encoding="utf-8"))
        summary["faithfulness"] = faith["summary"] | {
            "set_size": faith["set_size"], "trust_note": faith["trust_note"]}

    return {"markdown": econfig.SCORECARD_PATH.read_text(encoding="utf-8"),
            "summary": summary}


@app.get("/evaluation.md", response_class=PlainTextResponse)
def evaluation_markdown() -> str:
    """The scorecard as raw Markdown, for a client that renders it directly."""
    if not econfig.SCORECARD_PATH.exists():
        raise HTTPException(status_code=404, detail="EVALUATION.md is not present")
    return econfig.SCORECARD_PATH.read_text(encoding="utf-8")


@app.get("/examples")
def examples() -> dict:
    """Ready-made request bodies, so a client need not hardcode them.

    Sourced from the scenarios the test suite and the Phase-5 held-out set
    already use, rather than invented for the demo — what is shown is what is
    actually tested.
    """
    items = [
        {"id": "drought_reliability",
         "label": "Drought risk for Barmer, with reliability",
         "body": {"request": "What is the drought risk for Barmer over the next "
                             "three months, and how much should I trust each "
                             "month's number?"},
         "shows": "Per-horizon skill labels, including a 'no skill' horizon."},
        {"id": "both_risks",
         "label": "Drought and heat for Rajasthan",
         "body": {"request": "Give me the drought and heat stress risk for "
                             "Rajasthan, and say how reliable each one is."},
         "shows": "Heat returning observations only, with no forecast."},
        {"id": "crop_impact_wheat",
         "label": "Heat impact on wheat, Feb 2006",
         "body": {"request": "How did heat affect wheat in Rajasthan in February "
                             "2006, and what does it mean for yield?",
                  "crop": "wheat", "month": "2006-02"},
         "shows": "The one sourced yield coefficient, with its caveat."},
        {"id": "crop_impact_bajra",
         "label": "Drought impact on bajra",
         "body": {"request": "What is the drought impact on bajra in Barmer?",
                  "crop": "bajra", "region": "barmer"},
         "shows": "An explicit 'no sourced yield-impact estimate available'."},
        {"id": "impossible_request",
         "label": "Something the system cannot know",
         "body": {"request": "Give me the six-month drought forecast for "
                             "Jaisalmer district and the expected percentage "
                             "yield loss for mustard there."},
         "shows": "Three unsupported asks, each declined rather than invented."},
    ]
    return {
        "examples": items,
        "supported_regions": sorted(fconfig.REGIONS),
        "supported_crops": sorted(cconfig.CROPS),
        "cost_warning": (
            f"Each example costs {quota.TYPICAL_CALLS_PER_REPORT}-"
            f"{quota.WORST_CASE_CALLS_PER_REPORT} Gemini calls against a "
            f"{quota.DAILY_CALL_BUDGET}/day free-tier budget."),
        "quota": quota.status(),
    }


@app.get("/quota")
def quota_status() -> dict:
    """What is left today. No LLM call."""
    return quota.status()
