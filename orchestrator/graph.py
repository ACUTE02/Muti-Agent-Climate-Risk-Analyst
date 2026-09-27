"""The LangGraph orchestrator: route a request to tools, then write the report.

    parse_request -> call_tools -> fetch_type_c -> synthesize -> verify_grounding

Kept linear and inspectable. No reflection loops or multi-agent debate in this
first pass — the one cycle in the graph is a single regeneration when the
grounding checker rejects a report.

Tool routing is done with Gemini's function calling rather than hand-written
keyword matching: the model is given the real tools and decides which to
invoke. A deterministic fallback covers the case where the model returns no tool
calls at all, so a transient LLM failure degrades to "call the obvious tools"
rather than to an empty report.

Run standalone:  python -m orchestrator.graph "drought risk for Barmer"
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from crop_impact import config as cconfig
from crop_impact.tool import assess_crop_impact
from forecasting import config as fconfig
from forecasting.tool import forecast_drought_risk
from heat.tool import forecast_heat_stress_risk
from orchestrator import config
from orchestrator.grounding import (check_grounding, log_failure,
                                    warning_banner)
from retrieval import config as rconfig
from retrieval.external import fetch_external_sources
from retrieval.live import gather_with_deadline
from retrieval.outlooks import fetch_outlooks
from retrieval.tool import retrieve_context_tool

TOOLS = [forecast_drought_risk, forecast_heat_stress_risk, assess_crop_impact,
         retrieve_context_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class ReportState(TypedDict, total=False):
    request: str
    region: str | None
    crop: str | None
    risk_types: list[str] | None
    month: str | None
    tool_calls: list[dict]
    tool_outputs: dict[str, Any]
    retrieved_chunks: list[dict]
    type_c: dict
    external: dict
    report: str
    grounding: dict
    attempts: int
    warnings: list[str]


# Every chat call in this project goes through invoke_with_backoff, so this is
# the one honest place to count them. The API surfaces the count as "requests
# used today" — an actual tally rather than an estimate. Retries count, because
# a retried request is a real request against the cap.
CALL_TALLY: dict = {"date": None, "generate_content": 0}


def _tally_call() -> None:
    from datetime import date

    today = date.today().isoformat()
    if CALL_TALLY["date"] != today:
        CALL_TALLY.update(date=today, generate_content=0)
    CALL_TALLY["generate_content"] += 1


QUOTA_ERROR_MARKERS = ("429", "resource_exhausted")
# Gemini names the violated quota (e.g. GenerateRequestsPerDayPerProjectPerModel),
# which is how a daily cap is told apart from the 5-per-minute limit.
DAILY_QUOTA_MARKERS = ("perday", "per day", "per_day")


def is_quota_error(message: str) -> bool:
    message = message.lower()
    return any(marker in message for marker in QUOTA_ERROR_MARKERS)


def is_daily_quota_error(message: str) -> bool:
    return is_quota_error(message) and any(
        marker in message.lower() for marker in DAILY_QUOTA_MARKERS)


def invoke_with_backoff(model, messages, attempts: int = 3):
    """Retry a chat call on transient rate limits.

    Mirrors the embedding build's discipline. A per-minute limit is waited out;
    a *daily* free-tier cap (20 generate_content requests/day) is not, because
    it does not clear for hours and waiting only holds the HTTP request open.
    Either way the failure surfaces as a warning and an explicit "report not
    generated" banner rather than a silent empty answer.
    """
    import time

    last = None
    for attempt in range(attempts):
        try:
            _tally_call()
            return model.invoke(messages)
        except Exception as exc:
            last = exc
            if not is_quota_error(str(exc)) or is_daily_quota_error(str(exc)):
                raise
            if attempt < attempts - 1:
                time.sleep(30 * (attempt + 1))
    raise last


def get_chat_model(tools: bool = False):
    """Gemini 3.6 Flash, with the project's tools bound when requested."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from retrieval.embed import get_api_key

    model = ChatGoogleGenerativeAI(model=config.CHAT_MODEL,
                                   temperature=config.TEMPERATURE,
                                   google_api_key=get_api_key())
    return model.bind_tools(TOOLS) if tools else model


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
ROUTING_INSTRUCTION = """You route climate-risk requests to tools for India.

Available regions: {regions}. Supported risk types: drought (forecast, with
per-horizon skill) and heat stress (observations only — it has no forecast skill).
Supported crops for impact assessment: {crops}.

Call every tool needed to answer the request:
- forecast_drought_risk(region) for any drought question
- forecast_heat_stress_risk(region, month) for any heat question
- assess_crop_impact(region, crop, month) whenever the request names a crop or
  asks about yield, harvest or farming impact. It decides for itself which risk
  is binding, so call it instead of guessing that from the forecast tools.
- retrieve_context(query, k, doc_type) at least once, to ground definitions and
  reliability claims in real documents. Prefer two calls when the request touches
  both a definition ("what is a heat wave") and a reliability question ("how
  confident are you") — use doc_type="A" for definitions and "B" for this
  project's own measured evidence.

If a request names no region, use {default_region}. If it asks for something this
system does not cover, still call the tools that are relevant to what it does
cover. Do not answer in prose — only make tool calls."""


def _caller_constraints(state: ReportState) -> str:
    """The fields the API caller set explicitly, as text the router can read."""
    parts = [f"{field}={state[field]}" for field in ("region", "crop", "month")
             if state.get(field)]
    if state.get("risk_types"):
        parts.append(f"risk_types={', '.join(state['risk_types'])}")
    if not parts:
        return ""
    return ("\n\nCaller constraints (set explicitly by the user, must be used "
            "as given): " + "; ".join(parts) + ".")


def _required_calls(state: ReportState, calls: list[dict]) -> list[dict]:
    """Tools the caller's explicit risk_types/crop demand but the model skipped."""
    wanted = set()
    for risk in state.get("risk_types") or []:
        if risk == "drought":
            wanted.add("forecast_drought_risk")
        elif risk == "heat_stress":
            wanted.add("forecast_heat_stress_risk")
    if state.get("crop"):
        wanted.add("assess_crop_impact")

    present = {c["name"] for c in calls}
    return [c for c in _fallback_calls(state)
            if c["name"] in wanted and c["name"] not in present]


def parse_request(state: ReportState) -> ReportState:
    """Let the model choose the tools; fall back to the obvious ones."""
    prompt = ROUTING_INSTRUCTION.format(
        regions=", ".join(fconfig.REGIONS),
        crops=", ".join(cconfig.CROPS),
        default_region=fconfig.DEFAULT_REGION)

    calls: list[dict] = []
    try:
        response = invoke_with_backoff(
            get_chat_model(tools=True),
            [("system", prompt),
             ("human", state["request"] + _caller_constraints(state))])
        calls = list(getattr(response, "tool_calls", []) or [])
    except Exception as exc:                 # network/quota — degrade, don't die
        calls = []
        state.setdefault("warnings", []).append(
            f"tool routing fell back to defaults ({type(exc).__name__})")

    if not calls:
        calls = _fallback_calls(state)
    else:
        calls += _required_calls(state, calls)
    return {"tool_calls": calls, "attempts": 0,
            "warnings": state.get("warnings", [])}


def _fallback_calls(state: ReportState) -> list[dict]:
    """Deterministic routing, used only if function calling returned nothing."""
    region = state.get("region") or fconfig.DEFAULT_REGION
    wanted = state.get("risk_types") or list(config.DEFAULT_RISK_TYPES)
    calls = []
    if "drought" in wanted:
        calls.append({"name": "forecast_drought_risk",
                      "args": {"region": region}, "id": "fb_drought"})
    if "heat_stress" in wanted:
        args = {"region": region}
        if state.get("month"):
            args["month"] = state["month"]
        calls.append({"name": "forecast_heat_stress_risk",
                      "args": args, "id": "fb_heat"})
    if state.get("crop"):
        calls.append({"name": "assess_crop_impact",
                      "args": {"region": region, "crop": state["crop"],
                               **({"month": state["month"]} if state.get("month")
                                  else {})},
                      "id": "fb_crop"})
    calls.append({"name": "retrieve_context",
                  "args": {"query": state["request"], "k": config.RETRIEVAL_K},
                  "id": "fb_retrieve"})
    return calls


def call_tools(state: ReportState) -> ReportState:
    """Execute the chosen tools, keeping failures visible rather than fatal."""
    outputs: dict[str, Any] = {}
    chunks: list[dict] = []
    regions: set[str] = set()
    warnings = list(state.get("warnings", []))

    for call in state["tool_calls"]:
        name = call["name"]
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            warnings.append(f"model requested unknown tool {name!r}")
            continue

        # Explicit caller fields win over whatever the model passed — or left out.
        # Checking only args the model happened to include let a tool fall back
        # to its default region and be reported under the requested one.
        args = dict(call.get("args") or {})
        accepted = set(getattr(tool, "args", None) or ())
        for field in ("region", "crop", "month"):
            if state.get(field) and field in accepted:
                args[field] = state[field]
        try:
            result = tool.invoke(args)
        except Exception as exc:
            warnings.append(f"{name} failed: {type(exc).__name__}: {exc}")
            continue

        if name == "retrieve_context":
            chunks.extend(result)
        else:
            outputs[name] = result
            if isinstance(result, dict) and result.get("region"):
                regions.add(result["region"])

    if not regions:
        regions.add(state.get("region") or fconfig.DEFAULT_REGION)

    return {"tool_outputs": outputs, "retrieved_chunks": chunks,
            "region": sorted(regions)[0], "warnings": warnings}


def fetch_type_c(state: ReportState) -> ReportState:
    """Always fetch IMD's live outlooks — cheap, and they improve the report.

    An unavailable outlook is reported as unavailable, never silently omitted and
    never replaced by an undated cached copy.
    """
    warnings = list(state.get("warnings", []))

    def failed(label: str, exc: BaseException | None = None) -> dict:
        warnings.append(f"{label} fetch failed entirely")
        return {"any_unavailable": True,
                "error": f"{type(exc).__name__}: {exc}" if exc else "timed out"}

    def outlooks() -> dict:
        try:
            return fetch_outlooks()
        except Exception as exc:
            return {"outlooks": [], **failed("IMD outlook", exc)}

    # Phase 8: the other two live sources, on the same node and under the same
    # contract as IMD — attributed by name, never merged into this project's own
    # tool outputs, and reported as unavailable rather than substituted for.
    def externals() -> dict:
        try:
            return fetch_external_sources(region=state.get("region"),
                                          crop=state.get("crop"),
                                          month=state.get("month"))
        except Exception as exc:
            return {"sources": [], **failed("external source", exc)}

    # Both aggregators enforce the live-fetch deadline themselves; running them
    # side by side keeps the whole node near one deadline instead of two. The
    # outer limit is only a backstop for a step that hangs outside a request.
    payload, external = gather_with_deadline(
        [outlooks, externals], rconfig.LIVE_FETCH_DEADLINE_S + 15,
        lambda i: ({"outlooks": [], **failed("IMD outlook")} if i == 0
                   else {"sources": [], **failed("external source")}))

    if payload.get("any_unavailable"):
        warnings.append("at least one IMD outlook was unavailable")

    for source in external.get("sources", []):
        if not source.get("available"):
            warnings.append(
                f"{source['id']} unavailable: {source.get('reason', 'unknown')}")

    return {"type_c": payload, "external": external, "warnings": warnings}


def _render_sources(state: ReportState) -> str:
    lines = []
    for chunk in state.get("retrieved_chunks", []):
        citation = chunk.get("citation", "")
        lines.append(f"--- source: {chunk.get('source')} ({citation})\n"
                     f"{chunk.get('text', '')}")
    for outlook in state.get("type_c", {}).get("outlooks", []):
        if outlook.get("available"):
            lines.append(
                f"--- IMD LIVE OUTLOOK: {outlook['title']} "
                f"({outlook['citation']}, fetched {outlook['fetched_at']})\n"
                f"{outlook['excerpt']}")
        else:
            lines.append(f"--- IMD LIVE OUTLOOK UNAVAILABLE: {outlook['title']} "
                         f"— reason: {outlook.get('reason')}")
    # Phase 8's two extra live sources, rendered in the same shape as IMD's so
    # the model has no structural reason to treat them differently.
    for source in state.get("external", {}).get("sources", []):
        if source.get("available"):
            lines.append(
                f"--- EXTERNAL LIVE SOURCE ({source['publisher']}): "
                f"{source['title']} ({source['citation']}, fetched "
                f"{source['fetched_at']})\n{source['excerpt']}")
        else:
            lines.append(f"--- EXTERNAL LIVE SOURCE UNAVAILABLE: {source['title']} "
                         f"— reason: {source.get('reason')}")
    return "\n\n".join(lines) if lines else "(no retrieved sources)"

def _extract_text(content) -> str:
    """Gemini 3.x returns content as a list of blocks, not a plain string —
    keep only the actual text blocks, drop thinking/signature metadata."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)

def synthesize(state: ReportState) -> ReportState:
    """One Gemini call, with the checked-in prompt and the raw tool outputs."""
    system_prompt = config.SYNTHESIS_PROMPT_PATH.read_text(encoding="utf-8")

    retry_note = ""
    previous = state.get("grounding")
    if previous and previous.get("unverified_numbers"):
        retry_note = (
            "\n\nIMPORTANT — your previous draft was rejected. These figures "
            f"could not be verified against the source data: "
            f"{', '.join(previous['unverified_numbers'])}. Remove them or replace "
            "them with the exact values present in the tool outputs below. Do not "
            "restate any number you cannot find in the data.")

    human = (
        f"REQUEST: {state['request']}\n\n"
        f"TOOL OUTPUTS (JSON — the only permitted source of project figures):\n"
        f"{json.dumps(state.get('tool_outputs', {}), indent=2, default=str)}\n\n"
        f"RETRIEVED SOURCES AND LIVE OUTSIDE SOURCES "
        f"(IMD, NASA POWER, data.gov.in):\n{_render_sources(state)}"
        f"{retry_note}")

    try:
        response = invoke_with_backoff(get_chat_model(),
                                       [("system", system_prompt),
                                        ("human", human)])
        report = _extract_text(response.content)
    except Exception as exc:
        return {"report": "", "attempts": state.get("attempts", 0) + 1,
                "warnings": state.get("warnings", []) +
                [f"synthesis failed: {type(exc).__name__}: {exc}"]}

    return {"report": report, "attempts": state.get("attempts", 0) + 1}


def verify_grounding(state: ReportState) -> ReportState:
    """Mechanically check every number, and log any failure."""
    sources = list(state.get("retrieved_chunks", []))
    sources += [o for o in state.get("type_c", {}).get("outlooks", [])
                if o.get("available")]
    # External live sources are checkable exactly like IMD's outlook: their
    # excerpts go into the same source blob the number checker reads.
    sources += [s for s in state.get("external", {}).get("sources", [])
                if s.get("available")]

    report = state.get("report", "")
    if not report.strip():
        # A report that does not exist must never be reported as "grounded" —
        # zero numbers checked is a vacuous pass, the same class of empty metric
        # this project has caught twice before.
        return {"grounding": {"grounded": False, "unverified_numbers": [],
                              "total_checked": 0, "report_missing": True,
                              "reason": "synthesis produced no report"}}

    result = check_grounding(report, state.get("tool_outputs", {}), sources)
    if not result["grounded"]:
        log_failure(state["request"], result, state.get("report", ""),
                    state.get("attempts", 0))
    return {"grounding": result}


def finalise(state: ReportState) -> ReportState:
    """Attach the warning banner rather than pretending the report is clean."""
    report = state.get("report", "")
    grounding = state.get("grounding", {})

    if grounding.get("report_missing"):
        failures = [w for w in state.get("warnings", []) if "synthesis failed" in w]
        detail = failures[0] if failures else "synthesis returned nothing"
        return {"report": (
            "> **REPORT NOT GENERATED.** The synthesis step did not return a "
            f"report, so there is nothing to verify. Cause: {detail}\n\n"
            "The tool outputs were collected successfully and are available in "
            "the returned state; only the written summary is missing.")}

    if not grounding.get("grounded", True):
        report = warning_banner(grounding) + "\n" + report
    return {"report": report}


def _should_retry(state: ReportState) -> str:
    grounding = state.get("grounding", {})
    if grounding.get("grounded", False):
        return "finalise"
    # invoke_with_backoff has already spent its retries on a quota error, so a
    # regeneration would only repeat the same wait against the same limit.
    if grounding.get("report_missing") and any(
            "synthesis failed" in w and is_quota_error(w)
            for w in state.get("warnings", [])):
        return "finalise"
    if state.get("attempts", 0) < config.MAX_SYNTHESIS_ATTEMPTS:
        return "synthesize"
    return "finalise"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph():
    graph = StateGraph(ReportState)
    graph.add_node("parse_request", parse_request)
    graph.add_node("call_tools", call_tools)
    graph.add_node("fetch_type_c", fetch_type_c)
    graph.add_node("synthesize", synthesize)
    graph.add_node("verify_grounding", verify_grounding)
    graph.add_node("finalise", finalise)

    graph.set_entry_point("parse_request")
    graph.add_edge("parse_request", "call_tools")
    graph.add_edge("call_tools", "fetch_type_c")
    graph.add_edge("fetch_type_c", "synthesize")
    graph.add_edge("synthesize", "verify_grounding")
    graph.add_conditional_edges("verify_grounding", _should_retry,
                                {"synthesize": "synthesize",
                                 "finalise": "finalise"})
    graph.add_edge("finalise", END)
    return graph.compile()


def analyse(request: str, region: str | None = None,
            risk_types: list[str] | None = None,
            month: str | None = None, crop: str | None = None) -> dict:
    """Run the whole pipeline for one request and return the final state."""
    if region:
        fconfig.check_region(region)
    if crop:
        cconfig.check_crop(crop)
    initial: ReportState = {"request": request, "region": region,
                            "risk_types": risk_types, "month": month,
                            "crop": crop, "warnings": []}
    return build_graph().invoke(initial)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or \
        "What is the drought and heat risk for Barmer, and how reliable is it?"
    final = analyse(question)

    print(final["report"])
    print("\n" + "=" * 70)
    print(f"tools called : {[c['name'] for c in final.get('tool_calls', [])]}")
    print(f"grounding    : {final.get('grounding')}")
    if final.get("warnings"):
        print(f"warnings     : {final['warnings']}")
