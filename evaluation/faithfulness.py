"""Sentence-level faithfulness and answer relevance of real generated reports.

**This is the one place in the entire project where an LLM judges an LLM's
output, and that exception is deliberate, bounded and logged.**

Everywhere else the rule has been the opposite: `orchestrator/grounding.py` is
mechanical precisely because an LLM checking an LLM shares the failure mode being
checked, and `crop_impact/dominance.py` is plain Python for the same reason. The
exception holds here because the quantity being measured is different in kind.
`check_grounding()` verifies discrete facts — does this number appear in the
sources — which is exactly what mechanical matching is good at. Faithfulness is a
soft judgement about whether a *claim* is entailed by a passage, which has no
mechanical form. The two are complementary, not redundant:

    check_grounding()  ->  every NUMBER traces to a source        (high trust)
    this module        ->  every CLAIM is entailed by a source    (lower trust)

Read the scores here as indicative, not authoritative. A judge that shares
training data with the writer can agree with it for the wrong reasons, and this
module cannot detect that. Where the two disagree, the mechanical checker wins.

Quota: 1 judge call per report, with faithfulness and relevance folded into one
prompt to keep the cost down (the phase spec's stated default). See
``evaluation/config.py`` for the budget arithmetic that fixed the set size.

Run standalone:  python -m evaluation.faithfulness
"""

from __future__ import annotations

import json
import re

from evaluation import config

JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def load_requests() -> dict:
    return json.loads(config.REQUESTS_PATH.read_text(encoding="utf-8"))


def _render_sources(state: dict) -> str:
    """The same source material the report was written from, for the judge."""
    lines = [
        "TOOL OUTPUTS (JSON):",
        json.dumps(state.get("tool_outputs", {}), indent=2, default=str),
        "",
        "RETRIEVED CHUNKS:",
    ]
    for chunk in state.get("retrieved_chunks", []):
        lines.append(f"--- {chunk.get('source')} ({chunk.get('citation')})\n"
                     f"{chunk.get('text', '')}")
    for outlook in state.get("type_c", {}).get("outlooks", []):
        if outlook.get("available"):
            lines.append(f"--- IMD LIVE OUTLOOK: {outlook['title']} "
                         f"({outlook['citation']})\n{outlook['excerpt']}")
        else:
            lines.append(f"--- IMD LIVE OUTLOOK UNAVAILABLE: {outlook['title']} "
                         f"— reason: {outlook.get('reason')}")
    return "\n\n".join(lines)


def _parse_judgement(text: str) -> dict:
    """Judges sometimes wrap JSON in prose or a fence — recover it, don't guess."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.M).strip()
    match = JSON_BLOCK.search(cleaned)
    if not match:
        return {"parse_failed": True, "raw": text[:800]}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"parse_failed": True, "error": str(exc), "raw": text[:800]}


def judge_report(request: str, report: str, state: dict) -> dict:
    """One Gemini call scoring faithfulness and relevance together."""
    from orchestrator.graph import _extract_text, invoke_with_backoff

    if not report.strip():
        return {"parse_failed": True,
                "error": "no report was generated, so nothing could be judged"}

    from langchain_google_genai import ChatGoogleGenerativeAI
    from orchestrator import config as oconfig
    from retrieval.embed import get_api_key

    model = ChatGoogleGenerativeAI(model=oconfig.CHAT_MODEL,
                                   temperature=config.JUDGE_TEMPERATURE,
                                   google_api_key=get_api_key())

    system_prompt = config.JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    human = (f"REQUEST THAT WAS ASKED:\n{request}\n\n"
             f"SOURCE MATERIAL THE REPORT WAS GIVEN:\n{_render_sources(state)}\n\n"
             f"THE REPORT TO GRADE:\n{report}")

    response = invoke_with_backoff(model, [("system", system_prompt),
                                           ("human", human)])
    return _parse_judgement(_extract_text(response.content))


def run_item(item: dict) -> dict:
    """Generate one report and judge it. Costs 3-5 Gemini calls."""
    from orchestrator.graph import analyse

    state = analyse(item["request"])
    report = state.get("report", "")
    judgement = judge_report(item["request"], report, state)

    return {
        "id": item["id"],
        "request": item["request"],
        "tools_called": [c["name"] for c in state.get("tool_calls", [])],
        "expected_tools": item["targets"],
        "mechanical_grounding": state.get("grounding", {}),
        "judge": judgement,
        "warnings": state.get("warnings", []),
        "report": report,
    }


def summarise(items: list[dict]) -> dict:
    scored = [i for i in items if not i["judge"].get("parse_failed")]
    faith = [float(i["judge"]["faithfulness"]) for i in scored
             if "faithfulness" in i["judge"]]
    rel = [float(i["judge"]["relevance"]) for i in scored
           if "relevance" in i["judge"]]

    unsupported = sum(len(i["judge"].get("unsupported_claims", []))
                      for i in scored)
    grounded_all = all(i["mechanical_grounding"].get("grounded", False)
                       for i in items)

    return {
        "items_run": len(items),
        "items_scored": len(scored),
        "mean_faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
        "mean_relevance": round(sum(rel) / len(rel), 4) if rel else None,
        "total_unsupported_claims": unsupported,
        "mechanical_grounding_clean_on_every_item": grounded_all,
        "routing_correct_on_every_item": all(
            set(i["expected_tools"]).issubset(set(i["tools_called"]))
            for i in items),
    }


def main() -> dict:
    spec = load_requests()
    items = [run_item(item) for item in spec["requests"]]

    result = {
        "generated_at": config.timestamp(),
        "trust_note": (
            "This is the only LLM-judges-LLM measurement in this project. It is "
            "complementary to the mechanical grounding checker, not a substitute, "
            "and is explicitly lower-trust: where the two disagree, the "
            "mechanical checker wins."),
        "set_size": len(items),
        "size_rationale": spec["size_rationale"],
        "known_coverage_gap": spec["known_coverage_gap"],
        "summary": summarise(items),
        "items": items,
    }
    config.FAITHFULNESS_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result["summary"], indent=2))
    for item in items:
        judge = item["judge"]
        if judge.get("parse_failed"):
            print(f"  ! {item['id']}: judge output unparseable")
            continue
        print(f"  {item['id']}: faithfulness={judge.get('faithfulness')} "
              f"relevance={judge.get('relevance')} "
              f"unsupported={len(judge.get('unsupported_claims', []))}")
    print(f"\nwritten to {config.FAITHFULNESS_PATH}")
    return result


if __name__ == "__main__":
    main()
