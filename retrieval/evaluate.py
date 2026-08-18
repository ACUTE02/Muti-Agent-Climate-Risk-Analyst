"""Bounded retrieval evaluation — one honest pass, not a tuning loop.

Ten hand-authored queries, each with the source document that should answer it.
Reported as precision@k (did the right source appear at all) and MRR (how near the
top it landed).

**This is deliberately not RAGAS** — full RAG evaluation is Phase 5. And it is
deliberately run *once*: ten queries is far too small a set to tune against, and
this project has been careful about exactly that failure mode with the forecasting
models. If a category retrieves badly, that gets reported as a diagnosed problem,
not chased with repeated re-chunking until the numbers look better.

Run standalone:  python -m retrieval.evaluate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from retrieval import config


def load_queries() -> dict:
    return json.loads(config.EVAL_QUERIES_PATH.read_text(encoding="utf-8"))


def evaluate(k: int | None = None, use_doc_type_filter: bool = False) -> dict:
    """Run every eval query and score retrieval honestly.

    ``use_doc_type_filter`` is off by default: filtering by document type would
    make the task easier than the real one, where the Orchestrator will not know
    in advance whether an answer lives in a domain reference or in this project's
    own evidence.
    """
    from retrieval.tool import retrieve_context
    from retrieval.store import get_collection

    spec = load_queries()
    k = k or spec["k"]
    collection = get_collection()
    id_of = {meta["citation"]: meta["source_id"]
             for meta in collection.get(include=["metadatas"])["metadatas"]}

    rows = []
    for item in spec["queries"]:
        hits = retrieve_context(item["query"], k=k,
                                doc_type=item["doc_type"] if use_doc_type_filter
                                else None)
        retrieved_ids = [id_of.get(h["citation"], h["source"]) for h in hits]
        acceptable = set(item["acceptable_sources"])

        rank = next((i + 1 for i, sid in enumerate(retrieved_ids)
                     if sid in acceptable), None)
        rows.append({
            "id": item["id"],
            "query": item["query"],
            "doc_type": item["doc_type"],
            "acceptable_sources": item["acceptable_sources"],
            "retrieved_sources": retrieved_ids,
            "hit": rank is not None,
            "rank_of_first_correct": rank,
            "top_score": hits[0]["score"] if hits else None,
        })

    hits_a = [r for r in rows if r["doc_type"] == "A"]
    hits_b = [r for r in rows if r["doc_type"] == "B"]

    def precision(subset):
        return round(sum(r["hit"] for r in subset) / len(subset), 3) if subset else None

    def mrr(subset):
        if not subset:
            return None
        return round(sum(1 / r["rank_of_first_correct"] if r["hit"] else 0
                         for r in subset) / len(subset), 3)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": k,
        "doc_type_filter_used": use_doc_type_filter,
        "precision_at_k": precision(rows),
        "precision_type_A_domain_reference": precision(hits_a),
        "precision_type_B_project_evidence": precision(hits_b),
        "mrr": mrr(rows),
        "results": rows,
    }
    config.EVAL_RESULTS_PATH.write_text(json.dumps(result, indent=2,
                                                   ensure_ascii=False),
                                        encoding="utf-8")
    return result


def format_report(result: dict) -> str:
    lines = [
        f"Retrieval evaluation — precision@{result['k']}, "
        f"{len(result['results'])} hand-authored queries",
        "",
        "| Query | Type | Hit | Rank | Expected | Top retrieved |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["results"]:
        lines.append(
            f"| {row['id']} | {row['doc_type']} | "
            f"{'yes' if row['hit'] else 'NO'} | "
            f"{row['rank_of_first_correct'] or '-'} | "
            f"{', '.join(row['acceptable_sources'])} | "
            f"{row['retrieved_sources'][0] if row['retrieved_sources'] else '-'} |")

    lines += [
        "",
        f"- Overall precision@{result['k']}: **{result['precision_at_k']}**",
        f"- Type A (domain reference): {result['precision_type_A_domain_reference']}",
        f"- Type B (project evidence): {result['precision_type_B_project_evidence']}",
        f"- MRR: {result['mrr']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    outcome = evaluate()
    print(format_report(outcome))
    print(f"\nwrote {config.EVAL_RESULTS_PATH}")
