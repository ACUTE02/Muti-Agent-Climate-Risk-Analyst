"""Chunk-level evaluation: dense vs. hybrid vs. (either) + cross-encoder rerank.

Track A, Phase 5. A measurement, not a tuning loop -- same discipline as
retrieval/evaluate.py: run once, report what it says, including "no better".

This is NOT retrieval/evaluate.py's metric. That one is document-level (did the
right source document appear at all). This scores individual chunks against the
1015 human relevance judgments in eval_labels_chunk*.json, so its numbers are
named chunk_precision_at_5 / mrr and must never be reported as plain
"precision@5".

Pooling assumption (TREC-style): each query's labels cover only its pool
(BM25 top-20 + dense top-20 + grep, built in Phase 2). A retrieved chunk outside
that pool has no human judgment and is scored as NOT relevant. The per-config
share of top-5 results falling outside the pool is reported so the size of that
caveat is visible.

Queries are run with their own doc_type filter, because that is how the label
pools were built; retrieval/evaluate.py deliberately runs unfiltered instead.

Run standalone:  python -m retrieval.evaluate_hybrid
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from retrieval import config, tool
from retrieval import rerank as rerank_module
from retrieval.chunk import read_chunks
from retrieval.lexical import bm25_search

CONFIGS = [
    ("dense", False),
    ("dense", True),     # dense retrieval, then cross-encoder rerank
    ("hybrid", False),   # BM25+dense RRF fusion
    ("hybrid", True),    # hybrid, then cross-encoder rerank
]
PAIRS = [
    ("dense", "hybrid"),
    ("dense", "dense+rerank"),
    ("hybrid", "hybrid+rerank"),
]
RETRIEVE_DEPTH = config.RERANK_CANDIDATE_N   # candidates pulled per config
SCORE_DEPTH = 10                             # ranked list kept for MRR
P_AT = 5

QUERY_SETS = {
    "soft": (config.EVAL_QUERIES_PATH, config.RETRIEVAL_DIR / "eval_labels_chunk.json"),
    "hard": (config.RETRIEVAL_DIR / "eval_queries_hard.json",
             config.RETRIEVAL_DIR / "eval_labels_chunk_hard.json"),
}

REPORT_PATH = config.REPO_ROOT / "internal_docs" / "PHASE5_RESULTS.md"


def config_name(mode: str, rerank: bool) -> str:
    return f"{mode}+rerank" if rerank else mode


# --------------------------------------------------------------------------- #
# Scoring (pure)
# --------------------------------------------------------------------------- #
def chunk_precision_at_5(ranked_ids: list[str], labels: dict[str, bool]) -> float:
    """Share of the top 5 judged relevant; unjudged ids count as not relevant."""
    return sum(labels.get(i) is True for i in ranked_ids[:P_AT]) / P_AT


def mrr(ranked_ids: list[str], labels: dict[str, bool]) -> float:
    """1/rank of the first relevant id within the top SCORE_DEPTH, else 0."""
    for rank, chunk_id in enumerate(ranked_ids[:SCORE_DEPTH], start=1):
        if labels.get(chunk_id) is True:
            return 1.0 / rank
    return 0.0


def outside_pool_at_5(ranked_ids: list[str], labels: dict[str, bool]) -> int:
    return sum(i not in labels for i in ranked_ids[:P_AT])


def score_ranking(ranked_ids: list[str], labels: dict[str, bool]) -> dict:
    return {
        "chunk_precision_at_5": chunk_precision_at_5(ranked_ids, labels),
        "mrr": mrr(ranked_ids, labels),
        "outside_pool_at_5": outside_pool_at_5(ranked_ids, labels),
        "ranked_ids": ranked_ids[:SCORE_DEPTH],
    }


# --------------------------------------------------------------------------- #
# Retrieval (ids kept, since retrieve_context()'s public shape has none)
# --------------------------------------------------------------------------- #
def ranked_ids_for(query: str, doc_type: str | None, mode: str, rerank: bool,
                   query_vector: list[float]) -> list[str]:
    if mode == "dense":
        hits = tool._dense_search(query, RETRIEVE_DEPTH, doc_type, query_vector=query_vector)
    else:
        bm25_hits = bm25_search(query, n=config.BM25_TOP_N, doc_type=doc_type)
        dense_hits = tool._dense_search(query, config.DENSE_TOP_N, doc_type,
                                        query_vector=query_vector)
        hits = tool._rrf_fuse(bm25_hits, dense_hits, RETRIEVE_DEPTH)

    if not rerank:
        return [h["id"] for h in hits][:SCORE_DEPTH]

    candidates = [{**tool._format(h), "id": h["id"]} for h in hits]
    return [c["id"] for c in rerank_module.rerank(query, candidates, k=SCORE_DEPTH)]


def load_labels(path) -> dict[str, dict[str, bool]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    return {qid: {c["id"]: c["relevant"] for c in q["candidates"]}
            for qid, q in spec["queries"].items()}


def check_labels_match_corpus(labels_by_set: dict[str, dict[str, dict[str, bool]]],
                              corpus_ids: set[str]) -> None:
    """Refuse to score against labels whose chunk ids this corpus does not have.

    Unjudged chunks score as not relevant, so labels from a different build of
    the corpus would not error on their own — every config would just quietly
    score zero. Checked before any embedding call is spent.
    """
    missing = {chunk_id for labels in labels_by_set.values()
               for query in labels.values() for chunk_id in query} - corpus_ids
    if missing:
        raise ValueError(
            f"{len(missing)} labelled chunk ids are not in chunks.jsonl "
            f"(e.g. {sorted(missing)[:3]}). The labels were made against a "
            "different build of the corpus; re-key them with "
            "`python -m scripts.migrate_chunk_ids` before evaluating.")


def is_negative(query: dict) -> bool:
    return query.get("type") == "negative" or bool(query.get("excluded_from_mean"))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(rows: list[dict]) -> dict:
    """Mean metrics per config over the given per-query rows."""
    out = {}
    for mode, rerank in CONFIGS:
        name = config_name(mode, rerank)
        scores = [r["configs"][name] for r in rows]
        out[name] = {
            "n_queries": len(scores),
            "chunk_precision_at_5": _mean([s["chunk_precision_at_5"] for s in scores]),
            "mrr": _mean([s["mrr"] for s in scores]),
            "outside_pool_fraction_at_5": _mean([s["outside_pool_at_5"] / P_AT for s in scores]),
        }
    return out


def paired(rows: list[dict]) -> list[dict]:
    """Per-query win/tie/loss on chunk_precision_at_5. No significance test: with
    12-15 queries a p-value would be false precision."""
    out = []
    for base, other in PAIRS:
        deltas = [r["configs"][other]["chunk_precision_at_5"]
                  - r["configs"][base]["chunk_precision_at_5"] for r in rows]
        out.append({
            "baseline": base, "compared": other,
            "improved": sum(d > 1e-9 for d in deltas),
            "same": sum(abs(d) <= 1e-9 for d in deltas),
            "regressed": sum(d < -1e-9 for d in deltas),
        })
    return out


def evaluate_set(queries: list[dict], labels: dict[str, dict[str, bool]],
                 vectors: dict[str, list[float]]) -> dict:
    rows = []
    for q in queries:
        row = {"id": q["id"], "type": q.get("type"), "negative": is_negative(q),
               "configs": {}}
        for mode, rerank in CONFIGS:
            ranked = ranked_ids_for(q["query"], q.get("doc_type"), mode, rerank,
                                    vectors[q["id"]])
            row["configs"][config_name(mode, rerank)] = score_ranking(ranked, labels[q["id"]])
        rows.append(row)

    main = [r for r in rows if not r["negative"]]
    result = {
        "aggregate": aggregate(main),
        "paired": paired(main),
        "per_query": rows,
        "negatives": [r for r in rows if r["negative"]],
    }
    types = sorted({r["type"] for r in main if r["type"]})
    if types:
        result["by_type"] = {t: aggregate([r for r in main if r["type"] == t]) for t in types}
    return result


def evaluate() -> dict:
    loaded = {}
    for set_name, (queries_path, labels_path) in QUERY_SETS.items():
        queries = json.loads(queries_path.read_text(encoding="utf-8"))["queries"]
        loaded[set_name] = (queries, load_labels(labels_path))

    check_labels_match_corpus({name: labels for name, (_, labels) in loaded.items()},
                              {c["id"] for c in read_chunks()})

    vectors: dict[str, list[float]] = {}
    embedding_calls = 0
    for queries, _ in loaded.values():
        for q in queries:
            vectors[q["id"]] = tool.embed_query(q["query"])
            embedding_calls += 1

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric_note": ("chunk-level: chunk_precision_at_5 and mrr@10 against human "
                        "chunk labels. Not the document-level precision@k of "
                        "retrieval/evaluate.py."),
        "pooling_assumption": ("Chunks outside a query's labelled pool are unjudged "
                               "and scored as NOT relevant."),
        "doc_type_filter_used": True,
        "embedding_calls": embedding_calls,
        "retrieve_depth": RETRIEVE_DEPTH,
        "score_depth": SCORE_DEPTH,
        "rerank_model": config.RERANK_MODEL,
        "rrf_k": config.RRF_K,
        "sets": {name: evaluate_set(queries, labels, vectors)
                 for name, (queries, labels) in loaded.items()},
    }
    config.EVAL_RESULTS_HYBRID_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _names() -> list[str]:
    return [config_name(m, r) for m, r in CONFIGS]


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def _aggregate_table(agg: dict) -> list[str]:
    names = _names()
    lines = ["| Metric | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    for key, label in (("chunk_precision_at_5", "chunk P@5"), ("mrr", "MRR@10"),
                       ("outside_pool_fraction_at_5", "top-5 outside pool")):
        lines.append(f"| {label} | " + " | ".join(_fmt(agg[n][key]) for n in names) + " |")
    return lines


def _per_query_table(rows: list[dict]) -> list[str]:
    names = _names()
    lines = ["| Query | Type | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(names)]
    for r in rows:
        cells = [f"{c['chunk_precision_at_5']:.1f} / {c['mrr']:.2f} / {c['outside_pool_at_5']}"
                 for c in (r["configs"][n] for n in names)]
        lines.append(f"| {r['id']} | {r['type'] or '-'} | " + " | ".join(cells) + " |")
    return lines


def format_report(result: dict) -> str:
    names = _names()
    lines = [
        "# Track A, Phase 5 — chunk-level retrieval comparison",
        "",
        f"Generated {result['generated_at']}. Configs: {', '.join(names)}. "
        f"Each config retrieves {result['retrieve_depth']} candidates; rerank configs "
        f"re-order them with `{result['rerank_model']}`; hybrid fuses BM25 + dense "
        f"with RRF (k={result['rrf_k']}).",
        "",
        "**Metric.** Chunk-level `chunk_precision_at_5` (share of the top 5 chunks "
        "a human judged relevant) and `MRR@10` (1/rank of the first relevant chunk "
        "in the top 10, else 0). This is stricter than, and not comparable to, the "
        "document-level precision@k reported by `retrieval/evaluate.py`.",
        "",
        "**Pooling assumption.** Each query was labelled only over its Phase 2 "
        "pool (BM25 top-20 + dense top-20 + keyword grep). A retrieved chunk "
        "outside that pool has no human judgment and is scored as NOT relevant. "
        "\"top-5 outside pool\" below shows how often that happened per config; "
        "where it is non-zero, that config's P@5 is a lower bound. Here it is 0 "
        "for every config by construction, not by luck: the pools were built "
        "from the same BM25 top-20 and dense top-20 under the same doc_type "
        "filter, and rerank only re-orders those candidates. So every scored "
        "chunk is human-judged; what the pool cannot show is a relevant chunk "
        "that none of these retrievers surfaced at depth 20.",
        "",
        "**Filtering.** Every query ran with its own `doc_type` filter, matching how "
        "its label pool was built. `retrieval/evaluate.py` runs unfiltered, which "
        "is the harder, more realistic setting.",
        "",
        "**No significance testing.** 12 and 13 scored queries are too few for a "
        "p-value to mean anything; paired results are given as win/tie/loss counts.",
        "",
        f"Embedding calls for the whole run: {result['embedding_calls']}.",
    ]

    for set_name, title in (("soft", "Soft set"), ("hard", "Hard set")):
        s = result["sets"][set_name]
        n = s["aggregate"][names[0]]["n_queries"]
        lines += ["", f"## {title} ({n} scored queries)", ""]
        lines += _aggregate_table(s["aggregate"])

        if "by_type" in s:
            lines += ["", "### By query type (chunk P@5 / MRR@10)", "",
                      "| Type | n | " + " | ".join(names) + " |",
                      "|---|---|" + "---|" * len(names)]
            for t, agg in s["by_type"].items():
                cells = [f"{_fmt(agg[c]['chunk_precision_at_5'])} / {_fmt(agg[c]['mrr'])}"
                         for c in names]
                lines.append(f"| {t} | {agg[names[0]]['n_queries']} | " + " | ".join(cells) + " |")

        lines += ["", "### Paired comparison on chunk P@5 (per query)", ""]
        for p in s["paired"]:
            lines.append(f"- {p['compared']} vs {p['baseline']}: {p['improved']} improved, "
                         f"{p['same']} same, {p['regressed']} regressed")

        if s["negatives"]:
            lines += ["", "### Negative queries (no correct answer exists; excluded above)", "",
                      "| Query | " + " | ".join(names) + " |",
                      "|---|" + "---|" * len(names)]
            for r in s["negatives"]:
                lines.append(f"| {r['id']} | " + " | ".join(
                    f"{r['configs'][c]['chunk_precision_at_5']:.1f}" for c in names) + " |")

        lines += ["", "### Per query (chunk P@5 / MRR@10 / top-5 outside pool)", ""]
        lines += _per_query_table(s["per_query"])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    outcome = evaluate()
    report = format_report(outcome)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {config.EVAL_RESULTS_HYBRID_PATH}")
    print(f"wrote {REPORT_PATH}")
