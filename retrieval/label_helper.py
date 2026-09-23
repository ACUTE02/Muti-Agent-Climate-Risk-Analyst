"""TREC-style pooling tool for chunk-level relevance labels (Track A, Phase 2, Part 3).

eval_queries.json's acceptable_sources is per-document, not per-chunk: a 92-chunk
document (ndma_drought_guidelines) counts as a hit even if the retrieved chunk is
about something unrelated within that document. Chunk-level labels are needed for
strict P@5 and MRR to mean anything.

For each query, this pools every chunk that ANY reasonable retrieval method
surfaced -- BM25 top-N, dense top-N, and a plain keyword grep -- so labelling is
not biased toward only what one system (the one actually being evaluated) found.
The scaffold JSON this writes is hand-edited in place: every candidate starts
with "relevant": null, and a human fills each one in by reading the chunk text,
never by which pool found it or how highly it ranked.

Run standalone:
  python -m retrieval.label_helper --queries retrieval/eval_queries.json \
      --out retrieval/eval_labels_chunk.json
"""

from __future__ import annotations

import argparse
import json

from retrieval import config
from retrieval.chunk import read_chunks
from retrieval.lexical import bm25_search, get_bm25_index, resolve_source_type, tokenize
from retrieval.tool import _dense_search

MIN_KEYWORD_LEN = 3
PREVIEW_CHARS = 300
# A keyword that appears in this many chunks or more is too common on this
# 224-chunk corpus to be a useful grep signal -- "drought"/"month"/"reliable"
# show up almost everywhere, and OR-ing them in floods the pool with the bulk
# of the corpus. Measured from the BM25 index's own per-chunk term counts, not
# a hand-maintained stoplist, so it adapts to whatever the corpus actually is.
GREP_MAX_DOC_FREQ = 20
CACHE_PATH = config.CACHE_DIR / "label_pool_cache.json"


def _keyword_document_frequency(keyword: str) -> int:
    """Number of chunks containing `keyword` at least once, per the BM25 index."""
    doc_freqs = get_bm25_index().bm25.doc_freqs
    return sum(1 for freqs in doc_freqs if keyword in freqs)


def _default_keywords(query: str) -> list[str]:
    candidates = [t for t in tokenize(query) if len(t) >= MIN_KEYWORD_LEN]
    return [t for t in candidates if _keyword_document_frequency(t) < GREP_MAX_DOC_FREQ]


def _grep_pool(keywords: list[str], chunks: list[dict],
               source_type: str | None) -> list[dict]:
    hits = []
    for chunk in chunks:
        if source_type is not None and chunk["source_type"] != source_type:
            continue
        text_lower = chunk["text"].lower()
        if any(kw in text_lower for kw in keywords):
            hits.append(chunk)
    return hits


def pool_candidates(query: str, doc_type: str | None = None,
                    keywords: list[str] | None = None,
                    bm25_n: int = 20, dense_n: int = 20) -> list[dict]:
    """Union of BM25 top-N, dense top-N, and a keyword grep, deduped by chunk id.

    keywords defaults to lexical.tokenize(query) with very short tokens dropped
    and, per-corpus, tokens that appear in GREP_MAX_DOC_FREQ+ chunks also dropped
    (too common on this corpus to be a useful grep signal), if not given
    explicitly. Each candidate carries which pool(s) surfaced it
    ("pools": ["bm25", "dense", "grep"], possibly more than one), so the label
    file records provenance even though the label itself must be decided on the
    chunk's content, never on which pool found it or how highly it ranked.
    """
    if keywords is None:
        keywords = _default_keywords(query)

    all_chunks = read_chunks()
    chunks_by_id = {c["id"]: c for c in all_chunks}
    source_type = resolve_source_type(doc_type)

    pools: dict[str, set[str]] = {}

    for hit in bm25_search(query, n=bm25_n, doc_type=doc_type):
        pools.setdefault(hit["id"], set()).add("bm25")

    for hit in _dense_search(query, dense_n, doc_type=doc_type):
        pools.setdefault(hit["id"], set()).add("dense")

    if keywords:
        for chunk in _grep_pool(keywords, all_chunks, source_type):
            pools.setdefault(chunk["id"], set()).add("grep")

    candidates = []
    for chunk_id, pool_names in pools.items():
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        candidates.append({
            "id": chunk_id,
            "source_id": chunk["source_id"],
            "section": chunk["section"],
            "pools": sorted(pool_names),
            "chunk": chunk,
        })
    candidates.sort(key=lambda c: c["id"])
    return candidates


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _pooled_for_query(query_entry: dict, cache: dict) -> list[dict]:
    """Cache-aware wrapper around pool_candidates, keyed by query id.

    A cache hit (same query id, same query text, same doc_type) skips BM25,
    dense and grep entirely -- most importantly, it skips the Gemini embedding
    call, so re-running the tool while labelling doesn't re-embed queries
    already pooled.
    """
    query_id = query_entry["id"]
    query_text = query_entry["query"]
    doc_type = query_entry.get("doc_type")

    cached = cache.get(query_id)
    if (cached is not None and cached.get("query") == query_text
            and cached.get("doc_type") == doc_type):
        chunks_by_id = {c["id"]: c for c in read_chunks()}
        candidates = []
        for chunk_id, pool_names in cached["candidate_pools"].items():
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            candidates.append({
                "id": chunk_id,
                "source_id": chunk["source_id"],
                "section": chunk["section"],
                "pools": pool_names,
                "chunk": chunk,
            })
        candidates.sort(key=lambda c: c["id"])
        return candidates

    candidates = pool_candidates(query_text, doc_type=doc_type)
    cache[query_id] = {
        "query": query_text,
        "doc_type": doc_type,
        "candidate_pools": {c["id"]: c["pools"] for c in candidates},
    }
    return candidates


def build_scaffold(queries_path, out_path) -> dict:
    with open(queries_path, encoding="utf-8") as fh:
        spec = json.load(fh)

    cache = _load_cache()

    scaffold_queries = {}
    total_candidates = 0
    for query_entry in spec["queries"]:
        candidates = _pooled_for_query(query_entry, cache)
        total_candidates += len(candidates)
        scaffold_queries[query_entry["id"]] = {
            "query": query_entry["query"],
            "candidates": [
                {
                    "id": c["id"],
                    "source_id": c["source_id"],
                    "section": c["section"],
                    "pools": c["pools"],
                    "preview": c["chunk"]["text"][:PREVIEW_CHARS],
                    "relevant": None,
                }
                for c in candidates
            ],
        }

    _save_cache(cache)

    scaffold = {
        "description": (
            "Chunk-level relevance labels, TREC-style pooling (BM25 top-20 + "
            "dense top-20 + keyword grep). Each candidate's \"relevant\" field "
            "is filled by hand, judged on the chunk's text alone -- never on "
            "which pool found it or its rank. null = not yet labelled."
        ),
        "queries": scaffold_queries,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(scaffold, fh, ensure_ascii=False, indent=2)

    n_queries = len(spec["queries"])
    avg = total_candidates / n_queries if n_queries else 0.0
    print(f"{n_queries} queries, {total_candidates} total candidates, "
          f"{avg:.1f} candidates/query -> {out_path}")

    return scaffold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    build_scaffold(args.queries, args.out)
