"""Phase 2 deliverable: the retrieval tool the Orchestrator will call.

Every result carries a citation. That is the point of this phase — the Synthesis
agent must be able to answer "how reliable is the 2-month drought forecast" with
the measured +0.0766 from ``models/region_comparison.md`` and say where it came
from, rather than generating a plausible-sounding number.

Run standalone:  python -m retrieval.tool "your question here"
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from retrieval import config
from retrieval.embed import embed_query
from retrieval.lexical import bm25_search, resolve_source_type
from retrieval.store import get_collection

DOC_TYPE_FILTERS = config.DOC_TYPE_FILTERS
RETRIEVAL_MODES = ("dense", "hybrid")

# The metadata keys retrieval.store.build_store writes to Chroma, so a chunk that
# only BM25 found gets exactly the meta a dense hit would have carried.
_META_KEYS = ("source", "source_id", "source_type", "doc_type", "citation", "section")


def _dense_search(query: str, n: int, doc_type: str | None = None,
                  query_vector: list[float] | None = None) -> list[dict]:
    """Raw Chroma hits in Chroma order: {id, text, meta, distance}.

    query_vector lets a caller (Phase 5's eval run) embed a query once and
    reuse it, instead of re-embedding on every call against the Gemini quota.
    """
    source_type = resolve_source_type(doc_type)

    collection = get_collection()
    where = ({"source_type": source_type} if source_type else None)

    result = collection.query(
        query_embeddings=[query_vector if query_vector is not None else embed_query(query)],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {"id": id_, "text": text, "meta": meta, "distance": distance}
        for id_, text, meta, distance in zip(result["ids"][0],
                                             result["documents"][0],
                                             result["metadatas"][0],
                                             result["distances"][0])
    ]


def _format(hit: dict) -> dict:
    meta = hit["meta"]
    return {
        "text": hit["text"],
        "source": meta["source"],
        "source_type": meta["source_type"],
        "citation": meta["citation"],
        "section": meta.get("section", ""),
        # cosine distance -> similarity, so higher is better for a caller
        "score": round(1.0 - float(hit["distance"]), 4),
    }


def _rrf_fuse(bm25_hits: list[dict], dense_hits: list[dict], k: int) -> list[dict]:
    """Reciprocal Rank Fusion over two ranked lists, joined by chunk id.

    score(id) = sum over each list the id appears in of 1 / (RRF_K + rank),
    rank being the 1-based position within that list. Returns the top-k by
    fused score (ties by id ascending), shaped like _dense_search's hits so
    _format applies unchanged; "distance" is 1 - fused score, so the formatted
    "score" is the RRF score itself.
    """
    scores: dict[str, float] = {}
    docs: dict[str, tuple[str, dict]] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (config.RRF_K + rank)
        docs[hit["id"]] = (hit["text"], hit["meta"])

    for rank, hit in enumerate(bm25_hits, start=1):
        scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (config.RRF_K + rank)
        chunk = hit["chunk"]
        docs.setdefault(hit["id"], (chunk["text"], {key: chunk[key] for key in _META_KEYS}))

    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:k]
    return [
        {"id": chunk_id, "text": docs[chunk_id][0], "meta": docs[chunk_id][1],
         "distance": 1.0 - scores[chunk_id]}
        for chunk_id in ranked
    ]


def retrieve_context(query: str, k: int = 5, doc_type: str | None = None,
                     mode: str | None = None, rerank: bool = False) -> list[dict]:
    """
    Returns up to k chunks most relevant to query, each with:
      - text: the chunk content
      - source: document title
      - source_type: "project_evidence" or "domain_reference"
      - citation: URL, or a repo-relative path for project documents
      - score: similarity score
    doc_type, if given, restricts to "A" (domain reference) or "B" (project evidence).
    mode is "dense" or "hybrid" (BM25 + dense, RRF-fused; score is then the RRF
    score); None uses config.RETRIEVAL_MODE. rerank=True pulls
    config.RERANK_CANDIDATE_N candidates and re-orders them with a cross-encoder
    (needs requirements-rerank.txt), adding "rerank_score" to each result.
    """
    if mode is None:
        mode = config.RETRIEVAL_MODE
    if mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. Use one of {', '.join(repr(m) for m in RETRIEVAL_MODES)}, "
            "or None for config.RETRIEVAL_MODE.")

    n = config.RERANK_CANDIDATE_N if rerank else k

    if mode == "dense":
        candidates = [_format(hit) for hit in _dense_search(query, n, doc_type)]
    else:
        bm25_hits = bm25_search(query, n=config.BM25_TOP_N, doc_type=doc_type)
        dense_hits = _dense_search(query, config.DENSE_TOP_N, doc_type)
        candidates = [_format(hit) for hit in _rrf_fuse(bm25_hits, dense_hits, n)]

    if not rerank:
        return candidates

    from retrieval.rerank import rerank as rerank_candidates   # optional dependency
    return rerank_candidates(query, candidates, k)


@tool("retrieve_context")
def retrieve_context_tool(query: str, k: int = 5,
                          doc_type: str | None = None) -> list[dict]:
    """
    Retrieves passages relevant to a climate-risk question, with citations.

    Searches two kinds of document: authoritative domain references (IMD/NDMA/ICAR
    definitions and methodology) and this project's own measured evidence (its
    logs and result tables). Use it to ground any factual claim — especially any
    figure describing how reliable a forecast is, which must come from the project
    evidence rather than be generated. Set doc_type="A" for domain references only
    or "B" for project evidence only.
    """
    return retrieve_context(query, k=k, doc_type=doc_type)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or \
        "What is the measured skill score for the 2-month drought forecast?"
    print(f"Q: {question}\n")
    for i, hit in enumerate(retrieve_context(question, k=3), 1):
        print(f"{i}. [{hit['score']:.3f}] {hit['source']} — {hit['section'][:60]}")
        print(f"   {hit['citation']}")
        print(f"   {hit['text'][:220].replace(chr(10), ' ')}...\n")
