"""Cross-encoder re-ranking — Track A, Phase 4.

BM25, dense and RRF all score a query against each chunk independently and
cheaply. A cross-encoder reads the query and the chunk together through one
model and scores how well they actually match: too slow to run over the whole
corpus, but far more accurate at the top of a small candidate list. So the
pattern is retrieve a wider pool cheaply, re-rank just that pool, keep top k.

sentence-transformers (and torch under it) lives in requirements-rerank.txt,
not requirements.txt, so it is imported lazily inside get_reranker() — the rest
of retrieval, and the API image, must work without it installed.
"""

from __future__ import annotations

import functools

from retrieval import config


@functools.lru_cache(maxsize=1)
def get_reranker():
    """Lazy-loaded, process-cached CrossEncoder. Downloads the model from
    Hugging Face Hub on first use if not already cached locally — a one-time
    network/disk cost, not per-call."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def rerank(query: str, candidates: list[dict], k: int) -> list[dict]:
    """Re-order retrieve_context()-shaped candidates by cross-encoder score.

    Returns the top k, each a copy of the input dict with "rerank_score" added
    (the raw cross-encoder logit, not normalized). The retrieval "score" is left
    untouched. Equal rerank scores keep their original relative order (stable sort).
    """
    if k <= 0 or not candidates:
        return []

    scores = get_reranker().predict([(query, c["text"]) for c in candidates])

    scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:k]
