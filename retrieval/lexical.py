"""BM25 lexical retriever — Track A, Phase 1.

Dense retrieval (Gemini embeddings, cosine similarity in Chroma) is good at
paraphrase but blurs exact tokens. This corpus is full of exact tokens that
matter: skill scores (+0.0766, +0.0438), codes (SPI-3, GKMS, t+1, IOD), labels
(weak/directional). The grounding checker needs chunks that contain those exact
figures. BM25 rewards the chunk that literally contains the query's rare
tokens, and costs no API call.

The index is derived data, built lazily in memory from chunks.jsonl. It is
never persisted, so it cannot go stale relative to chunks.jsonl — but nor does
it notice a rebuilt chunks.jsonl on its own: after a rebuild, the process must
be restarted (or get_bm25_index.cache_clear() called).

Run standalone:  python -m retrieval.lexical "your question here"
"""

from __future__ import annotations

import functools
import re
from typing import NamedTuple

from rank_bm25 import BM25Okapi

from retrieval import config
from retrieval.chunk import read_chunks

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "do", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, keep numbers/codes like +0.0766 or SPI-3 as single tokens."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index(NamedTuple):
    bm25: BM25Okapi
    ids: list[str]
    chunks: list[dict]


@functools.lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    chunks = read_chunks()
    ids = [c["id"] for c in chunks]
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return BM25Index(bm25=bm25, ids=ids, chunks=chunks)


def resolve_source_type(doc_type: str | None) -> str | None:
    if doc_type is None:
        return None
    if doc_type not in config.DOC_TYPE_FILTERS:
        raise ValueError(
            f"Unknown doc_type {doc_type!r}. Use 'A' (domain reference), "
            "'B' (project evidence), or None for both.")
    return config.DOC_TYPE_FILTERS[doc_type]


def bm25_search(query: str, n: int = 5, doc_type: str | None = None) -> list[dict]:
    """Score the whole corpus (so IDF is computed over all chunks), then filter."""
    source_type = resolve_source_type(doc_type)

    query_tokens = tokenize(query)
    if n <= 0 or not query_tokens:
        return []

    index = get_bm25_index()
    scores = index.bm25.get_scores(query_tokens)

    results = []
    for chunk_id, chunk, score in zip(index.ids, index.chunks, scores):
        if source_type is not None and chunk["source_type"] != source_type:
            continue
        if score <= 0:
            continue
        results.append((chunk_id, chunk, float(score)))

    results.sort(key=lambda r: (-r[2], r[0]))

    return [
        {"id": chunk_id, "bm25_score": score, "rank": rank, "chunk": chunk}
        for rank, (chunk_id, chunk, score) in enumerate(results[:n], start=1)
    ]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or \
        "What is the measured skill score for the 2-month drought forecast?"
    print(f"Q: {question}\n")
    for hit in bm25_search(question, n=5):
        chunk = hit["chunk"]
        preview = chunk["text"][:150].replace("\n", " ")
        print(f"{hit['rank']}. [{hit['bm25_score']:.3f}] {chunk['source_id']}")
        print(f"   {preview}...\n")
