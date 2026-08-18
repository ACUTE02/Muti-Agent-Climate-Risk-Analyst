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
from retrieval.store import get_collection

DOC_TYPE_FILTERS = {
    "A": "domain_reference",
    "B": "project_evidence",
    "domain_reference": "domain_reference",
    "project_evidence": "project_evidence",
}


def retrieve_context(query: str, k: int = 5,
                     doc_type: str | None = None) -> list[dict]:
    """
    Returns up to k chunks most relevant to query, each with:
      - text: the chunk content
      - source: document title
      - source_type: "project_evidence" or "domain_reference"
      - citation: URL, or a repo-relative path for project documents
      - score: similarity score
    doc_type, if given, restricts to "A" (domain reference) or "B" (project evidence).
    """
    if doc_type is not None and doc_type not in DOC_TYPE_FILTERS:
        raise ValueError(
            f"Unknown doc_type {doc_type!r}. Use 'A' (domain reference), "
            "'B' (project evidence), or None for both.")

    collection = get_collection()
    where = ({"source_type": DOC_TYPE_FILTERS[doc_type]} if doc_type else None)

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, distance in zip(result["documents"][0],
                                    result["metadatas"][0],
                                    result["distances"][0]):
        hits.append({
            "text": text,
            "source": meta["source"],
            "source_type": meta["source_type"],
            "citation": meta["citation"],
            "section": meta.get("section", ""),
            # cosine distance -> similarity, so higher is better for a caller
            "score": round(1.0 - float(distance), 4),
        })
    return hits


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
