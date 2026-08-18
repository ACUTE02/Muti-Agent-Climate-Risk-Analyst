"""One command to build the retrieval index end to end.

    python -m retrieval.build

Fetch and verify sources -> chunk per type -> write the manifest -> embed ->
populate ChromaDB. Each stage is independently re-runnable, and the embedding
stage resumes from disk, so a rate limit or a dropped connection costs only the
remaining work.

The manifest is written even when the embedding stage is blocked on a missing API
key: knowing exactly what the corpus contains, what was dropped and why is useful
on its own, and it is the part that has to survive review.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from retrieval import config
from retrieval.chunk import build_chunks, write_chunks


def write_manifest(entries: list[dict], chunks: list[dict],
                   embedded: bool, note: str = "") -> dict:
    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk["doc_type"]] = by_type.get(chunk["doc_type"], 0) + 1

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": config.EMBED_MODEL,
        "embedding_dim": config.EMBED_DIM,
        "chunking": {
            "type_A": f"paragraph packing, ~{config.CHUNK_CHARS} chars with "
                      f"{config.CHUNK_OVERLAP_CHARS} char overlap",
            "type_B": "header-aware (## / ### boundaries), so a result table "
                      "stays with the caveat that qualifies it",
            "type_C": "not chunked and not indexed — fetched live, kept whole",
        },
        "chunk_counts": by_type,
        "total_chunks": len(chunks),
        "index_built": embedded,
        "note": note,
        "documents": entries,
        "verified_dead_and_excluded": [{
            "url": config.NDMA_DROUGHT_DEAD_PAGE,
            "checked_variants": ["/Drought", "/Droughts", "/drought",
                                 "/Natural-Hazards"],
            "result": "404 — NDMA publishes no drought hazard page at that path",
            "replaced_by": "ndma_drought_guidelines (the official 2010 NDMA "
                           "drought guidelines, NIDM-hosted PDF)",
        }],
        "live_sources_not_indexed": [
            {k: v for k, v in src.items() if k != "format"}
            for src in config.TYPE_C_SOURCES
        ],
    }
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2,
                                               ensure_ascii=False),
                                    encoding="utf-8")
    return manifest


def main(force_fetch: bool = False) -> dict:
    print("1/4  fetching and verifying sources")
    chunks, entries = build_chunks(force=force_fetch)
    for entry in entries:
        state = (f"{entry.get('chunks', 0):3d} chunks" if entry["usable"]
                 else f"DROPPED — {entry['dropped_reason']}")
        print(f"     [{entry['doc_type']}] {entry['id']:28s} {state}")

    print(f"\n2/4  chunking -> {len(chunks)} chunks")
    write_chunks(chunks)

    print("3/4  embedding")
    from retrieval.embed import MissingAPIKey, embed_corpus

    try:
        embeddings = embed_corpus(chunks)
    except MissingAPIKey as exc:
        print(f"     BLOCKED: {exc}")
        write_manifest(entries, chunks, embedded=False,
                       note="Corpus and chunks built; embedding/index blocked on "
                            "a missing Gemini API key.")
        print(f"\n     wrote {config.MANIFEST_PATH} (corpus recorded, index not built)")
        return {"chunks": len(chunks), "index_built": False}

    print("4/4  building ChromaDB store")
    from retrieval.store import build_store, store_stats

    build_store(chunks, embeddings)
    stats = store_stats()
    write_manifest(entries, chunks, embedded=True)
    print(f"     {stats['chunks']} chunks indexed: {stats['by_source_type']}")
    print(f"\nwrote {config.MANIFEST_PATH}")
    print(f"wrote {config.CHROMA_DIR}")
    return {"chunks": len(chunks), "index_built": True, **stats}


if __name__ == "__main__":
    import sys

    main(force_fetch="--force" in sys.argv)
