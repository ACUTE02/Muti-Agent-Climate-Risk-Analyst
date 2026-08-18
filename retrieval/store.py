"""ChromaDB store — populated at build time, queried at runtime.

Persisted to ``retrieval/chroma_store/`` so the build step is testable now,
before the Docker phase exists. Per the standing architecture decision the index
is baked into the image at build time rather than created at runtime, because the
free hosting tier has no persistent disk.

Embeddings are supplied explicitly rather than letting Chroma embed for us — the
task_type distinction in ``retrieval.embed`` is the whole point and Chroma's
default embedding function would silently bypass it.
"""

from __future__ import annotations

import chromadb

from retrieval import config


def get_client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def build_store(chunks: list[dict], embeddings: dict[str, list[float]],
                reset: bool = True):
    """Populate the collection from chunks plus their precomputed vectors."""
    client = get_client()
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass                     # collection did not exist yet

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine",
                  "embedding_model": config.EMBED_MODEL,
                  "embedding_dim": config.EMBED_DIM})

    usable = [c for c in chunks if c["id"] in embeddings]
    missing = len(chunks) - len(usable)
    if missing:
        raise ValueError(f"{missing} chunks have no embedding — re-run "
                         "`python -m retrieval.embed` before building the store")

    collection.add(
        ids=[c["id"] for c in usable],
        documents=[c["text"] for c in usable],
        embeddings=[embeddings[c["id"]] for c in usable],
        metadatas=[{
            "source": c["source"],
            "source_id": c["source_id"],
            "source_type": c["source_type"],
            "doc_type": c["doc_type"],
            "citation": c["citation"],
            "section": c["section"],
        } for c in usable],
    )
    return collection


def get_collection():
    """The collection for querying. Raises with instructions if not built."""
    client = get_client()
    try:
        return client.get_collection(config.COLLECTION_NAME)
    except Exception as exc:
        raise FileNotFoundError(
            "The retrieval index has not been built yet — run "
            "`python -m retrieval.build`.") from exc


def store_stats() -> dict:
    collection = get_collection()
    got = collection.get(include=["metadatas"])
    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for meta in got["metadatas"]:
        by_type[meta["source_type"]] = by_type.get(meta["source_type"], 0) + 1
        by_source[meta["source"]] = by_source.get(meta["source"], 0) + 1
    return {"chunks": len(got["ids"]), "by_source_type": by_type,
            "by_source": by_source}
