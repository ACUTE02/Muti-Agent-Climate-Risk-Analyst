"""Gemini embeddings — batched, backed off, and resumable.

The Heat Stress agent's Open-Meteo pull hit a real per-minute rate limit mid-build
and taught the lesson the hard way: an embedding build that restarts from zero on
a 429 will never finish on a free tier. So progress is written to disk as it goes,
and a re-run picks up where it stopped.

``task_type`` matters and is not cosmetic: ``gemini-embedding-001`` produces
*different* vectors for the same text depending on whether it is being stored or
queried, and mixing them measurably degrades retrieval. Documents are embedded
with RETRIEVAL_DOCUMENT at build time, queries with RETRIEVAL_QUERY at call time.

Run standalone:  python -m retrieval.embed
"""

from __future__ import annotations

import json
import os
import time

from retrieval import config

EMBEDDINGS_PATH = config.CACHE_DIR / "embeddings.jsonl"


class MissingAPIKey(RuntimeError):
    """Raised with instructions rather than a bare KeyError."""


def _load_dotenv_if_present() -> None:
    """Allow the key in a gitignored .env rather than only a shell variable.

    A shell export is lost when the terminal restarts and is invisible to a
    process started earlier; a .env file avoids both traps. `.env` is already in
    .gitignore, so the key cannot be committed by accident.
    """
    env_path = config.REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:                     # parse it ourselves rather than fail
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def get_api_key() -> str:
    _load_dotenv_if_present()
    for var in config.API_KEY_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    raise MissingAPIKey(
        "No Gemini API key found. Set one of "
        f"{', '.join(config.API_KEY_ENV_VARS)} and re-run. A free key comes from "
        "https://aistudio.google.com/apikey — the corpus and chunks are already "
        "built, so only the embedding step needs it. A .env file at the repo "
        "root with GEMINI_API_KEY=... also works and is gitignored."
    )


def get_client():
    from google import genai

    return genai.Client(api_key=get_api_key())


def embed_texts(texts: list[str], task_type: str, client=None,
                batch_size: int | None = None, verbose: bool = False
                ) -> list[list[float]]:
    """Embed a list of texts, batching and backing off on rate limits."""
    from google.genai import types

    client = client or get_client()
    batch_size = batch_size or config.EMBED_BATCH_SIZE
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(config.EMBED_MAX_RETRIES):
            try:
                response = client.models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=config.EMBED_DIM),
                )
                vectors.extend(list(e.values) for e in response.embeddings)
                break
            except Exception as exc:
                message = str(exc).lower()
                retriable = ("429" in message or "quota" in message
                             or "rate" in message or "unavailable" in message
                             or "503" in message)
                if not retriable or attempt == config.EMBED_MAX_RETRIES - 1:
                    raise
                wait = config.EMBED_BACKOFF_SECONDS * (attempt + 1)
                if verbose:
                    print(f"  rate limited, waiting {wait}s "
                          f"(attempt {attempt + 1})")
                time.sleep(wait)
        if verbose:
            print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    return vectors


def _load_progress() -> dict[str, list[float]]:
    if not EMBEDDINGS_PATH.exists():
        return {}
    done = {}
    with EMBEDDINGS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                done[record["id"]] = record["embedding"]
    return done


def embed_corpus(chunks: list[dict], verbose: bool = True) -> dict[str, list[float]]:
    """Embed every chunk, skipping any already on disk from an earlier run."""
    done = _load_progress()
    pending = [c for c in chunks if c["id"] not in done]
    if verbose:
        print(f"{len(done)} chunks already embedded, {len(pending)} to go")
    if not pending:
        return done

    client = get_client()
    batch_size = config.EMBED_BATCH_SIZE
    with EMBEDDINGS_PATH.open("a", encoding="utf-8") as fh:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            vectors = embed_texts([c["text"] for c in batch],
                                  config.TASK_TYPE_DOCUMENT,
                                  client=client, batch_size=batch_size,
                                  verbose=verbose)
            for chunk, vector in zip(batch, vectors):
                done[chunk["id"]] = vector
                fh.write(json.dumps({"id": chunk["id"], "embedding": vector}) + "\n")
            fh.flush()          # resumable: progress survives a crash mid-build
            if verbose:
                print(f"  saved {min(start + batch_size, len(pending))}"
                      f"/{len(pending)}")
    return done


def embed_query(query: str, client=None) -> list[float]:
    """One query vector, with the query-side task type."""
    return embed_texts([query], config.TASK_TYPE_QUERY, client=client)[0]


if __name__ == "__main__":
    from retrieval.chunk import read_chunks

    corpus = read_chunks()
    try:
        embeddings = embed_corpus(corpus)
        print(f"{len(embeddings)} embeddings ready "
              f"({config.EMBED_DIM}-dim, {config.EMBED_MODEL})")
    except MissingAPIKey as exc:
        print(f"BLOCKED: {exc}")
