"""One-time re-key of chunk ids from the old salted hash() to sha1 content ids.

    python -m scripts.migrate_chunk_ids            # dry run: report only
    python -m scripts.migrate_chunk_ids --apply    # back up, then rewrite

Chunk ids used to be built with Python's per-process salted ``hash()``, so the
ids in the current ``chunks.jsonl``, the Chroma store, the embedding cache and
the committed chunk-level labels can never be reproduced by a rebuild — a fresh
``retrieval.build`` (every Docker build does one) would resolve none of the
1015 labelled candidates. ``retrieval.chunk`` now derives ids from the text
(sha1), and this script moves the existing artifacts onto that id space.

Only ids change. Chunk text, relevance judgments and stored rankings are
untouched, the Chroma store is rebuilt from the cached vectors (no embedding
call), and everything rewritten is first copied to a timestamped backup under
``retrieval/cache/`` (gitignored). Running it again on migrated data reports
that there is nothing to do.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime

from retrieval import config
from retrieval.chunk import chunk_id, read_chunks, write_chunks
from retrieval.embed import EMBEDDINGS_PATH

LABEL_FILES = [config.RETRIEVAL_DIR / "eval_labels_chunk.json",
               config.RETRIEVAL_DIR / "eval_labels_chunk_hard.json"]
POOL_CACHE_PATH = config.CACHE_DIR / "label_pool_cache.json"


def build_mapping(chunks: list[dict]) -> dict[str, str]:
    mapping = {c["id"]: chunk_id(c["source_id"], c["text"]) for c in chunks}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("two chunks would share a sha1 id — refusing to migrate")
    return mapping


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8", newline="\n")


def rekey_labels(data: dict, mapping: dict[str, str]) -> int:
    changed = 0
    for query in data["queries"].values():
        for candidate in query["candidates"]:
            new = mapping[candidate["id"]]
            changed += new != candidate["id"]
            candidate["id"] = new
    return changed


def rekey_results(data: dict, mapping: dict[str, str]) -> int:
    changed = 0
    for result_set in data["sets"].values():
        for row in result_set["per_query"] + result_set.get("negatives", []):
            for scores in row["configs"].values():
                new = [mapping[i] for i in scores["ranked_ids"]]
                changed += sum(a != b for a, b in zip(new, scores["ranked_ids"]))
                scores["ranked_ids"] = new
    return changed


def rekey_pool_cache(data: dict, mapping: dict[str, str]) -> int:
    changed = 0
    for entry in data.values():
        pools = entry["candidate_pools"]
        entry["candidate_pools"] = {mapping.get(k, k): v for k, v in pools.items()}
        changed += sum(k in mapping and mapping[k] != k for k in pools)
    return changed


def _check_everything_maps(mapping: dict[str, str]) -> None:
    """Every id the labels and stored rankings use must be a current chunk."""
    missing = set()
    for path in LABEL_FILES:
        for query in _load_json(path)["queries"].values():
            missing |= {c["id"] for c in query["candidates"]} - mapping.keys()
    if config.EVAL_RESULTS_HYBRID_PATH.exists():
        results = _load_json(config.EVAL_RESULTS_HYBRID_PATH)
        for result_set in results["sets"].values():
            for row in result_set["per_query"] + result_set.get("negatives", []):
                for scores in row["configs"].values():
                    missing |= set(scores["ranked_ids"]) - mapping.keys()
    if missing:
        raise ValueError(f"{len(missing)} labelled/ranked ids are not in chunks.jsonl, "
                         f"e.g. {sorted(missing)[:3]} — the labels were not made "
                         "against this corpus, so they cannot be re-keyed from it")


def _backup(paths: list, chroma: bool) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = config.CACHE_DIR / f"chunk_id_migration_backup_{stamp}"
    target.mkdir(parents=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, target / path.name)
    if chroma:
        shutil.copytree(config.CHROMA_DIR, target / "chroma_store")
    return str(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true",
                        help="back up and rewrite (default is a dry run)")
    args = parser.parse_args(argv)

    chunks = read_chunks()
    mapping = build_mapping(chunks)
    to_change = sum(old != new for old, new in mapping.items())
    print(f"{len(chunks)} chunks, {to_change} ids would change")
    if not to_change:
        print("nothing to do — ids are already content-derived")
        return 0

    _check_everything_maps(mapping)
    print("every labelled and ranked id maps onto a current chunk")

    embeddings = {}
    with EMBEDDINGS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                embeddings[record["id"]] = record["embedding"]
    missing_vectors = [i for i in mapping if i not in embeddings]
    if missing_vectors:
        raise ValueError(f"{len(missing_vectors)} chunks have no cached vector — "
                         "the store could not be rebuilt without re-embedding")
    print(f"all {len(chunks)} vectors present in the embedding cache "
          f"({len(embeddings) - len(chunks)} stale entries will be dropped)")

    if not args.apply:
        print("\ndry run — re-run with --apply to back up and rewrite")
        return 0

    touched = [config.CHUNKS_PATH, EMBEDDINGS_PATH, POOL_CACHE_PATH,
               config.EVAL_RESULTS_HYBRID_PATH, *LABEL_FILES]
    print(f"backup: {_backup(touched, chroma=True)}")

    new_chunks = [{**c, "id": mapping[c["id"]]} for c in chunks]
    write_chunks(new_chunks)

    new_vectors = {mapping[old]: embeddings[old] for old in mapping}
    with EMBEDDINGS_PATH.open("w", encoding="utf-8") as fh:
        for chunk in new_chunks:
            fh.write(json.dumps({"id": chunk["id"],
                                 "embedding": new_vectors[chunk["id"]]}) + "\n")

    from retrieval.store import build_store
    build_store(new_chunks, new_vectors, reset=True)

    for path in LABEL_FILES:
        data = _load_json(path)
        print(f"{path.name}: {rekey_labels(data, mapping)} candidate ids re-keyed")
        _write_json(path, data)

    if config.EVAL_RESULTS_HYBRID_PATH.exists():
        data = _load_json(config.EVAL_RESULTS_HYBRID_PATH)
        print(f"{config.EVAL_RESULTS_HYBRID_PATH.name}: "
              f"{rekey_results(data, mapping)} ranked ids re-keyed")
        _write_json(config.EVAL_RESULTS_HYBRID_PATH, data)

    if POOL_CACHE_PATH.exists():
        data = _load_json(POOL_CACHE_PATH)
        print(f"{POOL_CACHE_PATH.name}: {rekey_pool_cache(data, mapping)} ids re-keyed")
        _write_json(POOL_CACHE_PATH, data)

    print("\nmigration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
