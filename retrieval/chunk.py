"""Chunking — a different strategy per document type, for a reason.

**Type B (this project's markdown)** is already structured with one topic per
section: a phase, a result table, a verdict. Header-aware chunking keeps a table
together with the caveat that qualifies it, which matters enormously here — a
retrieved skill score without its "weak/directional" label is exactly the kind of
half-truth this corpus exists to prevent.

**Type A (external prose)** has no such structure, so it gets paragraph packing
with overlap, the standard approach.

Every chunk carries its source metadata, because a retrieved fact without a
citation is indistinguishable from a model's invention.

Run standalone:  python -m retrieval.chunk
"""

from __future__ import annotations

import json
import re

from retrieval import config
from retrieval.sources import SourceText

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def chunk_markdown_by_heading(source: SourceText, doc_type: str = "B") -> list[dict]:
    """Split on ## / ### boundaries, keeping each section whole where possible.

    A section longer than the chunk budget is split further on paragraph breaks,
    but every piece keeps the heading path so the citation stays precise.
    """
    lines = source.text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    buffer: list[str] = []

    for line in lines:
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) <= 3:
            if buffer:
                sections.append((heading, buffer))
            heading = match.group(2).strip()
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, buffer))

    chunks = []
    for heading, body in sections:
        text = "\n".join(body).strip()
        if len(text) < config.MIN_CHUNK_CHARS:
            continue
        for part in _split_long(text):
            chunks.append(_chunk_record(source, part, doc_type, heading))
    return chunks


def chunk_prose_with_overlap(source: SourceText, doc_type: str = "A") -> list[dict]:
    """Pack paragraphs up to the chunk budget, carrying an overlap tail forward."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", source.text) if p.strip()]

    chunks: list[dict] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= config.CHUNK_CHARS:
            current = candidate
            continue
        if current:
            chunks.append(_chunk_record(source, current, doc_type))
            current = (current[-config.CHUNK_OVERLAP_CHARS:] + "\n\n" + para).strip()
        else:                                   # a single oversized paragraph
            for part in _split_long(para):
                chunks.append(_chunk_record(source, part, doc_type))
            current = ""
    if current and len(current) >= config.MIN_CHUNK_CHARS:
        chunks.append(_chunk_record(source, current, doc_type))
    return chunks


def _split_long(text: str) -> list[str]:
    """Break an oversized block on paragraph, then sentence, then hard limit."""
    if len(text) <= config.CHUNK_CHARS:
        return [text]

    pieces, current = [], ""
    for unit in re.split(r"(?<=[.!?])\s+|\n", text):
        candidate = f"{current} {unit}".strip()
        if len(candidate) <= config.CHUNK_CHARS:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = unit[:config.CHUNK_CHARS]
    if current:
        pieces.append(current)
    return [p for p in pieces if len(p) >= config.MIN_CHUNK_CHARS]


def _chunk_record(source: SourceText, text: str, doc_type: str,
                  heading: str = "") -> dict:
    return {
        "id": f"{source.source_id}::{abs(hash(text)) % (10 ** 12):012d}",
        "text": text,
        "source": source.title,
        "source_id": source.source_id,
        "source_type": config.TYPE_LABELS[doc_type],
        "doc_type": doc_type,
        "citation": source.citation,
        "section": heading,
        "chars": len(text),
    }


def build_chunks(force: bool = False) -> tuple[list[dict], list[dict]]:
    """Fetch every Type A/B source, chunk it, and record what was dropped."""
    from retrieval.sources import load_project_document, load_source

    chunks: list[dict] = []
    manifest_entries: list[dict] = []

    for src in config.TYPE_A_SOURCES:
        loaded = load_source(src, force=force)
        entry = {**{k: v for k, v in src.items() if k != "format"},
                 "doc_type": "A", "format": src["format"],
                 "usable": loaded.usable, "stats": loaded.stats,
                 "dropped_reason": loaded.reason}
        if loaded.usable:
            produced = chunk_prose_with_overlap(loaded, "A")
            chunks += produced
            entry["chunks"] = len(produced)
        manifest_entries.append(entry)

    for src in config.TYPE_B_SOURCES:
        loaded = load_project_document(src)
        entry = {**src, "doc_type": "B", "usable": loaded.usable,
                 "stats": loaded.stats, "dropped_reason": loaded.reason}
        if loaded.usable:
            produced = chunk_markdown_by_heading(loaded, "B")
            chunks += produced
            entry["chunks"] = len(produced)
        manifest_entries.append(entry)

    return chunks, manifest_entries


def write_chunks(chunks: list[dict]) -> None:
    with config.CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def read_chunks() -> list[dict]:
    if not config.CHUNKS_PATH.exists():
        raise FileNotFoundError("no chunks yet — run `python -m retrieval.build`")
    with config.CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    produced, entries = build_chunks()
    write_chunks(produced)
    by_type: dict[str, int] = {}
    for chunk in produced:
        by_type[chunk["doc_type"]] = by_type.get(chunk["doc_type"], 0) + 1
    print(f"{len(produced)} chunks: {by_type}")
    for entry in entries:
        state = f"{entry.get('chunks', 0)} chunks" if entry["usable"] else \
                f"DROPPED — {entry['dropped_reason']}"
        print(f"  [{entry['doc_type']}] {entry['id']:28s} {state}")
