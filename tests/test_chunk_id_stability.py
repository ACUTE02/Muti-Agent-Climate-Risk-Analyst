"""Chunk-ID stability (Track A, Phase 2, Part 1).

_chunk_record() used to build a chunk's id suffix from Python's built-in
hash(), which is salted per process by default — every `python -m
retrieval.build` run would silently reassign every chunk ID. The fix is a
content-derived hash (sha1). A within-process repeat call couldn't have caught
the original bug, since hash() is self-consistent within one process; the real
regression test has to cross a process boundary.
"""

from __future__ import annotations

import subprocess
import sys

from retrieval.chunk import _chunk_record
from retrieval.sources import SourceText

_SUBPROCESS_SCRIPT = """
from retrieval.chunk import _chunk_record
from retrieval.sources import SourceText

source = SourceText("fixture_doc", "Fixture Doc", "irrelevant", "fixture://x", usable=True)
texts = ["First fixed chunk of text.", "Second fixed chunk of text."]
for text in texts:
    print(_chunk_record(source, text, "A")["id"])
"""


def test_chunk_ids_are_deterministic_across_processes():
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT],
            capture_output=True, text=True, check=True,
        )
        outputs.append([line.strip() for line in result.stdout.splitlines() if line.strip()])

    assert len(outputs[0]) == 2
    assert outputs[0] == outputs[1]


def test_chunk_id_is_deterministic_within_a_process():
    source = SourceText("fixture_doc", "Fixture Doc", "irrelevant", "fixture://x", usable=True)
    text = "A repeated fixed chunk of text."

    first = _chunk_record(source, text, "A")["id"]
    second = _chunk_record(source, text, "A")["id"]

    assert first == second
