"""Characterization test for retrieval.tool's dense search path.

Written against the unchanged retrieve_context() to lock its exact behaviour —
output dict shape, collection.query kwargs, and the doc_type ValueError path —
before retrieval/tool.py is refactored to move the Chroma query into a private
_dense_search() helper (Track A, Phase 1). It must pass unedited both before
and after that refactor.
"""

from __future__ import annotations

import pytest

from retrieval import tool

FAKE_VECTOR = [0.1] * 768


class FakeCollection:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


CHROMA_RESULT = {
    "ids": [["c1", "c2", "c3"]],
    "documents": [["Text one", "Text two", "Text three"]],
    "metadatas": [[
        {"source": "Doc One", "source_type": "domain_reference",
         "citation": "http://example.com/one", "section": "Intro"},
        {"source": "Doc Two", "source_type": "project_evidence",
         "citation": "models/two.md"},  # no "section" key — locks the "" default
        {"source": "Doc Three", "source_type": "domain_reference",
         "citation": "http://example.com/three", "section": "Methods"},
    ]],
    "distances": [[0.1, 0.3333, 0.5]],
}

EXPECTED_HITS = [
    {"text": "Text one", "source": "Doc One", "source_type": "domain_reference",
     "citation": "http://example.com/one", "section": "Intro", "score": 0.9},
    {"text": "Text two", "source": "Doc Two", "source_type": "project_evidence",
     "citation": "models/two.md", "section": "", "score": 0.6667},
    {"text": "Text three", "source": "Doc Three", "source_type": "domain_reference",
     "citation": "http://example.com/three", "section": "Methods", "score": 0.5},
]


@pytest.fixture
def fake_collection(monkeypatch):
    collection = FakeCollection(CHROMA_RESULT)
    monkeypatch.setattr(tool, "get_collection", lambda: collection)
    monkeypatch.setattr(tool, "embed_query", lambda query: FAKE_VECTOR)
    return collection


def test_retrieve_context_returns_exact_expected_hits(fake_collection):
    result = tool.retrieve_context("some question", k=3)
    assert result == EXPECTED_HITS


def test_query_kwargs_no_doc_type(fake_collection):
    tool.retrieve_context("some question", k=3)
    assert len(fake_collection.calls) == 1
    kwargs = fake_collection.calls[0]
    assert kwargs["n_results"] == 3
    assert kwargs["where"] is None
    assert kwargs["include"] == ["documents", "metadatas", "distances"]
    assert kwargs["query_embeddings"] == [FAKE_VECTOR]


def test_query_kwargs_doc_type_b(fake_collection):
    tool.retrieve_context("some question", k=3, doc_type="B")
    kwargs = fake_collection.calls[0]
    assert kwargs["where"] == {"source_type": "project_evidence"}


def test_invalid_doc_type_raises_and_does_not_query(monkeypatch):
    calls = []
    monkeypatch.setattr(tool, "get_collection", lambda: calls.append("called"))
    monkeypatch.setattr(tool, "embed_query", lambda query: FAKE_VECTOR)

    with pytest.raises(ValueError):
        tool.retrieve_context("some question", k=3, doc_type="X")

    assert calls == []
