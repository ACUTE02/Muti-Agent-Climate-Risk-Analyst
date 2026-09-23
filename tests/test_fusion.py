"""RRF hybrid fusion + retrieve_context's mode switch (Track A, Phase 3).

No Gemini calls: the pure tests use hand-built hit lists, and the real-corpus
tests replace embed_query with a vector already stored in the Chroma index, so
they exercise the real BM25 index and the real Chroma store offline. Ad hoc
queries only -- nothing here reads the eval label files.
"""

from __future__ import annotations

import json

import pytest

from retrieval import config, tool

needs_corpus = pytest.mark.skipif(
    not config.CHUNKS_PATH.exists(),
    reason="no chunks yet — run `python -m retrieval.build`",
)


def _chunk(chunk_id: str, source_type: str = "domain_reference") -> dict:
    return {"id": chunk_id, "text": f"text of {chunk_id}", "source": f"Source {chunk_id}",
            "source_id": f"src_{chunk_id}", "source_type": source_type, "doc_type": "A",
            "citation": f"http://example.com/{chunk_id}", "section": f"Section {chunk_id}",
            "chars": 20}


def _bm25_hit(chunk_id: str, rank: int) -> dict:
    return {"id": chunk_id, "bm25_score": 10.0 / rank, "rank": rank, "chunk": _chunk(chunk_id)}


def _dense_hit(chunk_id: str, distance: float) -> dict:
    chunk = _chunk(chunk_id)
    meta = {key: chunk[key] for key in tool._META_KEYS}
    return {"id": chunk_id, "text": chunk["text"], "meta": meta, "distance": distance}


def _fused_score(hit: dict) -> float:
    return 1.0 - hit["distance"]


# --------------------------------------------------------------------------- #
# Pure _rrf_fuse
# --------------------------------------------------------------------------- #
class TestRRFFuse:
    def test_scores_match_hand_computed_rrf(self):
        bm25 = [_bm25_hit("a", 1), _bm25_hit("b", 2)]
        dense = [_dense_hit("b", 0.1), _dense_hit("c", 0.2)]

        fused = tool._rrf_fuse(bm25, dense, k=10)

        assert [h["id"] for h in fused] == ["b", "a", "c"]
        scores = {h["id"]: _fused_score(h) for h in fused}
        assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)   # bm25 rank 2 + dense rank 1
        assert scores["a"] == pytest.approx(1 / 61)            # bm25 only, rank 1
        assert scores["c"] == pytest.approx(1 / 62)            # dense only, rank 2

    def test_ties_break_by_id_ascending(self):
        fused = tool._rrf_fuse([_bm25_hit("y", 1)], [_dense_hit("z", 0.1)], k=10)
        assert [h["id"] for h in fused] == ["y", "z"]
        assert _fused_score(fused[0]) == pytest.approx(_fused_score(fused[1]))

    def test_truncates_to_k(self):
        bm25 = [_bm25_hit(c, i) for i, c in enumerate("abcd", start=1)]
        assert len(tool._rrf_fuse(bm25, [], k=2)) == 2

    def test_is_deterministic(self):
        bm25 = [_bm25_hit("a", 1), _bm25_hit("b", 2), _bm25_hit("d", 3)]
        dense = [_dense_hit("c", 0.1), _dense_hit("a", 0.2), _dense_hit("b", 0.3)]
        assert tool._rrf_fuse(bm25, dense, k=4) == tool._rrf_fuse(bm25, dense, k=4)

    def test_bm25_only_hit_carries_full_dense_meta_shape(self):
        fused = tool._rrf_fuse([_bm25_hit("a", 1)], [], k=1)
        assert set(fused[0]["meta"]) == set(tool._META_KEYS)
        formatted = tool._format(fused[0])
        assert formatted == {
            "text": "text of a", "source": "Source a", "source_type": "domain_reference",
            "citation": "http://example.com/a", "section": "Section a",
            "score": round(1 / 61, 4),
        }

    def test_overlapping_hit_keeps_dense_meta(self):
        dense = _dense_hit("a", 0.1)
        dense["meta"]["section"] = "from dense"
        fused = tool._rrf_fuse([_bm25_hit("a", 1)], [dense], k=1)
        assert fused[0]["meta"]["section"] == "from dense"


# --------------------------------------------------------------------------- #
# mode switch wiring (no corpus, no API)
# --------------------------------------------------------------------------- #
class TestModeSwitch:
    def test_unknown_mode_raises_before_any_search(self, monkeypatch):
        calls = []
        monkeypatch.setattr(tool, "_dense_search", lambda *a, **k: calls.append("dense"))
        monkeypatch.setattr(tool, "bm25_search", lambda *a, **k: calls.append("bm25"))

        with pytest.raises(ValueError, match="hybrid"):
            tool.retrieve_context("q", k=5, mode="bm25")
        assert calls == []

    def test_hybrid_pulls_configured_top_n_with_doc_type(self, monkeypatch):
        recorded = {}

        def fake_bm25(query, n=5, doc_type=None):
            recorded["bm25"] = (query, n, doc_type)
            return [_bm25_hit("a", 1)]

        def fake_dense(query, n, doc_type=None, query_vector=None):
            recorded["dense"] = (query, n, doc_type)
            return [_dense_hit("b", 0.1)]

        monkeypatch.setattr(tool, "bm25_search", fake_bm25)
        monkeypatch.setattr(tool, "_dense_search", fake_dense)

        result = tool.retrieve_context("q", k=1, doc_type="B", mode="hybrid")

        assert recorded["bm25"] == ("q", config.BM25_TOP_N, "B")
        assert recorded["dense"] == ("q", config.DENSE_TOP_N, "B")
        assert len(result) == 1

    def test_default_mode_is_still_dense(self):
        assert config.RETRIEVAL_MODE == "dense"

    def test_llm_facing_tool_does_not_expose_mode(self):
        # the tool is bound to Gemini; retrieval mode stays a config decision
        assert set(tool.retrieve_context_tool.args) == {"query", "k", "doc_type"}


# --------------------------------------------------------------------------- #
# Real corpus + real Chroma store, embedding stubbed
# --------------------------------------------------------------------------- #
@pytest.fixture
def offline_query_vector(monkeypatch):
    try:
        got = tool.get_collection().get(limit=1, include=["embeddings"])
    except Exception:
        pytest.skip("Chroma store unavailable")
    vector = [float(x) for x in got["embeddings"][0]]
    monkeypatch.setattr(tool, "embed_query", lambda query: vector)
    return vector


QUERY = "drought early warning and declaration"


@needs_corpus
def test_hybrid_smoke_on_real_corpus(offline_query_vector):
    hybrid = tool.retrieve_context(QUERY, k=5, mode="hybrid")
    dense = tool.retrieve_context(QUERY, k=5, mode="dense")

    assert 0 < len(hybrid) <= 5
    assert all(set(r) == set(dense[0]) for r in hybrid)
    scores = [r["score"] for r in hybrid]
    assert scores == sorted(scores, reverse=True)


@needs_corpus
def test_hybrid_fused_ids_are_unique_on_real_corpus(offline_query_vector):
    bm25_hits = tool.bm25_search(QUERY, n=config.BM25_TOP_N)
    dense_hits = tool._dense_search(QUERY, config.DENSE_TOP_N)
    fused = tool._rrf_fuse(bm25_hits, dense_hits, k=len(bm25_hits) + len(dense_hits))

    ids = [h["id"] for h in fused]
    assert len(ids) == len(set(ids))
    assert set(ids) == {h["id"] for h in bm25_hits} | {h["id"] for h in dense_hits}


@needs_corpus
def test_default_and_dense_mode_are_byte_identical(offline_query_vector):
    default = json.dumps(tool.retrieve_context(QUERY, k=5))
    dense = json.dumps(tool.retrieve_context(QUERY, k=5, mode="dense"))
    direct = json.dumps([tool._format(h) for h in tool._dense_search(QUERY, 5)])
    assert default == dense == direct


@needs_corpus
@pytest.mark.parametrize("doc_type,source_type", [("A", "domain_reference"),
                                                  ("B", "project_evidence")])
def test_hybrid_respects_doc_type_filter(offline_query_vector, doc_type, source_type):
    results = tool.retrieve_context("drought", k=5, doc_type=doc_type, mode="hybrid")
    assert results
    assert all(r["source_type"] == source_type for r in results)
