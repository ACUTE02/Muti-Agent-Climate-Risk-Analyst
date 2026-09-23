"""Cross-encoder re-ranking (Track A, Phase 4).

Every test here needs sentence-transformers (requirements-rerank.txt), so the
whole module is skipped when it isn't installed. No Gemini calls: real-corpus
tests stub embed_query with a vector already stored in the Chroma index. Ad hoc
queries only -- nothing here reads the eval label files.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from retrieval import config, rerank as rerank_module, tool  # noqa: E402

needs_corpus = pytest.mark.skipif(
    not config.CHUNKS_PATH.exists(),
    reason="no chunks yet — run `python -m retrieval.build`",
)


def _candidate(text: str, score: float = 0.5) -> dict:
    return {"text": text, "source": "S", "source_type": "domain_reference",
            "citation": "http://example.com", "section": "", "score": score}


class FakeModel:
    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return [self._scores[text] for _, text in pairs]


# --------------------------------------------------------------------------- #
# rerank() on hand-built candidates
# --------------------------------------------------------------------------- #
def test_better_match_ranks_first_with_real_model():
    query = "What temperature defines a heat wave?"
    good = _candidate("A heat wave is declared when the maximum temperature reaches "
                      "at least 40 degrees Celsius over the plains.", score=0.1)
    bad = _candidate("The cooperative society distributes seeds and fertiliser "
                     "to member farmers each sowing season.", score=0.9)

    result = rerank_module.rerank(query, [bad, good], k=2)

    assert result[0]["text"] == good["text"]
    assert result[0]["rerank_score"] > result[1]["rerank_score"]
    assert result[0]["score"] == 0.1          # retrieval score left untouched


@pytest.mark.parametrize("candidates,k", [([], 5), ([_candidate("x")], 0)])
def test_empty_input_or_zero_k_never_calls_model(monkeypatch, candidates, k):
    def boom():
        raise AssertionError("model must not be loaded")

    monkeypatch.setattr(rerank_module, "get_reranker", boom)
    assert rerank_module.rerank("q", candidates, k) == []


def test_equal_scores_preserve_original_order(monkeypatch):
    monkeypatch.setattr(rerank_module, "get_reranker",
                        lambda: FakeModel({"first": 1.0, "second": 1.0, "third": 1.0}))
    result = rerank_module.rerank(
        "q", [_candidate("first"), _candidate("second"), _candidate("third")], k=3)
    assert [c["text"] for c in result] == ["first", "second", "third"]


def test_truncates_to_top_k_by_rerank_score(monkeypatch):
    scores = {"a": -2.0, "b": 3.5, "c": 0.1, "d": 7.0, "e": 1.2}
    monkeypatch.setattr(rerank_module, "get_reranker", lambda: FakeModel(scores))

    result = rerank_module.rerank("q", [_candidate(t) for t in "abcde"], k=3)

    assert [c["text"] for c in result] == ["d", "b", "e"]
    assert [c["rerank_score"] for c in result] == [7.0, 3.5, 1.2]


def test_does_not_mutate_input_candidates(monkeypatch):
    monkeypatch.setattr(rerank_module, "get_reranker", lambda: FakeModel({"a": 1.0}))
    original = _candidate("a")
    rerank_module.rerank("q", [original], k=1)
    assert "rerank_score" not in original


# --------------------------------------------------------------------------- #
# retrieve_context(rerank=True) wiring and the LLM-facing tool
# --------------------------------------------------------------------------- #
def test_rerank_pulls_wider_candidate_pool(monkeypatch):
    recorded = {}

    def fake_dense(query, n, doc_type=None, query_vector=None):
        recorded["n"] = n
        return [{"id": f"c{i}", "text": f"t{i}", "distance": 0.1 * i,
                 "meta": {"source": "S", "source_type": "domain_reference",
                          "citation": "c", "section": ""}}
                for i in range(n)]

    monkeypatch.setattr(tool, "_dense_search", fake_dense)
    monkeypatch.setattr(rerank_module, "get_reranker",
                        lambda: FakeModel({f"t{i}": float(i) for i in range(40)}))

    result = tool.retrieve_context("q", k=3, mode="dense", rerank=True)

    assert recorded["n"] == config.RERANK_CANDIDATE_N
    assert [r["text"] for r in result] == [f"t{config.RERANK_CANDIDATE_N - 1 - i}"
                                           for i in range(3)]


def test_llm_facing_tool_exposes_neither_rerank_nor_mode():
    assert set(tool.retrieve_context_tool.args) == {"query", "k", "doc_type"}


# --------------------------------------------------------------------------- #
# Real corpus + real Chroma store + real cross-encoder, embedding stubbed
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


@needs_corpus
@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_rerank_on_real_corpus(offline_query_vector, mode):
    label_files = [config.RETRIEVAL_DIR / "eval_labels_chunk.json",
                   config.RETRIEVAL_DIR / "eval_labels_chunk_hard.json"]
    before = {p: p.stat().st_mtime_ns for p in label_files if p.exists()}

    results = tool.retrieve_context("drought relief measures for livestock", k=5,
                                    mode=mode, rerank=True)

    assert 0 < len(results) <= 5
    assert all("score" in r and "rerank_score" in r for r in results)
    rerank_scores = [r["rerank_score"] for r in results]
    assert rerank_scores == sorted(rerank_scores, reverse=True)
    assert {p: p.stat().st_mtime_ns for p in before} == before
