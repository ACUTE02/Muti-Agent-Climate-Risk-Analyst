"""retrieval.evaluate_hybrid — chunk-level dense/hybrid/rerank comparison (Track A, Phase 5).

Pure scoring tests need nothing. The real-corpus smoke test runs real BM25,
Chroma and the cross-encoder, with embed_query stubbed to a vector already in
the index, so no test here makes a live Gemini call.
"""

from __future__ import annotations

import json

import pytest

from retrieval import config, evaluate_hybrid as eh, tool

needs_corpus = pytest.mark.skipif(
    not config.CHUNKS_PATH.exists(),
    reason="no chunks yet — run `python -m retrieval.build`",
)

LABEL_FILES = [config.RETRIEVAL_DIR / "eval_labels_chunk.json",
               config.RETRIEVAL_DIR / "eval_labels_chunk_hard.json"]


@pytest.fixture(autouse=True, scope="module")
def label_files_untouched():
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in LABEL_FILES if p.exists()}
    yield
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in before}
    assert after == before


# --------------------------------------------------------------------------- #
# Pure scoring
# --------------------------------------------------------------------------- #
LABELS = {"a": True, "b": False, "c": True, "d": False, "e": False, "f": True}


class TestScoring:
    def test_precision_counts_relevant_in_top_5(self):
        assert eh.chunk_precision_at_5(["a", "b", "c", "d", "e", "f"], LABELS) == 2 / 5

    def test_unjudged_ids_count_as_not_relevant(self):
        ranked = ["x", "a", "y", "z", "w"]
        assert eh.chunk_precision_at_5(ranked, LABELS) == 1 / 5
        assert eh.mrr(ranked, LABELS) == 1 / 2
        assert eh.outside_pool_at_5(ranked, LABELS) == 4

    def test_short_list_still_divides_by_5(self):
        assert eh.chunk_precision_at_5(["a"], LABELS) == 1 / 5

    def test_mrr_uses_first_relevant_rank(self):
        assert eh.mrr(["b", "d", "c", "a"], LABELS) == 1 / 3

    def test_mrr_is_zero_when_nothing_relevant_in_top_10(self):
        ranked = ["b", "d", "e"] + [f"u{i}" for i in range(7)] + ["a"]  # "a" at rank 11
        assert eh.mrr(ranked, LABELS) == 0.0


class TestAggregation:
    def test_negative_queries_excluded_from_main_aggregate(self, monkeypatch):
        rankings = {"q_pos": ["a", "b", "c", "d", "e"], "q_neg": ["b", "d", "e", "x", "y"]}
        monkeypatch.setattr(eh, "ranked_ids_for",
                            lambda query, doc_type, mode, rerank, vec: rankings[query])
        queries = [
            {"id": "q_pos", "query": "q_pos", "doc_type": "A", "type": "exact_figure"},
            {"id": "q_neg", "query": "q_neg", "doc_type": None, "type": "negative",
             "excluded_from_mean": True},
        ]
        labels = {"q_pos": LABELS, "q_neg": LABELS}

        result = eh.evaluate_set(queries, labels, {"q_pos": [0.0], "q_neg": [0.0]})

        dense = result["aggregate"]["dense"]
        assert dense["n_queries"] == 1
        assert dense["chunk_precision_at_5"] == 2 / 5
        assert [r["id"] for r in result["negatives"]] == ["q_neg"]
        assert set(result["by_type"]) == {"exact_figure"}
        assert all(sum(p[k] for k in ("improved", "same", "regressed")) == 1
                   for p in result["paired"])

    def test_paired_counts_direction(self, monkeypatch):
        def ranking(query, doc_type, mode, rerank, vec):
            better = mode == "hybrid"
            return ["a", "c", "f", "b", "d"] if better else ["b", "d", "e", "a", "x"]

        monkeypatch.setattr(eh, "ranked_ids_for", ranking)
        result = eh.evaluate_set([{"id": "q", "query": "q", "doc_type": "A"}],
                                 {"q": LABELS}, {"q": [0.0]})
        dense_vs_hybrid = next(p for p in result["paired"] if p["compared"] == "hybrid")
        assert (dense_vs_hybrid["improved"], dense_vs_hybrid["same"],
                dense_vs_hybrid["regressed"]) == (1, 0, 0)


# --------------------------------------------------------------------------- #
# Labels must belong to this corpus build
# --------------------------------------------------------------------------- #
class TestLabelCorpusGuard:
    def test_passes_when_every_label_id_is_in_the_corpus(self):
        eh.check_labels_match_corpus({"soft": {"q": LABELS}}, set(LABELS) | {"z"})

    def test_raises_when_a_label_id_is_missing(self):
        with pytest.raises(ValueError, match="migrate_chunk_ids"):
            eh.check_labels_match_corpus({"soft": {"q": LABELS}}, {"a", "b"})

    def test_evaluate_refuses_before_spending_any_embedding_call(self, monkeypatch):
        monkeypatch.setattr(eh, "read_chunks", lambda: [])

        def no_embedding(query):
            raise AssertionError("must fail before embedding")

        monkeypatch.setattr(tool, "embed_query", no_embedding)
        with pytest.raises(ValueError, match="not in chunks.jsonl"):
            eh.evaluate()


@needs_corpus
def test_committed_labels_resolve_against_the_current_corpus():
    """Catches a rebuild or id-scheme change that would orphan the labels."""
    from retrieval.chunk import read_chunks

    corpus_ids = {c["id"] for c in read_chunks()}
    eh.check_labels_match_corpus({p.name: eh.load_labels(p) for p in LABEL_FILES},
                                 corpus_ids)


# --------------------------------------------------------------------------- #
# Real corpus smoke test, all 4 configs, no Gemini
# --------------------------------------------------------------------------- #
@needs_corpus
def test_one_soft_query_through_all_configs(monkeypatch):
    pytest.importorskip("sentence_transformers")
    try:
        got = tool.get_collection().get(limit=1, include=["embeddings"])
    except Exception:
        pytest.skip("Chroma store unavailable")
    vector = [float(x) for x in got["embeddings"][0]]

    def no_live_embedding(query):
        raise AssertionError("smoke test must not embed live")

    monkeypatch.setattr(tool, "embed_query", no_live_embedding)

    spec = json.loads(config.EVAL_QUERIES_PATH.read_text(encoding="utf-8"))
    query = spec["queries"][0]
    labels = eh.load_labels(LABEL_FILES[0])

    result = eh.evaluate_set([query], labels, {query["id"]: vector})

    row = result["per_query"][0]
    assert set(row["configs"]) == {"dense", "dense+rerank", "hybrid", "hybrid+rerank"}
    for scores in row["configs"].values():
        assert set(scores) == {"chunk_precision_at_5", "mrr", "outside_pool_at_5", "ranked_ids"}
        assert 0 < len(scores["ranked_ids"]) <= eh.SCORE_DEPTH
        assert len(set(scores["ranked_ids"])) == len(scores["ranked_ids"])
        assert 0.0 <= scores["chunk_precision_at_5"] <= 1.0
