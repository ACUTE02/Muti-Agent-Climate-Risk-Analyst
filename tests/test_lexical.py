"""retrieval.lexical — BM25 lexical retriever (Track A, Phase 1).

Offline, no API key. Pure-function tests (tokenizer, resolve_source_type) run
unconditionally. Everything touching the corpus/index is skipped if
chunks.jsonl doesn't exist; the Chroma ID-parity check is separately skipped
if the store is unavailable, since it needs neither embedding nor Gemini.
"""

from __future__ import annotations

import pytest

from retrieval import config
from retrieval.chunk import read_chunks
from retrieval.lexical import bm25_search, get_bm25_index, resolve_source_type, tokenize

needs_corpus = pytest.mark.skipif(
    not config.CHUNKS_PATH.exists(),
    reason="no chunks yet — run `python -m retrieval.build`",
)


class TestTokenize:
    def test_keeps_numbers_and_codes_as_single_tokens(self):
        tokens = tokenize("Skill +0.0766 at SPI-3, threshold 4.5")
        assert "0.0766" in tokens
        assert "spi-3" in tokens
        assert "4.5" in tokens

    def test_lowercases_and_drops_stopwords(self):
        assert tokenize("What is the GKMS") == ["gkms"]


class TestResolveSourceType:
    def test_none_maps_to_none(self):
        assert resolve_source_type(None) is None

    def test_a_and_domain_reference_map_to_domain_reference(self):
        assert resolve_source_type("A") == "domain_reference"
        assert resolve_source_type("domain_reference") == "domain_reference"

    def test_b_and_project_evidence_map_to_project_evidence(self):
        assert resolve_source_type("B") == "project_evidence"
        assert resolve_source_type("project_evidence") == "project_evidence"

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError):
            resolve_source_type("X")


@needs_corpus
def test_index_covers_every_chunk_exactly_once():
    index = get_bm25_index()
    assert len(index.ids) == len(read_chunks())
    assert len(set(index.ids)) == len(index.ids)


@needs_corpus
def test_bm25_ids_equal_chroma_ids():
    from retrieval.store import get_collection

    try:
        collection = get_collection()
        chroma_ids = set(collection.get(include=[])["ids"])
    except Exception:
        pytest.skip("Chroma store unavailable")

    assert set(get_bm25_index().ids) == chroma_ids


@needs_corpus
def test_sanity_gkms_query_top_hit():
    results = bm25_search("GKMS standard operating procedure", n=1)
    assert results[0]["chunk"]["source_id"] == "imd_gkms_sop"


@needs_corpus
def test_results_are_sorted_positive_and_ranked():
    results = bm25_search("drought", n=10)
    assert len(results) <= 10
    scores = [r["bm25_score"] for r in results]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))


@needs_corpus
def test_determinism():
    first = [r["id"] for r in bm25_search("drought", n=10)]
    second = [r["id"] for r in bm25_search("drought", n=10)]
    assert first == second


@needs_corpus
@pytest.mark.parametrize("doc_type", ["A", "B"])
def test_filter_parity(doc_type):
    results = bm25_search("drought", n=20, doc_type=doc_type)
    assert results
    expected_source_type = resolve_source_type(doc_type)
    assert all(r["chunk"]["source_type"] == expected_source_type for r in results)


@needs_corpus
@pytest.mark.parametrize("query", ["", "   ", "the of and", "zzqxv"])
def test_edge_case_queries_return_empty_list(query):
    assert bm25_search(query, n=5) == []


@needs_corpus
def test_n_zero_returns_empty_list():
    assert bm25_search("drought", n=0) == []


def test_invalid_doc_type_raises():
    with pytest.raises(ValueError):
        bm25_search("drought", doc_type="X")
