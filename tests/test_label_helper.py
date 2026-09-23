"""retrieval.label_helper — TREC-style pooling for chunk-level labels (Track A, Phase 2).

Fully offline: the dense side is always stubbed out, so no Gemini call is ever
made by this test file, and the corpus (read_chunks) is replaced with a small
synthetic fixture so results don't depend on the real chunks.jsonl.
"""

from __future__ import annotations

import json

import pytest

from retrieval import label_helper

CHUNKS = [
    {"id": "c1", "text": "The drought forecast skill score is +0.0766 at t+2.",
     "source": "Doc A", "source_id": "docA", "source_type": "project_evidence",
     "doc_type": "B", "citation": "models/a.md", "section": "Results", "chars": 50},
    {"id": "c2", "text": "Heat wave season in the plains begins around April.",
     "source": "Doc B", "source_id": "docB", "source_type": "domain_reference",
     "doc_type": "A", "citation": "http://example.com/b", "section": "Intro", "chars": 52},
    {"id": "c3", "text": "Unrelated filler text about something else entirely.",
     "source": "Doc C", "source_id": "docC", "source_type": "domain_reference",
     "doc_type": "A", "citation": "http://example.com/c", "section": "Other", "chars": 53},
]


@pytest.fixture(autouse=True)
def fixed_corpus(monkeypatch):
    monkeypatch.setattr(label_helper, "read_chunks", lambda: CHUNKS)
    # keeps _default_keywords() from calling the real (Gemini-free, but still
    # real-corpus-backed) BM25 index; treat every keyword as rare in tests
    # that don't specifically exercise the document-frequency filter.
    monkeypatch.setattr(label_helper, "_keyword_document_frequency", lambda keyword: 0)


@pytest.fixture(autouse=True)
def no_dense_calls(monkeypatch):
    # every test must stub this explicitly if it wants dense hits; default to none
    # so an unstubbed test can never reach the real (Gemini-backed) _dense_search.
    monkeypatch.setattr(label_helper, "_dense_search", lambda *a, **k: [])


class TestGrepPool:
    def test_matches_chunk_containing_keyword(self):
        hits = label_helper._grep_pool(["drought"], CHUNKS, source_type=None)
        assert [c["id"] for c in hits] == ["c1"]

    def test_misses_chunk_without_keyword(self):
        hits = label_helper._grep_pool(["heat"], CHUNKS, source_type=None)
        ids = [c["id"] for c in hits]
        assert "c2" in ids
        assert "c1" not in ids
        assert "c3" not in ids


class TestDefaultKeywords:
    def test_drops_short_tokens(self, monkeypatch):
        monkeypatch.setattr(label_helper, "_keyword_document_frequency", lambda kw: 0)
        assert "at" not in label_helper._default_keywords("heat wave at noon")

    def test_drops_keywords_above_the_document_frequency_threshold(self, monkeypatch):
        common = {"drought"}
        monkeypatch.setattr(
            label_helper, "_keyword_document_frequency",
            lambda kw: label_helper.GREP_MAX_DOC_FREQ if kw in common else 0,
        )
        keywords = label_helper._default_keywords("drought skill score")
        assert "drought" not in keywords
        assert "skill" in keywords
        assert "score" in keywords


class TestPoolCandidates:
    def test_dedupes_by_id(self, monkeypatch):
        monkeypatch.setattr(label_helper, "bm25_search",
                            lambda query, n=20, doc_type=None: [{"id": "c1"}])
        monkeypatch.setattr(label_helper, "_dense_search",
                            lambda query, n, doc_type=None: [{"id": "c1"}])

        candidates = label_helper.pool_candidates("drought", keywords=["drought"])

        assert len(candidates) == 1
        assert candidates[0]["id"] == "c1"

    def test_chunk_found_by_two_pools_shows_both(self, monkeypatch):
        monkeypatch.setattr(label_helper, "bm25_search",
                            lambda query, n=20, doc_type=None: [{"id": "c1"}])
        monkeypatch.setattr(label_helper, "_dense_search",
                            lambda query, n, doc_type=None: [{"id": "c1"}])

        candidates = label_helper.pool_candidates("drought", keywords=[])

        assert len(candidates) == 1
        assert candidates[0]["pools"] == ["bm25", "dense"]

    def test_grep_only_hit_is_included(self, monkeypatch):
        monkeypatch.setattr(label_helper, "bm25_search",
                            lambda query, n=20, doc_type=None: [])
        monkeypatch.setattr(label_helper, "_dense_search",
                            lambda query, n, doc_type=None: [])

        candidates = label_helper.pool_candidates("heat", keywords=["heat"])

        assert len(candidates) == 1
        assert candidates[0]["id"] == "c2"
        assert candidates[0]["pools"] == ["grep"]

    def test_no_pool_hit_means_no_candidate(self, monkeypatch):
        monkeypatch.setattr(label_helper, "bm25_search",
                            lambda query, n=20, doc_type=None: [])
        monkeypatch.setattr(label_helper, "_dense_search",
                            lambda query, n, doc_type=None: [])

        candidates = label_helper.pool_candidates("nothing matches", keywords=["zzqxv"])

        assert candidates == []


class TestScaffoldRoundTrip:
    def test_writes_scaffold_with_null_labels_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setattr(label_helper, "CACHE_PATH", tmp_path / "label_pool_cache.json")
        monkeypatch.setattr(label_helper.config, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(label_helper, "bm25_search",
                            lambda query, n=20, doc_type=None: [{"id": "c1"}])
        monkeypatch.setattr(label_helper, "_dense_search",
                            lambda query, n, doc_type=None: [])

        queries_path = tmp_path / "queries.json"
        queries_path.write_text(json.dumps({
            "description": "test",
            "k": 5,
            "queries": [
                {"id": "q1", "query": "drought skill", "doc_type": "B",
                 "acceptable_sources": ["docA"]},
            ],
        }), encoding="utf-8")
        out_path = tmp_path / "scaffold.json"

        label_helper.build_scaffold(str(queries_path), str(out_path))

        with open(out_path, encoding="utf-8") as fh:
            scaffold = json.load(fh)

        candidates = scaffold["queries"]["q1"]["candidates"]
        assert len(candidates) == 1
        assert candidates[0]["relevant"] is None

    def test_cache_hit_skips_dense_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(label_helper, "CACHE_PATH", tmp_path / "label_pool_cache.json")
        monkeypatch.setattr(label_helper.config, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(label_helper, "bm25_search",
                            lambda query, n=20, doc_type=None: [{"id": "c1"}])

        dense_calls = []

        def counting_dense(query, n, doc_type=None):
            dense_calls.append(query)
            return []

        monkeypatch.setattr(label_helper, "_dense_search", counting_dense)

        queries_path = tmp_path / "queries.json"
        queries_path.write_text(json.dumps({
            "description": "test", "k": 5,
            "queries": [{"id": "q1", "query": "drought skill", "doc_type": "B",
                        "acceptable_sources": ["docA"]}],
        }), encoding="utf-8")
        out_path = tmp_path / "scaffold.json"

        label_helper.build_scaffold(str(queries_path), str(out_path))
        assert len(dense_calls) == 1

        label_helper.build_scaffold(str(queries_path), str(out_path))
        assert len(dense_calls) == 1  # second run hit the cache, no new dense call
