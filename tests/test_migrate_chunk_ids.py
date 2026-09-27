"""scripts.migrate_chunk_ids — re-keying chunk ids must change ids and nothing else.

Pure tests on synthetic data: no corpus, no Chroma, no network.
"""

from __future__ import annotations

import copy

import pytest

from retrieval.chunk import chunk_id
from scripts import migrate_chunk_ids as mig

CHUNKS = [
    {"id": "docA::111111111111", "source_id": "docA", "text": "first chunk"},
    {"id": "docB::222222222222", "source_id": "docB", "text": "second chunk"},
]


def test_mapping_uses_the_same_id_function_as_the_chunker():
    mapping = mig.build_mapping(CHUNKS)
    assert mapping == {c["id"]: chunk_id(c["source_id"], c["text"]) for c in CHUNKS}


def test_mapping_refuses_a_collision():
    twins = [{**CHUNKS[0], "id": "x::1"}, {**CHUNKS[0], "id": "x::2"}]
    with pytest.raises(ValueError, match="share a sha1 id"):
        mig.build_mapping(twins)


def test_rekey_labels_changes_ids_and_keeps_judgments():
    mapping = mig.build_mapping(CHUNKS)
    labels = {"queries": {"q1": {"query": "?", "candidates": [
        {"id": "docA::111111111111", "relevant": True, "preview": "p"},
        {"id": "docB::222222222222", "relevant": False, "preview": "p"}]}}}
    before = copy.deepcopy(labels)

    changed = mig.rekey_labels(labels, mapping)

    assert changed == 2
    after = labels["queries"]["q1"]["candidates"]
    assert [c["id"] for c in after] == [mapping[c["id"]]
                                        for c in before["queries"]["q1"]["candidates"]]
    assert [c["relevant"] for c in after] == [True, False]
    assert all(c["preview"] == "p" for c in after)


def test_rekey_results_keeps_scores_and_order():
    mapping = mig.build_mapping(CHUNKS)
    scores = {"chunk_precision_at_5": 0.2, "mrr": 1.0, "outside_pool_at_5": 0,
              "ranked_ids": ["docB::222222222222", "docA::111111111111"]}
    results = {"sets": {"soft": {"per_query": [{"configs": {"dense": dict(scores)}}],
                                 "negatives": []}}}

    mig.rekey_results(results, mapping)

    out = results["sets"]["soft"]["per_query"][0]["configs"]["dense"]
    assert out["ranked_ids"] == [mapping["docB::222222222222"],
                                 mapping["docA::111111111111"]]
    assert (out["chunk_precision_at_5"], out["mrr"]) == (0.2, 1.0)


def test_rekey_labels_fails_loudly_on_an_unknown_id():
    labels = {"queries": {"q": {"candidates": [{"id": "gone::0", "relevant": True}]}}}
    with pytest.raises(KeyError):
        mig.rekey_labels(labels, mig.build_mapping(CHUNKS))
