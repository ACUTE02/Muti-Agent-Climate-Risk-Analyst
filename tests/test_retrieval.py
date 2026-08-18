"""Retrieval corpus and chunking contracts.

These run offline against the cached corpus — no API key and no built index
required, so they guard the parts that are decided at build time: what is in the
corpus, what was dropped and why, and whether every chunk can be cited.
"""

import json

import pytest

from retrieval import config
from retrieval.chunk import (chunk_markdown_by_heading, chunk_prose_with_overlap,
                             read_chunks)
from retrieval.sources import SourceText, load_project_document

pytestmark = pytest.mark.skipif(
    not config.MANIFEST_PATH.exists(),
    reason="corpus not built — run `python -m retrieval.build`",
)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chunks():
    return read_chunks()


# --------------------------------------------------------------------------- #
# Corpus composition
# --------------------------------------------------------------------------- #
def test_manifest_records_every_configured_source(manifest):
    recorded = {d["id"] for d in manifest["documents"]}
    expected = {s["id"] for s in config.TYPE_A_SOURCES + config.TYPE_B_SOURCES}
    assert recorded == expected


def test_dead_ndma_drought_page_is_not_used(manifest):
    """It is a hard 404 and must not be re-added on the assumption it exists."""
    urls = [s["url"] for s in config.TYPE_A_SOURCES]
    assert config.NDMA_DROUGHT_DEAD_PAGE not in urls
    dead = manifest["verified_dead_and_excluded"]
    assert dead[0]["url"] == config.NDMA_DROUGHT_DEAD_PAGE
    assert dead[0]["replaced_by"]


def test_corpus_covers_drought_domain_not_only_heat():
    """A domain-reference corpus that was heat-only would let a drought question
    fall back on the model's own recall — the exact failure this phase exists to
    prevent. Both the official guidelines and the SPI methodology are required."""
    ids = {s["id"] for s in config.TYPE_A_SOURCES}
    assert {"ndma_drought_guidelines", "nih_spi_methodology"} <= ids
    assert len(config.TYPE_A_SOURCES) == 6


def test_eval_set_actually_tests_the_drought_sources():
    """The blind spot that made this necessary: a query set with no drought
    queries would score perfectly while testing none of the drought corpus."""
    import json as _json
    spec = _json.loads(config.EVAL_QUERIES_PATH.read_text(encoding="utf-8"))
    targeted = {s for q in spec["queries"] for s in q["acceptable_sources"]}
    assert "nih_spi_methodology" in targeted
    assert "ndma_drought_guidelines" in targeted
    assert len(spec["queries"]) >= 10


def test_every_type_a_source_is_targeted_by_some_query():
    """No indexed domain document should go entirely untested."""
    import json as _json
    spec = _json.loads(config.EVAL_QUERIES_PATH.read_text(encoding="utf-8"))
    targeted = {s for q in spec["queries"] for s in q["acceptable_sources"]}
    untested = {s["id"] for s in config.TYPE_A_SOURCES} - targeted
    assert not untested, f"no eval query targets: {sorted(untested)}"


def test_every_indexed_document_produced_chunks(manifest):
    for doc in manifest["documents"]:
        if doc["usable"]:
            assert doc.get("chunks", 0) > 0, f"{doc['id']} produced no chunks"
        else:
            assert doc["dropped_reason"], f"{doc['id']} dropped without a reason"


def test_both_document_types_are_represented(manifest):
    assert manifest["chunk_counts"].get("A", 0) > 0
    assert manifest["chunk_counts"].get("B", 0) > 0


def test_live_outlooks_are_not_baked_into_the_index(manifest, chunks):
    """Type C changes every season — it must never be in the static index."""
    live_ids = {s["id"] for s in config.TYPE_C_SOURCES}
    assert not (live_ids & {c["source_id"] for c in chunks})
    assert {s["id"] for s in manifest["live_sources_not_indexed"]} == live_ids


def test_live_outlooks_declare_which_agents_they_serve():
    for src in config.TYPE_C_SOURCES:
        assert src["relevant_to"], f"{src['id']} has no relevant_to tag"
        assert set(src["relevant_to"]) <= {"drought", "heat_stress"}


# --------------------------------------------------------------------------- #
# Chunk quality and citability
# --------------------------------------------------------------------------- #
def test_every_chunk_carries_citation_metadata(chunks):
    for chunk in chunks:
        assert chunk["text"].strip()
        assert chunk["source"]
        assert chunk["citation"]
        assert chunk["source_type"] in {"domain_reference", "project_evidence"}


def test_chunk_ids_are_unique(chunks):
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_no_chunk_is_a_bare_heading_fragment(chunks):
    for chunk in chunks:
        assert chunk["chars"] >= config.MIN_CHUNK_CHARS


def test_project_evidence_chunks_keep_their_section_heading(chunks):
    evidence = [c for c in chunks if c["source_type"] == "project_evidence"]
    assert evidence
    assert sum(1 for c in evidence if c["section"]) / len(evidence) > 0.8


def test_header_aware_chunking_keeps_a_table_with_its_caveat():
    """The reason Type B is chunked by heading: a skill score separated from its
    'weak/directional' label is exactly the half-truth this corpus prevents."""
    markdown = (
        "# Doc\n\n"
        "## Results\n\n"
        "| Horizon | Skill |\n|---|---|\n| t+2 | +0.0766 |\n\n"
        "That is weak/directional, not validated, and should not be relied on.\n\n"
        "## Something else\n\n"
        "Unrelated prose that must not be pulled into the results section. "
        + "Filler. " * 30
    )
    source = SourceText("doc", "Doc", markdown, "models/x.md", usable=True)
    produced = chunk_markdown_by_heading(source)

    results = [c for c in produced if "+0.0766" in c["text"]]
    assert len(results) == 1
    assert "weak/directional" in results[0]["text"]
    assert "Unrelated prose" not in results[0]["text"]


def test_prose_chunking_overlaps_consecutive_chunks():
    paragraphs = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(12))
    source = SourceText("p", "P", paragraphs, "http://x", usable=True)
    produced = chunk_prose_with_overlap(source)

    assert len(produced) > 1
    for chunk in produced:
        assert chunk["chars"] <= config.CHUNK_CHARS + config.CHUNK_OVERLAP_CHARS + 200
    tail = produced[0]["text"][-60:]
    assert tail.strip() in produced[1]["text"]


def test_project_documents_load_from_the_repo():
    for src in config.TYPE_B_SOURCES:
        loaded = load_project_document(src)
        assert loaded.usable, f"{src['id']}: {loaded.reason}"
        assert loaded.citation == src["path"]        # repo-relative, not a URL


def test_the_measured_skill_numbers_are_actually_in_the_corpus(chunks):
    """The corpus must be able to answer 'how reliable is it' from evidence."""
    evidence = " ".join(c["text"] for c in chunks
                        if c["source_type"] == "project_evidence")
    for number in ("+0.2622", "+0.0766", "-0.0145", "+0.0378"):
        assert number in evidence, f"{number} missing from project evidence"
