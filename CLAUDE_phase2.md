# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 2: Retrieval Agent (RAG)

**Where this sits:** the Forecasting and Heat Stress agents are closed (see `PROJECT_LOG.md`) — two callable tools exist, `forecast_drought_risk()` and `forecast_heat_stress_risk()`. Neither of them explains itself in plain language or cites authority for the thresholds/methods they use. This phase builds the piece that does: a retrieval tool the eventual Orchestrator/Synthesis agent (Phase 3) will call to ground its answers in real documents instead of the LLM's own unverified recall.

**Zero-cost constraints, already decided in the tech spec — do not relitigate:** Gemini for embeddings and generation, ChromaDB baked into the Docker image at build time (no persistent disk on the free hosting tier), no paid vector DB service.

---

## 1. The corpus — two deliberately different kinds of documents

**Do not scrape broadly or open-endedly.** This is a bounded, curated corpus, not a live web-scraping pipeline — matches the project's zero-cost, reproducible-build discipline (the corpus is embedded once at build time from a fixed document set, not fetched live per query).

**Type A — domain reference documents** (authoritative definitions/methodology, so the system can explain *why* a threshold is what it is, with a real citation instead of an LLM guess):
- IMD Heat Wave FAQ — `https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf` (already cited in `heat/target.py`)
- NDMA Heat Wave hazard page — `https://ndma.gov.in/Natural-Hazards/Heat-Wave` (already cited)
- NDMA National Disaster Management Guidelines: Management of Drought — `https://nidm.gov.in/PDF/pubs/NDMA/13.pdf` (backup mirror: `https://www.droughtmanagement.info/literature/GovIndia_management_of_drought_2010.pdf`). Note: `https://ndma.gov.in/Natural-Hazards/Drought` and its plausible variants (`/Droughts`, `/drought`, `/Natural-Hazards`) are confirmed **dead (404)** — NDMA no longer publishes a drought hazard page at that path; this NIDM-hosted PDF is the same official 2010 NDMA guidelines document, verified fetchable and substantive (~70pp, covers drought classification, early warning, and response).
- NIH Roorkee — Standardized Precipitation Index (SPI) methodology — `https://nihroorkee.gov.in/sites/default/files/uploadfiles/SPINov2011.pdf`. Directly relevant beyond general drought context: this document explains the same gamma-distribution SPI methodology (citing the same McKee et al. 1993 source) that `forecasting/spi.py` implements — a real, authoritative Indian source grounding the project's own methodology choice, not just drought in general.
- IMD GKMS Agromet Advisory Service SOP — `https://mausam.imd.gov.in/imd_latest/contents/pdf/gkms_sop.pdf` (describes the actual government methodology for district-level agromet advisories — real, citable, India-specific)
- ATARI Jodhpur Agro-Advisory Services (Rajasthan-specific) — `https://atarijodhpur.res.in/Agro%20Advisory%20Services.pdf`

Verify each URL is actually fetchable and text-extractable before committing to it in the corpus manifest — if one is dead or unparseable, drop it and note why, don't force it in.

**Type B — this project's own evidence documents** (so the system can answer "how confident are you" with the actual measured number, not an invented one):
- `PROJECT_LOG.md`
- `models/region_comparison.md`
- `models/heat_region_comparison.md`

This is the more important half of the corpus, not a bonus. It directly prevents the failure mode this whole project has been designed around avoiding: an LLM confidently stating a skill/confidence figure it made up. When the eventual Synthesis agent is asked "how reliable is the 2-month drought forecast," it must retrieve the real answer (+0.0766/+0.0438, "weak/directional") from these documents, not generate one.

**Type C — current official outlooks (enriches the report, does not replace the model — for both Drought and Heat Stress).** Two live IMD sources, different timescales:
- **Seasonal outlook** — `https://mausam.imd.gov.in/responsive/seasonal_forecast.php`. IMD's own current seasonal bulletin covers *both* rainfall and temperature (it explicitly issues a combined "Rainfall and Temperature Outlook"), so this single source enriches both the Drought Agent's and the Heat Stress Agent's reports — it is not drought-only.
- **Extended range forecast** — `https://internal.imd.gov.in/section/nhac/dynamic/extended.pdf`. A ~2-week-ahead rainfall + temperature forecast, updated regularly. More near-term and tactical than the seasonal bulletin — this is the one that actually matters for a heat-wave-watch-style statement ("above-normal Tmax expected over the next two weeks"), since Heat Phase 1/1.1 established the project's own model has no forecasting skill at that timescale.

For either source: when the eventual Synthesis agent writes up a region's risk (drought or heat), it can add "IMD's current [seasonal/extended-range] outlook: [whatever IMD is currently saying]" alongside the project's own measured result — giving the user a fuller, better-grounded report, especially valuable for Heat Stress where the project's own forecast is genuinely absent. **This is not a comparison feature and not a substitute for the project's own model/observation** — the project's own SPI-3 forecast (and the Heat Stress Agent's observed heat wave counts) stay clearly attributed as this project's own measured result; IMD's outlook is cited separately, by name, as IMD's own statement. Never blend the two into one unattributed number.

Unlike Type A/B, Type C is **not static** — it changes every season, so it cannot be baked into the Docker image once at build time. Fetch it at query/report time (with the same defensive-fetch discipline as every external source in this project: verify the page parses, cache the result with a timestamp, and if the fetch fails, the report should say "IMD's current outlook was unavailable" rather than silently omitting context or using a stale cached value without saying so).

Save the corpus manifest (source, URL or path, fetch date, doc type A/B/C) to `retrieval/corpus_manifest.json`.

## 2. Chunking — different strategy per document type

- **Type B (this project's own markdown):** chunk by `##`/`###` heading boundaries — these documents are already well-structured with one topic per section (a phase, a table, a verdict). Header-aware chunking keeps a full result (e.g., the Phase 1.4 dedicated-model table plus its surrounding explanation) together in one chunk rather than splitting a table from its caveat.
- **Type A (external PDFs/HTML):** paragraph-based chunking with overlap (target ~500 tokens per chunk, ~50 token overlap) — standard practice for less-structured prose. Extract PDF text defensively (check the extraction actually produced readable text, not garbage/empty — some government PDFs are scanned images; if a PDF's extracted text is mostly empty or unreadable, drop it and say so rather than silently indexing noise).
- **Type C (IMD's current outlooks, seasonal + extended-range):** not chunked into the baked index at all — fetched fresh and passed to the report as a short, whole excerpt per source (a paragraph or two, not split), since it needs to read as "here is what IMD is currently saying," not as fragments competing for retrieval ranking against Type A/B. Tag each excerpt with which agent(s) it's relevant to (seasonal → both; extended-range → primarily Heat Stress, given the timescale match to that agent's gap).
- Every chunk keeps its source metadata (document title, URL/path, section heading if available) — required for citation at query time.

## 3. Embeddings

- Model: `gemini-embedding-001` (current Gemini API embedding model, free tier available).
- Output dimension: 768 (Google's own guidance allows scaling down from the 3072 default via Matryoshka Representation Learning without a proportional quality loss — 768 keeps the baked ChromaDB index small, which matters for a free-tier Docker image).
- Set `task_type="RETRIEVAL_DOCUMENT"` when embedding corpus chunks at build time, and `task_type="RETRIEVAL_QUERY"` when embedding a user's query at retrieval time — these are different embeddings by design in this model; using the wrong one for either side measurably hurts retrieval quality.
- **Rate limits, defensively:** the Heat Stress Agent's Open-Meteo fetch already hit a real per-minute rate limit on Barmer — treat the embedding API the same way. Batch requests, add exponential backoff on 429s, and log progress so a partial build can resume rather than restart from zero.

## 4. Vector store

- ChromaDB, built and populated at Docker image build time (per the standing zero-cost architecture decision — no persistent disk on the free hosting tier means the index must be baked in, not created at runtime).
- For local development before containerization exists (Phase 6), persist to `retrieval/chroma_store/` so the build step is testable now without waiting for the Docker phase.

## 5. The retrieval tool

```python
def retrieve_context(query: str, k: int = 5, doc_type: str | None = None) -> list[dict]:
    """
    Returns up to k chunks most relevant to query, each with:
      - text: the chunk content
      - source: document title
      - source_type: "project_evidence" or "domain_reference"
      - citation: URL, or a repo-relative path for project documents
      - score: similarity score
    doc_type, if given, restricts to "A" (domain reference) or "B" (project evidence).
    """
```

## 6. Evaluation — bounded, not full RAGAS yet

Full RAG evaluation (RAGAS or similar) is Phase 5 per the original build plan — do not build that machinery now. For this phase, write ~10-12 hand-authored test queries with a known-correct expected source document for each (a mix of Type A and Type B questions, e.g. "What is the IMD threshold for a severe heat wave?" → should retrieve the IMD FAQ; "What is the measured skill score for the 2-month drought forecast?" → should retrieve `region_comparison.md`'s horizon table). **Include at least one query specifically targeting the drought-domain Type A sources** — e.g. "How is the Standardized Precipitation Index calculated?" → should retrieve the NIH Roorkee SPI paper; "What are the three types of drought and how does India's early warning system work?" → should retrieve the NDMA drought guidelines. Since Heat Phase 1.1 already used a hardcoded-baseline bug as a cautionary example in this project, apply the same instinct here: a corpus with zero drought-domain Type A documents evaluated on zero drought-domain Type A queries would report a misleadingly clean precision number without ever actually testing that half of the corpus — so this addition isn't optional polish, it closes a real blind spot in the original 10-query set. Report retrieval precision — did the correct source appear in the top-k for each — honestly. If retrieval quality is poor for a category, say so; do not tune chunking/embedding choices repeatedly against this same query set until numbers look good — that's overfitting to a tiny eval set, and this project has been careful about that failure mode with the forecasting models. One honest pass, report the result.

## Stopping rule

This is a first pass at retrieval infrastructure, not a fully tuned system — the goal is a working, citable retrieval tool over a real, bounded corpus, with an honest first read on quality. If the 10-query check reveals a systemic problem (e.g., Type A retrieval works but Type B doesn't, or vice versa), report that plainly; a targeted follow-up phase to fix a specific, diagnosed problem is reasonable, but do not open-endedly re-chunk/re-embed chasing marginal gains on the eval set.

**Note if you're picking this up after an earlier partial run:** if a corpus/index already exists without the two drought sources above (check `retrieval/corpus_manifest.json` — it should list 6 Type A documents, not 4), rebuild it. Re-run the embedding step for the two new documents (existing embeddings for unchanged documents don't need to be redone if the build script is already resumable/incremental), then re-run the full evaluation set including the new drought-domain queries. Update `PROJECT_LOG.md`'s Phase 2 entry to close out the "known gap" it currently documents, rather than leaving that note stale once the gap is fixed.

## Definition of Done

- [ ] `retrieval/corpus_manifest.json` — every Type A source verified fetchable/extractable before inclusion, dead/unparseable ones dropped and noted
- [ ] Type B (project's own documents) chunked header-aware; Type A chunked paragraph-based with overlap
- [ ] Type C (IMD's live seasonal + extended-range outlooks) fetched fresh, not baked, kept as whole excerpts with clear IMD attribution, tagged by which agent(s) they're relevant to, and a graceful "unavailable" fallback if either fetch fails
- [ ] Every chunk carries source metadata for citation
- [ ] Embeddings via `gemini-embedding-001`, 768-dim, correct `task_type` for document vs. query embedding
- [ ] Embedding build is rate-limit-defensive (backoff, resumable) — same discipline as the Open-Meteo fetch
- [ ] ChromaDB populated and persisted locally (`retrieval/chroma_store/`)
- [ ] `retrieve_context()` implemented, returns citations not just text
- [ ] 10-query hand-authored evaluation set, precision reported honestly per query
- [ ] `PROJECT_LOG.md` updated with corpus composition, chunking choices, and the evaluation result
- [ ] `.gitignore` extended (the ChromaDB store itself is regenerable from the corpus + embeddings; keep the manifest and eval results tracked, not the binary index)

## When done

Report the final corpus (what's in it, what was dropped and why), the chunking/embedding choices made, and the 10-query precision result, honestly. This sets up Phase 3 (Orchestrator + Synthesis), which will be the first thing to actually call `retrieve_context()` in combination with the forecasting tools.
