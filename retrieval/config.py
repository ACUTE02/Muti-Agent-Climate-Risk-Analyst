"""Retrieval Agent configuration — the corpus definition is the single source of truth.

Three document types, deliberately different in kind and handled differently:

* **Type A — domain reference.** External authority (IMD, NDMA, ICAR) for
  definitions and methodology, so the system can explain *why* a threshold is what
  it is with a real citation instead of an LLM's recall.
* **Type B — this project's own evidence.** The measured record. This is the more
  important half: it is what stops a language model inventing a skill score when
  asked "how reliable is the 2-month forecast". The real answer (+0.0766 /
  +0.0438, "weak/directional") is written down and retrievable.
* **Type C — live official outlooks.** Not baked into the index: they change every
  season, so they are fetched at report time and passed through whole, attributed
  to IMD by name, never blended into this project's own numbers.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_DIR = REPO_ROOT / "retrieval"
CACHE_DIR = RETRIEVAL_DIR / "cache"          # downloaded source bytes
CHROMA_DIR = RETRIEVAL_DIR / "chroma_store"  # regenerable index
MANIFEST_PATH = RETRIEVAL_DIR / "corpus_manifest.json"
CHUNKS_PATH = RETRIEVAL_DIR / "chunks.jsonl"
EVAL_QUERIES_PATH = RETRIEVAL_DIR / "eval_queries.json"
EVAL_RESULTS_PATH = RETRIEVAL_DIR / "eval_results.json"
OUTLOOK_CACHE_PATH = CACHE_DIR / "outlooks.json"

for _d in (CACHE_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768          # scaled down from the 3072 default via Matryoshka;
#                          keeps the baked Chroma index small for free-tier hosting
TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"    # corpus chunks, at build time
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"          # user queries, at retrieval time
EMBED_BATCH_SIZE = 32
EMBED_MAX_RETRIES = 5
EMBED_BACKOFF_SECONDS = 20   # 429s on the free tier are per-minute quotas
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")

COLLECTION_NAME = "climate_risk_corpus"

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
# ~500 tokens with ~50 overlap, in characters (~4 chars/token for this material).
CHUNK_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 120        # below this a chunk is a heading fragment, not content

# A PDF page yielding less than this is probably a scanned image, not text.
MIN_CHARS_PER_PAGE = 200
MIN_ALPHA_RATIO = 0.5

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
FETCH_TIMEOUT = 120

# --------------------------------------------------------------------------- #
# Type A — domain reference documents
# --------------------------------------------------------------------------- #
# NDMA's Drought hazard *page* (https://ndma.gov.in/Natural-Hazards/Drought) was
# checked and is a hard 404, as are /Droughts, /drought and /Natural-Hazards —
# NDMA does not publish a drought hazard page at that path. The official 2010 NDMA
# drought guidelines survive as a NIDM-hosted PDF, which is used instead; the dead
# page is still recorded in the manifest so the substitution is visible.
NDMA_DROUGHT_DEAD_PAGE = "https://ndma.gov.in/Natural-Hazards/Drought"

TYPE_A_SOURCES = [
    {
        "id": "imd_heatwave_faq",
        "title": "IMD FAQ on Heat Wave",
        "url": "https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf",
        "format": "pdf",
        "publisher": "India Meteorological Department",
        "why": "Authoritative definition of the heat wave and severe heat wave "
               "criteria this project's heat/target.py implements.",
    },
    {
        "id": "ndma_heatwave",
        "title": "NDMA Heat Wave hazard page",
        "url": "https://ndma.gov.in/Natural-Hazards/Heat-Wave",
        "format": "html",
        "publisher": "National Disaster Management Authority",
        "why": "National-authority statement of the same criteria, plus the "
               "advisory context around them.",
    },
    {
        "id": "ndma_drought_guidelines",
        "title": "NDMA National Disaster Management Guidelines: Management of Drought",
        "url": "https://nidm.gov.in/PDF/pubs/NDMA/13.pdf",
        "backup_url": "https://www.droughtmanagement.info/literature/"
                      "GovIndia_management_of_drought_2010.pdf",
        "format": "pdf",
        "publisher": "National Disaster Management Authority (hosted by NIDM)",
        "why": "The official national drought guidelines — classification, early "
               "warning and response. Replaces the dead NDMA drought hazard page, "
               "and balances a corpus that would otherwise be heat-only on the "
               "domain-reference side.",
    },
    {
        "id": "nih_spi_methodology",
        "title": "NIH Roorkee — Standardized Precipitation Index methodology",
        "url": "https://nihroorkee.gov.in/sites/default/files/uploadfiles/SPINov2011.pdf",
        "format": "pdf",
        "publisher": "National Institute of Hydrology, Roorkee",
        "why": "Authoritative Indian source for the same gamma-fit SPI method "
               "(McKee et al. 1993) this project implements in "
               "forecasting/split.py — it grounds the project's own methodology "
               "choice, not just drought in general.",
    },
    {
        "id": "imd_gkms_sop",
        "title": "IMD Standard Operating Procedure for Agromet Advisory Services",
        "url": "https://mausam.imd.gov.in/imd_latest/contents/pdf/gkms_sop.pdf",
        "format": "pdf",
        "publisher": "India Meteorological Department",
        "why": "The actual government methodology for district-level agromet "
               "advisories — what a real advisory looks like and how it is made.",
    },
    {
        "id": "atari_jodhpur_agroadvisory",
        "title": "ICAR-ATARI Jodhpur Agro-Advisory Services",
        "url": "https://atarijodhpur.res.in/Agro%20Advisory%20Services.pdf",
        "format": "pdf",
        "publisher": "ICAR-ATARI Zone-II, Jodhpur",
        "why": "Rajasthan-specific advisory practice, matching this project's "
               "two modelled regions.",
    },
]

# --------------------------------------------------------------------------- #
# Type B — this project's own measured evidence
# --------------------------------------------------------------------------- #
TYPE_B_SOURCES = [
    {
        "id": "project_log",
        "title": "Project Log — Multi-Agent Climate Risk Analyst",
        "path": "PROJECT_LOG.md",
        "why": "The full decision and result record, including every negative "
               "result and why it was accepted.",
    },
    {
        "id": "drought_region_comparison",
        "title": "Drought — model / region comparison",
        "path": "models/region_comparison.md",
        "why": "The measured per-horizon drought skill scores, which the "
               "Synthesis agent must cite rather than invent.",
    },
    {
        "id": "heat_region_comparison",
        "title": "Heat Stress — model / region comparison",
        "path": "models/heat_region_comparison.md",
        "why": "The measured heat stress results, establishing that heat "
               "forecasting has no usable skill with this data.",
    },
]

# --------------------------------------------------------------------------- #
# Type C — live official outlooks (never baked into the index)
# --------------------------------------------------------------------------- #
TYPE_C_SOURCES = [
    {
        "id": "imd_extended_range",
        "title": "IMD Extended Range Forecast (next two weeks)",
        "url": "https://internal.imd.gov.in/section/nhac/dynamic/extended.pdf",
        "format": "pdf",
        "publisher": "India Meteorological Department",
        "relevant_to": ["heat_stress", "drought"],
        "why": "~2-week rainfall and temperature outlook. Primarily valuable for "
               "Heat Stress, where this project's own model has no skill at all.",
    },
    {
        "id": "imd_seasonal",
        "title": "IMD Seasonal Forecast page",
        "url": "https://mausam.imd.gov.in/responsive/seasonal_forecast.php",
        "format": "html",
        "publisher": "India Meteorological Department",
        "relevant_to": ["drought", "heat_stress"],
        "why": "IMD's seasonal outlook hub. Note: the page itself is a navigation "
               "hub whose extractable text is menu items plus a rotating press "
               "release marquee — the actual seasonal bulletins are linked PDFs. "
               "The current press-release statements are extracted; the manifest "
               "records this limitation rather than implying a full outlook.",
    },
]

TYPE_LABELS = {"A": "domain_reference", "B": "project_evidence", "C": "live_outlook"}
