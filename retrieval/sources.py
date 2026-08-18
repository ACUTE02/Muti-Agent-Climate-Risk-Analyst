"""Defensive fetching and text extraction for the corpus.

Same discipline as every external fetch in this project: download, then *verify
the extraction actually produced readable text* before trusting it. Several Indian
government PDFs are scanned images; indexing their empty or garbled extraction
would silently poison retrieval, so a source that fails the check is dropped and
the reason recorded in the manifest rather than forced in.

Run standalone:  python -m retrieval.sources
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from retrieval import config


@dataclass
class SourceText:
    source_id: str
    title: str
    text: str
    citation: str
    usable: bool
    reason: str = ""
    stats: dict = field(default_factory=dict)


def fetch_bytes(url: str, cache_name: str | None = None,
                force: bool = False) -> bytes:
    """Download with a browser UA, caching to disk so a rebuild is offline."""
    cache_path = config.CACHE_DIR / cache_name if cache_name else None
    if cache_path and cache_path.exists() and not force:
        return cache_path.read_bytes()

    last_error = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers={"User-Agent": config.USER_AGENT},
                                timeout=config.FETCH_TIMEOUT)
            resp.raise_for_status()
            if cache_path:
                cache_path.write_bytes(resp.content)
            return resp.content
        except Exception as exc:           # network flakiness, not a logic error
            last_error = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last_error}")


def extract_pdf_text(raw: bytes, cache_name: str) -> tuple[str, dict]:
    """Extract text and measure whether it is actually text."""
    path = config.CACHE_DIR / cache_name
    if not path.exists():
        path.write_bytes(raw)

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())

    compact = " ".join(text.split())
    alpha_ratio = (sum(c.isalpha() for c in compact) / len(compact)
                   if compact else 0.0)
    stats = {
        "pages": len(pages),
        "chars": len(compact),
        "chars_per_page": round(len(compact) / max(len(pages), 1), 1),
        "alpha_ratio": round(alpha_ratio, 3),
    }
    return text, stats


def extract_html_text(raw: bytes) -> tuple[str, dict]:
    """Strip chrome, keep body prose."""
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    lines = [" ".join(line.split())
             for line in soup.get_text("\n").splitlines()]
    text = "\n".join(line for line in lines if line)

    compact = " ".join(text.split())
    alpha_ratio = (sum(c.isalpha() for c in compact) / len(compact)
                   if compact else 0.0)
    return text, {"html_title": title, "chars": len(compact),
                  "alpha_ratio": round(alpha_ratio, 3)}


def load_source(source: dict, force: bool = False) -> SourceText:
    """Fetch one Type A/C source and judge whether it is usable."""
    fmt = source["format"]
    cache_name = f"{source['id']}.{fmt}"
    citation = source["url"]
    try:
        raw = fetch_bytes(source["url"], cache_name, force=force)
    except RuntimeError as exc:
        backup = source.get("backup_url")
        if not backup:
            return SourceText(source["id"], source["title"], "", source["url"],
                              usable=False, reason=f"fetch failed: {exc}")
        try:                       # documented mirror, used only if the primary dies
            raw = fetch_bytes(backup, cache_name, force=force)
            citation = backup
        except RuntimeError as backup_exc:
            return SourceText(source["id"], source["title"], "", source["url"],
                              usable=False,
                              reason=f"primary and backup both failed: {backup_exc}")

    if fmt == "pdf":
        text, stats = extract_pdf_text(raw, cache_name)
        scanned = (stats["chars_per_page"] < config.MIN_CHARS_PER_PAGE
                   or stats["alpha_ratio"] < config.MIN_ALPHA_RATIO)
        if scanned:
            return SourceText(source["id"], source["title"], text, citation,
                              usable=False, stats=stats,
                              reason=("extraction looks like a scanned image: "
                                      f"{stats['chars_per_page']} chars/page, "
                                      f"alpha ratio {stats['alpha_ratio']}"))
    else:
        text, stats = extract_html_text(raw)
        if stats["chars"] < 500:
            return SourceText(source["id"], source["title"], text, citation,
                              usable=False, stats=stats,
                              reason=f"only {stats['chars']} chars of text — "
                                     "page is missing, empty or JS-rendered")
        if "page not found" in text.lower()[:400]:
            return SourceText(source["id"], source["title"], text, citation,
                              usable=False, stats=stats,
                              reason="server returned a 'page not found' body")

    return SourceText(source["id"], source["title"], text, citation,
                      usable=True, stats=stats)


def load_project_document(source: dict) -> SourceText:
    """Type B: this project's own markdown, read from the repo."""
    path = config.REPO_ROOT / source["path"]
    if not path.exists():
        return SourceText(source["id"], source["title"], "", source["path"],
                          usable=False, reason=f"missing file {source['path']}")

    text = path.read_text(encoding="utf-8")
    return SourceText(source["id"], source["title"], text, source["path"],
                      usable=True,
                      stats={"chars": len(" ".join(text.split())),
                             "headings": sum(1 for line in text.splitlines()
                                             if line.startswith("#"))})


if __name__ == "__main__":
    print("=== Type A ===")
    for src in config.TYPE_A_SOURCES:
        result = load_source(src)
        flag = "OK  " if result.usable else "DROP"
        print(f"{flag} {result.source_id:28s} {result.stats} {result.reason}")

    print("\n=== Type B ===")
    for src in config.TYPE_B_SOURCES:
        result = load_project_document(src)
        flag = "OK  " if result.usable else "DROP"
        print(f"{flag} {result.source_id:28s} {result.stats} {result.reason}")

    print("\n=== Type C (live, not indexed) ===")
    for src in config.TYPE_C_SOURCES:
        result = load_source(src)
        flag = "OK  " if result.usable else "DROP"
        print(f"{flag} {result.source_id:28s} {result.stats} {result.reason}")
