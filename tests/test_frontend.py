"""Phase 7 §2.4 — the frontend is really wired to the real API.

No browser and no Playwright: the thing worth testing is the *contract* between
the page and the API, and that is checkable without a rendering engine. What
these tests actually assert:

  1. The page is served by the app and loads only local assets.
  2. It contains no canned report data — the design export's `CANNED` object and
     its unsupported regions (punjab/maharashtra/karnataka) must not have
     survived into the shipped code. This is the one thing most likely to be
     copied across by accident, so it is asserted rather than trusted.
  3. Every endpoint the page calls exists on the app.
  4. Replaying the chip-click path end to end — GET /examples, take an example,
     build the request body the way app.js builds it, POST /report — reaches the
     orchestrator with exactly the example's own body.

(4) mirrors app.js's `applyExample` + `buildRequestBody` in Python. That is a
deliberate duplication: it means a change to either side that breaks the
agreement makes this test fail rather than quietly shipping a chip that sends
something other than what it advertised.

The orchestrator is monkeypatched exactly as in test_api.py — no live quota is
spent here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from forecasting import config as fconfig

client = TestClient(app)

FRONTEND = fconfig.REPO_ROOT / "frontend"
INDEX = FRONTEND / "index.html"
APP_JS = FRONTEND / "app.js"

pytestmark = pytest.mark.skipif(not INDEX.exists(),
                                reason="frontend/ is not present in this checkout")


# --------------------------------------------------------------------------- #
# The page is served, and is self-contained
# --------------------------------------------------------------------------- #
def test_frontend_is_served_by_the_api():
    """Phase 7 §2.3 option (a): one process, one port, whole local site."""
    response = client.get("/app/")
    assert response.status_code == 200
    assert "Agricultural Risk Query Assistant" in response.text
    for asset in ("app.js", "styles.css", "nocturne.css"):
        assert client.get(f"/app/{asset}").status_code == 200, asset


def _code_without_comments(path: Path) -> str:
    """Strip comments so this checks shipped behaviour, not prose about it."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)      # /* ... */
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)     # <!-- ... -->
    text = re.sub(r"(?m)^\s*//.*$", " ", text)              # whole-line //
    return text.lower()


def test_no_canned_report_data_survived_the_design_export():
    """The export shipped fake reports and four unsupported regions. Neither
    may exist in the served app — a demo that invents a report is exactly the
    failure five phases of grounding work exist to prevent."""
    code = _code_without_comments(APP_JS) + _code_without_comments(INDEX)

    assert "const canned" not in code
    for unsupported in ("punjab", "maharashtra", "karnataka"):
        assert unsupported not in code, (
            f"{unsupported} is not a region this project supports")

    # Distinctive strings from the export's fake reports.
    for fake in ("reservoir storage", "vidarbha", "monitoring stations",
                 "canal water release", "yavatmal"):
        assert fake not in code, f"fabricated report text survived: {fake!r}"

    supported = set(client.get("/examples").json()["supported_regions"])
    assert supported <= {"rajasthan", "barmer"}, "test needs updating for new regions"


def test_frontend_loads_no_third_party_assets():
    """Everything the page needs is local, apart from the design system's own
    Google Fonts import, which is inside the verbatim nocturne.css copy."""
    html = INDEX.read_text(encoding="utf-8")
    for url in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert not url.startswith("http"), f"third-party asset in index.html: {url}"


# --------------------------------------------------------------------------- #
# Every endpoint the page calls actually exists
# --------------------------------------------------------------------------- #
def test_every_endpoint_the_page_calls_exists():
    called = set(re.findall(r'\$\{API\}(/[a-z]+)', APP_JS.read_text(encoding="utf-8")))
    assert called, "no API calls found in app.js — the regex or the file changed"

    routes = {getattr(r, "path", None) for r in app.routes}
    for path in called:
        assert path in routes, f"app.js calls {path}, which the API does not serve"


# --------------------------------------------------------------------------- #
# The chip-click path, replayed end to end
# --------------------------------------------------------------------------- #
def _apply_example_then_build_body(example: dict) -> dict:
    """Python mirror of app.js applyExample() + buildRequestBody().

    A chip fills the form from the example's body; Ask then rebuilds a body from
    that form state. Round-tripping through the form is the point: a field the
    form cannot represent would be silently dropped here, exactly as in the page.
    """
    body = example["body"]

    # applyExample(): example body -> form fields
    form = {
        "query": body.get("request", ""),
        "region": body.get("region", ""),
        "crop": body.get("crop", ""),
        "risk": (body.get("risk_types") or [""])[0],
        "year": "",
        "month": "",
    }
    month = body.get("month", "")
    if re.fullmatch(r"\d{4}-\d{2}", month or ""):
        form["year"], form["month"] = month.split("-")

    # buildRequestBody(): form fields -> request body
    rebuilt = {"request": form["query"].strip()}
    if form["region"]:
        rebuilt["region"] = form["region"]
    if form["crop"]:
        rebuilt["crop"] = form["crop"]
    if form["risk"]:
        rebuilt["risk_types"] = [form["risk"]]
    if form["year"] and form["month"]:
        rebuilt["month"] = f"{form['year']}-{form['month']}"
    return rebuilt


def test_examples_survive_the_form_round_trip():
    """Every chip must send exactly what /examples advertised for it."""
    examples = client.get("/examples").json()["examples"]
    assert examples

    for example in examples:
        rebuilt = _apply_example_then_build_body(example)
        assert rebuilt == example["body"], (
            f"chip '{example['id']}' would send {rebuilt}, "
            f"but /examples advertises {example['body']}")


def test_clicking_a_chip_reaches_the_orchestrator_with_that_examples_body(
        monkeypatch):
    """The full path a user takes: load examples, click a chip, press Ask."""
    seen = {}

    def fake_analyse(**kwargs):
        seen.update(kwargs)
        return {
            "request": kwargs["request"],
            "report": "Forecast SPI-3 is -1.62 at t+1.",
            "grounding": {"grounded": True, "unverified_numbers": [],
                          "total_checked": 3, "source_numbers_available": 9},
            "tool_calls": [], "tool_outputs": {}, "retrieved_chunks": [],
            "warnings": [],
        }

    monkeypatch.setattr("orchestrator.graph.analyse", fake_analyse)

    examples = client.get("/examples").json()["examples"]
    chosen = next(e for e in examples if e["id"] == "crop_impact_wheat")
    body = _apply_example_then_build_body(chosen)

    response = client.post("/report", json=body)
    assert response.status_code == 200

    # What the orchestrator received matches the example, field for field.
    assert seen["request"] == chosen["body"]["request"]
    assert seen["crop"] == chosen["body"]["crop"]
    assert seen["month"] == chosen["body"]["month"]
    assert seen["region"] == chosen["body"].get("region")

    payload = response.json()
    assert payload["grounding"]["status"] == "clean"
    assert "quota" in payload


def test_a_429_is_distinguishable_from_a_generic_failure(monkeypatch):
    """app.js shows the quota message only for 429, so 429 must be reachable."""
    def fake_analyse(**kwargs):
        return {
            "request": kwargs["request"],
            "report": "",
            "grounding": {"report_missing": True, "grounded": False,
                          "unverified_numbers": [], "total_checked": 0},
            "tool_calls": [], "tool_outputs": {}, "retrieved_chunks": [],
            "warnings": ["429 RESOURCE_EXHAUSTED: quota exceeded"],
        }

    monkeypatch.setattr("orchestrator.graph.analyse", fake_analyse)
    response = client.post("/report", json={"request": "drought risk for barmer"})
    assert response.status_code == 429
    assert "quota" in response.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# Phase 8 Part 1 — English copy, unchanged API values
# --------------------------------------------------------------------------- #
def test_no_devanagari_remains_in_the_shipped_frontend():
    """Part 1: visible copy is English now. Checked by codepoint range rather
    than by looking for particular words, so nothing can hide."""
    for path in (INDEX, APP_JS, FRONTEND / "styles.css"):
        text = path.read_text(encoding="utf-8")
        offenders = [line.strip()[:60] for line in text.splitlines()
                     if re.search(r"[\u0900-\u097F]", line)]
        assert not offenders, f"{path.name} still has Devanagari: {offenders[:3]}"


def test_the_page_declares_itself_english():
    assert '<html lang="en">' in INDEX.read_text(encoding="utf-8")


def test_api_values_were_not_translated_along_with_the_labels():
    """Only the visible text changed. The values sent to the API must not."""
    html = INDEX.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")

    assert '<option value="drought">' in html
    assert '<option value="heat_stress">' in html
    # The label maps are keyed by the API's own values.
    for value in ("rajasthan", "barmer", "bajra", "wheat"):
        assert f"{value}:" in js, f"{value} is no longer a label-map key"

    supported = client.get("/examples").json()
    for region in supported["supported_regions"]:
        assert f"{region}:" in js or region in js


def test_english_labels_are_present():
    html = INDEX.read_text(encoding="utf-8")
    for label in ("Ask your question", "Your question", "Region", "Crop",
                  "Risk type", "Example questions", "Drought", "Heat stress"):
        assert label in html, f"missing English label: {label}"


# --------------------------------------------------------------------------- #
# Phase 8 Part 4 — outside sources stay visibly outside
# --------------------------------------------------------------------------- #
def test_frontend_renders_external_sources_in_their_own_attributed_block():
    html = INDEX.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")

    assert 'id="external-block"' in html
    assert "not this project's figures" in html, (
        "the external block must say whose figures these are not")
    assert "renderExternal" in js
    # Publisher must be printed with every external line.
    external_fn = js.split("function renderExternal")[1].split("\nfunction ")[0]
    assert "source.publisher" in external_fn
    assert "unavailable" in external_fn, "a failed source must still be shown"
