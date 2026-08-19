# CLAUDE.md — Phase 7: Audit Fixes + Real Frontend

**Where this sits:** two independent pieces of leftover work, bundled into one dispatch because the user wants them done together. Neither touches the other's files, so do them in either order — but keep them as **separate, clearly-labelled units of work** in `PROJECT_LOG.md` and (later) as separate commits, the same discipline already used for Phase 5 vs Phase 6.

**Do not touch test behaviour or measured results** without the explicit numeric-check discipline spelled out in §1.3 below. This is the same rule every prior phase has followed.

---

## Part 1 — Audit fixes (`AUDIT_REPORT.md`, all four items approved by the user)

### 1.1 Fix the real bug — `api_key_present` false positive

`api/app.py:221` and `scripts/setup.py:161` both do a substring search for `"API_KEY"` in the raw `.env` file text, so they report a key as present even for a commented-out line, an empty value, or an unrelated key like `OPENAI_API_KEY`.

- Replace both checks with the existing `get_api_key()` helper in `retrieval/embed.py` (already correct — checks the three specific env-var names for non-empty values). Do not write a new resolver.
- After the fix, re-run (or add, if missing) a test that a `.env` containing only `# GEMINI_API_KEY=` (commented out) or `OPENAI_API_KEY=xyz` (wrong key) reports `api_key_present: false`.

### 1.2 Safe cleanup — dead code and stale gitignore entries

- Remove `outlooks_for()` from `retrieval/outlooks.py` — confirmed zero callers anywhere in the repo (both by the audit and independently by the Cowork session that dispatched this).
- Remove the two stale `.gitignore` lines: `*.chroma/` and `reports/eval/*.json.tmp` — confirmed they match no real path in this project.
- Run the full test suite after both removals to confirm nothing depended on them (it shouldn't, per the dead-code analysis, but confirm rather than assume).

### 1.3 Skill-score formula consolidation — numeric-check required

The formula `1 - rmse/rmse_baseline` is duplicated 6x across `evaluate.py`, `t1_model.py`, `heat/model.py`, `heat/phase11.py`. Extract it into one shared function (put it wherever `check_grounding()` lives, or a new small `metrics.py` if that's cleaner — your call).

**This is approved but not free** — every one of those 6 call sites produced a number that is already published in `PROJECT_LOG.md` and/or `EVALUATION.md`. Before replacing any call site:

1. Capture the current output of each of the 6 call sites (a snapshot script is fine, doesn't need to be a permanent test).
2. Swap in the shared function.
3. Re-run and diff against the snapshot — every value must match **exactly** (these are deterministic calculations, not stochastic ones, so exact match is the right bar, not "close enough").
4. If any value differs, stop and report it — do not silently accept a different number under an assumption that the shared version is "more correct." A changed number invalidates a phase's published result and needs the user's explicit sign-off, not a silent fix.
5. Once confirmed identical, this is safe to land.

### 1.4 Requirements consolidation

Replace the current 5 files (`requirements-phase1/2/3/6.txt`, `requirements-api.txt`) with 2:

- `requirements.txt` — everything needed to run the app and the test suite (the current `requirements-api.txt` contents plus whatever phase1-3 packages the API/tests actually import — check, don't assume).
- `requirements-training.txt` — adds TensorFlow, matplotlib, and anything else only the LSTM ablation path needs (`train.py`, `lstm_small.py`, `evaluate.py`'s plotting).

Also fix the two gaps the audit found: declare `pydantic` explicitly in `requirements.txt` (currently only a FastAPI transitive dependency — fragile), and declare `python-dotenv` (currently undeclared but used behind a `try/except`).

Update `README.md`/`SETUP_FROM_CLEAN.md` install instructions to reference the new file names.

---

## Part 2 — Real frontend, wired to the actual API

**Context:** the user designed a UI mockup in a separate design tool (Claude Design) and exported it. The export is now in this repo at `design_export/` — `design_export/AgriRisk Query Assistant.dc.html`, `design_export/support.js`, `design_export/_ds/nocturne/`.

**Read this carefully before starting:** that export is a **visual prototype only**. It uses a proprietary template runtime (`x-dc` custom elements, `{{ }}` bindings, `sc-if`/`sc-for`, a `DCLogic` base class in `support.js`) that does not call any real backend — the "Ask" button's `ask()` method just flips a local boolean and shows **hardcoded fake report data** (`CANNED` object at the bottom of the `.dc.html` file, keyed by `rajasthan`/`punjab`/`maharashtra`/`karnataka`). Do not ship this file as-is or wire it up as a thin layer — the fake data must not survive into the working app, and note that `punjab`/`maharashtra`/`karnataka` keys reference regions this project doesn't support at all (a leftover from an earlier design-tool mistake, already corrected in the visible UI but not in the underlying JS).

### 2.1 What to build

A plain, single-page HTML/JS/CSS app (per the original Phase 6 §2 constraint: no new heavy frontend toolchain) at `frontend/index.html` (+ `frontend/app.js`, `frontend/styles.css` if you prefer splitting it out) that:

- **Matches the visual design** in `design_export/` — same layout, same dark theme, same Hindi labels, same colour scheme (pull the actual CSS values from `design_export/_ds/nocturne/styles.css` rather than re-guessing them), same grounding-status banner styling (`banner-clean`/`banner-warn`, green/amber dot), same claim tags (`tag-accent` for Validated, `tag-outline` for Weak/Directional, `tag-neutral` for No skill).
- **Fields:** Region (Rajasthan/Barmer/koi bhi), Crop (Bajra/Wheat/koi bhi), Risk type (Drought/Heat/koi bhi) as the three visible filters; Month tucked into a collapsed "Advanced options" section with a Year input + Month dropdown that combine into `YYYY-MM` before being sent — this part of the design mockup is already correct, keep it.
- **Example query chips:** fetch from the real `GET /examples` endpoint at page load and render as clickable chips — do not hardcode the `CANNED` examples from the export. Clicking a chip should populate the form fields and query text from that example's actual request body.
- **Ask button:** calls `POST /report` with the real form state as the request body. While waiting, show a loading state (the API can take several seconds — it's making real Gemini calls). On success, render the real response: grounding banner from `grounding.status`, report text from `report`, the `horizon_confidence` list as distinct labelled tags (never collapsed into one number — this is the one thing Phase 6's schema exists to protect), `missing_data` entries as the italic "data not available" lines, exactly as the mockup shows.
- **Errors:** a 429 (quota exhausted) must show the quota-exhausted message the mockup already has a slot for, not a generic error. Other errors (400 for bad region/crop, 500) get a plain, honest error message — no fabricated content standing in for a failed call.
- **Footer:** call `GET /quota` (or use the `quota` field returned inline on `/report`) to show real `requests remaining today`, not a client-side-decremented fake counter like the export does.
- **Health dot:** call `GET /health` once on page load to set the "Backend + ChromaDB connected" indicator honestly (green only if `status: "ok"`, amber/red with the actual reason if `"degraded"`).

### 2.2 CORS / local dev

The API already has `CORSMiddleware` enabled for local dev per Phase 6 — confirm the frontend can be served from a different port (e.g. a trivial `python -m http.server` in `frontend/`) and still successfully call the API at its `localhost:8000` origin without a CORS error. If it doesn't, the fix belongs in the API's CORS config, not a frontend workaround.

### 2.3 Docker

Decide (and state which you picked, with reasoning) whether the frontend:
- (a) ships as static files served by the same FastAPI app (`StaticFiles` mount) so `docker run` gives you the whole local site on one port, or
- (b) stays a separate `python -m http.server` step outside the container for now, documented in `SETUP_FROM_CLEAN.md`.

(a) is probably nicer for the "one command, working local site" goal Phase 6 was aiming at, but don't rebuild/repush the image as part of this task unless you also re-verify `docker build && docker run` end-to-end the same way Phase 6 did — don't claim it works without actually running it.

### 2.4 Tests

- At minimum, a smoke test (Playwright or even a simple `requests`-based check against a running `uvicorn` process) that loads the page, fetches `/examples`, clicks a chip, and confirms the request body sent to `/report` matches the example. Mock/monkeypatch the orchestrator the same way the API tests already do — don't spend live quota on this.

---

## Definition of Done

- [ ] `api_key_present` bug fixed in both files, tested
- [ ] `outlooks_for()` and the 2 stale `.gitignore` lines removed, full suite still passes
- [ ] Skill-score formula consolidated, with the before/after numeric diff explicitly reported (all-match or discrepancy flagged, not silently resolved)
- [ ] `requirements.txt` + `requirements-training.txt` replace the 5 old files, `pydantic`/`python-dotenv` declared, docs updated
- [ ] `frontend/` — real working single-page app, matching `design_export/`'s look, calling the real API (no hardcoded/fake report data anywhere in the shipped code)
- [ ] `PROJECT_LOG.md` updated with both units of work clearly separated (audit fixes vs. frontend), and the skill-score numeric-check result stated explicitly

## When done

Report each of Part 1's four items separately (what changed, what the numeric check showed for §1.3), and for Part 2 confirm exactly what a person needs to run (which commands, which ports) to see the real frontend talking to the real backend locally — the same "not assumed, actually run" standard Phase 6 held itself to for `docker build`/`docker run`.
