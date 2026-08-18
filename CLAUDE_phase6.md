# CLAUDE.md — Multi-Agent Climate Risk Analyst
## Phase 6: Local Site (Backend API) — no public deployment yet

**UPDATE (Aug 2026):** The frontend is being built separately by the user with a design tool (Claude Design), NOT by this phase. **Do not build a frontend, do not create any HTML/JS/CSS UI files.** This phase is now backend-only: FastAPI service + Docker + setup automation. Section 2 below is kept for reference only, describing what the externally-built frontend needs from this API (so the API contract matches) — do not implement any of it as files in this repo.

**Where this sits:** all five agents/tools work and are tested (drought, heat, retrieval, crop impact, orchestrator+synthesis, plus the Phase 5 evaluation suite). Nothing has a face — every result so far has been read from a terminal or a `.json`/`.md` file. This phase makes the system runnable as a local API: a separately-built frontend will call it, ask a question, and show the same grounded report the tests have been checking all along — no shortcuts, no separate "demo" logic that bypasses the real pipeline.

**Deployment is explicitly out of scope for this phase.** The user will decide when and where to deploy later (once there's budget) — this phase produces something that runs correctly on `localhost` and is container-ready, not something pushed anywhere public. Do not sign up for, configure, or push to any hosting service.

---

## 1. Backend — thin wrapper, not a rewrite

- FastAPI, one process, wrapping `orchestrator.graph.analyse()` directly — the API must call the exact same function the tests already exercise, not a reimplementation. If the API needs a slightly different return shape than `analyse()` gives, adapt it in the API layer, don't fork the orchestrator logic.
- Minimum endpoints:
  - `POST /report` — body: `{request, region?, risk_types?, month?}`, returns the full `ReportState` result (report text, grounding result, tool outputs, warnings).
  - `GET /health` — cheap, no LLM call, just confirms the app started and the ChromaDB index is reachable.
  - `GET /evaluation` — serves `EVALUATION.md`'s content (or a small JSON summary of it) so the frontend can show "how good is this system" without a new evaluation run.
- **Quota is a first-class concern in the API, not an afterthought.** Every `/report` call costs 3-5 Gemini calls (routing + crop-impact + synthesis, per Phase 3/4's own accounting) against a 20/day cap. The API must:
  - Return a clear, specific error (not a generic 500) when a call fails due to quota exhaustion — reuse the existing `invoke_with_backoff`/warning pattern rather than inventing new error handling.
  - Optionally expose a request counter or a "requests remaining today" indicator if that's cheap to track — a demo that silently stops working at request #5 is a bad experience; one that says why is fine.

## 2. Frontend contract — reference only, DO NOT BUILD THIS

*Not part of this phase's deliverable — the user is building this separately with a design tool. Read this section only to make sure the API (Section 1) exposes what an external frontend will need; do not create any UI files.*

- The externally-built frontend will need, from the API responses, everything required to surface the project's honesty mechanisms without extra client-side guessing:
  - The **grounding status** (clean / warning) as an explicit field in the `/report` response, not something the frontend has to infer from text.
  - "No skill"/"weak/directional"/"validated" labels present as distinct fields/values in the response — never collapsed server-side into a single confidence number.
  - When `forecast_available: False` (heat) or a crop/risk combination has no sourced coefficient, the response must say so explicitly (a clear field/flag), not omit the field or leave it null with no explanation.
- `/report` must be callable cross-origin from a separately hosted/served frontend during local dev — enable permissive CORS for localhost origins (FastAPI's `CORSMiddleware`) since the design-tool frontend will likely run on a different local port.
- The pre-canned example queries (reuse `tests/test_orchestrator.py`'s `SCENARIOS` and the crop-impact eval set) should still be exposed by the API as a small `GET /examples` endpoint (list of ready-made request bodies) so the external frontend can fetch and render them as clickable chips, instead of hardcoding them client-side.

## 3. The "database disappears on a fresh clone" problem — fix the actual cause

This came up as a specific complaint before this phase started: something about the local database not surviving a download/reopen cycle. Root-cause it properly rather than working around it:

- Confirm exactly which paths are git-ignored (`data/`, `models/*.pkl`/`.h5`, the ChromaDB persistent directory) — a fresh clone legitimately won't have these; that's expected, not a bug, but it must be **documented and automated**, not silently broken.
- Add a single setup script/command (`make setup` or `python -m scripts.setup`, whichever fits the project's existing conventions) that runs the full regeneration sequence in the right order — fetch data, train models, build the ChromaDB index — so "clone → one command → working local site" is real, not aspirational. This should build on (not duplicate) whatever `SETUP_FROM_CLEAN.md` the codebase-audit task produces — check whether that file already exists before writing a second version of the same instructions.
- Verify the ChromaDB store's path handling is relative to the repo root, not an absolute path baked in at build time — a store built at `E:\Muti-Agent-Climate-Risk-Analyst\...` must still work if the repo is cloned to a different drive/folder.

## 4. Containerization — build it, don't push it

- A `Dockerfile` that produces a working image locally (`docker build` + `docker run` succeeds and the local site works inside the container) — this is the artifact that makes deployment a later decision rather than a rebuild.
- Per the project's standing architecture decision, ChromaDB is baked into the image at build time (no free host offers persistent disk) — the Dockerfile must run the index-build step during `docker build`, not expect it to exist on a mounted volume.
- The `.env`/API key must be supplied at `docker run` time (`-e GEMINI_API_KEY=...` or an env file), never baked into the image.
- Do not write deployment-platform-specific configuration (no Hugging Face Spaces `README.md` frontmatter, no Cloud Run YAML) — that's the next decision, made when the user says so.

## 5. Tests

- API tests using FastAPI's test client, mocking the orchestrator call for the fast/offline majority (assert routing, error handling, response shape) — do not make these live-Gemini by default, same quota discipline as every other phase.
- One smoke test, gated the same way as the other live tests, that hits `/report` for real and checks the response is well-formed.
- A test that `/health` doesn't call Gemini (grep the handler's source, same pattern as `test_checker_is_not_an_llm`).

## Stopping rule

This is a working local demo, not a production API — do not add auth, rate limiting beyond the quota-awareness above, or a database beyond what already exists (ChromaDB + the flat evidence files). If something looks like it needs those, note it as a future-deployment concern and move on.

## Definition of Done

- [ ] FastAPI backend wrapping `analyse()` directly, `/report` `/health` `/evaluation` `/examples` endpoints
- [ ] API responses carry grounding status, per-label honesty fields, and explicit missing-data flags as structured fields (no frontend built in this phase — see Section 2)
- [ ] CORS enabled for local frontend dev origins
- [ ] `SETUP_FROM_CLEAN.md` (shared with/building on the audit task's output) — clone-to-working-API in one documented sequence
- [ ] `Dockerfile` — builds and runs locally, ChromaDB baked in at build time, API key supplied at runtime only
- [ ] Tests: offline API tests (majority), one gated live smoke test, no-LLM-call test for `/health`
- [ ] `PROJECT_LOG.md` updated with what was built and the quota-per-request reality now visible in the API itself

## When done

Report whether `docker build && docker run` actually produces a working local site end-to-end, what the setup-from-clean sequence turned out to need that wasn't obvious in advance, and confirm no deployment credentials or platform-specific config were added — this phase should leave the project exactly as portable as before, just packaged.
