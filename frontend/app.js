/* Agricultural Risk Query Assistant — the real client.
 *
 * Every value rendered here arrives from the API. There is deliberately no
 * canned report data, no client-side quota counter, and no region list beyond
 * what GET /examples reports as supported: the design export carried all three,
 * and all three were fictions that would outlive the mockup if copied across.
 *
 * The one rule this file exists to protect: horizon_confidence entries are
 * rendered as separate labelled tags and are never averaged, thresholded, or
 * collapsed into a single confidence number. That distinction is the whole
 * point of the response schema.
 */

// Where the API lives.
//
// Same origin by default: when FastAPI serves this page at /app, the API is
// right there, whatever port uvicorn happens to be on. Guessing port 8000
// because the page is not on 8000 would be wrong exactly when it matters —
// a second backend on another port would have its UI silently talk to the
// first one.
//
// The only cases that cannot be same-origin are a file:// open and a separate
// static server (python -m http.server). For those, ?api=http://host:port is
// the explicit answer and localhost:8000 is the documented default.
const API = (() => {
  const override = new URLSearchParams(location.search).get("api");
  if (override) return override.replace(/\/$/, "");
  if (location.protocol === "file:") return "http://localhost:8000";
  // Served by the API itself? Then the page sits under its /app mount.
  if (location.pathname.startsWith("/app")) return location.origin;
  // Otherwise this is a plain static server; the API is elsewhere.
  return "http://localhost:8000";
})();

const $ = (id) => document.getElementById(id);

// Display names for the values the backend supports. The keys are the values
// actually sent to the API and must not change; only the labels are copy. A
// value the backend reports but this map does not know still appears —
// capitalised — rather than being silently dropped.
const REGION_LABELS = { rajasthan: "Rajasthan", barmer: "Barmer" };
const CROP_LABELS = { bajra: "Bajra (pearl millet)", wheat: "Wheat" };

const state = { examples: [], activeExampleId: null, busy: false };

/* ── helpers ─────────────────────────────────────────────────────────────── */

function titleCase(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function fillSelect(select, values, labels) {
  const keep = select.querySelector('option[value=""]');
  select.replaceChildren(keep);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labels[value] || titleCase(value);
    select.appendChild(option);
  }
}

function setDot(dotEl, textEl, cls, text) {
  dotEl.className = dotEl.className.replace(/dot-(green|amber|red|grey)/, "") + " " + cls;
  textEl.textContent = text;
}

/* The claim tag classes the design specifies, driven by the measured label. */
function tagClassFor(label) {
  if (label === "validated") return "tag-accent";
  if (label.startsWith("no skill")) return "tag-neutral";
  return "tag-outline";               // "weak/directional" and anything new
}

function tagTextFor(label) {
  if (label === "validated") return "Validated";
  if (label.startsWith("no skill")) return "No skill";
  return "Weak/Directional";
}

/* ── page load: /health, /quota, /examples ───────────────────────────────── */

async function loadHealth() {
  try {
    const response = await fetch(`${API}/health`);
    const body = await response.json();
    const ok = body.status === "ok";

    // Green only for a genuine "ok". A degraded backend says what is missing.
    if (ok) {
      setDot($("health-dot"), $("health-text"), "dot-green",
             `Backend + ChromaDB connected (${body.chroma_chunks} chunks)`);
      setDot($("footer-dot"), $("footer-text"), "dot-green", "All systems operational");
    } else {
      const reasons = [];
      if (!body.chroma_index_ready) reasons.push("ChromaDB index missing");
      if (!body.forecast_artifacts_ready) {
        reasons.push(`${body.missing_artifacts.length} forecast artifact(s) missing`);
      }
      if (!body.api_key_present) reasons.push("no API key");
      const why = reasons.join("; ") || "degraded";
      setDot($("health-dot"), $("health-text"), "dot-amber", `Degraded — ${why}`);
      setDot($("footer-dot"), $("footer-text"), "dot-amber", "Degraded");
    }
  } catch (err) {
    setDot($("health-dot"), $("health-text"), "dot-red",
           `Backend not reachable at ${API}`);
    setDot($("footer-dot"), $("footer-text"), "dot-red", "Backend down");
  }
}

function renderQuota(quota) {
  if (!quota) return;
  const left = quota.calls_remaining_today;
  $("quota-line").textContent =
    `${left} of ${quota.daily_call_budget} Gemini calls remaining today`;
  const exhausted = left <= 0;
  $("ask").disabled = exhausted || state.busy;
  $("quota-warning").classList.toggle("hidden", !exhausted);
  if (exhausted) {
    $("quota-warning").textContent =
      "Today's quota is used up — please try again tomorrow.";
  }
}

async function loadQuota() {
  try {
    renderQuota(await (await fetch(`${API}/quota`)).json());
  } catch (err) {
    $("quota-line").textContent = "Quota unknown — the backend did not respond";
  }
}

async function loadExamples() {
  const chips = $("chips");
  try {
    const body = await (await fetch(`${API}/examples`)).json();
    state.examples = body.examples || [];

    fillSelect($("f-region"), body.supported_regions || [], REGION_LABELS);
    fillSelect($("f-crop"), body.supported_crops || [], CROP_LABELS);
    if (body.cost_warning) $("cost-warning").textContent = body.cost_warning;
    renderQuota(body.quota);

    chips.replaceChildren();
    if (!state.examples.length) {
      chips.append(Object.assign(document.createElement("span"),
        { className: "hint", textContent: "No examples are available" }));
      return;
    }
    for (const example of state.examples) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = example.label;
      chip.title = example.shows || "";
      chip.setAttribute("aria-pressed", "false");
      chip.addEventListener("click", () => applyExample(example));
      chips.appendChild(chip);
    }
  } catch (err) {
    chips.replaceChildren(Object.assign(document.createElement("span"),
      { className: "hint", textContent: `Could not load examples from ${API}/examples` }));
  }
}

/* Clicking a chip fills the form from that example's ACTUAL request body, so
   what gets sent on Ask is what /examples advertised. */
function applyExample(example) {
  const body = example.body || {};
  $("f-query").value = body.request || "";
  $("f-region").value = body.region || "";
  $("f-crop").value = body.crop || "";
  $("f-risk").value = (body.risk_types && body.risk_types[0]) || "";

  if (body.month && /^\d{4}-\d{2}$/.test(body.month)) {
    const [year, month] = body.month.split("-");
    $("f-year").value = year;
    $("f-month").value = month;
  } else {
    $("f-year").value = "";
    $("f-month").value = "";
  }
  updateMonthPreview();

  state.activeExampleId = example.id;
  for (const chip of $("chips").children) {
    if (chip.setAttribute) {
      chip.setAttribute("aria-pressed", String(chip.textContent === example.label));
    }
  }
}

/* ── request body ────────────────────────────────────────────────────────── */

function combinedMonth() {
  const year = $("f-year").value.trim();
  const month = $("f-month").value;
  return year && month ? `${year}-${month}` : null;
}

function updateMonthPreview() {
  const month = combinedMonth();
  $("month-preview").textContent = month ? `Format: ${month}` : "";
  $("month-preview").classList.toggle("hidden", !month);
}

function buildRequestBody() {
  const body = { request: $("f-query").value.trim() };
  const region = $("f-region").value;
  const crop = $("f-crop").value;
  const risk = $("f-risk").value;
  const month = combinedMonth();

  if (region) body.region = region;
  if (crop) body.crop = crop;
  if (risk) body.risk_types = [risk];
  if (month) body.month = month;
  return body;
}

/* ── rendering a real response ───────────────────────────────────────────── */

function showBanner(cls, dotCls, text) {
  $("banner").className = `banner ${cls}`;
  $("banner-dot").className = `dot dot-lg ${dotCls}`;
  $("banner-text").textContent = text;
}

function renderGroundingBanner(grounding) {
  const status = grounding && grounding.status;
  if (status === "clean") {
    showBanner("banner-clean", "dot-green",
      "Clean — every figure was checked against its source");
  } else if (status === "warning") {
    const n = (grounding.unverified_numbers || []).length;
    showBanner("banner-warn", "dot-amber",
      `Warning — ${n} figure${n === 1 ? "" : "s"} could not be verified against a source`);
  } else {
    // "not_generated" or anything unexpected: say so, do not guess clean.
    showBanner("banner-warn", "dot-amber",
      `Grounding status: ${status || "unknown"}`);
  }
  if (grounding && grounding.explanation) $("banner").title = grounding.explanation;
}

/* The synthesis agent writes Markdown, so rendering the report as flat text
 * leaves literal "##" and "**" on screen. This handles the subset it actually
 * emits — headings, rules, bullets, bold, inline code — and nothing else.
 *
 * Everything is built with createElement + textContent, never innerHTML: the
 * report is LLM output, and the one thing a client must never do with LLM
 * output is hand it to an HTML parser. Unhandled syntax degrades to plain text
 * rather than disappearing.
 */
function appendInline(parent, text) {
  // Split on **bold** and `code`, keeping the delimiters' contents.
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  for (const part of parts) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      parent.appendChild(strong);
    } else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      const code = document.createElement("code");
      code.textContent = part.slice(1, -1);
      parent.appendChild(code);
    } else {
      parent.appendChild(document.createTextNode(part));
    }
  }
}

function renderMarkdownBlock(container, block) {
  const lines = block.split("\n");

  // A block that is only a rule.
  if (/^\s*(-{3,}|\*{3,})\s*$/.test(block)) {
    container.append(Object.assign(document.createElement("div"),
      { className: "hr" }));
    return;
  }

  let list = null;
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) continue;

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    const bullet = line.match(/^(\s*)[-*]\s+(.*)$/);

    if (heading) {
      list = null;
      const level = Math.min(heading[1].length + 2, 6);   // "#" -> h3
      const el = document.createElement(`h${level}`);
      el.style.margin = "var(--space-3) 0 var(--space-2)";
      appendInline(el, heading[2]);
      container.appendChild(el);
    } else if (bullet) {
      if (!list) {
        list = document.createElement("ul");
        list.style.margin = "0 0 var(--space-2)";
        list.style.paddingLeft = "1.2em";
        container.appendChild(list);
      }
      const item = document.createElement("li");
      item.className = "claim-text";
      // Nested bullets keep their indent rather than being flattened away.
      if (bullet[1].length >= 2) item.style.marginLeft = "1em";
      appendInline(item, bullet[2]);
      list.appendChild(item);
    } else {
      list = null;
      const p = document.createElement("p");
      p.className = "report-para";
      appendInline(p, line);
      container.appendChild(p);
    }
  }
}

function renderReportText(text) {
  const container = $("report-body");
  container.replaceChildren();
  const blocks = (text || "").split(/\n{2,}/).filter((b) => b.trim());
  if (!blocks.length) {
    container.append(Object.assign(document.createElement("p"),
      { className: "missing-line", textContent: "No report text was returned." }));
    return;
  }
  for (const block of blocks) renderMarkdownBlock(container, block);
}

/* Each horizon stays its own row with its own measured label and skill score.
   Never averaged into one number — see the file header. */
function renderHorizons(horizons) {
  const block = $("claims-block");
  const list = $("claims");
  list.replaceChildren();
  if (!horizons || !horizons.length) {
    block.classList.add("hidden");
    return;
  }
  for (const h of horizons) {
    const row = document.createElement("div");
    row.className = "claim-row";

    const tag = document.createElement("span");
    tag.className = `tag ${tagClassFor(h.label || "")}`;
    tag.textContent = tagTextFor(h.label || "");

    const text = document.createElement("span");
    text.className = "claim-text";
    text.textContent =
      `${h.horizon}: skill ${Number(h.skill_score).toFixed(4)} ` +
      `(${h.method}) — ${h.label}`;

    row.append(tag, text);
    list.appendChild(row);
  }
  block.classList.remove("hidden");
}

function renderMissing(flags) {
  const block = $("missing-block");
  const list = $("missing");
  list.replaceChildren();
  if (!flags || !flags.length) {
    block.classList.add("hidden");
    return;
  }
  for (const flag of flags) {
    list.append(Object.assign(document.createElement("p"),
      { className: "missing-line", textContent: `${flag.what} — ${flag.reason}` }));
  }
  block.classList.remove("hidden");
}

function renderSources(sources) {
  const block = $("sources-block");
  const list = $("sources");
  list.replaceChildren();
  if (!sources || !sources.length) {
    block.classList.add("hidden");
    return;
  }
  for (const source of sources) {
    list.append(Object.assign(document.createElement("p"),
      { className: "source-line", textContent: source.citation || source.source || "" }));
  }
  block.classList.remove("hidden");
}

/* Live third-party sources, each labelled with its publisher.
 *
 * Rendered in their own block, never merged into the horizons or the report
 * figures: IMD's, NASA's and data.gov.in's numbers belong to them, and the UI's
 * job is to keep that visible. Unavailable sources are shown too, with the
 * reason — "we did not ask" and "they had nothing" are different facts.
 */
function renderExternal(sources) {
  const block = $("external-block");
  const list = $("external");
  list.replaceChildren();
  if (!sources || !sources.length) {
    block.classList.add("hidden");
    return;
  }
  for (const source of sources) {
    const line = document.createElement("p");
    line.className = source.available ? "source-line" : "missing-line";
    line.textContent = source.available
      ? `${source.publisher}: ${source.excerpt}`
      : `${source.publisher} — unavailable: ${source.reason}`;
    line.title = source.citation || "";
    list.appendChild(line);
  }
  block.classList.remove("hidden");
}

function renderReport(body) {
  $("query-label").textContent = body.request || "";
  renderGroundingBanner(body.grounding);
  renderReportText(body.report);
  renderHorizons(body.horizon_confidence);
  renderMissing(body.missing_data);
  renderExternal(body.external_sources);
  renderSources(body.retrieved_sources);
  renderQuota(body.quota);
  $("result").classList.remove("hidden");
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* A failed call renders as a failure. No placeholder report ever stands in. */
function renderError(title, detail) {
  $("query-label").textContent = $("f-query").value.trim() || "Your question";
  showBanner("banner-error", "dot-red", title);
  $("banner").title = "";
  renderReportText(detail);
  renderHorizons(null);
  renderMissing(null);
  renderExternal(null);
  renderSources(null);
  $("result").classList.remove("hidden");
}

/* ── ask ─────────────────────────────────────────────────────────────────── */

function setBusy(busy) {
  state.busy = busy;
  $("ask").disabled = busy;
  $("ask-spinner").classList.toggle("hidden", !busy);
  $("ask-label").textContent = busy
    ? "Preparing your report…"
    : "Ask";
}

async function ask() {
  const body = buildRequestBody();
  if (!body.request) {
    renderError("The question is empty",
                "Type a question first, or pick one of the example queries.");
    return;
  }

  setBusy(true);
  try {
    const response = await fetch(`${API}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    let payload = null;
    try { payload = await response.json(); } catch (err) { /* non-JSON error body */ }

    if (response.ok) {
      renderReport(payload);
      return;
    }

    const detail = payload && payload.detail
      ? (typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail))
      : `HTTP ${response.status}`;

    if (response.status === 429) {
      // Quota gets the mockup's own message plus the API's real explanation.
      $("quota-warning").textContent =
        "Today's quota is used up — please try again tomorrow.";
      $("quota-warning").classList.remove("hidden");
      renderError("Today's quota is used up — please try again tomorrow.", detail);
    } else if (response.status === 400) {
      renderError("The request was not accepted (400)", detail);
    } else if (response.status === 422) {
      renderError("The request format was invalid (422)", detail);
    } else {
      renderError(`The report could not be generated (HTTP ${response.status})`, detail);
    }
    loadQuota();
  } catch (err) {
    renderError("Could not reach the backend",
      `The request never got to ${API}/report. ${err}`);
  } finally {
    setBusy(false);
    loadQuota();
  }
}

/* ── wiring ──────────────────────────────────────────────────────────────── */

$("ask").addEventListener("click", ask);
$("f-year").addEventListener("input", updateMonthPreview);
$("f-month").addEventListener("change", updateMonthPreview);
$("f-query").addEventListener("input", () => {
  state.activeExampleId = null;
  for (const chip of $("chips").children) {
    if (chip.setAttribute) chip.setAttribute("aria-pressed", "false");
  }
});

loadHealth();
loadExamples();
loadQuota();
