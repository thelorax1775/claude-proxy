// Backends panel: the ordered failover list plus the add-backend form.

import { api } from "./api.js";
import { $, escapeHtml } from "./ui.js";
import { store, emit } from "./store.js";

const PRESETS = [
  { label: "Groq", base_url: "https://api.groq.com/openai/v1", big: "llama-3.3-70b-versatile", small: "llama-3.1-8b-instant" },
  { label: "Google AI Studio (Gemini)", base_url: "https://generativelanguage.googleapis.com/v1beta/openai", big: "gemini-2.0-flash", small: "gemini-2.0-flash" },
  { label: "Cerebras", base_url: "https://api.cerebras.ai/v1", big: "llama-3.3-70b", small: "llama3.1-8b" },
  { label: "OpenRouter", base_url: "https://openrouter.ai/api/v1", big: "meta-llama/llama-3.3-70b-instruct:free", small: "meta-llama/llama-3.1-8b-instruct:free" },
  { label: "NVIDIA NIM", base_url: "https://integrate.api.nvidia.com/v1", big: "meta/llama-3.3-70b-instruct", small: "meta/llama-3.1-8b-instruct" },
  { label: "Local Ollama", base_url: "http://localhost:11434/v1", big: "llama3.1:8b", small: "llama3.1:8b" },
  { label: "Open WebUI gateway", base_url: "http://localhost:8080/api", big: "", small: "" },
  {
    label: "Ollama Cloud",
    base_url: "https://ollama.com/v1",
    big: "", small: "",
    note: "Best-known guess -- verify the exact base URL and grab a key at ollama.com/settings/keys, then confirm with “Detect models” before trusting it.",
  },
  { label: "Custom", base_url: "", big: "", small: "" },
];

function fillPresets() {
  const select = $("preset-select");
  select.innerHTML = PRESETS.map((p, i) => `<option value="${i}">${escapeHtml(p.label)}</option>`).join("");
  select.onchange = () => {
    const preset = PRESETS[select.value];
    $("f-base-url").value = preset.base_url;
    $("f-big-model").value = preset.big;
    $("f-small-model").value = preset.small;
    if (!$("f-name").value) {
      $("f-name").value = preset.label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
    }
    $("preset-note").textContent = preset.note || "";
  };
  select.onchange();
}

// Which card has its edit form open, if any. Only one at a time: two open
// forms with an unsaved key field between them is a good way to paste a key
// into the wrong backend.
let editing = null;

// name -> usage row from /api/usage. Kept apart from store.backends because it
// arrives later and can fail on its own: the list must draw immediately with
// whatever the proxy already knows, and gain the meters when they show up.
let usageByName = {};

/** At or above this, treat the allowance as gone. The provider reports a
 *  rounded fraction, so 0.999 and 1.0 mean the same thing in practice. */
const SPENT_AT = 99.5;

function meterFor(name) {
  const u = usageByName[name];
  if (!u) return "";

  if (u.supported === false) {
    return `<div class="meter"><span class="meter-unknown"
      title="${escapeHtml(u.reason || "")}">usage not reported by this provider</span></div>`;
  }
  if (u.error) {
    return `<div class="meter"><span class="meter-unknown">usage unavailable — ${escapeHtml(u.error)}</span></div>`;
  }
  if (typeof u.used_pct !== "number") return "";

  const spent = u.used_pct >= SPENT_AT;
  // The bar shows magnitude; this string is what actually states it, so it is
  // never omitted and never the only thing that changes colour.
  const label = spent ? "weekly quota spent" : `${u.remaining_pct}% left this week`;

  const detail = [
    `${u.used_pct}% of the weekly allowance used`,
    u.requests ? `${u.requests} requests` : "",
    (u.models || []).map((m) => `${m.name}: ${m.requests}`).join(", "),
    typeof u.session_pct === "number" ? `session window ${u.session_pct}%` : "",
    u.checked_at ? `checked ${new Date(u.checked_at * 1000).toLocaleTimeString()}` : "",
  ].filter(Boolean).join(" · ");

  return `
    <div class="meter ${spent ? "is-spent" : ""}" title="${escapeHtml(detail)}">
      <div class="meter-track" role="meter" aria-valuemin="0" aria-valuemax="100"
           aria-valuenow="${u.used_pct}"
           aria-label="${escapeHtml(`${name}: ${label}`)}">
        <div class="meter-fill" style="width: ${u.used_pct}%"></div>
      </div>
      <span class="meter-value">${escapeHtml(label)}</span>
    </div>`;
}

/** Fetched after the cards are drawn, so a slow provider never delays the
 *  list. Failure is silent by design -- the cards are still correct without
 *  it, and a red banner over "we couldn't tell you a nice-to-have number"
 *  would be worse than no number. */
export async function loadUsage({ refresh = false } = {}) {
  try {
    const data = await api(`/api/usage${refresh ? "?refresh=1" : ""}`);
    usageByName = Object.fromEntries((data.backends || []).map((u) => [u.name, u]));
    render();
  } catch {
    /* leave the cards as they are */
  }
}

function editForm(b) {
  return `
    <form class="form edit-form" data-edit>
      <label>Name
        <input name="name" value="${escapeHtml(b.name)}" autocomplete="off">
      </label>
      <label>Base URL
        <input name="base_url" value="${escapeHtml(b.base_url)}" autocomplete="off">
      </label>
      <label>API key
        <span class="hint">${b.has_key ? "leave blank to keep the saved key" : "none saved"}</span>
        <input name="api_key" type="password" placeholder="${b.has_key ? "unchanged" : "paste here, never into chat"}" autocomplete="off">
      </label>
      <button type="button" class="ghost-btn" data-act="detect">Detect models</button>
      <p class="hint" data-detect-hint></p>
      <label>Default model
        <input name="big_model" list="model-options" value="${escapeHtml(b.big_model)}" autocomplete="off">
      </label>
      <label>Fast model <span class="hint">used for haiku-mapped calls</span>
        <input name="small_model" list="model-options" value="${escapeHtml(b.small_model)}" autocomplete="off">
      </label>
      <div class="row">
        <button type="submit" class="primary-btn">Save</button>
        <button type="button" class="ghost-btn" data-act="cancel">Cancel</button>
      </div>
      <p class="hint error" data-edit-error></p>
    </form>`;
}

function render() {
  const list = $("backend-list");
  if (store.backends.length === 0) {
    list.innerHTML = `<p class="hint">Nothing configured yet — add one below to start.</p>`;
    return;
  }

  list.innerHTML = store.backends.map((b, i) => `
    <div class="card ${editing === b.name ? "is-selected" : ""}" data-name="${escapeHtml(b.name)}">
      <div class="row">
        <span class="title"><span class="ordinal">${i + 1}</span>${escapeHtml(b.name)}</span>
        <span class="actions">
          <button class="icon-btn" data-act="edit" title="Edit models, URL and key">&#9998;</button>
          <button class="icon-btn" data-act="up" ${i === 0 ? "disabled" : ""} title="Move up">&uarr;</button>
          <button class="icon-btn" data-act="down" ${i === store.backends.length - 1 ? "disabled" : ""} title="Move down">&darr;</button>
          <button class="icon-btn danger" data-act="delete" title="Remove">&times;</button>
        </span>
      </div>
      <div class="sub">${escapeHtml(b.base_url)}</div>
      <div class="tags">
        <span class="tag">${escapeHtml(b.big_model || "no model")}</span>
        ${b.small_model && b.small_model !== b.big_model ? `<span class="tag">${escapeHtml(b.small_model)}</span>` : ""}
        <span class="tag ${b.has_key ? "accent" : ""}">${b.has_key ? "key set" : "no key"}</span>
        ${b.cooling ? `<span class="tag bad" title="Skipped until it's had a rest — it recently rate-limited or rejected its key">cooling ${b.cooling}s</span>` : ""}
      </div>
      ${meterFor(b.name)}
      ${editing === b.name ? editForm(b) : ""}
    </div>
  `).join("");

  list.querySelectorAll("button[data-act]").forEach((btn) => {
    btn.onclick = () => cardAction(btn.closest(".card").dataset.name, btn.dataset.act, btn);
  });
  list.querySelectorAll("form[data-edit]").forEach((form) => {
    form.onsubmit = (e) => saveEdit(e, form.closest(".card").dataset.name);
  });
}

/** Only the fields that actually changed, so a PATCH can't overwrite
 *  something the form wasn't showing. A blank key is left out entirely --
 *  the server reads a blank as "keep the saved one", but not sending it at
 *  all says the same thing more plainly. */
function changedFields(form, current) {
  const data = Object.fromEntries(new FormData(form));
  const patch = {};
  for (const field of ["name", "base_url", "big_model", "small_model"]) {
    const value = (data[field] || "").trim();
    if (value !== (current[field] || "")) patch[field] = value;
  }
  if ((data.api_key || "").trim()) patch.api_key = data.api_key.trim();
  return patch;
}

async function saveEdit(event, name) {
  event.preventDefault();
  const form = event.target;
  const errorEl = form.querySelector("[data-edit-error]");
  errorEl.textContent = "";

  const current = store.backends.find((b) => b.name === name);
  const patch = changedFields(form, current);
  if (Object.keys(patch).length === 0) {
    editing = null;
    render();
    return;
  }

  try {
    await api(`/api/backends/${encodeURIComponent(name)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    editing = null;
    await loadBackends();
  } catch (err) {
    errorEl.textContent = err.message;
  }
}

async function detectForSaved(name, card) {
  const hint = card.querySelector("[data-detect-hint]");
  hint.textContent = "Probing…";
  const typedKey = card.querySelector("input[name=api_key]").value;
  const typedUrl = card.querySelector("input[name=base_url]").value.trim();
  try {
    // Naming the backend lets the server fall back to the stored key, which
    // the browser never has.
    const data = await api("/api/probe-models", {
      method: "POST",
      body: JSON.stringify({ name, base_url: typedUrl, api_key: typedKey }),
    });
    $("model-options").innerHTML = data.models.map((m) => `<option value="${escapeHtml(m)}">`).join("");
    hint.textContent = data.models.length
      ? `Found ${data.models.length} model(s) — the model fields now autocomplete.`
      : data.error
        ? `Couldn't auto-detect: ${data.error}. Type the model id manually.`
        : "No models returned — type the model id manually.";
  } catch (err) {
    hint.textContent = `Couldn't auto-detect: ${err.message}. Type the model id manually.`;
  }
}

async function cardAction(name, act, btn) {
  try {
    if (act === "edit") {
      editing = editing === name ? null : name;
      render();
      return;
    }
    if (act === "cancel") {
      editing = null;
      render();
      return;
    }
    if (act === "detect") {
      await detectForSaved(name, btn.closest(".card"));
      return;
    }
    if (act === "delete") {
      if (!confirm(`Remove backend "${name}"?`)) return;
      await api(`/api/backends/${encodeURIComponent(name)}`, { method: "DELETE" });
      if (editing === name) editing = null;
    } else {
      const order = store.backends.map((b) => b.name);
      const from = order.indexOf(name);
      const to = act === "up" ? from - 1 : from + 1;
      [order[from], order[to]] = [order[to], order[from]];
      await api("/api/backends/reorder", { method: "POST", body: JSON.stringify({ order }) });
    }
    await loadBackends();
  } catch (err) {
    alert(err.message);
  }
}

export async function loadBackends() {
  try {
    const data = await api("/api/backends");
    store.backends = data.backends;
    render();
    // Not awaited: the meters fill in when the providers answer.
    loadUsage();
  } catch (err) {
    store.backends = [];
    $("backend-list").innerHTML =
      `<p class="hint error">Couldn't load backends: ${escapeHtml(err.message)}. Is the proxy still running?</p>`;
  }
  emit("backends");
}

export function initBackends() {
  fillPresets();

  $("add-form").onsubmit = async (e) => {
    e.preventDefault();
    $("add-error").textContent = "";
    try {
      await api("/api/backends", {
        method: "POST",
        body: JSON.stringify({
          name: $("f-name").value.trim(),
          base_url: $("f-base-url").value.trim(),
          api_key: $("f-api-key").value,
          big_model: $("f-big-model").value.trim(),
          small_model: $("f-small-model").value.trim(),
        }),
      });
      e.target.reset();
      fillPresets();
      $("add-backend").open = false;
      await loadBackends();
    } catch (err) {
      $("add-error").textContent = err.message;
    }
  };

  $("refresh-usage").onclick = async (e) => {
    const btn = e.target;
    btn.disabled = true;
    const original = btn.textContent;
    btn.textContent = "Checking…";
    await loadUsage({ refresh: true });
    btn.textContent = original;
    btn.disabled = false;
  };

  $("detect-btn").onclick = async () => {
    $("detect-hint").textContent = "Probing…";
    try {
      const data = await api("/api/probe-models", {
        method: "POST",
        body: JSON.stringify({ base_url: $("f-base-url").value.trim(), api_key: $("f-api-key").value }),
      });
      $("model-options").innerHTML = data.models.map((m) => `<option value="${escapeHtml(m)}">`).join("");
      $("detect-hint").textContent = data.models.length
        ? `Found ${data.models.length} model(s) — the model fields now autocomplete.`
        : data.error
          ? `Couldn't auto-detect: ${data.error}. Type the model id manually.`
          : "No models returned — type the model id manually.";
    } catch (err) {
      $("detect-hint").textContent = `Couldn't auto-detect: ${err.message}. Type the model id manually.`;
    }
  };
}
