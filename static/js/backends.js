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

function render() {
  const list = $("backend-list");
  if (store.backends.length === 0) {
    list.innerHTML = `<p class="hint">Nothing configured yet — add one below to start.</p>`;
    return;
  }

  list.innerHTML = store.backends.map((b, i) => `
    <div class="card" data-name="${escapeHtml(b.name)}">
      <div class="row">
        <span class="title"><span class="ordinal">${i + 1}</span>${escapeHtml(b.name)}</span>
        <span class="actions">
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
      </div>
    </div>
  `).join("");

  list.querySelectorAll("button[data-act]").forEach((btn) => {
    btn.onclick = () => cardAction(btn.closest(".card").dataset.name, btn.dataset.act);
  });
}

async function cardAction(name, act) {
  try {
    if (act === "delete") {
      if (!confirm(`Remove backend "${name}"?`)) return;
      await api(`/api/backends/${encodeURIComponent(name)}`, { method: "DELETE" });
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
  } catch (err) {
    store.backends = [];
    $("backend-list").innerHTML =
      `<p class="hint error">Couldn't load backends: ${escapeHtml(err.message)}. Check the proxy key up top.</p>`;
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
