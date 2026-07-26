const PRESETS = [
  { label: "Groq", base_url: "https://api.groq.com/openai/v1", big: "llama-3.3-70b-versatile", small: "llama-3.1-8b-instant" },
  { label: "Google AI Studio (Gemini)", base_url: "https://generativelanguage.googleapis.com/v1beta/openai", big: "gemini-2.0-flash", small: "gemini-2.0-flash" },
  { label: "Cerebras", base_url: "https://api.cerebras.ai/v1", big: "llama-3.3-70b", small: "llama3.1-8b" },
  { label: "OpenRouter", base_url: "https://openrouter.ai/api/v1", big: "meta-llama/llama-3.3-70b-instruct:free", small: "meta-llama/llama-3.1-8b-instruct:free" },
  { label: "NVIDIA NIM", base_url: "https://integrate.api.nvidia.com/v1", big: "meta/llama-3.3-70b-instruct", small: "meta/llama-3.1-8b-instruct" },
  { label: "Local Ollama", base_url: "http://localhost:11434/v1", big: "llama3.1:8b", small: "llama3.1:8b" },
  { label: "Open WebUI gateway", base_url: "http://localhost:8080/api", big: "", small: "" },
  { label: "Custom", base_url: "", big: "", small: "" },
];

let backends = [];
let history = [];

const $ = (id) => document.getElementById(id);

function proxyKey() {
  return $("proxy-key").value;
}

async function api(path, opts = {}) {
  const resp = await fetch(path, {
    ...opts,
    headers: { "Content-Type": "application/json", "X-Proxy-Key": proxyKey(), ...opts.headers },
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
  return data;
}

function fillPresets() {
  const sel = $("preset-select");
  sel.innerHTML = PRESETS.map((p, i) => `<option value="${i}">${p.label}</option>`).join("");
  sel.onchange = () => {
    const p = PRESETS[sel.value];
    $("f-base-url").value = p.base_url;
    $("f-big-model").value = p.big;
    $("f-small-model").value = p.small;
    if (!$("f-name").value) $("f-name").value = p.label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  };
  sel.onchange();
}

function renderBackendList() {
  const el = $("backend-list");
  if (backends.length === 0) {
    el.innerHTML = `<div class="hint">No backends configured yet.</div>`;
  } else {
    el.innerHTML = backends.map((b, i) => `
      <div class="backend-card" data-name="${b.name}">
        <div class="row">
          <span class="name">${i + 1}. ${b.name}</span>
          <span>
            <button class="icon-btn" data-act="up" ${i === 0 ? "disabled" : ""} title="Move up">&uarr;</button>
            <button class="icon-btn" data-act="down" ${i === backends.length - 1 ? "disabled" : ""} title="Move down">&darr;</button>
            <button class="icon-btn danger" data-act="delete" title="Remove">&times;</button>
          </span>
        </div>
        <div class="url">${b.base_url}</div>
        <div class="models">${b.big_model || "?"} / ${b.small_model || "?"}</div>
        <span class="badge ${b.has_key ? "" : "dim"}">${b.has_key ? "key set" : "no key needed"}</span>
      </div>
    `).join("");
  }

  el.querySelectorAll("button[data-act]").forEach((btn) => {
    btn.onclick = () => handleCardAction(btn.closest(".backend-card").dataset.name, btn.dataset.act);
  });

  renderChatSelectors();
}

async function handleCardAction(name, act) {
  try {
    if (act === "delete") {
      if (!confirm(`Remove backend "${name}"?`)) return;
      await api(`/api/backends/${encodeURIComponent(name)}`, { method: "DELETE" });
    } else if (act === "up" || act === "down") {
      const order = backends.map((b) => b.name);
      const i = order.indexOf(name);
      const j = act === "up" ? i - 1 : i + 1;
      [order[i], order[j]] = [order[j], order[i]];
      await api("/api/backends/reorder", { method: "POST", body: JSON.stringify({ order }) });
    }
    await loadBackends();
  } catch (err) {
    alert(err.message);
  }
}

function renderChatSelectors() {
  const backendSel = $("chat-backend");
  const prevBackend = backendSel.value;
  backendSel.innerHTML = backends.map((b) => `<option value="${b.name}">${b.name}</option>`).join("");
  if (backends.some((b) => b.name === prevBackend)) backendSel.value = prevBackend;
  backendSel.onchange = renderModelSelector;
  renderModelSelector();
}

function renderModelSelector() {
  const modelSel = $("chat-model");
  const b = backends.find((x) => x.name === $("chat-backend").value);
  const prevModel = modelSel.value;
  if (!b) { modelSel.innerHTML = ""; return; }
  const models = [...new Set([b.big_model, b.small_model].filter(Boolean))];
  modelSel.innerHTML = models.map((m) => `<option value="${m}">${m}</option>`).join("");
  if (models.includes(prevModel)) modelSel.value = prevModel;
}

async function loadBackends() {
  try {
    const data = await api("/api/backends");
    backends = data.backends;
    renderBackendList();
  } catch (err) {
    $("backend-list").innerHTML = `<div class="hint error">Couldn't load backends: ${err.message}. Check the proxy key above.</div>`;
  }
}

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
    await loadBackends();
  } catch (err) {
    $("add-error").textContent = err.message;
  }
};

$("detect-btn").onclick = async () => {
  $("detect-hint").textContent = "Probing...";
  try {
    const data = await api("/api/probe-models", {
      method: "POST",
      body: JSON.stringify({ base_url: $("f-base-url").value.trim(), api_key: $("f-api-key").value }),
    });
    const list = $("model-options");
    list.innerHTML = data.models.map((m) => `<option value="${m}">`).join("");
    $("detect-hint").textContent = data.models.length
      ? `Found ${data.models.length} model(s) -- pick from the dropdown in the model fields.`
      : (data.error ? `Couldn't auto-detect: ${data.error}. Type the model id manually.` : "No models returned -- type manually.");
  } catch (err) {
    $("detect-hint").textContent = `Couldn't auto-detect: ${err.message}. Type the model id manually.`;
  }
};

function appendBubble(role, text) {
  const messages = $("messages");
  const empty = messages.querySelector(".empty-state");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

$("clear-btn").onclick = () => {
  history = [];
  $("messages").innerHTML = `<div class="empty-state">Add a backend on the left, pick it above, and start chatting.</div>`;
};

$("composer-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
$("send-btn").onclick = sendMessage;

async function sendMessage() {
  const input = $("composer-input");
  const text = input.value.trim();
  if (!text) return;
  const backendName = $("chat-backend").value;
  const model = $("chat-model").value;
  if (!backendName || !model) {
    $("chat-hint").textContent = "Configure and select a backend + model first.";
    return;
  }
  $("chat-hint").textContent = "";

  input.value = "";
  history.push({ role: "user", content: text });
  appendBubble("user", text);
  const assistantBubble = appendBubble("assistant", "");

  $("send-btn").disabled = true;
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Proxy-Key": proxyKey() },
      body: JSON.stringify({ backend: backendName, model, messages: history, stream: true }),
    });

    if (!resp.ok || !resp.body) {
      const data = await resp.json().catch(() => ({}));
      assistantBubble.textContent = `[error] ${data.detail || data.error || resp.status}`;
      history.pop();
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let full = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        try {
          const chunk = JSON.parse(data);
          if (chunk.error) {
            full += `\n[error] ${chunk.error}`;
            assistantBubble.textContent = full;
            continue;
          }
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) {
            full += delta;
            assistantBubble.textContent = full;
            $("messages").scrollTop = $("messages").scrollHeight;
          }
        } catch { /* ignore partial/non-JSON lines */ }
      }
    }
    history.push({ role: "assistant", content: full });
  } catch (err) {
    assistantBubble.textContent = `[error] ${err.message}`;
    history.pop();
  } finally {
    $("send-btn").disabled = false;
  }
}

$("proxy-key").value = localStorage.getItem("proxyKey") || "";
$("proxy-key").addEventListener("change", () => {
  localStorage.setItem("proxyKey", $("proxy-key").value);
});

fillPresets();
loadBackends();
