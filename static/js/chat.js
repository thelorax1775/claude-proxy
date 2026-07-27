// The test chat: backend/model pickers, the agent + sandbox toggles, and the
// SSE rendering for plain streaming, mid-conversation failover, and agent
// tool rounds.

import { postStream, sseJson } from "./api.js";
import { $, escapeHtml, renderRich, shorten, autoGrow } from "./ui.js";
import { store, on, emit, prefs } from "./store.js";
import { selectedPaths } from "./workdirs.js";
import { selectedGroups, maxRounds } from "./groups.js";
import { ensureSession, saveCurrent, newSession } from "./sessions.js";

let inFlight = null;
// What the last agent run was scoped to, so the scope notice only shows up
// when it actually changes rather than above every single message.
let lastContextKey = "";

const EMPTY_STATE = `
  <div class="empty-state">
    <strong>Nothing here yet</strong>
    <span>Pick a backend and model above, then say something. Turn on
    <em>Agent tools</em> below to let the model work on real directories,
    repositories and the web.</span>
  </div>`;

function messagesEl() {
  return $("messages");
}

/** Follow the stream only while the reader is already at the bottom --
 *  yanking the view down while someone is scrolled up reading is the other
 *  half of "the chat gets messed up as it gets longer". */
function scrollDown(force = false) {
  const el = messagesEl();
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  if (force || nearBottom) el.scrollTop = el.scrollHeight;
}

function clearEmptyState() {
  const empty = messagesEl().querySelector(".empty-state");
  if (empty) empty.remove();
}

function addMessage(role, text, { error = false, streaming = false } = {}) {
  clearEmptyState();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}${error ? " is-error" : ""}`;
  wrap.innerHTML = `<span class="who">${role === "user" ? "you" : "assistant"}</span><div class="body"></div>`;
  const body = wrap.querySelector(".body");
  body.innerHTML = renderRich(text);
  if (streaming) body.classList.add("cursor");
  messagesEl().appendChild(wrap);
  // Your own message always scrolls into view; everything else respects
  // where you were reading.
  scrollDown(role === "user");
  return body;
}

function addNotice(text) {
  clearEmptyState();
  const wrap = document.createElement("div");
  wrap.className = "msg notice";
  wrap.innerHTML = `<div class="body"></div>`;
  wrap.querySelector(".body").textContent = text;
  messagesEl().appendChild(wrap);
  scrollDown();
}

/** Shown when a message uses up its tool-round budget. The run paused with a
 *  status report; this just sends the follow-up that resumes it. */
function addContinue(rounds) {
  clearEmptyState();
  const wrap = document.createElement("div");
  wrap.className = "msg notice continue-notice";
  wrap.innerHTML = `<div class="body">Paused after ${rounds} tool rounds — the chat is intact.
    <button class="ghost-btn small" data-act="continue">Continue</button></div>`;
  wrap.querySelector("button").onclick = () => {
    wrap.remove();
    $("composer-input").value = "Continue where you left off.";
    sendMessage();
  };
  messagesEl().appendChild(wrap);
  scrollDown();
}

function addToolCall(name, args) {
  clearEmptyState();
  const card = document.createElement("details");
  card.className = "tool-call running";
  const summaryArg = args.command || args.path || args.pattern || "";
  card.innerHTML = `
    <summary>
      <span class="tool-name">${escapeHtml(name)}</span>
      <span class="tool-arg">${escapeHtml(shorten(summaryArg, 80))}</span>
      <span class="tool-state">running…</span>
    </summary>
    <pre class="args-block"></pre>
    <pre class="output-block">waiting for output…</pre>`;
  card.querySelector(".args-block").textContent = JSON.stringify(args, null, 2);
  messagesEl().appendChild(card);
  scrollDown();
  return card;
}

function finishToolCall(card, output) {
  const failed = output.startsWith("[error]");
  card.className = `tool-call ${failed ? "failed" : "done"}`;
  card.querySelector(".tool-state").textContent = failed ? "failed" : "done";
  card.querySelector(".output-block").textContent = output || "(no output)";
  if (failed) card.open = true;
  scrollDown();
}

// ---------------------------------------------------------------------------
// transcript <-> saved session
// ---------------------------------------------------------------------------

function record(event) {
  store.events.push(event);
}

function sessionState(backend, model, agent, sandbox, groups) {
  return {
    backend, model, agent, sandbox, groups,
    workdirs: selectedPaths(),
    messages: store.history,
    events: store.events,
  };
}

/** Rebuild the visible chat from a saved session's event list. */
function replayTranscript() {
  const el = messagesEl();
  el.innerHTML = "";
  lastContextKey = "";

  for (const event of store.events) {
    if (event.t === "user") addMessage("user", event.text);
    else if (event.t === "assistant") addMessage("assistant", event.text, { error: event.error });
    else if (event.t === "notice") addNotice(event.text);
    else if (event.t === "tool") finishToolCall(addToolCall(event.name, event.args || {}), event.output || "");
  }

  if (store.events.length === 0) el.innerHTML = EMPTY_STATE;
  renderChatTitle();
  scrollDown();
}

function renderChatTitle() {
  const current = store.sessions.find((s) => s.id === store.sessionId);
  $("chat-title").textContent = current ? current.title : "";
}

// ---------------------------------------------------------------------------
// selectors + toggles
// ---------------------------------------------------------------------------

function renderBackendSelector() {
  const select = $("chat-backend");
  const wanted = select.value || prefs.get("backend", "");
  select.innerHTML = store.backends
    .map((b) => `<option value="${escapeHtml(b.name)}">${escapeHtml(b.name)}</option>`)
    .join("");
  if (store.backends.some((b) => b.name === wanted)) select.value = wanted;
  renderModelSelector();
}

function renderModelSelector() {
  const select = $("chat-model");
  const backend = store.backends.find((b) => b.name === $("chat-backend").value);
  if (!backend) {
    select.innerHTML = "";
    return;
  }
  const wanted = select.value || prefs.get("model", "");
  const models = [...new Set([backend.big_model, backend.small_model].filter(Boolean))];
  select.innerHTML = models.map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
  if (models.includes(wanted)) select.value = wanted;
}

export function agentOn() {
  return $("agent-mode").checked;
}

export function sandboxOn() {
  return $("sandbox-mode").checked;
}

function syncToggles() {
  const sandboxToggle = $("sandbox-mode");
  const sandboxRow = $("sandbox-row");
  const on = agentOn();

  // The sandbox switch only means anything while agent mode is on.
  sandboxToggle.disabled = !on;
  sandboxRow.classList.toggle("is-locked", !on);
  sandboxRow.classList.toggle("warn", on && !sandboxToggle.checked);
  $("sandbox-note").textContent = sandboxToggle.checked
    ? "stay inside the selected directories"
    : "file tools may go anywhere on that machine";

  const chip = $("mode-chip");
  chip.classList.toggle("is-hidden", !on);
  if (on) {
    const dirs = selectedPaths();
    const primary = dirs[0] ? dirs[0].split("/").filter(Boolean).pop() || dirs[0] : "workspace";
    const extra = dirs.length > 1 ? ` +${dirs.length - 1}` : "";
    chip.classList.toggle("unsandboxed", !sandboxToggle.checked);
    chip.textContent = sandboxToggle.checked
      ? `agent · sandboxed · ${primary}${extra}`
      : `agent · UNSANDBOXED · ${primary}${extra}`;
  }
  emit("mode");
}

function setBusy(busy) {
  const button = $("send-btn");
  button.textContent = busy ? "Stop" : "Send";
  button.classList.toggle("is-busy", busy);
}

// ---------------------------------------------------------------------------
// sending
// ---------------------------------------------------------------------------

async function sendMessage() {
  const input = $("composer-input");
  const text = input.value.trim();
  if (!text) return;

  const backend = $("chat-backend").value;
  const model = $("chat-model").value;
  if (!backend || !model) {
    $("chat-hint").textContent = "Add and select a backend + model first.";
    return;
  }
  $("chat-hint").textContent = "";
  prefs.set("backend", backend);
  prefs.set("model", model);

  const agent = agentOn();
  const sandbox = sandboxOn();
  const groups = selectedGroups();

  input.value = "";
  autoGrow(input);
  store.history.push({ role: "user", content: text });
  record({ t: "user", text });
  addMessage("user", text);
  await ensureSession(text, sessionState(backend, model, agent, sandbox, groups));

  // Plain chat streams tokens into one bubble; agent mode emits whole
  // rounds, so its bubbles are created as events arrive.
  let liveBody = agent ? null : addMessage("assistant", "", { streaming: true });
  const toolCards = new Map();
  const agentTexts = [];
  let streamed = "";
  let errorText = "";
  let hitLimit = 0;
  // What an agent run actually added to the model-facing conversation: the
  // assistant's tool calls and every tool result, not just the text. Without
  // it, Continue resumes with a summary and the model redoes all the work.
  let agentHistory = null;

  /** Nothing usable came back. The user's message is on screen and in the
   *  saved transcript, so it has to stay in the model-facing history too --
   *  silently dropping it is what makes the two diverge, and the next message
   *  then refers to a question the model was never shown. Record what
   *  happened as the assistant turn instead. */
  const settle = (reason) => store.history.push({ role: "assistant", content: reason });

  inFlight = new AbortController();
  setBusy(true);

  try {
    const resp = await postStream("/api/chat", {
      backend,
      model,
      messages: store.history,
      stream: true,
      agent,
      sandbox,
      groups,
      max_rounds: maxRounds(),
      workdirs: selectedPaths(),
    }, inFlight.signal);

    if (!resp.ok || !resp.body) {
      const data = await resp.json().catch(() => ({}));
      const message = data.detail || data.error || `HTTP ${resp.status}`;
      if (liveBody) {
        liveBody.classList.remove("cursor");
        liveBody.closest(".msg").classList.add("is-error");
        liveBody.innerHTML = renderRich(message);
      } else {
        addMessage("assistant", message, { error: true });
      }
      record({ t: "assistant", text: message, error: true });
      settle(message);
      return;
    }

    for await (const chunk of sseJson(resp)) {
      if (chunk.agent_context) {
        const { roots, sandbox: sandboxed } = chunk.agent_context;
        // A clone during the run adds a directory -- refresh the panel so the
        // scope shown on screen matches what the model is actually using.
        if (chunk.workdirs_changed) emit("workdirs-stale");
        const key = JSON.stringify([roots, sandboxed]);
        if (key !== lastContextKey) {
          lastContextKey = key;
          const notice = `${sandboxed ? "Sandboxed" : "UNSANDBOXED"} agent run · ${roots.join(" · ")}`;
          addNotice(notice);
          record({ t: "notice", text: notice });
        }
        continue;
      }

      if (chunk.claude_proxy_switch) {
        const notice = `Switched to "${chunk.claude_proxy_switch}" (${chunk.claude_proxy_model}) after the previous backend ran out.`;
        addNotice(notice);
        record({ t: "notice", text: notice });
        $("chat-backend").value = chunk.claude_proxy_switch;
        renderModelSelector();
        $("chat-model").value = chunk.claude_proxy_model;
        continue;
      }

      if (chunk.agent_text) {
        agentTexts.push(chunk.agent_text);
        addMessage("assistant", chunk.agent_text);
        record({ t: "assistant", text: chunk.agent_text });
        continue;
      }

      if (chunk.agent_tool_call) {
        const { id, name, arguments: args } = chunk.agent_tool_call;
        toolCards.set(id, { card: addToolCall(name, args || {}), name, args: args || {} });
        continue;
      }

      if (chunk.agent_tool_result) {
        const { id, name, output } = chunk.agent_tool_result;
        const entry = toolCards.get(id);
        if (entry) finishToolCall(entry.card, output);
        else finishToolCall(addToolCall(name, {}), output);
        record({ t: "tool", name, args: entry ? entry.args : {}, output });
        continue;
      }

      if (chunk.agent_history) {
        agentHistory = chunk.agent_history;
        continue;
      }

      if (chunk.agent_limit) {
        hitLimit = chunk.agent_limit;
        continue;
      }

      if (chunk.agent_done) continue;

      if (chunk.error) {
        errorText = chunk.error;
        addMessage("assistant", chunk.error, { error: true });
        record({ t: "assistant", text: chunk.error, error: true });
        continue;
      }

      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) {
        streamed += delta;
        if (liveBody) {
          liveBody.innerHTML = renderRich(streamed);
          liveBody.classList.add("cursor");
          scrollDown();
        }
      }
    }

    if (agentHistory) {
      // The whole run: assistant turns, tool calls, tool results. Continue
      // (and any later message) then carries on from the work itself.
      store.history.push(...agentHistory);
      if (errorText) settle(errorText);
    } else {
      const finalText = agent ? agentTexts.join("\n\n") : streamed;
      if (finalText) store.history.push({ role: "assistant", content: finalText });
      else if (errorText) settle(errorText);
    }
    if (!agent && streamed) record({ t: "assistant", text: streamed });
    if (hitLimit) addContinue(hitLimit);
  } catch (err) {
    if (err.name === "AbortError") {
      addNotice("Stopped.");
      record({ t: "notice", text: "Stopped." });
      // Whatever the model managed to say before Stop -- in agent mode that's
      // the round texts, since nothing streams as deltas there.
      const partial = agent ? agentTexts.join("\n\n") : streamed;
      if (partial) {
        store.history.push({ role: "assistant", content: partial });
        if (!agent) record({ t: "assistant", text: partial });
      } else {
        settle("(stopped by the user before I replied)");
      }
    } else {
      addMessage("assistant", err.message, { error: true });
      record({ t: "assistant", text: err.message, error: true });
      settle(err.message);
    }
  } finally {
    if (liveBody) liveBody.classList.remove("cursor");
    inFlight = null;
    setBusy(false);
    input.focus();
    await saveCurrent(sessionState(backend, model, agent, sandbox, groups));
  }
}

export function initChat() {
  const input = $("composer-input");

  messagesEl().innerHTML = EMPTY_STATE;

  $("chat-backend").onchange = () => {
    renderModelSelector();
    prefs.set("backend", $("chat-backend").value);
  };
  $("chat-model").onchange = () => prefs.set("model", $("chat-model").value);

  $("send-btn").onclick = () => {
    if (inFlight) inFlight.abort();
    else sendMessage();
  };

  $("clear-btn").onclick = () => {
    newSession();
    $("composer-input").focus();
  };

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!inFlight) sendMessage();
    }
  });
  input.addEventListener("input", () => autoGrow(input));

  $("agent-mode").checked = prefs.get("agent", false);
  $("sandbox-mode").checked = prefs.get("sandbox", true);
  $("agent-mode").onchange = () => {
    prefs.set("agent", $("agent-mode").checked);
    syncToggles();
    $("chat-hint").textContent = $("agent-mode").checked
      ? "Agent mode needs a model with tool/function calling support."
      : "";
  };
  $("sandbox-mode").onchange = () => {
    prefs.set("sandbox", $("sandbox-mode").checked);
    syncToggles();
  };

  on("backends", renderBackendSelector);
  on("settings", syncToggles);
  on("workdirs", syncToggles);
  on("sessions", renderChatTitle);
  on("session-load", () => {
    replayTranscript();
    syncToggles();
  });
  on("composer-filled", () => autoGrow($("composer-input")));

  renderBackendSelector();
  syncToggles();
}
