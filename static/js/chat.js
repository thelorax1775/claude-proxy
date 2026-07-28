// The test chat: backend/model pickers, the agent + sandbox toggles, and the
// SSE rendering for plain streaming, mid-conversation failover, and agent
// tool rounds.

import { api, postStream, sseJson } from "./api.js";
import { approvalMode } from "./composer.js";
import { $, escapeHtml, renderRich, shorten, autoGrow } from "./ui.js";
import { store, on, emit, prefs } from "./store.js";
import { selectedPaths } from "./workdirs.js";
import { selectedGroups, maxRounds } from "./groups.js";
import { ensureSession, saveCurrent, newSession } from "./sessions.js";

// Matches main.OVERFLOW: the server says which failures are "too long" rather
// than "broken", because only that one has an action attached.
const OVERFLOW = "context_overflow";

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

/** The card a manual-mode run parks on. The run is blocked on the server
 *  while this is on screen, so it says which tool and shows the exact
 *  arguments -- approving "bash" without seeing the command would be theatre. */
function addApproval(id, name, args, group) {
  clearEmptyState();
  const risky = ["shell", "files_write", "github_write", "ssh"].includes(group);
  const card = document.createElement("div");
  card.className = `msg approval${risky ? " risky" : ""}`;
  card.dataset.approvalId = id;
  card.innerHTML = `
    <div class="body">
      <div class="approval-head">
        <span class="tool-name">${escapeHtml(name)}</span>
        <span class="tag ${risky ? "bad" : ""}">${escapeHtml(group)}</span>
      </div>
      <pre class="code-block"></pre>
      <div class="button-row approval-actions">
        <button class="primary-btn" data-decision="approve">Approve</button>
        <button class="ghost-btn" data-decision="deny">Deny</button>
        <button class="ghost-btn" data-decision="approve_all">Approve all remaining</button>
      </div>
      <p class="hint approval-note"></p>
    </div>`;
  card.querySelector("pre").textContent = JSON.stringify(args, null, 2);

  card.querySelectorAll("[data-decision]").forEach((button) => {
    button.onclick = async () => {
      card.querySelectorAll("[data-decision]").forEach((b) => { b.disabled = true; });
      try {
        await api(`/api/approvals/${encodeURIComponent(id)}`, {
          method: "POST",
          body: JSON.stringify({ decision: button.dataset.decision }),
        });
      } catch (err) {
        // Re-enable: the usual cause is a stale card whose run already moved
        // on, but if it was a transient failure the buttons should still work.
        card.querySelector(".approval-note").textContent = err.message;
        card.querySelectorAll("[data-decision]").forEach((b) => { b.disabled = false; });
      }
    };
  });

  messagesEl().appendChild(card);
  scrollDown(true);
  return card;
}

/** Replace the buttons with what was decided, so a scrolled-back transcript
 *  reads as a record rather than a set of live controls. */
function finishApproval(card, decision) {
  if (!card) return;
  const wording = {
    approve: "approved",
    approve_all: "approved — and the rest of this message runs without asking",
    deny: "denied",
    timeout: "no answer in time, so it was not run",
  }[decision] || decision;
  card.classList.add("is-resolved");
  card.classList.toggle("denied", decision === "deny" || decision === "timeout");
  const actions = card.querySelector(".approval-actions");
  if (actions) actions.remove();
  card.querySelector(".approval-note").textContent = wording;
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

// The checklist is a single panel pinned above the composer, not a card in the
// transcript. write_todos replaces its whole list on every call, so a
// transcript entry per update would be eight near-identical lists, and the one
// you want -- the current one -- would be the hardest to find.
function renderTasks(items) {
  const panel = $("agent-tasks");
  if (!panel) return;
  if (!items || !items.length) {
    panel.classList.add("is-hidden");
    panel.innerHTML = "";
    return;
  }
  const done = items.filter((i) => i.status === "completed").length;
  panel.classList.remove("is-hidden");
  panel.innerHTML = `
    <div class="task-head">Plan <span class="hint">${done}/${items.length} done</span></div>
    <ul class="task-list">
      ${items.map((i) => `
        <li class="task ${escapeHtml(i.status)}">
          <span class="task-mark" aria-hidden="true"></span>
          <span class="task-text">${escapeHtml(i.content)}</span>
        </li>`).join("")}
    </ul>`;
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

function sessionState(backend, model, agent, sandbox, groups, lan, approval) {
  return {
    backend, model, agent, sandbox, groups, lan, approval,
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
    else if (event.t === "approval") {
      finishApproval(addApproval(event.id || "", event.name, event.args || {}, event.group || ""),
                     event.decision);
    }
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

export function lanOn() {
  return $("lan-mode").checked;
}

function syncToggles() {
  const sandboxToggle = $("sandbox-mode");
  const sandboxRow = $("sandbox-row");
  const lanToggle = $("lan-mode");
  const lanRow = $("lan-row");
  const on = agentOn();
  // The server decides whether the LAN switch can do anything at all; with
  // AGENT_WEB_ALLOW_PRIVATE unset it stays locked rather than pretending.
  const lanAvailable = store.settings.lan_available !== false;

  // The sandbox switch only means anything while agent mode is on.
  sandboxToggle.disabled = !on;
  sandboxRow.classList.toggle("is-locked", !on);
  sandboxRow.classList.toggle("warn", on && !sandboxToggle.checked);
  $("sandbox-note").textContent = sandboxToggle.checked
    ? "stay inside the selected directories"
    : "file tools may go anywhere on that machine";

  // Same rule as sandbox, plus the operator's ceiling. A locked switch is
  // forced off so what's on screen matches what the run will actually get.
  if (!lanAvailable) lanToggle.checked = false;
  lanToggle.disabled = !on || !lanAvailable;
  lanRow.classList.toggle("is-locked", !on || !lanAvailable);
  lanRow.classList.toggle("warn", on && lanAvailable && lanToggle.checked);
  $("lan-note").textContent = !lanAvailable
    ? "blocked on the proxy (AGENT_WEB_ALLOW_PRIVATE)"
    : lanToggle.checked
      ? "web_fetch may reach this machine's own network"
      : "public internet only";

  const chip = $("mode-chip");
  chip.classList.toggle("is-hidden", !on);
  if (on) {
    const dirs = selectedPaths();
    const primary = dirs[0] ? dirs[0].split("/").filter(Boolean).pop() || dirs[0] : "workspace";
    const extra = dirs.length > 1 ? ` +${dirs.length - 1}` : "";
    chip.classList.toggle("unsandboxed", !sandboxToggle.checked);
    const lanBit = lanToggle.checked && lanAvailable ? " · LAN" : "";
    chip.textContent = sandboxToggle.checked
      ? `agent · sandboxed · ${primary}${extra}${lanBit}`
      : `agent · UNSANDBOXED · ${primary}${extra}${lanBit}`;
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

/** Every index in the history where a fresh exchange begins.
 *
 *  A role:"user" message is always a turn boundary -- tool results are
 *  role:"tool", never "user" -- so cutting immediately before one can never
 *  strand a tool result whose tool_call the assistant message announced. That
 *  pairing is exactly what a naive "drop the first N messages" would break,
 *  and providers reject the result. */
function turnStarts(history) {
  return history.reduce((acc, m, i) => (m.role === "user" ? acc.concat(i) : acc), []);
}

/** Drop roughly the older half of the conversation, on a turn boundary.
 *  Returns the number of messages dropped, or 0 if there was nothing safe to
 *  drop (a single exchange that doesn't fit can only be started over). */
function shortenHistory() {
  const starts = turnStarts(store.history);
  if (starts.length <= 1) return 0;
  // Never drop the final turn: that's the message the user is waiting on.
  const cut = starts[Math.min(Math.max(1, Math.floor(starts.length / 2)), starts.length - 1)];
  store.history.splice(0, cut);
  return cut;
}

/** Shown when no backend would take the conversation. This is the one error
 *  the user can actually act on, and the action is mechanical, so offer to do
 *  it rather than describing it. */
function addOverflowNotice(message, retry) {
  clearEmptyState();
  const droppable = turnStarts(store.history).length > 1;
  const wrap = document.createElement("div");
  wrap.className = "msg notice continue-notice";
  wrap.innerHTML = `<div class="body"></div>`;
  const body = wrap.querySelector(".body");
  body.textContent = `${message} `;
  if (droppable) {
    const button = document.createElement("button");
    button.className = "ghost-btn small";
    button.textContent = "Drop the older half and retry";
    button.onclick = () => {
      const dropped = shortenHistory();
      wrap.remove();
      addNotice(`Dropped the oldest ${dropped} message(s) from what the model sees. `
                + `They're still in the transcript above.`);
      retry();
    };
    body.appendChild(button);
  } else {
    body.textContent += "This single exchange is too big on its own, so there is "
      + "nothing older to drop — start a new chat.";
  }
  messagesEl().appendChild(wrap);
  scrollDown();
}

async function sendMessage({ resend = false } = {}) {
  const input = $("composer-input");
  const text = resend ? "" : input.value.trim();
  if (!resend && !text) return;

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
  const lan = lanOn();
  const approval = approvalMode();
  const groups = selectedGroups();

  // A resend replays the history as it stands -- the user's message is
  // already in it, and re-adding it would both duplicate the bubble and make
  // the conversation longer, which is the opposite of what a retry after an
  // overflow needs to do.
  if (!resend) {
    input.value = "";
    autoGrow(input);
    store.history.push({ role: "user", content: text });
    record({ t: "user", text });
    addMessage("user", text);
  }
  await ensureSession(text || "(retry)",
                      sessionState(backend, model, agent, sandbox, groups, lan, approval));

  // Plain chat streams tokens into one bubble; agent mode emits whole
  // rounds, so its bubbles are created as events arrive.
  let liveBody = agent ? null : addMessage("assistant", "", { streaming: true });
  const toolCards = new Map();
  const approvalCards = new Map();
  const agentTexts = [];
  let streamed = "";
  let errorText = "";
  let overflow = "";
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
      lan,
      approval,
      groups,
      max_rounds: maxRounds(),
      workdirs: selectedPaths(),
    }, inFlight.signal);

    if (!resp.ok || !resp.body) {
      const data = await resp.json().catch(() => ({}));
      const message = data.detail || data.error || `HTTP ${resp.status}`;
      if (liveBody) liveBody.closest(".msg").remove();
      if (data.error_kind === OVERFLOW) {
        // Deliberately no settle(): recording the failure as an assistant turn
        // would make the conversation longer, which is what just failed.
        record({ t: "notice", text: message });
        addOverflowNotice(message, () => sendMessage({ resend: true }));
        return;
      }
      addMessage("assistant", message, { error: true });
      record({ t: "assistant", text: message, error: true });
      settle(message);
      return;
    }

    for await (const chunk of sseJson(resp)) {
      if (chunk.agent_context) {
        // lan here is what the server granted after clamping, which is not
        // necessarily what the switch asked for -- show that, not the switch.
        const { roots, sandbox: sandboxed, lan: lanGranted } = chunk.agent_context;
        // A clone during the run adds a directory -- refresh the panel so the
        // scope shown on screen matches what the model is actually using.
        if (chunk.workdirs_changed) emit("workdirs-stale");
        const key = JSON.stringify([roots, sandboxed, lanGranted]);
        if (key !== lastContextKey) {
          lastContextKey = key;
          const notice = `${sandboxed ? "Sandboxed" : "UNSANDBOXED"} agent run · ${roots.join(" · ")}`
            + (lanGranted ? " · LAN access on" : "");
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

      if (chunk.agent_approval_request) {
        // The run is blocked server-side until one of these buttons is
        // pressed, so this card is the whole reason the stream has gone quiet.
        const { id, name, arguments: args, group } = chunk.agent_approval_request;
        approvalCards.set(id, {
          card: addApproval(id, name, args || {}, group || ""),
          name, args: args || {}, group: group || "",
        });
        continue;
      }

      if (chunk.agent_approval_resolved) {
        const { id, decision } = chunk.agent_approval_resolved;
        const entry = approvalCards.get(id);
        finishApproval(entry && entry.card, decision);
        if (entry) {
          record({ t: "approval", id, name: entry.name, args: entry.args,
                   group: entry.group, decision });
        }
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

      if (chunk.agent_tasks) {
        // One panel that changes, rather than eight near-identical checklists
        // down the transcript: write_todos replaces its whole list each call.
        renderTasks(chunk.agent_tasks);
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

      if (chunk.agent_done) {
        // The summary is the model's account of what it did; this is the
        // list's. When they disagree, the list is the one that was written
        // down before the work rather than after it.
        if (chunk.unfinished && chunk.unfinished.length) {
          addMessage("assistant",
            `⚠ The run ended with ${chunk.unfinished.length} task(s) still open:\n` +
            chunk.unfinished.map((t) => `• ${t}`).join("\n"));
        }
        continue;
      }

      if (chunk.error) {
        if (chunk.error_kind === OVERFLOW) {
          overflow = chunk.error;
          continue;
        }
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
    if (overflow) {
      // No settle() -- see the same note on the non-streaming path. Whatever
      // tool work the run got through stays in the history above; only the
      // failure itself is left out, because recording it would grow the
      // conversation that just proved too big.
      if (liveBody && !streamed) liveBody.closest(".msg").remove();
      record({ t: "notice", text: overflow });
      addOverflowNotice(overflow, () => sendMessage({ resend: true }));
    }
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
    await saveCurrent(sessionState(backend, model, agent, sandbox, groups, lan, approval));
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
  // Defaults to off: reaching the local network is the sort of thing you opt
  // into per session, not something a remembered tick silently leaves on.
  $("lan-mode").checked = prefs.get("lan", false);
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
  $("lan-mode").onchange = () => {
    prefs.set("lan", $("lan-mode").checked);
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
