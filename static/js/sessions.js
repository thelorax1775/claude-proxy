// Saved chats. The server keeps both the model-facing message list and the
// transcript the UI replays (tool cards included), so reopening a chat shows
// what actually happened rather than just the text.

import { api } from "./api.js";
import { $, escapeHtml, relativeTime, shorten } from "./ui.js";
import { store, emit } from "./store.js";

function render() {
  const list = $("session-list");
  if (store.sessions.length === 0) {
    list.innerHTML = `<p class="hint">No saved chats yet — send a message and one appears here.</p>`;
    return;
  }

  list.innerHTML = store.sessions.map((s) => `
    <div class="card ${s.id === store.sessionId ? "is-selected" : ""}" data-id="${escapeHtml(s.id)}">
      <div class="row">
        <span class="title" data-act="open">${escapeHtml(shorten(s.title, 40))}</span>
        <span class="actions">
          <button class="icon-btn" data-act="rename" title="Rename">&#9998;</button>
          <button class="icon-btn danger" data-act="delete" title="Delete">&times;</button>
        </span>
      </div>
      <div class="sub plain">${s.message_count} message${s.message_count === 1 ? "" : "s"} · ${escapeHtml(relativeTime(s.updated))}${
        s.model ? ` · ${escapeHtml(shorten(s.model, 22))}` : ""}</div>
    </div>`).join("");

  list.querySelectorAll(".card").forEach((card) => {
    const id = card.dataset.id;
    card.querySelector('[data-act="open"]').onclick = () => openSession(id);
    card.querySelector('[data-act="rename"]').onclick = () => renameSession(id);
    card.querySelector('[data-act="delete"]').onclick = () => deleteSession(id);
  });
}

export async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    store.sessions = data.sessions;
  } catch {
    store.sessions = [];
  }
  render();
  emit("sessions");
}

async function openSession(id) {
  try {
    const session = await api(`/api/sessions/${id}`);
    store.sessionId = session.id;
    store.history = session.messages || [];
    store.events = session.events || [];
    emit("session-load");
    render();
  } catch (err) {
    alert(err.message);
  }
}

async function renameSession(id) {
  const current = store.sessions.find((s) => s.id === id);
  const title = prompt("Rename chat", current ? current.title : "");
  if (title === null) return;
  await api(`/api/sessions/${id}`, { method: "PATCH", body: JSON.stringify({ title }) });
  await loadSessions();
}

async function deleteSession(id) {
  if (!confirm("Delete this chat?")) return;
  await api(`/api/sessions/${id}`, { method: "DELETE" });
  if (store.sessionId === id) newSession();
  await loadSessions();
}

export function newSession() {
  store.sessionId = null;
  store.history = [];
  store.events = [];
  emit("session-load");
  render();
}

/** Called before the first message of a chat: creates the server-side row so
 *  everything after it has somewhere to save to. */
export async function ensureSession(firstText, state) {
  if (store.sessionId) return store.sessionId;
  try {
    const session = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ title: titleFrom(firstText), ...state }),
    });
    store.sessionId = session.id;
    await loadSessions();
  } catch { /* an unsaved chat still works; it just won't be in the list */ }
  return store.sessionId;
}

export async function saveCurrent(state) {
  if (!store.sessionId) return;
  try {
    await api(`/api/sessions/${store.sessionId}`, { method: "PATCH", body: JSON.stringify(state) });
    await loadSessions();
  } catch { /* keep chatting even if the save failed */ }
}

function titleFrom(text) {
  const line = (text || "").trim().split("\n")[0].replace(/\s+/g, " ");
  return line.length > 60 ? `${line.slice(0, 60)}…` : line || "New chat";
}

export function initSessions() {
  $("new-chat").onclick = newSession;
  render();
}
