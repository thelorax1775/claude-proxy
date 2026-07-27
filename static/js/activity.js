// Activity panel: the proxy's recent backend attempts, so failover is
// something you can watch instead of infer.

import { api } from "./api.js";
import { $, escapeHtml, relativeTime } from "./ui.js";

function outcomeTag(outcome) {
  if (outcome === "ok") return `<span class="tag good">ok</span>`;
  return `<span class="tag bad">${escapeHtml(outcome)}</span>`;
}

export async function loadActivity() {
  const list = $("activity-list");
  try {
    const data = await api("/status");
    const attempts = [...data.recent_attempts].reverse();
    if (attempts.length === 0) {
      list.innerHTML = `<p class="hint">No requests yet.</p>`;
      return;
    }
    list.innerHTML = attempts.map((a) => `
      <div class="card">
        <div class="row">
          <span class="title">${escapeHtml(a.backend)}</span>
          <span class="hint">${escapeHtml(relativeTime(a.time))}</span>
        </div>
        <div class="tags">${outcomeTag(a.outcome)}</div>
      </div>
    `).join("");
  } catch (err) {
    list.innerHTML = `<p class="hint error">Couldn't load activity: ${escapeHtml(err.message)}</p>`;
  }
}

export function initActivity() {
  $("refresh-activity").onclick = loadActivity;
}
