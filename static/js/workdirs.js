// Working directories: which places on the proxy's machine agent mode may
// touch. The server owns the registered list; ticking a box only decides
// which of those this browser session puts in scope for its next message.

import { api } from "./api.js";
import { $, escapeHtml } from "./ui.js";
import { store, emit, prefs } from "./store.js";

/** Selected paths in registration order -- the first is the one bash starts in. */
export function selectedPaths() {
  return store.workdirs.map((d) => d.path).filter((p) => store.selectedDirs.has(p));
}

function persistSelection() {
  prefs.set("selectedDirs", [...store.selectedDirs]);
}

function render() {
  const list = $("workdir-list");
  if (store.workdirs.length === 0) {
    list.innerHTML = `<p class="hint">No directories registered.</p>`;
    return;
  }

  const primary = selectedPaths()[0];
  list.innerHTML = store.workdirs.map((d) => {
    const checked = store.selectedDirs.has(d.path);
    return `
      <div class="card ${checked ? "is-selected" : ""} ${d.exists ? "" : "is-missing"}" data-path="${escapeHtml(d.path)}">
        <div class="row">
          <label class="pick">
            <input type="checkbox" data-act="pick" ${checked ? "checked" : ""}>
            <span class="title">${escapeHtml(d.label)}</span>
          </label>
          <span class="actions">
            <button class="icon-btn danger" data-act="remove" ${d.default ? "disabled" : ""}
              title="${d.default ? "The default workspace can't be removed" : "Remove from the list"}">&times;</button>
          </span>
        </div>
        <div class="sub">${escapeHtml(d.path)}</div>
        <div class="tags">
          ${d.path === primary ? `<span class="tag accent">primary</span>` : ""}
          ${d.default ? `<span class="tag">default workspace</span>` : ""}
          ${d.exists ? "" : `<span class="tag bad">missing</span>`}
          ${d.exists && !d.writable ? `<span class="tag bad">read-only</span>` : ""}
        </div>
      </div>
    `;
  }).join("");

  list.querySelectorAll("[data-act]").forEach((el) => {
    const path = el.closest(".card").dataset.path;
    if (el.dataset.act === "pick") {
      el.onchange = () => {
        if (el.checked) store.selectedDirs.add(path);
        else store.selectedDirs.delete(path);
        persistSelection();
        render();
        emit("workdirs");
      };
    } else {
      el.onclick = () => removeWorkdir(path);
    }
  });
}

async function removeWorkdir(path) {
  if (!confirm(`Stop offering "${path}" to agent mode?`)) return;
  try {
    await api(`/api/workdirs?path=${encodeURIComponent(path)}`, { method: "DELETE" });
    store.selectedDirs.delete(path);
    persistSelection();
    await loadWorkdirs();
  } catch (err) {
    $("workdir-error").textContent = err.message;
  }
}

export async function loadWorkdirs() {
  try {
    const data = await api("/api/workdirs");
    store.workdirs = data.workdirs;

    const known = new Set(store.workdirs.map((d) => d.path));
    const saved = prefs.get("selectedDirs", null);
    const base = saved === null ? [store.settings.default_workspace] : saved;
    store.selectedDirs = new Set(base.filter((p) => known.has(p)));
    if (store.selectedDirs.size === 0 && store.workdirs.length) {
      // Never leave the session with nothing in scope -- the server would
      // silently fall back to the workspace anyway, so make that visible.
      store.selectedDirs.add(store.workdirs[0].path);
    }
    persistSelection();
    render();
  } catch (err) {
    store.workdirs = [];
    $("workdir-list").innerHTML = `<p class="hint error">Couldn't load directories: ${escapeHtml(err.message)}</p>`;
  }
  emit("workdirs");
}

export function initWorkdirs() {
  $("add-workdir-form").onsubmit = async (e) => {
    e.preventDefault();
    $("workdir-error").textContent = "";
    const path = $("f-workdir").value.trim();
    if (!path) return;
    try {
      const added = await api("/api/workdirs", {
        method: "POST",
        body: JSON.stringify({ path, label: $("f-workdir-label").value.trim() }),
      });
      store.selectedDirs.add(added.path);
      persistSelection();
      e.target.reset();
      await loadWorkdirs();
    } catch (err) {
      $("workdir-error").textContent = err.message;
    }
  };
}
