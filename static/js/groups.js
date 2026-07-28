// Tool groups: which families of tools a run may call. The server decides
// what's *available* (GitHub needs a token); the boxes decide what's used.

import { api } from "./api.js";
import { $, escapeHtml } from "./ui.js";
import { store, on, emit, prefs } from "./store.js";

const LABELS = {
  shell: ["Shell", "run commands with bash"],
  files: ["Files", "read, write, edit, find and search files"],
  files_write: ["Files: move & delete", "rename, move and delete files — not reversible"],
  git: ["Git", "status, diff, log, branch, commit, push, pull, stash"],
  github: ["GitHub", "browse repos, PRs, issues, branches, CI and code"],
  github_write: ["GitHub write", "issues, comments, pull requests, reviews, merges"],
  web: ["Web", "fetch pages, search, and call APIs"],
  ssh: ["SSH", "run commands and copy files on registered remote hosts"],
};

// Fallback only: the server sends a `reason` per group, which is the thing
// actually shown. Only the GitHub groups can be unavailable at all.
const NEEDS = {
  github: "needs a GitHub token",
  github_write: "needs a GitHub token",
};

/** Groups this session wants, minus anything the server can't offer. */
export function selectedGroups() {
  return store.settings.groups
    .filter((g) => g.available && store.groups.has(g.name))
    .map((g) => g.name);
}

export function maxRounds() {
  return prefs.get("maxRounds", 0) || store.settings.max_agent_iterations;
}

function persist() {
  prefs.set("groups", [...store.groups]);
}

/** Same as ticking the box in the Agent tab, callable from the composer menu. */
export function toggleGroup(name, on) {
  if (on) store.groups.add(name);
  else store.groups.delete(name);
  persist();
  render();
  emit("mode");
}

function renderStatus() {
  const s = store.settings;
  $("agent-status").className = "callout";
  $("agent-status").innerHTML = `<strong>Agent mode runs on the proxy's machine.</strong> Up to
    ${s.max_agent_iterations} tool rounds per message, ${s.tool_timeout}s per command
    (a call can ask for up to ${s.bash_max_timeout}s).
    With the sandbox on, the file tools stay inside the ticked directories; with it off
    they can reach anything the proxy's user can. Either way a shell command can walk
    out on its own — the machine is the real boundary, and the SSH group reaches past
    it entirely.`;
}

function render() {
  renderStatus();

  const list = $("group-list");
  list.innerHTML = store.settings.groups.map((group) => {
    const [label, blurb] = LABELS[group.name] || [group.name, ""];
    const on = group.available && store.groups.has(group.name);
    return `
      <div class="card group-card ${on ? "is-selected" : ""}" data-group="${escapeHtml(group.name)}">
        <div class="row">
          <label class="pick">
            <input type="checkbox" ${on ? "checked" : ""} ${group.available ? "" : "disabled"}>
            <span class="title">${escapeHtml(label)}</span>
          </label>
          <span class="hint">${group.tools.length} tool${group.tools.length === 1 ? "" : "s"}</span>
        </div>
        <div class="sub plain">${escapeHtml(blurb)}</div>
        ${group.available ? "" : `<div class="tags"><span class="tag bad">${escapeHtml(group.reason || NEEDS[group.name] || "unavailable")}</span></div>`}
      </div>`;
  }).join("");

  list.querySelectorAll("input[type=checkbox]").forEach((box) => {
    const name = box.closest(".group-card").dataset.group;
    box.onchange = () => {
      if (box.checked) store.groups.add(name);
      else store.groups.delete(name);
      persist();
      render();
      emit("mode");
    };
  });
}

function renderSearch() {
  const search = store.integrations.search || {};
  const select = $("f-search-provider");
  select.innerHTML = (search.providers || []).map(
    (p) => `<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`).join("");
  if (search.provider) select.value = search.provider;
  $("search-state").textContent = search.configured ? "key saved" : "not set";
}

function renderRounds() {
  const field = $("f-max-rounds");
  field.max = store.settings.round_ceiling || 500;
  field.value = maxRounds();
  $("rounds-hint").textContent =
    `Running out doesn't end the chat — the model reports where it got to and you can continue. Max ${field.max}.`;
}

export function initGroups() {
  const saved = prefs.get("groups", null);
  store.groups = new Set(saved || store.settings.default_groups);

  $("f-max-rounds").onchange = () => {
    const value = parseInt($("f-max-rounds").value, 10);
    const ceiling = store.settings.round_ceiling || 500;
    prefs.set("maxRounds", Math.max(1, Math.min(value || store.settings.max_agent_iterations, ceiling)));
    renderRounds();
  };

  $("search-form").onsubmit = async (e) => {
    e.preventDefault();
    const body = { search_provider: $("f-search-provider").value };
    const key = $("f-search-key").value.trim();
    if (key) body.search_api_key = key;
    try {
      store.integrations = await api("/api/integrations", { method: "PUT", body: JSON.stringify(body) });
      $("f-search-key").value = "";
      $("search-hint").textContent = "Saved.";
      renderSearch();
    } catch (err) {
      $("search-hint").textContent = err.message;
    }
  };

  on("settings", () => {
    // A saved selection can name a group this build no longer has. Dropping
    // those can leave the selection empty, and empty stays empty: unticking
    // everything is a thing you're allowed to mean, and re-ticking the
    // defaults behind your back would hand the model a shell you switched off.
    // Defaults apply once, in initGroups, when nothing was ever saved.
    const known = new Set(store.settings.groups.map((g) => g.name));
    const kept = [...store.groups].filter((g) => known.has(g));
    if (kept.length !== store.groups.size) {
      store.groups = new Set(kept);
      persist();
    }
    render();
    renderSearch();
    renderRounds();
  });

  render();
  renderSearch();
  renderRounds();
}
