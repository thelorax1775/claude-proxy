// Boot: wire the shell (tabs, panel, status pills), load the server's
// settings, then hand off to the panels.

import { api } from "./api.js";
import { $ } from "./ui.js";
import { store, on, emit, prefs } from "./store.js";
import { initBackends, loadBackends } from "./backends.js";
import { initWorkdirs, loadWorkdirs } from "./workdirs.js";
import { initSshHosts, loadSshHosts } from "./sshhosts.js";
import { initGroups, selectedGroups } from "./groups.js";
import { initGithub, loadRepos } from "./github.js";
import { initSessions, loadSessions } from "./sessions.js";
import { initActivity, loadActivity } from "./activity.js";
import { initChat, agentOn, sandboxOn } from "./chat.js";
import { initCli } from "./cli.js";
import { initComposer, approvalMode } from "./composer.js";

function renderPills() {
  const count = store.backends.length;
  const pills = [`<span class="pill">${count} backend${count === 1 ? "" : "s"}</span>`];

  if (agentOn()) {
    pills.push(`<span class="pill on">agent on</span>`);
    pills.push(sandboxOn()
      ? `<span class="pill">sandboxed</span>`
      : `<span class="pill warn">unsandboxed</span>`);
    const dirs = store.selectedDirs.size;
    pills.push(`<span class="pill">${dirs} director${dirs === 1 ? "y" : "ies"}</span>`);
    pills.push(`<span class="pill">${selectedGroups().length} tool group(s)</span>`);
    pills.push(approvalMode() === "manual"
      ? `<span class="pill on">approvals on</span>`
      : `<span class="pill warn">no approvals</span>`);
  } else {
    pills.push(`<span class="pill off">agent off</span>`);
  }
  if (store.integrations.github && store.integrations.github.configured) {
    pills.push(`<span class="pill on">github</span>`);
  }
  $("status-pills").innerHTML = pills.join("");
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach((p) => p.classList.toggle("is-hidden", p.dataset.panel !== name));
  prefs.set("tab", name);
  if (name === "github") loadRepos();
}

function initShell() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => showTab(tab.dataset.tab);
  });

  const app = $("app");
  const startCollapsed = window.innerWidth <= 860 ? true : prefs.get("navCollapsed", false);
  app.classList.toggle("nav-collapsed", startCollapsed);
  $("nav-toggle").onclick = () => {
    const collapsed = app.classList.toggle("nav-collapsed");
    prefs.set("navCollapsed", collapsed);
  };

  $("dirs-shortcut").onclick = () => {
    app.classList.remove("nav-collapsed");
    showTab("agent");
  };

  showTab(prefs.get("tab", "chats"));
}

async function loadSettings() {
  try {
    const settings = await api("/api/settings");
    store.settings = { ...store.settings, ...settings };
    if (settings.integrations) store.integrations = settings.integrations;
  } catch { /* the panels surface their own load errors */ }
  emit("settings");
  renderPills();
}

async function loadAll() {
  await loadSettings();
  await Promise.all([loadBackends(), loadWorkdirs(), loadSshHosts(), loadSessions()]);
  renderPills();
}

// Panels that need settings loaded first are initialised after loadSettings;
// the rest can wire their DOM immediately.
initShell();
initBackends();
initWorkdirs();
initSshHosts();
initActivity();
initSessions();
initChat();
initCli();
initComposer();

on("backends", renderPills);
on("mode", renderPills);
on("workdirs", renderPills);
on("sessions", renderPills);
on("workdirs-stale", loadWorkdirs);
on("settings-stale", async () => {
  await loadSettings();
  renderPills();
});

loadAll().then(() => {
  initGroups();
  initGithub();
  loadActivity();
  // initShell() picked the remembered tab before any of this existed, so if
  // that tab was GitHub its repo list was skipped -- the token wasn't loaded
  // yet, so it looked disconnected. Now that it is, fill it in.
  if (prefs.get("tab", "chats") === "github") loadRepos();
});
