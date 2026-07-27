// Boot: wire the shell (tabs, panel, proxy key, status pills), load the
// server's settings, then hand off to the panels.

import { api } from "./api.js";
import { $ } from "./ui.js";
import { store, on, emit, prefs } from "./store.js";
import { initBackends, loadBackends } from "./backends.js";
import { initWorkdirs, loadWorkdirs } from "./workdirs.js";
import { initActivity, loadActivity } from "./activity.js";
import { initChat, agentOn, sandboxOn } from "./chat.js";

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
  } else {
    pills.push(`<span class="pill off">agent off</span>`);
  }
  $("status-pills").innerHTML = pills.join("");
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach((p) => p.classList.toggle("is-hidden", p.dataset.panel !== name));
  prefs.set("tab", name);
  if (name === "activity") loadActivity();
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
    $("f-workdir").focus();
  };

  showTab(prefs.get("tab", "backends"));
}

async function loadSettings() {
  try {
    store.settings = { ...store.settings, ...(await api("/api/settings")) };
  } catch { /* the panels surface their own load errors */ }
  emit("settings");
  renderPills();
}

async function loadAll() {
  await loadSettings();
  await Promise.all([loadBackends(), loadWorkdirs()]);
  renderPills();
}

initShell();
initBackends();
initWorkdirs();
initActivity();
initChat();

on("backends", renderPills);
on("mode", renderPills);
on("workdirs", renderPills);
loadAll();
