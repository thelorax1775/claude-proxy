// The two composer menus: the permission-mode dropdown, and the "+" menu for
// uploading files, changing scope, and cloning a repo without leaving the
// chat. Everything here is a shortcut to state that also lives in the side
// panel -- both views read the same store, so neither can drift.

import { api } from "./api.js";
import { $, escapeHtml } from "./ui.js";
import { store, on, emit, prefs } from "./store.js";
import { toggleDir, loadWorkdirs } from "./workdirs.js";
import { toggleGroup } from "./groups.js";

const MODES = {
  manual: ["Manually approve", "each tool call waits for you"],
  auto: ["Automatically approve", "runs as it goes; Stop still works"],
  skip: ["Skip all approvals", "no prompts at all"],
};

export function approvalMode() {
  const mode = prefs.get("approval", "skip");
  return mode in MODES ? mode : "skip";
}

function setMode(mode) {
  prefs.set("approval", mode in MODES ? mode : "skip");
  renderMode();
  emit("mode");
}

function renderMode() {
  const mode = approvalMode();
  const [label] = MODES[mode];
  // The warning glyph only on "skip": it's the one mode where nothing stops a
  // command between the model deciding to run it and it running.
  $("perm-btn").innerHTML =
    `${mode === "skip" ? "&#9888; " : ""}${escapeHtml(label)} <span class="caret">&#9662;</span>`;
  $("perm-btn").classList.toggle("warn", mode === "skip");
  for (const item of $("perm-menu").querySelectorAll("[data-mode]")) {
    item.classList.toggle("is-checked", item.dataset.mode === mode);
  }
}

// ---------------------------------------------------------------------------
// menu open/close
// ---------------------------------------------------------------------------

function closeMenus(except) {
  for (const id of ["perm-menu", "plus-menu"]) {
    if (id !== except) $(id).classList.add("is-hidden");
  }
}

function toggleMenu(id) {
  const menu = $(id);
  const wasOpen = !menu.classList.contains("is-hidden");
  closeMenus();
  if (!wasOpen) {
    menu.classList.remove("is-hidden");
    if (id === "plus-menu") renderPlusMenu();
  }
}

// ---------------------------------------------------------------------------
// "+" menu contents
// ---------------------------------------------------------------------------

function renderPlusMenu() {
  const dirs = $("menu-dirs");
  dirs.innerHTML = store.workdirs.length
    ? store.workdirs.map((d) => `
        <label class="menu-check" title="${escapeHtml(d.path)}">
          <input type="checkbox" data-dir="${escapeHtml(d.path)}"
            ${store.selectedDirs.has(d.path) ? "checked" : ""}>
          <span>${escapeHtml(d.label)}</span>
          ${d.exists ? "" : `<em class="bad">missing</em>`}
        </label>`).join("")
    : `<p class="hint">No directories registered.</p>`;

  const groups = $("menu-groups");
  groups.innerHTML = (store.settings.groups || []).map((g) => `
      <label class="menu-check">
        <input type="checkbox" data-group="${escapeHtml(g.name)}"
          ${store.groups.has(g.name) ? "checked" : ""}>
        <span>${escapeHtml(g.name)}</span>
        <em>${g.tools.length}</em>
      </label>`).join("");

  dirs.querySelectorAll("input[data-dir]").forEach((box) => {
    box.onchange = () => toggleDir(box.dataset.dir, box.checked);
  });
  groups.querySelectorAll("input[data-group]").forEach((box) => {
    box.onchange = () => toggleGroup(box.dataset.group, box.checked);
  });
}

/** Where an upload lands: the first ticked directory, i.e. the same place
 *  bash starts, so "I dropped a file in" and "look in the working directory"
 *  agree with each other. */
function uploadTarget() {
  const selected = store.workdirs.filter((d) => store.selectedDirs.has(d.path));
  return (selected[0] || store.workdirs[0] || {}).path || "";
}

async function uploadFiles(files) {
  const status = $("upload-status");
  const target = uploadTarget();
  if (!target) {
    status.textContent = "No working directory to upload into.";
    return;
  }
  const done = [];
  for (const file of files) {
    status.textContent = `Uploading ${file.name}…`;
    try {
      const resp = await fetch(
        `/api/upload?name=${encodeURIComponent(file.name)}&path=${encodeURIComponent(target)}`,
        { method: "POST", body: file });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      done.push(data.name);
    } catch (err) {
      status.textContent = `${file.name}: ${err.message}`;
      return;
    }
  }
  status.textContent = `Uploaded ${done.join(", ")} to ${target}`;
  // Mention them in the composer rather than sending anything: what to do
  // with the file is still the user's sentence to write.
  const input = $("composer-input");
  const mention = done.map((n) => `${target}/${n}`).join("\n");
  input.value = input.value ? `${input.value}\n${mention}` : `${mention}\n`;
  emit("composer-filled");
  input.focus();
}

async function cloneRepo(event) {
  event.preventDefault();
  const repo = $("menu-clone-repo").value.trim();
  const hint = $("menu-clone-hint");
  if (!repo) return;
  hint.textContent = "Cloning…";
  try {
    const data = await api("/api/github/clone", {
      method: "POST", body: JSON.stringify({ repo }),
    });
    hint.textContent = data.message || "Cloned.";
    $("menu-clone-repo").value = "";
    await loadWorkdirs();
    renderPlusMenu();
  } catch (err) {
    hint.textContent = err.message;
  }
}

// ---------------------------------------------------------------------------

export function initComposer() {
  $("perm-btn").onclick = (e) => { e.stopPropagation(); toggleMenu("perm-menu"); };
  $("plus-btn").onclick = (e) => { e.stopPropagation(); toggleMenu("plus-menu"); };

  for (const item of $("perm-menu").querySelectorAll("[data-mode]")) {
    item.onclick = () => { setMode(item.dataset.mode); closeMenus(); };
  }

  // Clicks inside a menu shouldn't close it -- ticking three boxes in a row is
  // the normal case, not an accident.
  for (const id of ["perm-menu", "plus-menu"]) {
    $(id).onclick = (e) => e.stopPropagation();
  }
  document.addEventListener("click", () => closeMenus());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeMenus();
    if (e.key === "u" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      $("file-input").click();
    }
  });

  $("menu-upload").onclick = () => $("file-input").click();
  $("file-input").onchange = async () => {
    if ($("file-input").files.length) await uploadFiles([...$("file-input").files]);
    $("file-input").value = "";
  };
  $("menu-clone-form").onsubmit = cloneRepo;

  // Drag a file anywhere onto the chat pane and it goes to the same place the
  // + menu would have put it.
  const drop = document.querySelector(".chat");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragging"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    drop.classList.remove("dragging");
    const files = [...(e.dataTransfer?.files || [])];
    if (files.length) {
      $("plus-menu").classList.remove("is-hidden");
      await uploadFiles(files);
    }
  });

  on("workdirs", () => {
    if (!$("plus-menu").classList.contains("is-hidden")) renderPlusMenu();
  });
  on("settings", () => {
    if (!$("plus-menu").classList.contains("is-hidden")) renderPlusMenu();
  });

  renderMode();
}
