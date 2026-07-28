// SSH hosts: the machines agent mode's ssh tools may connect to.
//
// The server owns this list and the tools take a name from it, so this panel
// is the allowlist. There is no per-session ticking the way working
// directories have -- the SSH tool group's checkbox is the on/off switch.

import { api } from "./api.js";
import { $, escapeHtml } from "./ui.js";
import { store, emit } from "./store.js";

function render() {
  const list = $("ssh-host-list");
  if (!store.sshAvailable) {
    list.innerHTML = `<p class="hint">The <code>ssh</code> command isn't installed on the proxy's
      machine, so the SSH tools can't run. Install an OpenSSH client there first.</p>`;
    return;
  }
  if (store.sshHosts.length === 0) {
    list.innerHTML = `<p class="hint">No hosts yet. Add one below to make the SSH tool group
      available — the model can only reach hosts on this list.</p>`;
    return;
  }

  list.innerHTML = store.sshHosts.map((h) => `
    <div class="card" data-name="${escapeHtml(h.name)}">
      <div class="row">
        <span class="title">${escapeHtml(h.name)}</span>
        <span class="actions">
          <button class="icon-btn" data-act="password"
                  title="${h.has_password ? "Change or remove the stored password" : "Store a password for this host"}">&#128273;</button>
          <button class="icon-btn" data-act="test" title="Connect and run a no-op">&#10003;</button>
          <button class="icon-btn danger" data-act="remove" title="Remove">&times;</button>
        </span>
      </div>
      <div class="sub">${escapeHtml(h.user ? `${h.user}@${h.host}` : h.host)}${h.port === 22 ? "" : `:${h.port}`}</div>
      <div class="tags">
        ${h.label && h.label !== h.name ? `<span class="tag">${escapeHtml(h.label)}</span>` : ""}
        ${h.has_password
          ? `<span class="tag">password</span>`
          : h.key_path
            ? `<span class="tag ${h.key_exists ? "" : "bad"}">${h.key_exists ? "key file" : "key file missing"}</span>`
            : `<span class="tag">default keys</span>`}
      </div>
      <form class="form password-form is-hidden" data-password-form>
        <label>Password for ${escapeHtml(h.user || "this host")}
          <input type="password" data-password autocomplete="new-password"
                 placeholder="${h.has_password ? "stored — type to replace" : "typed here, never in chat"}">
        </label>
        <span class="actions">
          <button type="submit" class="primary-btn">Save</button>
          ${h.has_password ? `<button type="button" class="ghost-btn" data-act="clear-password">Remove</button>` : ""}
        </span>
        <p class="hint">Stored in plain text in <code>ssh_hosts.json</code> (0600).
          It never reaches the model or a chat message.</p>
      </form>
      <p class="hint" data-test-result></p>
    </div>
  `).join("");

  list.querySelectorAll(".card").forEach((card) => {
    const name = card.dataset.name;
    const form = card.querySelector("[data-password-form]");

    card.querySelectorAll("button[data-act]").forEach((btn) => {
      btn.onclick = () => {
        if (btn.dataset.act === "remove") return removeHost(name);
        if (btn.dataset.act === "test") return testHost(name, card);
        if (btn.dataset.act === "clear-password") return savePassword(name, "", card);
        form.classList.toggle("is-hidden");
        if (!form.classList.contains("is-hidden")) form.querySelector("[data-password]").focus();
      };
    });

    form.onsubmit = (e) => {
      e.preventDefault();
      savePassword(name, form.querySelector("[data-password]").value, card);
    };
  });
}

// The password goes browser -> server -> ssh_hosts.json and stops there. It is
// never put in the store, never re-rendered into a card, and the field is
// cleared by the reload that follows.
async function savePassword(name, password, card) {
  const out = card.querySelector("[data-test-result]");
  out.classList.remove("error");
  out.textContent = password ? "Saving…" : "Removing…";
  try {
    await api(`/api/ssh-hosts/${encodeURIComponent(name)}/password`, {
      method: "POST",
      body: JSON.stringify({ password }),
    });
    await loadSshHosts();
  } catch (err) {
    out.textContent = err.message;
    out.classList.add("error");
  }
}

async function testHost(name, card) {
  const out = card.querySelector("[data-test-result]");
  out.textContent = "Connecting…";
  out.classList.remove("error");
  try {
    const data = await api(`/api/ssh-hosts/${encodeURIComponent(name)}/test`, { method: "POST" });
    out.textContent = data.ok
      ? `Connected in ${data.latency_ms}ms.`
      : `Couldn't connect: ${data.output}`;
    out.classList.toggle("error", !data.ok);
  } catch (err) {
    out.textContent = err.message;
    out.classList.add("error");
  }
}

async function removeHost(name) {
  if (!confirm(`Stop offering SSH host "${name}" to agent mode?`)) return;
  try {
    await api(`/api/ssh-hosts?name=${encodeURIComponent(name)}`, { method: "DELETE" });
    await loadSshHosts();
    // Removing the last host takes the whole group away, so the tool-group
    // list has to be told.
    emit("settings-stale");
  } catch (err) {
    $("ssh-error").textContent = err.message;
  }
}

export async function loadSshHosts() {
  try {
    const data = await api("/api/ssh-hosts");
    store.sshHosts = data.hosts;
    store.sshAvailable = data.ssh_available;
    render();
  } catch (err) {
    store.sshHosts = [];
    $("ssh-host-list").innerHTML =
      `<p class="hint error">Couldn't load SSH hosts: ${escapeHtml(err.message)}</p>`;
  }
  emit("sshHosts");
}

export function initSshHosts() {
  $("add-ssh-form").onsubmit = async (e) => {
    e.preventDefault();
    $("ssh-error").textContent = "";
    try {
      await api("/api/ssh-hosts", {
        method: "POST",
        body: JSON.stringify({
          name: $("f-ssh-name").value.trim(),
          host: $("f-ssh-host").value.trim(),
          user: $("f-ssh-user").value.trim(),
          port: Number($("f-ssh-port").value) || 22,
          key_path: $("f-ssh-key").value.trim(),
          // Not trimmed: leading and trailing spaces can be part of a
          // password, and a form that quietly ate one would fail as a wrong
          // password with nothing to see.
          password: $("f-ssh-password").value,
        }),
      });
      e.target.reset();
      await loadSshHosts();
      // The first host makes the SSH group usable.
      emit("settings-stale");
    } catch (err) {
      $("ssh-error").textContent = err.message;
    }
  };
}
