// GitHub panel: connect a token, browse repositories, clone one into a
// working directory, and look at open PRs/issues without leaving the app.

import { api } from "./api.js";
import { $, escapeHtml, shorten } from "./ui.js";
import { store, emit } from "./store.js";

let repos = [];
let openRepo = "";
// Why the stored token didn't check out, if it didn't. Kept next to the token
// rather than in the general error line, because it's the status box's story.
let tokenError = "";

function connected() {
  return Boolean(store.integrations.github && store.integrations.github.configured);
}

function renderStatus() {
  const box = $("github-status");
  if (!connected()) {
    box.className = "callout";
    box.innerHTML = `<strong>Not connected.</strong> Paste a personal access token with
      <code>repo</code> scope below. It's stored on the proxy's machine, never sent to a
      model, and never shown back here.`;
    return;
  }
  if (tokenError) {
    // Say what's wrong here rather than leaving "checking…" on screen with an
    // error stranded somewhere else.
    box.className = "callout error";
    box.innerHTML = `<strong>That token didn't work.</strong> ${escapeHtml(tokenError)}
      <br>Paste a fresh one below, or <em>Disconnect</em> to clear it.`;
    return;
  }

  const who = store.githubUser;
  if (!who) {
    box.className = "callout ok";
    box.innerHTML = `<strong>Token saved.</strong> Checking who it belongs to…`;
    return;
  }

  // A classic token reports its scopes; a fine-grained one doesn't (it uses
  // per-repository permissions instead), so an absent list means "not
  // applicable" rather than "none". Showing the gap here beats discovering it
  // when a clone of a private repo mysteriously 404s.
  const scopes = who.scopes || [];
  const missingRepo = who.scopes_reported && !scopes.includes("repo");
  box.className = `callout ${missingRepo ? "warn" : "ok"}`;
  const scopeLine = who.scopes_reported
    ? `Scopes: <code>${scopes.length ? escapeHtml(scopes.join(", ")) : "none"}</code>.`
    : `Fine-grained token — access is set per repository on GitHub.`;
  box.innerHTML = `<strong>Connected as ${escapeHtml(who.login)}.</strong> ${scopeLine}
    ${missingRepo
      ? `<br>Without the <code>repo</code> scope this can only see public repositories —
         cloning or reading a private one will fail.`
      : `The agent's GitHub tools are available; tick <em>GitHub write</em> in the Agent tab
         to let it open issues and PRs.`}`;
}

function renderRepos() {
  const list = $("repo-list");
  if (!connected()) {
    list.innerHTML = `<p class="hint">Connect a token to list repositories.</p>`;
    return;
  }
  const filter = $("repo-filter").value.trim().toLowerCase();
  const shown = repos.filter((r) => !filter || r.full_name.toLowerCase().includes(filter));
  if (shown.length === 0) {
    list.innerHTML = `<p class="hint">${repos.length ? "Nothing matches that filter." : "No repositories loaded yet."}</p>`;
    return;
  }

  list.innerHTML = shown.slice(0, 60).map((r) => `
    <div class="card" data-repo="${escapeHtml(r.full_name)}">
      <div class="row">
        <span class="title">${escapeHtml(r.full_name)}</span>
        <span class="actions">
          <button class="icon-btn" data-act="clone" title="Clone onto the proxy and add as a working directory">&darr;</button>
          <button class="icon-btn" data-act="open" title="Show pull requests and issues">&rsaquo;</button>
        </span>
      </div>
      ${r.description ? `<div class="sub plain">${escapeHtml(shorten(r.description, 90))}</div>` : ""}
      <div class="tags">
        ${r.private ? `<span class="tag">private</span>` : ""}
        ${r.default_branch ? `<span class="tag">${escapeHtml(r.default_branch)}</span>` : ""}
      </div>
    </div>`).join("");

  list.querySelectorAll("button[data-act]").forEach((btn) => {
    const repo = btn.closest(".card").dataset.repo;
    btn.onclick = () => (btn.dataset.act === "clone" ? cloneRepo(repo, btn) : showRepo(repo));
  });
}

async function cloneRepo(repo, btn) {
  btn.disabled = true;
  const previous = btn.textContent;
  btn.textContent = "…";
  try {
    await api("/api/github/clone", { method: "POST", body: JSON.stringify({ repo }) });
    emit("workdirs-stale");
  } catch (err) {
    alert(`Clone failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = previous;
  }
}

function askAbout(text) {
  const input = $("composer-input");
  input.value = text;
  input.focus();
  emit("composer-filled");
}

async function showRepo(repo) {
  openRepo = repo;
  $("repo-detail-head").classList.remove("is-hidden");
  $("repo-detail-title").textContent = repo;
  const detail = $("repo-detail");
  detail.innerHTML = `<p class="hint">Loading…</p>`;
  try {
    const [pulls, issues] = await Promise.all([
      api(`/api/github/pulls?repo=${encodeURIComponent(repo)}`),
      api(`/api/github/issues?repo=${encodeURIComponent(repo)}`),
    ]);
    detail.innerHTML = `
      ${section("Open pull requests", pulls.pulls.map((p) => ({
        key: `pr-${p.number}`,
        title: `#${p.number} ${p.title}`,
        sub: `${p.user} · ${p.head} → ${p.base}${p.draft ? " · draft" : ""}`,
        ask: `Review pull request #${p.number} in ${repo}: read the diff and tell me what it changes and anything that looks wrong.`,
        url: p.url,
      })))}
      ${section("Open issues", issues.issues.map((i) => ({
        key: `issue-${i.number}`,
        title: `#${i.number} ${i.title}`,
        sub: `${i.user} · ${i.comments} comment(s)`,
        ask: `Read issue #${i.number} in ${repo} and summarise what's being asked for.`,
        url: i.url,
      })))}`;

    detail.querySelectorAll("button[data-ask]").forEach((btn) => {
      btn.onclick = () => askAbout(btn.dataset.ask);
    });
  } catch (err) {
    detail.innerHTML = `<p class="hint error">${escapeHtml(err.message)}</p>`;
  }
}

function section(heading, items) {
  if (items.length === 0) {
    return `<div class="section-head"><h2>${escapeHtml(heading)}</h2></div><p class="hint">None.</p>`;
  }
  return `<div class="section-head"><h2>${escapeHtml(heading)}</h2></div>` + items.map((item) => `
    <div class="card">
      <div class="row">
        <span class="title">${escapeHtml(shorten(item.title, 60))}</span>
        <span class="actions">
          <button class="icon-btn" data-ask="${escapeHtml(item.ask)}" title="Ask the model about this">&rarr;</button>
          ${item.url ? `<a class="icon-btn" href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer" title="Open on GitHub">&#8599;</a>` : ""}
        </span>
      </div>
      <div class="sub plain">${escapeHtml(item.sub)}</div>
    </div>`).join("");
}

export async function loadRepos() {
  if (!connected()) {
    renderStatus();
    renderRepos();
    return;
  }
  // Settled independently, not Promise.all: these answer two different
  // questions, and one failing used to bury the other's answer. The repo list
  // rejecting skipped the `store.githubUser = me` assignment entirely, so the
  // status box sat on "Checking who it belongs to…" while an unrelated error
  // printed underneath -- which reads exactly like the token check hanging.
  const [who, list] = await Promise.allSettled([
    api("/api/integrations/github/test", { method: "POST" }),
    api("/api/github/repos?limit=50"),
  ]);

  store.githubUser = who.status === "fulfilled" ? who.value : null;
  tokenError = who.status === "rejected" ? who.reason.message : "";

  repos = list.status === "fulfilled" ? list.value.repos : [];
  // Only worth saying separately when the token itself is fine; otherwise this
  // is just an echo of the token error, which the status box already explains.
  $("github-error").textContent =
    list.status === "rejected" && !tokenError ? list.reason.message : "";

  renderStatus();
  renderRepos();
}

export function initGithub() {
  $("github-form").onsubmit = async (e) => {
    e.preventDefault();
    $("github-error").textContent = "";
    const token = $("f-github-token").value.trim();
    if (!token) return;
    try {
      store.integrations = await api("/api/integrations", {
        method: "PUT", body: JSON.stringify({ github_token: token }),
      });
      $("f-github-token").value = "";
      await loadRepos();
      emit("settings-stale");
    } catch (err) {
      $("github-error").textContent = err.message;
    }
  };

  $("github-disconnect").onclick = async () => {
    if (!confirm("Remove the stored GitHub token?")) return;
    store.integrations = await api("/api/integrations", {
      method: "PUT", body: JSON.stringify({ github_token: "" }),
    });
    store.githubUser = null;
    repos = [];
    $("repo-detail").innerHTML = "";
    $("repo-detail-head").classList.add("is-hidden");
    renderStatus();
    renderRepos();
    emit("settings-stale");
  };

  $("refresh-repos").onclick = loadRepos;
  $("repo-filter").oninput = renderRepos;
  $("repo-back").onclick = () => {
    openRepo = "";
    $("repo-detail").innerHTML = "";
    $("repo-detail-head").classList.add("is-hidden");
  };

  renderStatus();
  renderRepos();
}
