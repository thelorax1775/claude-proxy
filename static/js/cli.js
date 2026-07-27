// The Claude Code tab: generates the one-time settings.json block that points
// the CLI at this proxy.
//
// The base URL defaults to whatever address this page was opened on, which is
// by definition an address that reaches the proxy. That beats asking the
// server, which usually only knows PROXY_HOST -- and "0.0.0.0" is not a URL
// anyone can use. It stays editable because the machine running `claude` isn't
// necessarily the machine running this browser.

import { $ } from "./ui.js";
import { prefs } from "./store.js";

// The proxy has no auth, so this value is never checked. Claude Code just
// refuses to start without something set, which is the only reason it exists.
const PLACEHOLDER_KEY = "local-claude-proxy";

function baseUrl() {
  const raw = ($("cli-base-url").value || "").trim().replace(/\/+$/, "");
  return raw || window.location.origin;
}

function settingsJson(url) {
  return JSON.stringify({
    $schema: "https://json.schemastore.org/claude-code-settings.json",
    env: {
      ANTHROPIC_BASE_URL: url,
      ANTHROPIC_API_KEY: PLACEHOLDER_KEY,
    },
  }, null, 2);
}

function render() {
  const url = baseUrl();
  $("cli-snippet").textContent = settingsJson(url);
  $("cli-bash").textContent =
    `export ANTHROPIC_BASE_URL=${url}\nexport ANTHROPIC_API_KEY=${PLACEHOLDER_KEY}\nclaude`;
  $("cli-pwsh").textContent =
    `$env:ANTHROPIC_BASE_URL = "${url}"\n$env:ANTHROPIC_API_KEY = "${PLACEHOLDER_KEY}"\nclaude`;
  $("cli-verify").textContent = `curl ${url}/health`;

  // Reaching the proxy over a non-loopback address only works if it's bound
  // to one, so say that here rather than letting the CLI fail with a refused
  // connection and no explanation.
  const remote = !/^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])(:|$)/.test(url);
  $("cli-remote-note").classList.toggle("is-hidden", !remote);
}

/** Clipboard access needs a secure context, and this UI is typically served
 *  over plain http on a LAN or Tailscale address -- so navigator.clipboard is
 *  usually unavailable exactly where it's most wanted. Fall back to the old
 *  execCommand path, and to selecting the text if even that is gone, so the
 *  button never just silently does nothing. */
async function copyText(source, button) {
  const text = source.textContent;
  const done = (ok) => {
    const original = button.dataset.label || button.textContent;
    button.dataset.label = original;
    button.textContent = ok ? "Copied" : "Press Ctrl+C";
    setTimeout(() => { button.textContent = button.dataset.label; }, 1600);
  };

  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return done(true);
    }
  } catch { /* fall through to execCommand */ }

  const scratch = document.createElement("textarea");
  scratch.value = text;
  // Off-screen rather than hidden: a display:none element can't be selected.
  scratch.style.position = "fixed";
  scratch.style.opacity = "0";
  document.body.appendChild(scratch);
  scratch.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch { ok = false; }
  document.body.removeChild(scratch);

  if (ok) return done(true);

  // Last resort: put it on screen selected, so Ctrl+C works.
  const range = document.createRange();
  range.selectNodeContents(source);
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
  done(false);
}

export function initCli() {
  const input = $("cli-base-url");
  input.value = prefs.get("cliBaseUrl", "") || window.location.origin;

  input.addEventListener("input", () => {
    prefs.set("cliBaseUrl", input.value.trim());
    render();
  });

  $("cli-reset-url").onclick = () => {
    input.value = window.location.origin;
    prefs.set("cliBaseUrl", input.value);
    render();
  };

  for (const button of document.querySelectorAll("[data-copy]")) {
    button.onclick = () => {
      const source = $(button.dataset.copy);
      if (source) copyText(source, button);
    };
  }

  render();
}
