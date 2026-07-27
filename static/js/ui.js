// DOM and text helpers shared by every panel.

export const $ = (id) => document.getElementById(id);

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMarks(escaped) {
  // Runs on already-escaped text, so the tags introduced here are the only
  // markup that can reach the DOM.
  return escaped
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
}

/** Just enough markdown for chat output: fenced code, inline code, bold.
 *  An unterminated fence renders as an open code block, which is what you
 *  want while a reply is still streaming in. */
export function renderRich(text) {
  return String(text)
    .split("```")
    .map((part, i) => {
      if (i % 2 === 1) {
        const firstBreak = part.indexOf("\n");
        const body = firstBreak === -1 ? part : part.slice(firstBreak + 1);
        return `<pre><code>${escapeHtml(body.replace(/\n$/, ""))}</code></pre>`;
      }
      return part
        .split(/\n{2,}/)
        .filter((block) => block.trim() !== "")
        .map((block) => `<p>${inlineMarks(escapeHtml(block))}</p>`)
        .join("");
    })
    .join("");
}

export function shorten(value, max = 70) {
  const text = String(value);
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

export function relativeTime(seconds) {
  const delta = Date.now() / 1000 - seconds;
  if (delta < 60) return `${Math.max(0, Math.round(delta))}s ago`;
  if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
  return new Date(seconds * 1000).toLocaleString();
}

export function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 192)}px`;
}
