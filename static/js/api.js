// Thin fetch wrappers around the proxy's own API. No auth header: the proxy
// runs unauthenticated, so reachability is the access control.

function errorText(data, status) {
  const detail = data.detail ?? data.error;
  if (!detail) return `HTTP ${status}`;
  if (typeof detail === "string") return detail;
  // FastAPI validation errors arrive as a list of objects.
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  return JSON.stringify(detail);
}

export async function api(path, opts = {}) {
  const resp = await fetch(path, {
    ...opts,
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(errorText(data, resp.status));
  return data;
}

export function postStream(path, body, signal) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

/** Parses an SSE body into decoded JSON payloads, skipping keepalives and
 *  the `[DONE]` sentinel. */
export async function* sseJson(resp) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop();
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const raw = line.slice(5).trim();
      if (!raw || raw === "[DONE]") continue;
      try {
        yield JSON.parse(raw);
      } catch { /* partial or non-JSON frame */ }
    }
  }
}
