// Shared UI state plus a two-line event bus, so the panels can react to each
// other (adding a backend refreshes the chat selectors, etc.) without any of
// them importing another panel's internals.

export const store = {
  settings: {
    max_agent_iterations: 0,
    tool_timeout: 0,
    default_workspace: "",
    clone_root: "",
    groups: [],
    default_groups: [],
  },
  integrations: { github: { configured: false }, search: { configured: false, providers: [] } },
  githubUser: null,
  backends: [],
  workdirs: [],
  selectedDirs: new Set(),
  groups: new Set(),
  sessions: [],
  sessionId: null,
  // history is what the model sees; events is what the UI replays.
  history: [],
  events: [],
};

const listeners = new Map();

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, []);
  listeners.get(event).push(fn);
}

export function emit(event) {
  for (const fn of listeners.get(event) || []) fn();
}

// Small wrapper so a locked-down browser (no localStorage) degrades to
// "nothing is remembered" instead of throwing on every read.
export const prefs = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(`claude-proxy.${key}`);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(`claude-proxy.${key}`, JSON.stringify(value));
    } catch { /* private mode / storage disabled */ }
  },
};
