"""Credentials for third-party integrations (GitHub, web search).

Same shape as backends.json: secrets live in a gitignored file on the
proxy's machine, are set through the web UI, and are never echoed back to
the browser -- the API only ever reports whether something is configured.
"""

import json
import os

INTEGRATIONS_PATH = os.getenv("INTEGRATIONS_CONFIG", "integrations.json")

# Every field the store knows about, with its default. Anything else sent by
# a client is rejected rather than quietly persisted.
FIELDS = {
    "github_token": "",
    # Overridable for GitHub Enterprise. A value saved through the API wins;
    # this is the default it starts from and the value clearing the field
    # returns to.
    "github_api": os.getenv("GITHUB_API", "https://api.github.com").rstrip("/") or "https://api.github.com",
    "search_provider": "brave",   # brave | tavily
    "search_api_key": "",
}

SECRET_FIELDS = {"github_token", "search_api_key"}
SEARCH_PROVIDERS = ("brave", "tavily")


def _load() -> dict:
    values = dict(FIELDS)
    if os.path.exists(INTEGRATIONS_PATH):
        try:
            with open(INTEGRATIONS_PATH) as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                values.update({k: v for k, v in stored.items() if k in FIELDS and isinstance(v, str)})
        except (OSError, ValueError):
            pass  # unreadable file -> start from defaults rather than refusing to boot
    return values


INTEGRATIONS = _load()


def save():
    # The temp file holds the same secrets the live file does, so it has to be
    # private from the moment it exists: created 0600, tightened again in case
    # a stale tmp was already there with looser modes, and only then moved
    # into place. os.replace keeps the tmp file's mode, so the live file is
    # never briefly world-readable.
    tmp = f"{INTEGRATIONS_PATH}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass  # best effort -- Windows and some mounts don't do POSIX modes
    with os.fdopen(fd, "w") as f:
        json.dump(INTEGRATIONS, f, indent=2)
    os.replace(tmp, INTEGRATIONS_PATH)


def get(field: str) -> str:
    return INTEGRATIONS.get(field, FIELDS.get(field, ""))


# Secrets this module doesn't own but that must still never reach a chat
# bubble -- the provider keys in backends.json, above all. Registered as
# callables rather than copied in, so rotating a key through the UI takes
# effect immediately and nothing here has to import main.
_SECRET_SOURCES = []

# Below this length a "secret" is more likely to appear by coincidence in
# ordinary output than to be the key itself, and blanking it would corrupt
# what the model reads.
MIN_REDACTABLE = 8


def register_secret_source(fn) -> None:
    """Register a callable returning an iterable of secret strings to blank
    out alongside the ones stored here."""
    _SECRET_SOURCES.append(fn)


def _all_secrets():
    for field in SECRET_FIELDS:
        yield INTEGRATIONS.get(field) or ""
    for source in _SECRET_SOURCES:
        try:
            for value in source() or ():
                yield value if isinstance(value, str) else ""
        except Exception:
            # A broken source must not be able to stop redaction happening at
            # all -- that would fail open, which is the one outcome to avoid.
            continue


def redact(text: str) -> str:
    """Blank every known secret out of a string. Anything on its way to a
    chat bubble, a log line or a saved session goes through here -- a tool
    that reads files can otherwise hand the model the token store itself,
    and `bash: env` can hand it every provider key at once."""
    if not isinstance(text, str):
        text = str(text)
    for value in _all_secrets():
        if value and len(value) >= MIN_REDACTABLE:
            text = text.replace(value, "***")
    return text


def update(values: dict) -> list[str]:
    """Apply a partial update. Returns the fields that changed. An empty
    string clears a field, which is how the UI disconnects an integration."""
    changed = []
    for field, value in values.items():
        if field not in FIELDS or not isinstance(value, str):
            continue
        value = value.strip()
        if field == "search_provider" and value not in SEARCH_PROVIDERS:
            continue
        if field == "github_api":
            value = value.rstrip("/") or FIELDS["github_api"]
        if INTEGRATIONS.get(field) != value:
            INTEGRATIONS[field] = value
            changed.append(field)
    if changed:
        save()
    return changed


def public() -> dict:
    """What the browser is allowed to know: configured or not, never the value."""
    return {
        "github": {
            "configured": bool(get("github_token")),
            "api": get("github_api"),
        },
        "search": {
            "configured": bool(get("search_api_key")),
            "provider": get("search_provider"),
            "providers": list(SEARCH_PROVIDERS),
        },
    }
