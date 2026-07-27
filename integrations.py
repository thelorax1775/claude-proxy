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
    "github_api": "https://api.github.com",
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
    tmp = f"{INTEGRATIONS_PATH}.tmp"
    with open(tmp, "w") as f:
        json.dump(INTEGRATIONS, f, indent=2)
    os.replace(tmp, INTEGRATIONS_PATH)
    try:
        os.chmod(INTEGRATIONS_PATH, 0o600)
    except OSError:
        pass  # best effort -- Windows and some mounts don't do POSIX modes


def get(field: str) -> str:
    return INTEGRATIONS.get(field, FIELDS.get(field, ""))


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
