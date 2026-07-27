"""Saved chat sessions.

One JSON file per conversation under SESSIONS_DIR, holding both the messages
the model sees and the transcript the UI replays (tool calls and their output
included, which the model-facing history doesn't carry). Small enough that
reading and rewriting a whole file per turn is fine, and it means a session
survives a browser reload or a proxy restart.
"""

import json
import os
import re
import time
import uuid

SESSIONS_DIR = os.getenv("SESSIONS_DIR", "sessions")
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "200"))

ID_RE = re.compile(r"^[a-f0-9]{12}$")


def _path(session_id: str) -> str:
    if not ID_RE.match(session_id or ""):
        raise ValueError("invalid session id")
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _read(session_id: str) -> dict | None:
    try:
        with open(_path(session_id)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write(session: dict):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path = _path(session["id"])
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(session, f, indent=2)
    os.replace(tmp, path)


def summary(session: dict) -> dict:
    """The listing view -- no message bodies, so the sidebar stays cheap."""
    return {
        "id": session["id"],
        "title": session.get("title") or "New chat",
        "created": session.get("created", 0),
        "updated": session.get("updated", 0),
        "message_count": len(session.get("messages", [])),
        "backend": session.get("backend", ""),
        "model": session.get("model", ""),
    }


def list_sessions() -> list[dict]:
    if not os.path.isdir(SESSIONS_DIR):
        return []
    out = []
    for name in os.listdir(SESSIONS_DIR):
        if not name.endswith(".json"):
            continue
        session = _read(name[:-5])
        if session and session.get("id"):
            out.append(summary(session))
    return sorted(out, key=lambda s: s["updated"], reverse=True)


def create(title: str = "", **fields) -> dict:
    now = time.time()
    session = {
        "id": uuid.uuid4().hex[:12],
        "title": (title or "New chat").strip()[:120],
        "created": now,
        "updated": now,
        "backend": fields.get("backend", ""),
        "model": fields.get("model", ""),
        "agent": bool(fields.get("agent", False)),
        "sandbox": bool(fields.get("sandbox", True)),
        "workdirs": fields.get("workdirs", []),
        "groups": fields.get("groups", []),
        "messages": fields.get("messages", []),
        "events": fields.get("events", []),
    }
    _write(session)
    _prune()
    return session


def get(session_id: str) -> dict | None:
    return _read(session_id)


UPDATABLE = ("title", "backend", "model", "agent", "sandbox", "workdirs", "groups", "messages", "events")


def update(session_id: str, changes: dict) -> dict | None:
    session = _read(session_id)
    if session is None:
        return None
    for field in UPDATABLE:
        if field in changes:
            session[field] = changes[field]
    if isinstance(session.get("title"), str):
        session["title"] = session["title"].strip()[:120] or "New chat"
    session["updated"] = time.time()
    _write(session)
    return session


def delete(session_id: str) -> bool:
    try:
        os.remove(_path(session_id))
        return True
    except (OSError, ValueError):
        return False


def title_from(text: str) -> str:
    """First line of the opening message, trimmed -- good enough as a name
    until someone renames it."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    line = re.sub(r"\s+", " ", line)
    return (line[:60] + "…") if len(line) > 60 else (line or "New chat")


def _prune():
    """Keep the newest MAX_SESSIONS files so an unattended proxy doesn't grow
    a session directory forever."""
    listing = list_sessions()
    for stale in listing[MAX_SESSIONS:]:
        delete(stale["id"])
