"""Saved chat sessions.

One JSON file per conversation under SESSIONS_DIR, holding both the messages
the model sees and the transcript the UI replays (tool calls and their output
included, which the model-facing history doesn't carry). One file is read and
rewritten per turn, which is fine at this size, and it means a session
survives a browser reload or a proxy restart. The sidebar listing is the one
place that touches every file, so it caches per file rather than re-parsing
transcripts that haven't changed.
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


# Building the sidebar means a summary per session, but a session file holds
# every message and every tool result -- a chat that read a few large files is
# megabytes, and the listing is refreshed after each turn. Cache the summary
# against the file's (mtime, size) so only files that actually changed are
# parsed again.
_SUMMARY_CACHE: dict[str, tuple[int, int, dict]] = {}


def _cached_summary(session_id: str, path: str) -> dict | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    cached = _SUMMARY_CACHE.get(session_id)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]
    session = _read(session_id)
    if not session or not session.get("id"):
        return None
    value = summary(session)
    _SUMMARY_CACHE[session_id] = (stat.st_mtime_ns, stat.st_size, value)
    return value


def list_sessions() -> list[dict]:
    if not os.path.isdir(SESSIONS_DIR):
        return []
    out = []
    live = set()
    for name in os.listdir(SESSIONS_DIR):
        if not name.endswith(".json"):
            continue
        session_id = name[:-5]
        if not ID_RE.match(session_id):
            continue
        live.add(session_id)
        entry = _cached_summary(session_id, os.path.join(SESSIONS_DIR, name))
        if entry:
            out.append(entry)
    for stale in _SUMMARY_CACHE.keys() - live:
        _SUMMARY_CACHE.pop(stale, None)
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
        "lan": bool(fields.get("lan", False)),
        "approval": fields.get("approval", "skip"),
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


UPDATABLE = ("title", "backend", "model", "agent", "sandbox", "lan", "approval",
             "workdirs", "groups", "messages", "events")


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
        _SUMMARY_CACHE.pop(session_id, None)
        return True
    except (OSError, ValueError):
        return False


def _prune():
    """Keep the newest MAX_SESSIONS files so an unattended proxy doesn't grow
    a session directory forever."""
    listing = list_sessions()
    for stale in listing[MAX_SESSIONS:]:
        delete(stale["id"])
