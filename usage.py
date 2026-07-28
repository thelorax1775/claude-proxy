"""How much of each backend's allowance is left.

A free-tier key that has run out doesn't announce itself. It 429s, the proxy
benches it for a cooldown, and the chain quietly gets shorter -- which looks
from the outside like the proxy being slow rather than an account being spent.
/status already shows which backends are cooling; this answers the question
that comes next, which is whether it will still be cooling in an hour or for
the rest of the week.

Only some providers can answer that, so the shape here is a registry: a
provider is recognised by the host in its base_url, and anything unrecognised
reports supported=False rather than guessing. Ollama Cloud is the one
implemented -- GET https://ollama.com/api/usage returns limits.weekly.usage as
a *fraction* of the weekly allowance already spent, so 0.957 means 4.3% left.

Nothing here is on the request path. The proxy does not consult usage before
choosing a backend and does not block on it; failover stays driven by what
upstream actually returns. This is reporting, and a provider being down for it
must never be a reason a chat can't be sent.
"""

import asyncio
import json
import logging
import time
from urllib.parse import urlsplit

log = logging.getLogger("claude-proxy")

# Answers are cached because the UI polls and six backends means six calls.
# Short enough that "I just switched keys" shows up quickly, long enough that
# an open Backends tab isn't a traffic generator of its own.
TTL_SECONDS = 120

# A usage lookup is a sideshow: if the provider is slow, say so and move on
# rather than making the page wait on it.
TIMEOUT_SECONDS = 10.0

_CACHE: dict[str, tuple[float, dict]] = {}


def _host(base_url: str) -> str:
    try:
        return (urlsplit(base_url).hostname or "").lower()
    except ValueError:
        return ""


def provider_for(base_url: str) -> str:
    """Which usage provider handles this backend, '' if none does."""
    host = _host(base_url)
    if host == "ollama.com" or host.endswith(".ollama.com"):
        return "ollama"
    return ""


def unsupported(base_url: str) -> dict:
    return {
        "supported": False,
        "provider": "",
        "reason": f"no usage API known for {_host(base_url) or 'this host'}",
    }


def _pct(fraction) -> float | None:
    """A 0..1 fraction as a percentage, clamped. The provider is the authority
    on how much is spent, but a value slightly over 1.0 (or a string) must not
    turn into a meter drawn past its own track."""
    try:
        value = float(fraction)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(1.0, value)) * 100, 1)


def parse_ollama(payload: dict) -> dict:
    """Normalise Ollama Cloud's /api/usage into the shape the UI renders.

    limits.weekly.usage is the fraction of the weekly allowance spent, and
    limits.session.usage the same for the shorter session window. Both are
    reported: the weekly one is what runs out for days, the session one is
    what makes a backend bounce for a few minutes.
    """
    limits = payload.get("limits") or {}
    weekly = limits.get("weekly") or {}
    session = limits.get("session") or {}
    activity = payload.get("activity") or {}

    models = []
    for entry in weekly.get("models") or []:
        if not isinstance(entry, dict):
            continue
        models.append({
            "name": str(entry.get("name") or "?"),
            "requests": int(entry.get("request_count") or 0),
        })
    models.sort(key=lambda m: m["requests"], reverse=True)

    used = _pct(weekly.get("usage"))
    return {
        "supported": True,
        "provider": "ollama",
        # None rather than 0 when the provider didn't say -- "we don't know"
        # and "nothing used" must not render the same.
        "used_pct": used,
        "remaining_pct": None if used is None else round(100.0 - used, 1),
        "session_pct": _pct(session.get("usage")),
        "window": "weekly",
        "requests": sum(m["requests"] for m in models),
        "models": models,
        "cost": str(activity.get("cost") or "").strip(),
    }


async def _fetch_ollama(client, backend: dict) -> dict:
    """The usage endpoint is a sibling of the OpenAI-compatible path, not a
    child of it: base_url is https://ollama.com/v1 and usage lives at
    /api/usage on the same host."""
    parts = urlsplit(backend["base_url"])
    url = f"{parts.scheme}://{parts.netloc}/api/usage"
    resp = await client.get(
        url,
        headers={"Authorization": f"Bearer {backend.get('api_key', '')}"},
        timeout=TIMEOUT_SECONDS,
    )
    if resp.status_code in (401, 403):
        return {"supported": True, "provider": "ollama",
                "error": "the key was rejected -- it may be revoked or expired"}
    if resp.status_code != 200:
        return {"supported": True, "provider": "ollama",
                "error": f"the usage API returned {resp.status_code}"}
    try:
        payload = resp.json()
    except (ValueError, json.JSONDecodeError):
        return {"supported": True, "provider": "ollama",
                "error": "the usage API returned something that wasn't JSON"}
    return parse_ollama(payload)


FETCHERS = {"ollama": _fetch_ollama}


async def for_backend(client, backend: dict, *, force: bool = False) -> dict:
    """Usage for one backend, from cache unless it's stale or force is set."""
    name = backend.get("name", "")
    provider = provider_for(backend.get("base_url", ""))
    if not provider:
        return {**unsupported(backend.get("base_url", "")), "name": name}

    cached = _CACHE.get(name)
    if cached and not force and time.monotonic() - cached[0] < TTL_SECONDS:
        return {**cached[1], "name": name, "cached": True}

    if not backend.get("api_key"):
        return {"name": name, "supported": True, "provider": provider,
                "error": "no key set for this backend"}

    try:
        result = await FETCHERS[provider](client, backend)
    except Exception as e:
        # Deliberately broad. Every failure mode here -- DNS, TLS, timeout, a
        # provider returning something unexpected -- has the same consequence,
        # which is that this card says why instead of the page breaking.
        log.warning("usage lookup for %s failed: %s", name, e)
        result = {"supported": True, "provider": provider,
                  "error": f"could not reach the usage API ({type(e).__name__})"}

    result["checked_at"] = time.time()
    _CACHE[name] = (time.monotonic(), result)
    return {**result, "name": name, "cached": False}


async def for_all(client, backends: list[dict], *, force: bool = False) -> list[dict]:
    """Every backend at once -- six sequential round trips is a visible pause,
    six concurrent ones is not."""
    if not backends:
        return []
    return list(await asyncio.gather(
        *(for_backend(client, b, force=force) for b in backends)))


def forget(name: str) -> None:
    """Drop a cached answer -- called when a backend's key or URL changes, so
    the card doesn't keep showing the old account's numbers."""
    _CACHE.pop(name, None)


# ---------------------------------------------------------------------------
# Reading the cache from the request path
#
# Everything below is called while a chat is being routed, so none of it may
# do I/O or block. It reads what happens to be cached and says "don't know"
# otherwise -- a backend is never demoted on a guess.
# ---------------------------------------------------------------------------

# The provider reports a rounded fraction, so 0.999 and 1.0 mean the same
# thing. Matches SPENT_AT in static/js/backends.js -- the meter and the router
# must not disagree about what "spent" means.
SPENT_AT = 99.5

# How old a figure may be and still steer routing. Longer than TTL_SECONDS on
# purpose: a stale number is still evidence, and re-checking is someone else's
# job. But not unbounded -- a weekly allowance resets, and a backend must not
# stay at the back of the chain forever on the strength of last week's reading.
ROUTING_MAX_AGE = 900


def cached_used_pct(name: str) -> float | None:
    """The last known percentage spent, or None if we don't know or the answer
    is too old to act on. Never fetches."""
    entry = _CACHE.get(name)
    if entry is None:
        return None
    recorded_at, row = entry
    if time.monotonic() - recorded_at > ROUTING_MAX_AGE:
        return None
    value = row.get("used_pct")
    return value if isinstance(value, (int, float)) else None


def is_spent(name: str) -> bool:
    """Whether this backend's allowance is known to be gone. False when
    unknown, so an unreachable usage API can never bench a working backend."""
    used = cached_used_pct(name)
    return used is not None and used >= SPENT_AT


def needs_refresh(backends: list[dict]) -> bool:
    """Whether a background re-read is worth doing.

    False when no configured backend has a usage provider at all, which keeps
    this entirely inert for setups it can't say anything about -- there is no
    point waking up to re-read nothing.
    """
    for b in backends:
        if not provider_for(b.get("base_url", "")):
            continue
        entry = _CACHE.get(b.get("name", ""))
        if entry is None or time.monotonic() - entry[0] >= TTL_SECONDS:
            return True
    return False
