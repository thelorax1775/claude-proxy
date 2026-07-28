"""
Anthropic Messages API <-> OpenAI-compatible translation proxy, with
failover across multiple backends (hosted free-tier APIs, local Ollama).

Point Claude Code at this server via ANTHROPIC_BASE_URL and it will
transparently rewrite requests/responses so Claude Code can drive any
backend that supports OpenAI-style function calling. Backends are tried
in order; on rate limit (429) or error, the next one in the list is tried.
"""

import asyncio
import collections
import json
import logging
import os
import re
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import agent_tools
import approvals
import github_api
import integrations
import sessions
import ssh_tools
import usage
import web_tools

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("claude-proxy")

# Last 50 backend attempts, newest last -- see GET /status
ATTEMPT_LOG = collections.deque(maxlen=50)


def load_backends() -> list[dict]:
    """Ordered list of {name, base_url, api_key, big_model, small_model}.
    Tried in order per request; first one that doesn't 429/5xx/error wins."""
    path = os.getenv("BACKENDS_CONFIG", "backends.json")
    if os.path.exists(path):
        with open(path) as f:
            backends = json.load(f)
        if not backends:
            raise RuntimeError(f"{path} is empty -- need at least one backend")
        return backends
    # Fallback: single backend from plain env vars (no backends.json present).
    return [{
        "name": "default",
        "base_url": os.getenv("BACKEND_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        "api_key": os.getenv("BACKEND_API_KEY", os.getenv("NVIDIA_API_KEY", "")),
        "big_model": os.getenv("BIG_MODEL", "meta/llama-3.3-70b-instruct"),
        "small_model": os.getenv("SMALL_MODEL", "meta/llama-3.1-8b-instruct"),
    }]


BACKENDS = load_backends()
BACKENDS_PATH = os.getenv("BACKENDS_CONFIG", "backends.json")

# Provider keys are secrets too. Without this, agent mode reading backends.json
# -- or just running `env` -- puts every key straight into a chat bubble and
# into the saved session on disk. Registered as a callable so a key changed
# through the UI is covered immediately.
integrations.register_secret_source(lambda: [b.get("api_key", "") for b in BACKENDS])


def save_backends():
    tmp = f"{BACKENDS_PATH}.tmp"
    with open(tmp, "w") as f:
        json.dump(BACKENDS, f, indent=2)
    os.replace(tmp, BACKENDS_PATH)

HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)

# Set by the test suite to an httpx.MockTransport so the whole failover and
# translation stack can be exercised without a network. None in normal use,
# which is httpx's own default.
UPSTREAM_TRANSPORT = None


def upstream_client(timeout=HTTP_TIMEOUT) -> httpx.AsyncClient:
    """Every call out to a backend goes through here, so there is one place to
    intercept them."""
    return httpx.AsyncClient(timeout=timeout, transport=UPSTREAM_TRANSPORT)

# Status codes worth failing over on. 401/403 included because a dead/expired
# free-tier key should not take the whole proxy down -- skip to the next one.
RETRYABLE_STATUS = {401, 403, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Context overflow.
#
# A conversation that outgrows a model's context window comes back as a 400,
# which is otherwise the shape of "your request is malformed" -- not worth
# retrying, because the next backend would say the same. Context is the
# exception: the whole point of a chain is that its backends differ, and they
# differ in context window as much as anything else. A conversation too long
# for an 8k model is ordinary for a 128k one, so this is worth failing over
# on rather than dead-ending the chat.
#
# There is no standard code for it. Every provider says it in prose, so this
# matches the prose; a false negative just restores the old behaviour of
# passing the 400 through.
# ---------------------------------------------------------------------------

CONTEXT_OVERFLOW_STATUS = {400, 413, 422}

CONTEXT_OVERFLOW_RE = re.compile(
    r"context[ _-]?length|context[ _-]?window|maximum context|"
    r"context_length_exceeded|string_above_max_length|"
    r"too many tokens|reduce the length|too long|"
    r"exceeds?\b[^.]{0,40}\btokens?\b|"
    r"\btokens?\b[^.]{0,40}\bexceed",
    re.I,
)


def is_context_overflow(status: int, body: str) -> bool:
    """Whether an upstream rejection means "this conversation no longer fits"
    rather than "this request is wrong"."""
    if status not in CONTEXT_OVERFLOW_STATUS:
        return False
    return bool(CONTEXT_OVERFLOW_RE.search(body or ""))


# What the UI shows when no backend would take the conversation. The chat is
# not broken and the history is not lost -- it just has to get shorter, and
# the browser offers to do that.
CONTEXT_OVERFLOW_MESSAGE = (
    "This conversation is now too long for every configured backend. "
    "Shorten it (or start a new chat) and the same backends will work again."
)

# Sentinel in call_chain_non_stream's hard_status position. Not an HTTP status:
# it says "every backend refused this as too long", which callers handle
# differently from an upstream status they should pass through.
OVERFLOW = "context_overflow"


def record_attempt(backend_name: str, outcome: str):
    entry = {"time": time.time(), "backend": backend_name, "outcome": outcome}
    ATTEMPT_LOG.append(entry)
    log.info("backend=%s outcome=%s", backend_name, outcome)


# ---------------------------------------------------------------------------
# Cooldown. Without it, a backend that just answered 429 goes straight back to
# the front of the chain on the very next request, so every request re-pays
# the rate limit before falling through. A dead key is worse: it fails every
# request forever, at the cost of a round trip each time.
#
# In memory only. A restart clearing it is correct -- the reason to skip a
# backend is something that just happened, not something worth persisting.
# ---------------------------------------------------------------------------

BACKEND_COOLDOWN = int(os.getenv("BACKEND_COOLDOWN", "60"))
# Longer, because a rejected key does not come back in a minute the way a
# rate limit does.
BACKEND_AUTH_COOLDOWN = int(os.getenv("BACKEND_AUTH_COOLDOWN", "300"))

# backend name -> monotonic time when it may be tried again
_COOLDOWN: dict[str, float] = {}


def cool_down(backend_name: str, seconds: float) -> None:
    if seconds > 0:
        _COOLDOWN[backend_name] = max(_COOLDOWN.get(backend_name, 0.0),
                                      time.monotonic() + seconds)


def cooling_for(backend_name: str) -> int:
    """Seconds until this backend is worth trying again; 0 when it is now."""
    remaining = _COOLDOWN.get(backend_name, 0.0) - time.monotonic()
    if remaining <= 0:
        _COOLDOWN.pop(backend_name, None)
        return 0
    return int(remaining) + 1


def clear_cooldown(backend_name: str) -> None:
    _COOLDOWN.pop(backend_name, None)


def _retry_after(resp) -> float:
    """Honour the provider's own Retry-After when it sends one -- it knows
    better than our default does."""
    raw = resp.headers.get("retry-after") if resp is not None else None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def note_failure(backend_name: str, status: int, resp=None) -> None:
    """Record a failed attempt and put the backend on the bench for a while."""
    record_attempt(backend_name, f"HTTP {status}")
    if status in (401, 403):
        cool_down(backend_name, BACKEND_AUTH_COOLDOWN)
    elif status in RETRYABLE_STATUS:
        cool_down(backend_name, _retry_after(resp) or BACKEND_COOLDOWN)


# How the chain is ordered. "least-used" puts whichever backend has the most
# allowance left at the front, so a pool of accounts drains evenly instead of
# one at a time; "configured" uses backends.json exactly as written.
BACKEND_ORDER = os.getenv("BACKEND_ORDER", "least-used").strip().lower()

# The knob that shipped first, still honoured: 0 meant "never reorder", which
# is what "configured" means now.
if os.getenv("BACKEND_DEMOTE_SPENT", "1").strip().lower() in ("0", "false", "no"):
    BACKEND_ORDER = "configured"

REORDERS = BACKEND_ORDER != "configured"

# Usage figures are rounded and move constantly, and the front of the chain is
# the end that gets used -- so ordering on the raw percentage would have the
# leader change every couple of minutes and the pool never settle. Ten-point
# bands mean a backend only overtakes another once it is meaningfully emptier,
# and accounts inside a band keep the order you wrote them in.
USAGE_BAND = 10.0

# Set while a background usage re-read is in flight, so a busy proxy starts one
# refresh rather than one per request.
_usage_refresh: asyncio.Task | None = None


def warm_usage_cache() -> None:
    """Kick off a usage re-read if the figures have gone stale, and return
    immediately.

    Routing reads the usage cache but must never wait on it, so something has
    to keep it warm. Doing it here means it is traffic that pays for it: a busy
    proxy re-checks every couple of minutes, an idle one never does, and there
    is no timer to manage or shut down. Failure is silent -- this only decides
    what order the chain is tried in, and getting that wrong is not worth a
    failed request.
    """
    global _usage_refresh
    if not REORDERS or not usage.needs_refresh(BACKENDS):
        return
    if _usage_refresh is not None and not _usage_refresh.done():
        return
    try:
        # Checked before the coroutine is built, not after: calling run() with
        # no loop to hand it to leaves an un-awaited coroutine behind.
        asyncio.get_running_loop()
    except RuntimeError:
        return

    async def run():
        try:
            async with upstream_client() as client:
                await usage.for_all(client, BACKENDS)
        except Exception as e:
            log.warning("background usage refresh failed: %s", e)

    _usage_refresh = asyncio.create_task(run())


def by_headroom(items, name_of):
    """`items` with the emptiest accounts pulled toward the front.

    Only backends we have a figure for move, and they move only between the
    slots they already occupied: a backend whose usage can't be read -- a
    provider with no usage API, or one that just failed to answer -- keeps its
    index exactly. So a paid endpoint deliberately configured first stays
    first instead of being shuffled behind a pool of free accounts, and a usage
    API outage degrades to the configured order rather than to a guess.
    """
    slots = [i for i, item in enumerate(items)
             if usage.cached_used_pct(name_of(item)) is not None]
    if len(slots) < 2:
        return items
    # Stable, so backends in the same band keep their configured order.
    ranked = sorted((items[i] for i in slots),
                    key=lambda x: usage.cached_used_pct(name_of(x)) // USAGE_BAND)
    out = list(items)
    for slot, item in zip(slots, ranked):
        out[slot] = item
    return out


def prioritise(items, name_of):
    """`items` ordered by how much allowance each backend has left: emptiest
    first, and anything known to be out of allowance last.

    Spent backends go to the very back rather than into a band, because "this
    one cannot answer at all" outranks "this one is fuller than that one", and
    that has to hold even when nothing else has a figure to compare against.

    Deliberately a reorder and not a filter. A spent account still answers
    sometimes -- the figure is rounded, the window may have just rolled over,
    and some providers keep serving at reduced priority -- so dropping it could
    turn a slow request into a failed one. Reordering costs nothing when the
    others work and loses nothing when they don't.
    """
    if not REORDERS or len(items) < 2:
        return items
    spent = [i for i in items if usage.is_spent(name_of(i))]
    head = [i for i in items if not usage.is_spent(name_of(i))]
    if not head:
        return items          # all equally bad; the configured order is the
                              # better tiebreak than an arbitrary one
    return by_headroom(head, name_of) + spent


def usable_backends(items, name_of):
    """`items` minus anything currently cooling down, ordered by how much
    allowance is left. If dropping the cooling ones would leave nothing, every
    one of them is returned instead: an answer from an over-quota account beats
    no answer at all, and the whole point of the chain is that it doesn't
    dead-end."""
    warm_usage_cache()
    usable = [i for i in items if not cooling_for(name_of(i))]
    if not usable and items:
        record_attempt("(chain)", "every backend cooling down -- trying anyway")
        usable = list(items)
    return prioritise(usable, name_of)


# ---------------------------------------------------------------------------
# Agent mode lives in agent_tools.py: the working-directory registry, the
# tool implementations, and the group registry that decides what a run may
# call. Only the loop that drives it stays here, since that needs the
# backend-failover machinery below.
# ---------------------------------------------------------------------------

MAX_AGENT_ITERATIONS = agent_tools.MAX_AGENT_ITERATIONS


def pick_model(claude_model: str, backend: dict) -> str:
    """Which of a backend's models answers this request. Read with .get rather
    than [] because backends.json is hand-editable: a missing small_model
    should fall back to the big one, not 500 every request that mentions
    haiku."""
    big = backend.get("big_model") or ""
    small = backend.get("small_model") or ""
    m = (claude_model or "").lower()
    if "haiku" in m:
        return small or big
    return big or small


def tool_args_dict(raw) -> dict:
    """A tool call's arguments as a dict. OpenAI specifies a JSON *string*,
    but llama.cpp's server, some vLLM builds and a few gateways send the
    object itself -- json.loads on a dict raises TypeError, which used to be
    an unhandled 500. Never raises; unparseable arguments become {}."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def tool_args_str(raw) -> str:
    """The same arguments as the JSON string every downstream consumer wants.
    A provider that sent an object gets it serialised here, once, rather than
    embedded raw into an input_json_delta the client can't parse."""
    if isinstance(raw, str):
        return raw
    if raw is None:
        return ""
    try:
        return json.dumps(raw)
    except (TypeError, ValueError):
        return ""


def first_choice(data: dict) -> dict:
    """The first choice of a completion, or {} if there isn't one. Providers do
    return an empty choices list -- content filters and some error shapes both
    do it -- and indexing [0] there is a 500 on /v1/messages or a stream that
    dies mid-agent-run with the work already done thrown away."""
    choices = (data or {}).get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


# ---------------------------------------------------------------------------
# Anthropic request -> OpenAI/NIM request
# ---------------------------------------------------------------------------

def system_to_text(system) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n\n".join(parts) if parts else None


def anthropic_messages_to_openai(system, messages):
    openai_messages = []
    sys_text = system_to_text(system)
    if sys_text:
        openai_messages.append({"role": "system", "content": sys_text})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        text_parts = []
        tool_calls = []
        tool_results = []

        for block in content or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif btype == "tool_result":
                tool_results.append(block)
            elif btype == "image":
                text_parts.append("[image omitted: this proxy does not forward images to NIM]")

        if tool_results:
            # Anthropic carries tool_result blocks inside a "user" message.
            # OpenAI wants each one as its own role:"tool" message.
            for tr in tool_results:
                tr_content = tr.get("content", "")
                if isinstance(tr_content, list):
                    tr_text = "\n".join(
                        b.get("text", "") for b in tr_content if isinstance(b, dict)
                    )
                else:
                    tr_text = str(tr_content)
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_use_id"],
                    "content": tr_text,
                })
            if text_parts:
                openai_messages.append({"role": role, "content": "\n".join(text_parts)})
            continue

        msg_obj = {"role": role, "content": "\n".join(text_parts) if text_parts else None}
        if tool_calls:
            msg_obj["tool_calls"] = tool_calls
        openai_messages.append(msg_obj)

    return openai_messages


def anthropic_tools_to_openai(tools):
    if not tools:
        return None
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def anthropic_tool_choice_to_openai(tool_choice):
    if not tool_choice:
        return None
    t = tool_choice.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return "auto"


def build_openai_payload(body: dict, backend: dict) -> dict:
    claude_model = body.get("model", "")
    payload = {
        "model": pick_model(claude_model, backend),
        "messages": anthropic_messages_to_openai(body.get("system"), body.get("messages", [])),
        "max_tokens": body.get("max_tokens", 1024),
        "stream": bool(body.get("stream", False)),
    }
    if "temperature" in body:
        payload["temperature"] = body["temperature"]
    if "top_p" in body:
        payload["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]

    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        payload["tools"] = tools
        choice = anthropic_tool_choice_to_openai(body.get("tool_choice"))
        if choice:
            payload["tool_choice"] = choice

    return payload


# ---------------------------------------------------------------------------
# OpenAI/NIM response -> Anthropic response (non-streaming)
# ---------------------------------------------------------------------------

FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def openai_response_to_anthropic(nim_data: dict, claude_model: str) -> dict:
    choice = first_choice(nim_data)
    message = choice.get("message") or {}
    content_blocks = []

    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in message.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            # A tool_use block with no name is unusable to Claude Code and
            # would be rejected; drop it rather than pass it through.
            log.warning("dropping a tool_call with no function name: %r", tc)
            continue
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": name,
            "input": tool_args_dict(fn.get("arguments")),
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = nim_data.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": claude_model,
        "content": content_blocks,
        "stop_reason": FINISH_REASON_MAP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE chunks -> Anthropic SSE events
# ---------------------------------------------------------------------------

def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Token estimation.
#
# No tokenizer: this project's dependency list is four packages and stays that
# way. Characters per token is crude but it only has to be good enough to
# decide "is this near the window" and "how much do we drop", and it has to be
# consistent everywhere -- one estimator, used by count_tokens, by the usage
# Claude Code reads, and by the trimmer.
#
# It biases HIGH on purpose. Under-reporting is what walks a conversation into
# an overflow; over-reporting only makes a client compact slightly early. The
# function itself stays honest though -- no fudge factor lives in here, so the
# safety margin can be tuned where it belongs, at the trim target.
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN_TEXT = 4      # prose
CHARS_PER_TOKEN_JSON = 3      # schemas and tool payloads tokenise denser
# Per-message and per-tool framing the provider pays for and the content
# itself doesn't show: role markers, delimiters, function-call scaffolding.
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_OVERHEAD_TOKENS = 8


def _value_tokens(value) -> int:
    """Tokens for one content value, whatever shape it arrives in: a string, an
    Anthropic block list, an OpenAI tool_calls list, a JSON schema.

    Anything that isn't a plain string is measured serialised, because the
    punctuation and keys are real tokens the provider is sent too. Being
    shape-agnostic is the point: a block type this has never seen still
    contributes its characters instead of vanishing from the count."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value) // CHARS_PER_TOKEN_TEXT
    try:
        return len(json.dumps(value, default=str)) // CHARS_PER_TOKEN_JSON
    except (TypeError, ValueError):
        return len(str(value)) // CHARS_PER_TOKEN_JSON


def estimate_tokens(system=None, messages=None, tools=None) -> int:
    """Rough input size of a request.

    Counts the things that actually make an agentic turn big -- tool schemas,
    tool_use inputs and tool_result payloads -- not just prose. Claude Code
    sends ~15 tool schemas on every request; leaving them out is what made this
    read a few hundred tokens for a 30k-token turn.

    Takes Anthropic bodies (/v1/messages) and OpenAI ones (/api/chat) alike."""
    total = _value_tokens(system if isinstance(system, str) else system_to_text(system))
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        total += MESSAGE_OVERHEAD_TOKENS
        total += _value_tokens(msg.get("content"))
        # OpenAI-shaped assistant turns carry their calls out of band; the
        # Anthropic equivalent is a tool_use block already inside content.
        total += _value_tokens(msg.get("tool_calls"))
    for tool in tools or []:
        total += TOOL_OVERHEAD_TOKENS + _value_tokens(tool)
    return total


def _output_tokens(upstream_usage, generated_chars: int) -> int:
    """What to report as output_tokens for a streamed response. Prefer the
    provider's own count when it sent one; otherwise the same rough estimate
    used everywhere else. Claude Code spends this on context and cost
    accounting, so a number in the wrong unit is worse than an approximate
    one in the right unit."""
    if isinstance(upstream_usage, dict):
        reported = upstream_usage.get("completion_tokens")
        if isinstance(reported, int) and reported >= 0:
            return reported
    return max(0, generated_chars // CHARS_PER_TOKEN_TEXT)


def backend_headers(backend: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if backend.get("api_key"):
        headers["Authorization"] = f"Bearer {backend['api_key']}"
    return headers


async def find_working_stream(client: httpx.AsyncClient, body: dict):
    """Try each backend in order for a streaming request. Returns
    (response, backend) for the first one that isn't rate-limited/erroring,
    or (None, None) if all backends were exhausted."""
    last_error = None
    overflowed = False
    # Snapshot: this loop spans awaits, and deleting or reordering a backend
    # from the UI mid-request would otherwise silently skip the next entry.
    for backend in usable_backends(list(BACKENDS), lambda b: b["name"]):
        payload = build_openai_payload(body, backend)
        payload["stream"] = True
        url = f"{backend['base_url']}/chat/completions"
        req = client.build_request("POST", url, headers=backend_headers(backend),
                                    json=payload, timeout=HTTP_TIMEOUT)
        try:
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as e:
            record_attempt(backend["name"], f"connection error: {e}")
            cool_down(backend["name"], BACKEND_COOLDOWN)
            last_error = str(e)
            continue
        if resp.status_code in RETRYABLE_STATUS:
            body_text = (await resp.aread()).decode("utf-8", "replace")
            note_failure(backend["name"], resp.status_code, resp)
            await resp.aclose()
            last_error = f"HTTP {resp.status_code}: {body_text[:300]}"
            continue
        if resp.status_code >= 400:
            body_text = (await resp.aread()).decode("utf-8", "replace")
            await resp.aclose()
            if is_context_overflow(resp.status_code, body_text):
                # Not this backend's fault and not a reason to bench it -- the
                # next message may well be short. Try one with a bigger window.
                record_attempt(backend["name"], "context overflow")
                overflowed = True
                last_error = f"HTTP {resp.status_code}: {body_text[:300]}"
                continue
            record_attempt(backend["name"], f"HTTP {resp.status_code} (non-retryable)")
            return None, None, f"HTTP {resp.status_code}: {body_text[:300]}"
        record_attempt(backend["name"], "ok")
        clear_cooldown(backend["name"])
        return resp, backend, None
    if overflowed:
        return None, None, OVERFLOW
    return None, None, last_error or "no backends configured"


async def stream_openai_as_anthropic(client: httpx.AsyncClient, body: dict, claude_model: str):
    resp, backend, error = await find_working_stream(client, body)
    if resp is None:
        if error == OVERFLOW:
            # invalid_request_error, not api_error: this is what Anthropic
            # returns for an over-long prompt, and it's what Claude Code reads
            # to decide to compact and retry rather than surface a failure.
            yield sse_event("error", {
                "type": "error",
                "error": {"type": "invalid_request_error",
                          "message": "prompt is too long: " + CONTEXT_OVERFLOW_MESSAGE},
            })
            return
        yield sse_event("error", {
            "type": "error",
            "error": {"type": "api_error", "message": f"all backends exhausted: {error}"},
        })
        return

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Estimated rather than 0. Providers rarely report prompt_tokens on a
    # stream, and a hardcoded zero told Claude Code every request was free --
    # the same blind spot count_tokens had.
    input_estimate = estimate_tokens(body.get("system"), body.get("messages"),
                                     body.get("tools"))

    yield sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": claude_model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": input_estimate, "output_tokens": 0},
        },
    })

    next_index = 0
    text_index = None
    text_open = False
    # openai tool-call index -> {"index": anthropic index or None until opened,
    # "id", "name", "pending": [argument fragments seen before the name was]}.
    # A block can't be opened until its name is known, because content_block_start
    # carries the name and there is no event to correct it afterwards.
    tool_state = {}
    last_tc_key = None
    synthetic_keys = 0
    finish_reason = "stop"
    # Fallback only. A provider that reports real usage wins over this.
    text_chars = 0
    upstream_usage = None

    try:
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            # Providers that volunteer usage put it on a chunk near the end,
            # often one with an empty choices list -- so read it before the
            # no-choices check below drops the chunk.
            if isinstance(chunk.get("usage"), dict):
                upstream_usage = chunk["usage"]

            choice = first_choice(chunk)
            if not choice:
                continue
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            if delta.get("content"):
                if not text_open:
                    text_index = next_index
                    next_index += 1
                    text_open = True
                    yield sse_event("content_block_start", {
                        "type": "content_block_start", "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                text_chars += len(delta["content"])
                yield sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": text_index,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                })

            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}

                # 'index' is how OpenAI ties fragments to a call, but several
                # OpenAI-compatible providers leave it out when streaming one
                # call at a time. Reading it with [] was a KeyError, i.e. a 500
                # in the middle of a stream Claude Code is already consuming.
                key = tc.get("index")
                if not isinstance(key, int):
                    current = tool_state.get(last_tc_key)
                    new_id = tc.get("id")
                    if current is not None and (not new_id or new_id == current["id"]):
                        key = last_tc_key          # more of the call in progress
                    else:
                        synthetic_keys += 1        # a different id means a new call
                        key = f"auto-{synthetic_keys}"
                last_tc_key = key

                state = tool_state.get(key)
                if state is None:
                    state = tool_state[key] = {"index": None, "id": "", "name": "", "pending": []}
                if tc.get("id"):
                    state["id"] = tc["id"]
                # First non-empty name wins. Providers send the name once, but
                # not always in the first fragment -- taking only the first
                # fragment's name is what produced tool_use blocks named "".
                if fn.get("name") and not state["name"]:
                    state["name"] = fn["name"]

                # A provider that sends the whole object instead of a JSON
                # string gets it serialised here; embedding it raw produced an
                # input_json_delta Claude Code silently failed to parse, which
                # reads as "the model forgot how to use tools".
                args_fragment = tool_args_str(fn.get("arguments"))
                if args_fragment:
                    text_chars += len(args_fragment)
                    state["pending"].append(args_fragment)

                if state["index"] is None and state["name"]:
                    if text_open:
                        yield sse_event("content_block_stop",
                                        {"type": "content_block_stop", "index": text_index})
                        text_open = False
                    state["index"] = next_index
                    next_index += 1
                    yield sse_event("content_block_start", {
                        "type": "content_block_start", "index": state["index"],
                        "content_block": {
                            "type": "tool_use",
                            "id": state["id"] or f"toolu_{uuid.uuid4().hex[:24]}",
                            "name": state["name"],
                            "input": {},
                        },
                    })
                if state["index"] is not None and state["pending"]:
                    for fragment in state["pending"]:
                        yield sse_event("content_block_delta", {
                            "type": "content_block_delta", "index": state["index"],
                            "delta": {"type": "input_json_delta", "partial_json": fragment},
                        })
                    state["pending"].clear()
    finally:
        await resp.aclose()

    if text_open:
        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index})
    for state in tool_state.values():
        if state["index"] is None:
            # Arguments streamed but a name never arrived. There is nothing
            # callable to hand Claude Code, so say so rather than emit a
            # nameless tool_use it will reject.
            log.warning("dropping a streamed tool call that never sent a function name")
            continue
        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": state["index"]})

    yield sse_event("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": FINISH_REASON_MAP.get(finish_reason, "end_turn"),
            "stop_sequence": None,
        },
        # Real usage when the provider volunteered it, otherwise the same
        # rough chars/4 estimate /v1/messages/count_tokens uses. This used to
        # report a count of SSE chunks, which is not a unit of anything.
        "usage": {"output_tokens": _output_tokens(upstream_usage, text_chars)},
    })
    yield sse_event("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backends": [b["name"] for b in BACKENDS],
        "agent_workspace": agent_tools.AGENT_WORKSPACE,
    }


@app.get("/api/usage")
async def backend_usage(refresh: int = 0):
    """How much of each backend's allowance is left, where the provider will
    say. Separate from /api/backends because it costs a round trip per backend
    -- listing backends stays instant, and the Backends tab asks for this
    after it has drawn the cards.

    Answers are cached (see usage.TTL_SECONDS); ?refresh=1 forces a re-read.
    """
    async with upstream_client() as client:
        rows = await usage.for_all(client, BACKENDS, force=bool(refresh))
    # `order` rides along so the Backends tab can say what the list actually
    # means without a second request: with least-used ordering the cards are no
    # longer the order the chain is tried in, and a caption claiming otherwise
    # is worse than none.
    return {"backends": rows, "ttl": usage.TTL_SECONDS, "order": BACKEND_ORDER}


@app.get("/status")
async def status():
    """Recent backend attempts -- watch failover happen in real time -- plus
    which backends are currently benched and for how long, so a backend that
    has quietly dropped out of the chain is visible rather than mysterious.

    `chain` answers the other question -- what order would the next request
    actually try these in -- since with least-used ordering that is no longer
    just backends.json read top to bottom. `demoted` is the tail of it: the
    backends whose allowance is spent, so they are tried last."""
    return {
        "recent_attempts": list(ATTEMPT_LOG),
        "cooling": {b["name"]: cooling_for(b["name"]) for b in BACKENDS
                    if cooling_for(b["name"])},
        "order": BACKEND_ORDER,
        "chain": [b["name"] for b in prioritise(list(BACKENDS), lambda b: b["name"])],
        "demoted": [b["name"] for b in BACKENDS
                    if REORDERS and usage.is_spent(b["name"])],
    }


# ---------------------------------------------------------------------------
# UI-facing API: manage backends.json and drive a manual test chat, without
# ever putting API keys through Claude Code or a chat transcript.
# ---------------------------------------------------------------------------

def public_backend(b: dict) -> dict:
    return {
        "name": b["name"],
        "base_url": b["base_url"],
        "big_model": b.get("big_model", ""),
        "small_model": b.get("small_model", ""),
        "has_key": bool(b.get("api_key")),
        # Seconds until this one is back in the chain, 0 when it's in it now.
        # Without this a backend simply vanishes from the failover order with
        # no way to see why.
        "cooling": cooling_for(b["name"]),
    }


@app.get("/api/backends")
async def list_backends():
    return {"backends": [public_backend(b) for b in BACKENDS]}


@app.post("/api/backends")
async def add_backend(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    base_url = (body.get("base_url") or "").strip().rstrip("/")
    if not name or not base_url:
        raise HTTPException(status_code=400, detail="name and base_url are required")
    if any(b["name"] == name for b in BACKENDS):
        raise HTTPException(status_code=409, detail=f"backend '{name}' already exists")

    big_model = (body.get("big_model") or "").strip()
    entry = {
        "name": name,
        "base_url": base_url,
        "api_key": body.get("api_key") or "",
        "big_model": big_model,
        "small_model": (body.get("small_model") or "").strip() or big_model,
    }
    BACKENDS.append(entry)
    save_backends()
    return public_backend(entry)


@app.patch("/api/backends/{name}")
async def update_backend(name: str, request: Request):
    """Partial update of one backend. Two conventions worth knowing:

    - A blank api_key means "leave the stored key alone", because the browser
      never has the key to send back -- an edit form with the key field left
      empty must not wipe it. To actually clear a key, send null.
    - 'name' may be changed here, so renaming a backend doesn't mean deleting
      it and losing the key with it.

    (This is the opposite of integrations.update, where blank clears. There the
    field is shown as configured/not; here it is edited alongside fields that
    are legitimately blank, so the safe reading of "" differs.)"""
    body = await request.json()
    for b in BACKENDS:
        if b["name"] != name:
            continue

        new_name = body.get("name")
        if isinstance(new_name, str) and new_name.strip() and new_name.strip() != name:
            new_name = new_name.strip()
            if any(other["name"] == new_name for other in BACKENDS):
                raise HTTPException(status_code=409, detail=f"backend '{new_name}' already exists")
            clear_cooldown(name)   # cooldown is keyed by name; don't strand it
            usage.forget(name)     # so is the usage cache
            b["name"] = new_name

        if "api_key" in body:
            if body["api_key"] is None:
                b["api_key"] = ""
            elif isinstance(body["api_key"], str) and body["api_key"].strip():
                b["api_key"] = body["api_key"].strip()
                # Someone replacing a key has just fixed the reason it was
                # benched; making them wait out the auth cooldown would be
                # absurd.
                clear_cooldown(b["name"])
            # A different key is a different account, so the cached figures
            # belong to something else now.
            usage.forget(b["name"])

        for field in ("base_url", "big_model", "small_model"):
            if field in body and isinstance(body[field], str):
                value = body[field].strip()
                b[field] = value.rstrip("/") if field == "base_url" else value
                if field == "base_url":
                    usage.forget(b["name"])

        save_backends()
        return public_backend(b)
    raise HTTPException(status_code=404, detail=f"backend '{name}' not found")


@app.delete("/api/backends/{name}")
async def delete_backend(name: str):
    for i, b in enumerate(BACKENDS):
        if b["name"] == name:
            del BACKENDS[i]
            clear_cooldown(name)
            usage.forget(name)
            save_backends()
            return {"ok": True}
    raise HTTPException(status_code=404, detail=f"backend '{name}' not found")


@app.post("/api/backends/reorder")
async def reorder_backends(request: Request):
    body = await request.json()
    order = body.get("order") or []
    by_name = {b["name"]: b for b in BACKENDS}
    if set(order) != set(by_name.keys()):
        raise HTTPException(status_code=400, detail="order must contain exactly the current backend names")
    BACKENDS[:] = [by_name[name] for name in order]
    save_backends()
    return {"backends": [public_backend(b) for b in BACKENDS]}


@app.get("/api/settings")
async def ui_settings():
    """What the UI needs to describe agent mode. The agent and sandbox
    toggles are the browser's call; lan_available is the one thing here that
    does bound a toggle, since reaching the local network is the operator's
    decision rather than the browser's."""
    return {
        "max_agent_iterations": MAX_AGENT_ITERATIONS,
        "round_ceiling": agent_tools.ROUND_CEILING,
        "tool_timeout": agent_tools.AGENT_TOOL_TIMEOUT,
        "bash_max_timeout": agent_tools.BASH_MAX_TIMEOUT,
        "ssh_timeout": ssh_tools.SSH_TIMEOUT,
        "ssh_max_timeout": ssh_tools.SSH_MAX_TIMEOUT,
        "default_workspace": agent_tools.AGENT_WORKSPACE,
        "clone_root": agent_tools.CLONE_ROOT,
        # Whether the operator allows the LAN switch to do anything. The UI
        # shows the switch either way, but locks it when this is false rather
        # than offering a toggle that silently can't take effect.
        "lan_available": web_tools.ALLOW_PRIVATE_CEILING,
        # No "available" flag: a group is never withheld from the UI. Whether
        # a credential is missing is the tool's own error to report when it
        # runs, not a reason to disable the box you'd tick to set it up.
        "groups": [
            {
                "name": group,
                "tools": [t["name"] for t in agent_tools.TOOLS if t["group"] == group],
            }
            for group in agent_tools.GROUPS
        ],
        "default_groups": agent_tools.DEFAULT_GROUPS,
        "integrations": integrations.public(),
    }


@app.get("/api/workdirs")
async def list_workdirs():
    return {"workdirs": [agent_tools.public_workdir(d) for d in agent_tools.WORKDIRS]}


@app.post("/api/workdirs")
async def add_workdir(request: Request):
    body = await request.json()
    raw = (body.get("path") or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="path is required")
    path = agent_tools.norm_path(raw)
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"'{path}' is not a directory on the proxy's machine")
    if any(d["path"] == path for d in agent_tools.WORKDIRS):
        raise HTTPException(status_code=409, detail=f"'{path}' is already a working directory")

    return agent_tools.public_workdir(
        agent_tools.register_workdir(path, (body.get("label") or "").strip()))


# ---------------------------------------------------------------------------
# SSH hosts -- the list of machines agent mode's ssh tools may connect to.
# Same shape as the working-directory registry: the server owns the list, and
# the model picks a name from it rather than inventing a destination.
# ---------------------------------------------------------------------------

@app.get("/api/ssh-hosts")
async def list_ssh_hosts():
    return {
        "hosts": [ssh_tools.public_host(h) for h in ssh_tools.HOSTS],
        # So the UI can explain an unusable group rather than showing an empty
        # list and leaving someone to wonder.
        "ssh_available": ssh_tools.ssh_available(),
    }


@app.post("/api/ssh-hosts")
async def add_ssh_host(request: Request):
    body = await request.json()
    # Same rule as a backend's API key: an absent or blank field keeps the
    # saved password, so editing a host's port doesn't mean retyping its
    # password into a form that has never displayed it. clear_password is the
    # only way to remove one, because "" has to mean "unchanged" for that to
    # work.
    password = body.get("password")
    if body.get("clear_password"):
        password = ""
    elif not password:
        password = None
    result = ssh_tools.register_host(
        name=(body.get("name") or "").strip(),
        host=(body.get("host") or "").strip(),
        user=(body.get("user") or "").strip(),
        port=body.get("port") or 22,
        key_path=(body.get("key_path") or "").strip(),
        label=(body.get("label") or "").strip(),
        password=password,
    )
    if isinstance(result, str):
        raise HTTPException(status_code=400, detail=result.removeprefix("[error] "))
    return ssh_tools.public_host(result)


@app.delete("/api/ssh-hosts")
async def delete_ssh_host(name: str):
    if not ssh_tools.remove_host(name):
        raise HTTPException(status_code=404, detail=f"no SSH host called '{name}'")
    return {"ok": True}


@app.post("/api/ssh-hosts/{name}/test")
async def test_ssh_host(name: str):
    """Connect and run `true`. This is what turns "the tool timed out" into
    "permission denied (publickey)" before a model is ever involved."""
    host = ssh_tools.find_host(name)
    if host is None:
        raise HTTPException(status_code=404, detail=f"no SSH host called '{name}'")
    started = time.monotonic()
    output = await ssh_tools.run_command(host, "true", timeout=20)
    elapsed = int((time.monotonic() - started) * 1000)
    ok = not output.startswith("[error]") and not output.startswith("[exit ")
    return {"ok": ok, "latency_ms": elapsed, "output": output}


@app.post("/api/approvals/{call_id}")
async def resolve_approval(call_id: str, request: Request):
    """Answer a tool call that a manual-mode run is parked on. Arrives on its
    own request because the run is mid-stream on another one."""
    body = await request.json()
    decision = (body.get("decision") or "").strip().lower()
    if decision not in approvals.DECISIONS:
        raise HTTPException(status_code=400,
                            detail=f"decision must be one of: {', '.join(approvals.DECISIONS)}")
    if not approvals.resolve(call_id, decision):
        # Usually a double click, or a run that was stopped while the card was
        # still on screen. Not an error worth a 500.
        raise HTTPException(status_code=404, detail="that call is no longer waiting for an answer")
    return {"ok": True, "decision": decision}


# Big enough for a log, a config file or a screenshot; small enough that a
# stray drag-and-drop of a video can't fill the container's disk.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@app.post("/api/upload")
async def upload_file(request: Request, name: str, path: str = ""):
    """Drop a file into one of the registered working directories.

    Takes the raw body rather than multipart/form-data on purpose: multipart
    would mean adding python-multipart as a dependency for one endpoint, and
    the browser can send a File object as a body directly."""
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="no file content received")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    # basename() alone handles "../../etc/passwd" and "C:\evil"; the explicit
    # checks cover what's left ("..", "", and a name that was all separators).
    safe = os.path.basename((name or "").replace("\\", "/")).strip()
    if not safe or safe in (".", ".."):
        raise HTTPException(status_code=400, detail=f"'{name}' is not a usable file name")

    target_dir = agent_tools.norm_path(path) if path else agent_tools.AGENT_WORKSPACE
    if not any(d["path"] == target_dir for d in agent_tools.WORKDIRS):
        raise HTTPException(status_code=400,
                            detail="uploads only go to a registered working directory")
    if not os.path.isdir(target_dir):
        raise HTTPException(status_code=400, detail=f"'{target_dir}' does not exist")

    dest = os.path.join(target_dir, safe)
    try:
        with open(dest, "wb") as f:
            f.write(raw)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"could not write the file: {e}")
    log.info("uploaded %s (%d bytes)", dest, len(raw))
    return {"path": dest, "name": safe, "bytes": len(raw), "directory": target_dir}


@app.delete("/api/workdirs")
async def delete_workdir(path: str):
    target = agent_tools.norm_path(path)
    for i, d in enumerate(agent_tools.WORKDIRS):
        if d["path"] == target:
            if d["default"]:
                raise HTTPException(status_code=400, detail="the default workspace can't be removed")
            del agent_tools.WORKDIRS[i]
            agent_tools.save_workdirs()
            return {"ok": True}
    raise HTTPException(status_code=404, detail=f"'{target}' is not a registered working directory")


# ---------------------------------------------------------------------------
# Integrations: tokens for GitHub and web search. Secrets go in, only
# "configured or not" comes back out.
# ---------------------------------------------------------------------------

@app.get("/api/integrations")
async def get_integrations():
    return integrations.public()


@app.put("/api/integrations")
async def put_integrations(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected an object")
    integrations.update(body)
    return integrations.public()


@app.post("/api/integrations/github/test")
async def test_github():
    """Verify the stored token and report who it belongs to, so the UI can
    show 'connected as <login>' instead of just 'a token is set'."""
    try:
        user = await github_api.whoami()
    except github_api.GitHubError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "login": user.get("login"),
        "name": user.get("name"),
        "url": user.get("html_url"),
        # So a token that works but can't reach private repositories says so
        # here, rather than at the moment a clone fails.
        "scopes": user.get("scopes", []),
        "scopes_reported": user.get("scopes_reported", False),
    }


# ---------------------------------------------------------------------------
# GitHub browsing for the UI itself (the agent has its own tools for this)
# ---------------------------------------------------------------------------

def _github_route(coro):
    """GitHubError is a user-facing configuration problem, not a 500."""
    async def wrapped():
        try:
            return await coro()
        except github_api.GitHubError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return wrapped()


def _login(item: dict) -> str:
    """GitHub sends "user": null for anything authored by a deleted account.
    One of those in a listing must not take the whole listing down."""
    user = item.get("user")
    return (user or {}).get("login", "") or "(unknown)"


def _ref(item: dict, side: str) -> str:
    return ((item.get(side) or {}).get("ref")) or ""


@app.get("/api/github/repos")
async def github_repos(query: str = "", limit: int = 30):
    repos = await _github_route(lambda: github_api.list_repos(limit=limit, query=query))
    return {"repos": [
        {
            "full_name": r.get("full_name"),
            "description": r.get("description") or "",
            "private": bool(r.get("private")),
            "default_branch": r.get("default_branch", ""),
            "pushed_at": r.get("pushed_at", ""),
            "url": r.get("html_url", ""),
        }
        for r in repos
    ]}


@app.get("/api/github/pulls")
async def github_pulls(repo: str, state: str = "open", limit: int = 20):
    pulls = await _github_route(lambda: github_api.list_pulls(repo, state=state, limit=limit))
    return {"pulls": [
        {
            "number": p.get("number"), "title": p.get("title") or "", "state": p.get("state") or "",
            "user": _login(p), "head": _ref(p, "head"), "base": _ref(p, "base"),
            "url": p.get("html_url") or "", "draft": bool(p.get("draft")),
        }
        for p in pulls
    ]}


@app.get("/api/github/issues")
async def github_issues(repo: str, state: str = "open", limit: int = 20):
    issues = await _github_route(lambda: github_api.list_issues(repo, state=state, limit=limit))
    return {"issues": [
        {
            "number": i.get("number"), "title": i.get("title") or "", "state": i.get("state") or "",
            "user": _login(i), "comments": i.get("comments", 0), "url": i.get("html_url") or "",
        }
        for i in issues
    ]}


@app.post("/api/github/clone")
async def github_clone(request: Request):
    """Clone a repo and register it as a working directory -- the same thing
    the agent's github_clone tool does, exposed for the UI's repo list."""
    body = await request.json()
    repo = (body.get("repo") or "").strip()
    if not repo:
        raise HTTPException(status_code=400, detail="repo is required")
    ctx = agent_tools.build_agent_context(None, True, ["github"])
    result = await agent_tools.run_github_clone(ctx, repo, (body.get("branch") or "").strip())
    if result.startswith("[error]"):
        raise HTTPException(status_code=400, detail=result.removeprefix("[error] "))
    return {"message": result, "workdirs": [agent_tools.public_workdir(d) for d in agent_tools.WORKDIRS]}


# ---------------------------------------------------------------------------
# Saved sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def get_sessions():
    return {"sessions": sessions.list_sessions()}


@app.post("/api/sessions")
async def post_session(request: Request):
    body = await request.json()
    return sessions.create(**{k: v for k, v in body.items() if k in ("title", *sessions.UPDATABLE)})


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@app.patch("/api/sessions/{session_id}")
async def patch_session(session_id: str, request: Request):
    body = await request.json()
    session = sessions.update(session_id, body)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return sessions.summary(session)


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


@app.post("/api/probe-models")
async def probe_models(request: Request):
    """Best-effort model discovery for the add-backend form. Never raises --
    the UI falls back to a free-text model field if this comes back empty."""
    body = await request.json()
    base_url = (body.get("base_url") or "").strip().rstrip("/")
    api_key = body.get("api_key") or ""

    # For a backend that already exists, the browser has no key to send -- the
    # API never gives one back. Naming the backend lets the server use the
    # stored key, which is what makes "Detect models" work when editing rather
    # than only when adding.
    saved_name = (body.get("name") or "").strip()
    if saved_name:
        backend = next((b for b in BACKENDS if b["name"] == saved_name), None)
        if backend is None:
            return {"models": [], "error": f"backend '{saved_name}' not found"}
        base_url = base_url or (backend.get("base_url") or "").rstrip("/")
        api_key = api_key or backend.get("api_key", "")

    if not base_url:
        return {"models": [], "error": "base_url is required"}

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with upstream_client(httpx.Timeout(15.0, connect=5.0)) as client:
            resp = await client.get(f"{base_url}/models", headers=headers)
    except httpx.HTTPError as e:
        return {"models": [], "error": str(e)}

    if resp.status_code >= 400:
        return {"models": [], "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    try:
        data = resp.json()
    except ValueError:
        return {"models": [], "error": "backend did not return JSON"}

    if isinstance(data, dict) and isinstance(data.get("data"), list):
        ids = [m.get("id") for m in data["data"] if isinstance(m, dict) and m.get("id")]
    elif isinstance(data, dict) and isinstance(data.get("models"), list):
        ids = [m.get("name") or m.get("model") for m in data["models"] if isinstance(m, dict)]
        ids = [i for i in ids if i]
    elif isinstance(data, list):
        ids = [m if isinstance(m, str) else m.get("id") for m in data]
        ids = [i for i in ids if i]
    else:
        ids = []

    return {"models": sorted(set(ids))}


def chat_fallback_chain(start_name: str, start_model: str):
    """Selected backend/model first, then every other configured backend
    (each using its own big_model) in their configured order -- so exhausting
    the chosen account transparently falls through the rest of the chain
    instead of dead-ending the conversation."""
    selected = next((b for b in BACKENDS if b["name"] == start_name), None)
    if selected is None:
        return None
    chain = [(selected, start_model or selected.get("big_model"))]
    chain += [(b, b.get("big_model")) for b in BACKENDS if b["name"] != start_name]
    return chain


def reorder_chain_start(chain, backend_name):
    """After a fallback succeeds, put that backend first so the next loop
    iteration doesn't waste a call re-trying the one that just failed."""
    idx = next((i for i, (b, _) in enumerate(chain) if b["name"] == backend_name), 0)
    return chain[idx:] + chain[:idx]


async def call_chain_non_stream(client: httpx.AsyncClient, chain, messages, tools=None):
    """Try each (backend, model) in order with a single non-streaming call.
    Returns (response_json, backend_name, model, error, hard_status).
    hard_status is set (and error is the raw body) only for a non-retryable
    upstream error -- callers should pass that straight through rather than
    treat it as the whole chain being exhausted."""
    last_error = None
    overflowed = False
    for backend, model in usable_backends(chain, lambda pair: pair[0]["name"]):
        payload = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        url = f"{backend['base_url']}/chat/completions"
        try:
            resp = await client.post(url, headers=backend_headers(backend), json=payload)
        except httpx.HTTPError as e:
            record_attempt(backend["name"], f"connection error: {e}")
            cool_down(backend["name"], BACKEND_COOLDOWN)
            last_error = str(e)
            continue
        if resp.status_code in RETRYABLE_STATUS:
            note_failure(backend["name"], resp.status_code, resp)
            last_error = f"HTTP {resp.status_code}"
            continue
        if resp.status_code >= 400:
            if is_context_overflow(resp.status_code, resp.text):
                record_attempt(backend["name"], "context overflow")
                overflowed = True
                last_error = f"HTTP {resp.status_code}"
                continue
            record_attempt(backend["name"], f"HTTP {resp.status_code} (non-retryable)")
            return None, None, None, resp.text, resp.status_code
        try:
            data = resp.json()
        except ValueError:
            # See try_backends_non_stream: a 200 with an unparseable body is
            # an unusable answer, not a reason to 500.
            record_attempt(backend["name"], "200 but the body was not JSON")
            last_error = f"{backend['name']} returned a non-JSON body"
            continue
        record_attempt(backend["name"], "ok")
        clear_cooldown(backend["name"])
        return data, backend["name"], model, None, None
    if overflowed:
        # OVERFLOW rather than a status, so the caller can tell the browser
        # what kind of failure this is and offer the one thing that fixes it.
        return None, None, None, CONTEXT_OVERFLOW_MESSAGE, OVERFLOW
    return None, None, None, last_error or "no backends configured", None


def sse_data(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# How long a paused tool call waits for someone to answer before it gives up,
# and how often to push a keepalive while it waits. The keepalive is an SSE
# comment: browsers and the UI's own parser both ignore it, but it stops an
# idle-connection timeout somewhere in the middle from killing a run that is
# simply waiting on a human.
APPROVAL_TIMEOUT = int(os.getenv("AGENT_APPROVAL_TIMEOUT", "600"))
APPROVAL_KEEPALIVE = 15
SSE_KEEPALIVE = ":\n\n"


async def agent_loop_gen(chain, messages, started_on, ctx: agent_tools.AgentContext, max_rounds: int):
    """Agentic mode for the web UI only: lets the model call bash/file tools
    scoped to the session's working directories on this container, looping
    tool-call -> execute -> feed result back -> repeat until it answers with
    plain text or MAX_AGENT_ITERATIONS is hit. Each round still gets the same
    account-failover treatment as plain chat."""
    yield sse_data({"agent_context": {"roots": ctx.roots, "sandbox": ctx.sandbox, "cwd": ctx.cwd,
                                      "lan": ctx.lan, "approval": ctx.approval}})
    # The system prompt is rebuilt per message rather than kept in history, so
    # toggling the sandbox or changing directories takes effect immediately.
    conv = [{"role": "system", "content": agent_tools.agent_system_prompt(ctx)}]
    conv += [m for m in messages if m.get("role") != "system"]
    # Everything from here on is what this run adds; it goes back to the
    # browser at the end so Continue resumes with the work, not just a summary.
    base_len = len(conv)
    tools = agent_tools.tools_for(ctx)
    # Signature of the call the model is currently repeating, and how many
    # times in a row. Only an unbroken run of identical calls is a stuck loop:
    # anything else in between (a write, a different read) can change what the
    # call would return, so the count starts over.
    repeated, repeats = "", 0
    # Set by "approve all remaining" and kept for the rest of this message.
    # Without it, manual mode on a hundred-round review is a hundred prompts,
    # which is a mode nobody would leave switched on.
    approved_all = False
    # Every tool-call id handed out during this message. Ids have to be unique
    # across the whole run, not just within a round: the browser keys its tool
    # cards by id, and approvals.create replaces an entry of the same name.
    seen_call_ids = set()
    async with upstream_client() as client:
        for _ in range(max_rounds):
            data, used_backend, used_model, error, hard_status = await call_chain_non_stream(
                client, chain, conv, tools=tools)
            if data is None:
                # Whatever tool work already happened is still worth keeping --
                # the next message shouldn't start over because one call failed.
                yield sse_data({"agent_history": conv[base_len:]})
                if hard_status == OVERFLOW:
                    # An agent run is the likeliest thing to outgrow a context
                    # window: every tool result is appended verbatim. Say so as
                    # itself, so the browser can offer to shorten instead of
                    # showing a provider's raw complaint.
                    yield sse_data({"error": error, "error_kind": OVERFLOW})
                    return
                prefix = f"HTTP {hard_status}" if hard_status else "all configured backends exhausted"
                yield sse_data({"error": f"{prefix}: {error}"})
                return
            if used_backend != started_on:
                yield sse_data({"claude_proxy_switch": used_backend, "claude_proxy_model": used_model})
                started_on = used_backend
                chain = reorder_chain_start(chain, used_backend)

            message = first_choice(data).get("message") or {}
            text = message.get("content") or ""
            tool_calls = [tc for tc in (message.get("tool_calls") or [])
                          if isinstance(tc, dict) and (tc.get("function") or {}).get("name")]

            if text:
                yield sse_data({"agent_text": text})

            if not tool_calls:
                if text:
                    conv.append({"role": "assistant", "content": text})
                yield sse_data({"agent_history": conv[base_len:]})
                yield sse_data({"agent_done": True})
                return

            if len(tool_calls) > agent_tools.MAX_CALLS_PER_ROUND:
                # max_rounds bounds rounds, not calls: a model that emits two
                # hundred calls in one round would otherwise run all of them
                # with no budget check in between.
                dropped = tool_calls[agent_tools.MAX_CALLS_PER_ROUND:]
                tool_calls = tool_calls[:agent_tools.MAX_CALLS_PER_ROUND]
            else:
                dropped = []

            # Give every call a unique id *before* the assistant message goes
            # into the history. The role:"tool" results below reference these
            # ids, and a result whose tool_call_id appears nowhere in the
            # preceding assistant message is a malformed conversation for every
            # round after this one -- which is what happened whenever a provider
            # omitted ids. Duplicates matter too: a provider that answers
            # "call_1" every round collides in the browser's card map and in the
            # approval registry.
            for tc in tool_calls + dropped:
                if not tc.get("id") or tc["id"] in seen_call_ids:
                    tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
                seen_call_ids.add(tc["id"])

            conv.append({"role": "assistant", "content": text or None,
                         "tool_calls": tool_calls + dropped})
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tool_args_dict(tc["function"].get("arguments"))
                call_id = tc["id"]
                yield sse_data({"agent_tool_call": {"id": call_id, "name": name, "arguments": args}})

                # Manual mode: stop here and wait for the browser before the
                # tool runs at all. The decision arrives on a separate request
                # (POST /api/approvals/{id}) and lands on the future this
                # registers, since a streaming response can't read from its
                # own request body mid-flight.
                refusal = ""
                if ctx.approval == "manual" and not approved_all:
                    future = approvals.create(call_id)
                    yield sse_data({"agent_approval_request": {
                        "id": call_id, "name": name, "arguments": args,
                        "group": agent_tools.group_of(name)}})
                    decision, waited = None, 0
                    try:
                        while waited < APPROVAL_TIMEOUT:
                            try:
                                # shield: wait_for cancels what it's waiting on
                                # when it times out, and the future has to
                                # survive that to still be answerable.
                                decision = await asyncio.wait_for(
                                    asyncio.shield(future), timeout=APPROVAL_KEEPALIVE)
                                break
                            except asyncio.TimeoutError:
                                waited += APPROVAL_KEEPALIVE
                                yield SSE_KEEPALIVE
                    finally:
                        # Covers the ordinary path, the timeout, and Stop being
                        # pressed while this is parked -- GeneratorExit lands on
                        # the yield above and still runs this.
                        approvals.cancel(call_id)

                    yield sse_data({"agent_approval_resolved": {
                        "id": call_id, "decision": decision or "timeout"}})
                    if decision == approvals.APPROVE_ALL:
                        approved_all = True
                    elif decision == approvals.DENY:
                        refusal = ("[denied] the user declined this call. Do not retry it -- "
                                   "carry on without it, or say what you needed it for.")
                    elif decision is None:
                        refusal = (f"[denied] nobody answered the approval request within "
                                   f"{APPROVAL_TIMEOUT}s, so the call was not run.")

                # Smaller models get stuck repeating one call forever. Answer
                # the third identical call in a row with a nudge instead of
                # running it again, so the round budget goes on actual work.
                signature = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
                repeats = repeats + 1 if signature == repeated else 1
                repeated = signature
                if refusal:
                    result = refusal
                elif repeats > 2:
                    result = (f"[error] this is the same call, with the same arguments, "
                              f"{repeats} times in a row with nothing in between. Repeating it "
                              "is not making progress -- do something different, or give your "
                              "final answer now.")
                else:
                    result = await agent_tools.execute_tool_call(ctx, name, args)
                yield sse_data({"agent_tool_result": {"id": call_id, "name": name, "output": result}})
                conv.append({"role": "tool", "tool_call_id": call_id, "content": result})
                # github_clone can add a working directory mid-run; tell the UI
                # so its directory list and scope chip stay honest.
                if name == "github_clone" and not result.startswith("[error]"):
                    yield sse_data({"agent_context": {"roots": ctx.roots, "sandbox": ctx.sandbox, "cwd": ctx.cwd,
                                                     "lan": ctx.lan, "approval": ctx.approval},
                                    "workdirs_changed": True})

            for tc in dropped:
                # Over the per-round cap, but still answered: every tool_call in
                # the assistant message needs a matching result or the next
                # round is a malformed conversation.
                refused = (f"[error] this round asked for {len(tool_calls) + len(dropped)} tool "
                           f"calls; only the first {agent_tools.MAX_CALLS_PER_ROUND} were run. "
                           "Ask for fewer at a time.")
                yield sse_data({"agent_tool_result": {"id": tc["id"],
                                                      "name": tc["function"]["name"],
                                                      "output": refused}})
                conv.append({"role": "tool", "tool_call_id": tc["id"], "content": refused})

        # Out of tool rounds. This is a pause, not a failure: ask once more
        # with no tools attached so the model reports where it got to, then
        # tell the UI it can offer to carry on. The conversation stays intact
        # either way -- a long review picks up exactly where it stopped.
        conv.append({
            "role": "user",
            "content": (f"You have used all {max_rounds} tool calls allowed for this message. "
                        "Do not request any more. Summarise what you have found so far and state "
                        "exactly what you were in the middle of, so you can pick it up next turn."),
        })
        data, _, _, _, _ = await call_chain_non_stream(client, chain, conv)
        final = ""
        if data is not None:
            final = ((first_choice(data).get("message") or {}).get("content") or "").strip()
        if final:
            conv.append({"role": "assistant", "content": final})
            yield sse_data({"agent_text": final})
        yield sse_data({"agent_history": conv[base_len:]})
        yield sse_data({"agent_limit": max_rounds})
        yield sse_data({"agent_done": True})


@app.post("/api/chat")
async def ui_chat(request: Request):
    """Manual test-chat UI. Starts with the backend/model the user picked;
    on 429/401/403/5xx/connection error, transparently falls through the
    rest of the configured backends (same conversation history) rather than
    erroring out -- this is what lets one account's quota running out just
    move to the next account instead of interrupting the chat."""
    body = await request.json()
    chain = chat_fallback_chain(body.get("backend"), body.get("model"))
    if chain is None:
        raise HTTPException(status_code=404, detail=f"backend '{body.get('backend')}' not found")

    messages = body.get("messages", [])
    stream = bool(body.get("stream", True))
    started_on = chain[0][0]["name"]

    if body.get("agent"):
        ctx = agent_tools.build_agent_context(
            body.get("workdirs"), body.get("sandbox", True), body.get("groups"),
            body.get("lan", False), body.get("approval", "skip"))
        max_rounds = agent_tools.clamp_rounds(body.get("max_rounds"))
        return StreamingResponse(agent_loop_gen(chain, messages, started_on, ctx, max_rounds),
                                 media_type="text/event-stream")

    if stream:
        client = upstream_client()

        async def gen():
            overflowed = False
            try:
                for backend, model in usable_backends(chain, lambda pair: pair[0]["name"]):
                    payload = {"model": model, "messages": messages, "stream": True}
                    url = f"{backend['base_url']}/chat/completions"
                    req = client.build_request("POST", url, headers=backend_headers(backend),
                                                json=payload, timeout=HTTP_TIMEOUT)
                    try:
                        resp = await client.send(req, stream=True)
                    except httpx.HTTPError as e:
                        record_attempt(backend["name"], f"connection error: {e}")
                        cool_down(backend["name"], BACKEND_COOLDOWN)
                        continue
                    if resp.status_code in RETRYABLE_STATUS:
                        note_failure(backend["name"], resp.status_code, resp)
                        await resp.aclose()
                        continue
                    if resp.status_code >= 400:
                        err = (await resp.aread()).decode("utf-8", "replace")
                        await resp.aclose()
                        if is_context_overflow(resp.status_code, err):
                            record_attempt(backend["name"], "context overflow")
                            overflowed = True
                            continue
                        yield f"data: {json.dumps({'error': err})}\n\n"
                        return
                    record_attempt(backend["name"], "ok")
                    clear_cooldown(backend["name"])
                    if backend["name"] != started_on:
                        yield f"data: {json.dumps({'claude_proxy_switch': backend['name'], 'claude_proxy_model': model})}\n\n"
                    try:
                        async for chunk in resp.aiter_raw():
                            yield chunk
                    finally:
                        await resp.aclose()
                    return
                if overflowed:
                    yield sse_data({"error": CONTEXT_OVERFLOW_MESSAGE, "error_kind": OVERFLOW})
                else:
                    yield f"data: {json.dumps({'error': 'all configured backends exhausted'})}\n\n"
            finally:
                await client.aclose()

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with upstream_client() as client:
        data, used_backend, used_model, error, hard_status = await call_chain_non_stream(client, chain, messages)
        if data is None:
            if hard_status == OVERFLOW:
                return JSONResponse(status_code=413,
                                    content={"error": error, "error_kind": OVERFLOW})
            if hard_status:
                return JSONResponse(status_code=hard_status, content={"error": error})
            raise HTTPException(status_code=502, detail=f"all configured backends exhausted: {error}")
        if used_backend != started_on:
            data["claude_proxy_switch"] = used_backend
            data["claude_proxy_model"] = used_model
        return JSONResponse(content=data)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def ui_index():
    return FileResponse("static/index.html")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """Claude Code asks this to decide when to compact. It used to count only
    text blocks and ignore body["tools"] entirely, so an agentic turn -- ~15
    tool schemas plus tool_result payloads -- reported a few hundred tokens for
    a thirty-thousand-token request, and Claude Code walked straight into the
    context window it was trying to stay inside."""
    body = await request.json()
    return {"input_tokens": max(1, estimate_tokens(
        body.get("system"), body.get("messages"), body.get("tools")))}


async def try_backends_non_stream(client: httpx.AsyncClient, body: dict):
    last_error = None
    overflow_resp = None
    # snapshot + cooldown filter -- see find_working_stream
    for backend in usable_backends(list(BACKENDS), lambda b: b["name"]):
        payload = build_openai_payload(body, backend)
        payload["stream"] = False
        url = f"{backend['base_url']}/chat/completions"
        try:
            resp = await client.post(url, headers=backend_headers(backend), json=payload)
        except httpx.HTTPError as e:
            record_attempt(backend["name"], f"connection error: {e}")
            cool_down(backend["name"], BACKEND_COOLDOWN)
            last_error = str(e)
            continue
        if resp.status_code in RETRYABLE_STATUS:
            note_failure(backend["name"], resp.status_code, resp)
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            continue
        if resp.status_code >= 400:
            if is_context_overflow(resp.status_code, resp.text):
                # A backend with a bigger window may still take it.
                record_attempt(backend["name"], "context overflow")
                overflow_resp = resp
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                continue
            record_attempt(backend["name"], f"HTTP {resp.status_code} (non-retryable)")
            return None, resp
        try:
            data = resp.json()
        except ValueError:
            # A 200 whose body isn't JSON: a gateway's HTML error page, or a
            # backend answering SSE to a non-streaming request. Treat it like
            # any other unusable answer and try the next one.
            record_attempt(backend["name"], "200 but the body was not JSON")
            last_error = f"{backend['name']} returned a non-JSON body"
            continue
        record_attempt(backend["name"], "ok")
        clear_cooldown(backend["name"])
        return data, None
    if overflow_resp is not None:
        # Hand the overflow back as itself rather than as "all backends
        # exhausted": Claude Code knows what to do with a context error (it
        # compacts and retries), and can do nothing with a 502.
        return None, overflow_resp
    raise HTTPException(status_code=502, detail=f"all backends exhausted: {last_error}")


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    claude_model = body.get("model", "")

    if body.get("stream"):
        client = upstream_client()

        async def gen():
            try:
                async for event in stream_openai_as_anthropic(client, body, claude_model):
                    yield event
            finally:
                await client.aclose()

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with upstream_client() as client:
        nim_data, error_resp = await try_backends_non_stream(client, body)
        if error_resp is not None:
            return JSONResponse(status_code=error_resp.status_code, content={
                "type": "error",
                "error": {"type": "api_error", "message": error_resp.text},
            })

    return JSONResponse(content=openai_response_to_anthropic(nim_data, claude_model))
