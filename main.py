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
import time
import uuid

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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


def save_backends():
    tmp = f"{BACKENDS_PATH}.tmp"
    with open(tmp, "w") as f:
        json.dump(BACKENDS, f, indent=2)
    os.replace(tmp, BACKENDS_PATH)

# Optional: require the client (Claude Code) to send this exact key so the
# proxy isn't wide open to anything that can reach the port.
PROXY_SHARED_KEY = os.getenv("PROXY_SHARED_KEY", "")

HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)

# Status codes worth failing over on. 401/403 included because a dead/expired
# free-tier key should not take the whole proxy down -- skip to the next one.
RETRYABLE_STATUS = {401, 403, 429, 500, 502, 503, 504}


def record_attempt(backend_name: str, outcome: str):
    entry = {"time": time.time(), "backend": backend_name, "outcome": outcome}
    ATTEMPT_LOG.append(entry)
    log.info("backend=%s outcome=%s", backend_name, outcome)


# ---------------------------------------------------------------------------
# Agent tools -- only reachable from the web UI's chat when explicitly
# opted into (never wired into /v1/messages -- Claude Code's own tools are
# unaffected). Everything is confined to AGENT_WORKSPACE. The real safety
# boundary is this container: it runs as the unprivileged `claude-proxy`
# system user, but a shell command isn't sandboxed beyond that -- treat
# this container as disposable, not as something holding anything you'd
# mind an LLM having shell access to.
# ---------------------------------------------------------------------------

AGENT_WORKSPACE = os.path.abspath(os.getenv("AGENT_WORKSPACE", "workspace"))
os.makedirs(AGENT_WORKSPACE, exist_ok=True)

AGENT_TOOL_TIMEOUT = 30
AGENT_OUTPUT_LIMIT = 8000
MAX_AGENT_ITERATIONS = 8

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the sandboxed workspace directory on this container. Output (stdout+stderr combined) is truncated if large.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file's contents from the sandboxed workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the workspace root"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or overwrite) a text file in the sandboxed workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories inside the sandboxed workspace (or a subdirectory of it).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Subdirectory relative to workspace root; defaults to '.'"}},
            },
        },
    },
]


def _truncate(text: str) -> str:
    if len(text) <= AGENT_OUTPUT_LIMIT:
        return text
    return text[:AGENT_OUTPUT_LIMIT] + f"\n...[truncated, {len(text)} bytes total]"


def resolve_workspace_path(rel_path: str) -> str:
    target = os.path.abspath(os.path.join(AGENT_WORKSPACE, rel_path or "."))
    if target != AGENT_WORKSPACE and not target.startswith(AGENT_WORKSPACE + os.sep):
        raise ValueError("path escapes the workspace directory")
    return target


async def run_bash(command: str) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            command, cwd=AGENT_WORKSPACE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=AGENT_TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            return f"[error] command timed out after {AGENT_TOOL_TIMEOUT}s"
        return _truncate(out.decode("utf-8", "replace"))
    except Exception as e:
        return f"[error] {e}"


def run_read_file(path: str) -> str:
    try:
        target = resolve_workspace_path(path)
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            return _truncate(f.read())
    except Exception as e:
        return f"[error] {e}"


def run_write_file(path: str, content: str) -> str:
    try:
        target = resolve_workspace_path(path)
        os.makedirs(os.path.dirname(target) or AGENT_WORKSPACE, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"[error] {e}"


def run_list_dir(path: str = ".") -> str:
    try:
        target = resolve_workspace_path(path)
        entries = sorted(os.listdir(target))
        return "\n".join(entries) if entries else "(empty)"
    except Exception as e:
        return f"[error] {e}"


async def execute_tool_call(name: str, args: dict) -> str:
    if name == "bash":
        return await run_bash(args.get("command", ""))
    if name == "read_file":
        return run_read_file(args.get("path", ""))
    if name == "write_file":
        return run_write_file(args.get("path", ""), args.get("content", ""))
    if name == "list_dir":
        return run_list_dir(args.get("path", "."))
    return f"[error] unknown tool '{name}'"


def pick_model(claude_model: str, backend: dict) -> str:
    m = (claude_model or "").lower()
    if "haiku" in m:
        return backend["small_model"]
    return backend["big_model"]


def check_auth(x_api_key: str | None, authorization: str | None):
    if not PROXY_SHARED_KEY:
        return
    token = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
    if token != PROXY_SHARED_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


async def require_proxy_key(x_proxy_key: str | None = Header(default=None, alias="X-Proxy-Key")):
    """Gates the web UI's /api/* routes. Separate from check_auth (which
    gates Claude Code's /v1/messages) because the browser can't send
    Anthropic-style headers -- same PROXY_SHARED_KEY value, different header.
    A no-op when PROXY_SHARED_KEY isn't set (e.g. localhost-only deployments)."""
    if not PROXY_SHARED_KEY:
        return
    if x_proxy_key != PROXY_SHARED_KEY:
        raise HTTPException(status_code=401, detail="invalid proxy key")


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
    choice = nim_data["choices"][0]
    message = choice.get("message", {})
    content_blocks = []

    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in message.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": tc["function"]["name"],
            "input": args,
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
    for backend in BACKENDS:
        payload = build_openai_payload(body, backend)
        payload["stream"] = True
        url = f"{backend['base_url']}/chat/completions"
        req = client.build_request("POST", url, headers=backend_headers(backend),
                                    json=payload, timeout=HTTP_TIMEOUT)
        try:
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as e:
            record_attempt(backend["name"], f"connection error: {e}")
            last_error = str(e)
            continue
        if resp.status_code in RETRYABLE_STATUS:
            body_text = (await resp.aread()).decode("utf-8", "replace")
            await resp.aclose()
            record_attempt(backend["name"], f"HTTP {resp.status_code}")
            last_error = f"HTTP {resp.status_code}: {body_text[:300]}"
            continue
        if resp.status_code >= 400:
            body_text = (await resp.aread()).decode("utf-8", "replace")
            await resp.aclose()
            record_attempt(backend["name"], f"HTTP {resp.status_code} (non-retryable)")
            return None, None, f"HTTP {resp.status_code}: {body_text[:300]}"
        record_attempt(backend["name"], "ok")
        return resp, backend, None
    return None, None, last_error or "no backends configured"


async def stream_openai_as_anthropic(client: httpx.AsyncClient, body: dict, claude_model: str):
    resp, backend, error = await find_working_stream(client, body)
    if resp is None:
        yield sse_event("error", {
            "type": "error",
            "error": {"type": "api_error", "message": f"all backends exhausted: {error}"},
        })
        return

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": claude_model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    next_index = 0
    text_index = None
    text_open = False
    tool_state = {}  # openai tc index -> {"index": anthropic_index, "opened": bool}
    finish_reason = "stop"
    completion_tokens_estimate = 0

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

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]

            if delta.get("content"):
                if not text_open:
                    text_index = next_index
                    next_index += 1
                    text_open = True
                    yield sse_event("content_block_start", {
                        "type": "content_block_start", "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                completion_tokens_estimate += 1
                yield sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": text_index,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                })

            for tc in delta.get("tool_calls") or []:
                idx = tc["index"]
                if idx not in tool_state:
                    if text_open:
                        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index})
                        text_open = False
                    a_index = next_index
                    next_index += 1
                    tool_state[idx] = {"index": a_index}
                    yield sse_event("content_block_start", {
                        "type": "content_block_start", "index": a_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": {},
                        },
                    })
                args_fragment = tc.get("function", {}).get("arguments")
                if args_fragment:
                    completion_tokens_estimate += 1
                    yield sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": tool_state[idx]["index"],
                        "delta": {"type": "input_json_delta", "partial_json": args_fragment},
                    })
    finally:
        await resp.aclose()

    if text_open:
        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index})
    for state in tool_state.values():
        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": state["index"]})

    yield sse_event("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": FINISH_REASON_MAP.get(finish_reason, "end_turn"),
            "stop_sequence": None,
        },
        "usage": {"output_tokens": completion_tokens_estimate},
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
    }


@app.get("/status")
async def status():
    """Recent backend attempts -- watch failover happen in real time."""
    return {"recent_attempts": list(ATTEMPT_LOG)}


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
    }


@app.get("/api/backends", dependencies=[Depends(require_proxy_key)])
async def list_backends():
    return {"backends": [public_backend(b) for b in BACKENDS]}


@app.post("/api/backends", dependencies=[Depends(require_proxy_key)])
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


@app.patch("/api/backends/{name}", dependencies=[Depends(require_proxy_key)])
async def update_backend(name: str, request: Request):
    body = await request.json()
    for b in BACKENDS:
        if b["name"] == name:
            for field in ("base_url", "api_key", "big_model", "small_model"):
                if field in body and body[field] is not None:
                    b[field] = body[field]
            save_backends()
            return public_backend(b)
    raise HTTPException(status_code=404, detail=f"backend '{name}' not found")


@app.delete("/api/backends/{name}", dependencies=[Depends(require_proxy_key)])
async def delete_backend(name: str):
    for i, b in enumerate(BACKENDS):
        if b["name"] == name:
            del BACKENDS[i]
            save_backends()
            return {"ok": True}
    raise HTTPException(status_code=404, detail=f"backend '{name}' not found")


@app.post("/api/backends/reorder", dependencies=[Depends(require_proxy_key)])
async def reorder_backends(request: Request):
    body = await request.json()
    order = body.get("order") or []
    by_name = {b["name"]: b for b in BACKENDS}
    if set(order) != set(by_name.keys()):
        raise HTTPException(status_code=400, detail="order must contain exactly the current backend names")
    BACKENDS[:] = [by_name[name] for name in order]
    save_backends()
    return {"backends": [public_backend(b) for b in BACKENDS]}


@app.post("/api/probe-models", dependencies=[Depends(require_proxy_key)])
async def probe_models(request: Request):
    """Best-effort model discovery for the add-backend form. Never raises --
    the UI falls back to a free-text model field if this comes back empty."""
    body = await request.json()
    base_url = (body.get("base_url") or "").strip().rstrip("/")
    api_key = body.get("api_key") or ""
    if not base_url:
        return {"models": [], "error": "base_url is required"}

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
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
    for backend, model in chain:
        payload = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        url = f"{backend['base_url']}/chat/completions"
        try:
            resp = await client.post(url, headers=backend_headers(backend), json=payload)
        except httpx.HTTPError as e:
            record_attempt(backend["name"], f"connection error: {e}")
            last_error = str(e)
            continue
        if resp.status_code in RETRYABLE_STATUS:
            record_attempt(backend["name"], f"HTTP {resp.status_code}")
            last_error = f"HTTP {resp.status_code}"
            continue
        if resp.status_code >= 400:
            record_attempt(backend["name"], f"HTTP {resp.status_code} (non-retryable)")
            return None, None, None, resp.text, resp.status_code
        record_attempt(backend["name"], "ok")
        return resp.json(), backend["name"], model, None, None
    return None, None, None, last_error or "no backends configured", None


def sse_data(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def agent_loop_gen(chain, messages, started_on):
    """Agentic mode for the web UI only: lets the model call bash/file tools
    scoped to AGENT_WORKSPACE on this container, looping tool-call ->
    execute -> feed result back -> repeat until it answers with plain text
    or MAX_AGENT_ITERATIONS is hit. Each round still gets the same
    account-failover treatment as plain chat."""
    conv = list(messages)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for _ in range(MAX_AGENT_ITERATIONS):
            data, used_backend, used_model, error, hard_status = await call_chain_non_stream(
                client, chain, conv, tools=AGENT_TOOLS)
            if data is None:
                prefix = f"HTTP {hard_status}" if hard_status else "all configured backends exhausted"
                yield sse_data({"error": f"{prefix}: {error}"})
                return
            if used_backend != started_on:
                yield sse_data({"claude_proxy_switch": used_backend, "claude_proxy_model": used_model})
                started_on = used_backend
                chain = reorder_chain_start(chain, used_backend)

            message = data["choices"][0]["message"]
            text = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            if text:
                yield sse_data({"agent_text": text})

            if not tool_calls:
                yield sse_data({"agent_done": True})
                return

            conv.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                call_id = tc.get("id", f"call_{uuid.uuid4().hex[:12]}")
                yield sse_data({"agent_tool_call": {"id": call_id, "name": name, "arguments": args}})
                result = await execute_tool_call(name, args)
                yield sse_data({"agent_tool_result": {"id": call_id, "name": name, "output": result}})
                conv.append({"role": "tool", "tool_call_id": call_id, "content": result})

        yield sse_data({"error": f"stopped after {MAX_AGENT_ITERATIONS} tool iterations without a final answer"})


@app.post("/api/chat", dependencies=[Depends(require_proxy_key)])
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
        return StreamingResponse(agent_loop_gen(chain, messages, started_on), media_type="text/event-stream")

    if stream:
        client = httpx.AsyncClient()

        async def gen():
            try:
                for backend, model in chain:
                    payload = {"model": model, "messages": messages, "stream": True}
                    url = f"{backend['base_url']}/chat/completions"
                    req = client.build_request("POST", url, headers=backend_headers(backend),
                                                json=payload, timeout=HTTP_TIMEOUT)
                    try:
                        resp = await client.send(req, stream=True)
                    except httpx.HTTPError as e:
                        record_attempt(backend["name"], f"connection error: {e}")
                        continue
                    if resp.status_code in RETRYABLE_STATUS:
                        await resp.aclose()
                        record_attempt(backend["name"], f"HTTP {resp.status_code}")
                        continue
                    if resp.status_code >= 400:
                        err = (await resp.aread()).decode("utf-8", "replace")
                        await resp.aclose()
                        yield f"data: {json.dumps({'error': err})}\n\n"
                        return
                    record_attempt(backend["name"], "ok")
                    if backend["name"] != started_on:
                        yield f"data: {json.dumps({'claude_proxy_switch': backend['name'], 'claude_proxy_model': model})}\n\n"
                    try:
                        async for chunk in resp.aiter_raw():
                            yield chunk
                    finally:
                        await resp.aclose()
                    return
                yield f"data: {json.dumps({'error': 'all configured backends exhausted'})}\n\n"
            finally:
                await client.aclose()

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        data, used_backend, used_model, error, hard_status = await call_chain_non_stream(client, chain, messages)
        if data is None:
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
    body = await request.json()
    text = system_to_text(body.get("system")) or ""
    for msg in body.get("messages", []):
        c = msg.get("content")
        if isinstance(c, str):
            text += c
        else:
            for block in c or []:
                if block.get("type") == "text":
                    text += block.get("text", "")
    # Rough estimate only -- good enough for Claude Code's context accounting.
    return {"input_tokens": max(1, len(text) // 4)}


async def try_backends_non_stream(client: httpx.AsyncClient, body: dict):
    last_error = None
    for backend in BACKENDS:
        payload = build_openai_payload(body, backend)
        payload["stream"] = False
        url = f"{backend['base_url']}/chat/completions"
        try:
            resp = await client.post(url, headers=backend_headers(backend), json=payload)
        except httpx.HTTPError as e:
            record_attempt(backend["name"], f"connection error: {e}")
            last_error = str(e)
            continue
        if resp.status_code in RETRYABLE_STATUS:
            record_attempt(backend["name"], f"HTTP {resp.status_code}")
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            continue
        if resp.status_code >= 400:
            record_attempt(backend["name"], f"HTTP {resp.status_code} (non-retryable)")
            return None, resp
        record_attempt(backend["name"], "ok")
        return resp.json(), None
    raise HTTPException(status_code=502, detail=f"all backends exhausted: {last_error}")


@app.post("/v1/messages")
async def messages(request: Request, x_api_key: str | None = Header(default=None),
                    authorization: str | None = Header(default=None)):
    check_auth(x_api_key, authorization)

    body = await request.json()
    claude_model = body.get("model", "")

    if body.get("stream"):
        client = httpx.AsyncClient()

        async def gen():
            try:
                async for event in stream_openai_as_anthropic(client, body, claude_model):
                    yield event
            finally:
                await client.aclose()

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        nim_data, error_resp = await try_backends_non_stream(client, body)
        if error_resp is not None:
            return JSONResponse(status_code=error_resp.status_code, content={
                "type": "error",
                "error": {"type": "api_error", "message": error_resp.text},
            })

    return JSONResponse(content=openai_response_to_anthropic(nim_data, claude_model))
