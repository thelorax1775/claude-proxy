"""Shared test setup.

Both `main` and `agent_tools` do real work at import time: `agent_tools`
creates AGENT_WORKSPACE on disk, `main` loads backends.json, `integrations`
reads the token store. So every state path has to be redirected *before* the
first import, which is what `_prepare_env()` does at module scope here --
conftest.py is imported before any test module, so this runs first.

The upshot is that running the suite never reads or writes the developer's
real backends.json, sessions/ or workspace/.
"""

import json
import os
import sys
import tempfile

import httpx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Two backends, in failover order, so ordering assertions have something to
# say. Keys are long enough to be redactable (integrations.MIN_REDACTABLE).
TEST_BACKENDS = [
    {
        "name": "primary",
        "base_url": "https://primary.test/v1",
        "api_key": "sk-primary-000000000000",
        "big_model": "primary-big",
        "small_model": "primary-small",
    },
    {
        "name": "secondary",
        "base_url": "https://secondary.test/v1",
        "api_key": "sk-secondary-00000000000",
        "big_model": "secondary-big",
        "small_model": "secondary-small",
    },
]


def _prepare_env():
    state = tempfile.mkdtemp(prefix="claude-proxy-tests-")
    backends_path = os.path.join(state, "backends.json")
    with open(backends_path, "w") as f:
        json.dump(TEST_BACKENDS, f)
    os.environ.update({
        "BACKENDS_CONFIG": backends_path,
        "AGENT_WORKSPACE": os.path.join(state, "workspace"),
        "AGENT_DIRS_CONFIG": os.path.join(state, "workdirs.json"),
        "INTEGRATIONS_CONFIG": os.path.join(state, "integrations.json"),
        "SESSIONS_DIR": os.path.join(state, "sessions"),
    })
    return state


STATE_DIR = _prepare_env()

import main  # noqa: E402  -- must follow _prepare_env()


@pytest.fixture
def app():
    """The FastAPI app with all module-global state reset, so a test that adds
    a backend, benches one, or fills the attempt log can't leak into the next
    one. Cooldowns especially: a 429 in one test would otherwise take that
    backend out of the chain for the next sixty seconds of the suite."""
    def reset():
        main.BACKENDS[:] = [dict(b) for b in TEST_BACKENDS]
        main.ATTEMPT_LOG.clear()
        main._COOLDOWN.clear()

    reset()
    yield main
    reset()


def sse(*chunks) -> str:
    """An OpenAI-style SSE body: one `data:` frame per chunk, then [DONE]."""
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


def mock_client(handler) -> httpx.AsyncClient:
    """An AsyncClient whose requests are answered by `handler(request)`
    instead of the network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def by_host(responses: dict, default=None):
    """Build a MockTransport handler that answers per backend hostname, e.g.
    {"primary.test": httpx.Response(429), "secondary.test": httpx.Response(200, ...)}.
    Records each request it saw on the returned handler's `.seen` list."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        for host, response in responses.items():
            if request.url.host == host:
                return response() if callable(response) else response
        if default is not None:
            return default() if callable(default) else default
        return httpx.Response(500, text=f"no stub for {request.url.host}")

    handler.seen = seen
    return handler


def anthropic_events(raw: str) -> list:
    """Parse an Anthropic SSE body into [(event_name, payload), ...]."""
    events = []
    for frame in raw.split("\n\n"):
        name, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if name is not None:
            events.append((name, data))
    return events
