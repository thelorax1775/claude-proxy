"""Context overflow.

A conversation that outgrows a model's window comes back as a 400, which is
otherwise "your request is malformed" and not worth retrying. Context is the
exception: backends differ in window size, so it's worth trying the next one,
and if none will take it the answer has to be something the UI can act on
rather than a provider's raw complaint.
"""

import httpx
import pytest

import main
from conftest import by_host, sse


def completion(text="ok"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text},
                                                  "finish_reason": "stop"}]})


def overflow(body, status=400):
    return httpx.Response(status, text=body)


def hosts_tried(handler):
    return [r.url.host for r in handler.seen]


# Real messages, as each provider actually words it.
REAL_OVERFLOW_BODIES = [
    # OpenAI / most compatible gateways
    '{"error":{"message":"This model\'s maximum context length is 8192 tokens. '
    'However, your messages resulted in 9001 tokens.","code":"context_length_exceeded"}}',
    # Groq
    '{"error":{"message":"Please reduce the length of the messages or completion."}}',
    # Anthropic-style wording, seen through gateways
    '{"error":{"message":"prompt is too long: 250000 tokens > 200000 maximum"}}',
    # Cerebras
    '{"error":{"code":"context_length_exceeded","message":"context length exceeded"}}',
    # Ollama
    '{"error":"input is too long for this model\'s context window"}',
    # Mistral-style
    '{"message":"Too many tokens in prompt"}',
]


@pytest.mark.parametrize("body", REAL_OVERFLOW_BODIES)
def test_real_provider_wordings_are_recognised(body):
    assert main.is_context_overflow(400, body) is True


@pytest.mark.parametrize("status", [400, 413, 422])
def test_the_statuses_providers_actually_use(status):
    assert main.is_context_overflow(status, "maximum context length is 8192 tokens") is True


def test_other_400s_are_not_mistaken_for_it():
    """A malformed request must still be a hard failure -- retrying it on
    every backend is slower and no more likely to work."""
    assert main.is_context_overflow(400, '{"error":"model not found"}') is False
    assert main.is_context_overflow(400, '{"error":"invalid api key"}') is False
    assert main.is_context_overflow(400, '{"error":"tool_choice is not supported"}') is False
    assert main.is_context_overflow(400, "") is False
    assert main.is_context_overflow(400, None) is False


def test_other_statuses_are_never_overflow():
    body = "maximum context length is 8192 tokens"
    for status in (200, 401, 429, 500, 503):
        assert main.is_context_overflow(status, body) is False


# ---------------------------------------------------------------------------
# Failing over on it
# ---------------------------------------------------------------------------

async def test_overflow_falls_through_to_a_bigger_backend(app):
    """The whole point of the chain is that the backends differ, and context
    window is one of the things they differ in."""
    handler = by_host({
        "primary.test": overflow('{"error":"maximum context length is 8192 tokens"}'),
        "secondary.test": completion("the big one took it"),
    })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        data, err = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data["choices"][0]["message"]["content"] == "the big one took it"
    assert hosts_tried(handler) == ["primary.test", "secondary.test"]
    assert any(e["outcome"] == "context overflow" for e in main.ATTEMPT_LOG)


async def test_a_plain_400_still_stops_at_the_first_backend(app):
    handler = by_host({"primary.test": overflow('{"error":"model not found"}')})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        data, error_resp = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data is None
    assert error_resp.status_code == 400
    assert hosts_tried(handler) == ["primary.test"]


async def test_overflow_does_not_bench_the_backend(app):
    """It isn't the backend's fault and the next message may well be short --
    cooling it down would take a working account out of the chain for nothing."""
    handler = by_host({
        "primary.test": overflow('{"error":"maximum context length is 8192 tokens"}'),
        "secondary.test": completion(),
    })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert main.cooling_for("primary") == 0


async def test_claude_code_gets_the_overflow_back_as_itself(app):
    """When every backend refuses it, /v1/messages returns the provider's own
    context error rather than a 502. Claude Code knows how to react to one
    (it compacts and retries) and can do nothing with the other."""
    body = '{"error":"maximum context length is 8192 tokens"}'
    handler = by_host({"primary.test": overflow(body), "secondary.test": overflow(body)})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        data, error_resp = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data is None
    assert error_resp.status_code == 400
    assert "context length" in error_resp.text
    assert hosts_tried(handler) == ["primary.test", "secondary.test"]


# ---------------------------------------------------------------------------
# What the web UI is told
# ---------------------------------------------------------------------------

async def test_the_ui_chain_reports_overflow_as_its_own_kind(app):
    body = '{"error":"maximum context length is 8192 tokens"}'
    handler = by_host({"primary.test": overflow(body), "secondary.test": overflow(body)})
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        data, backend, model, error, hard = await main.call_chain_non_stream(
            client, chain, [{"role": "user", "content": "hi"}])
    assert data is None
    assert hard == main.OVERFLOW
    assert error == main.CONTEXT_OVERFLOW_MESSAGE


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_non_streaming_chat_returns_a_typed_413(client):
    body = '{"error":"maximum context length is 8192 tokens"}'
    handler = by_host({"primary.test": overflow(body), "secondary.test": overflow(body)})
    main.UPSTREAM_TRANSPORT = httpx.MockTransport(handler)
    try:
        resp = await client.post("/api/chat", json={
            "backend": "primary", "model": "primary-big", "stream": False,
            "messages": [{"role": "user", "content": "hi"}]})
    finally:
        main.UPSTREAM_TRANSPORT = None

    assert resp.status_code == 413
    assert resp.json()["error_kind"] == main.OVERFLOW
    # The message has to be something a person can act on, not a provider blob.
    assert "too long" in resp.json()["error"]


async def test_streaming_chat_emits_the_typed_error(client):
    import json
    body = '{"error":"maximum context length is 8192 tokens"}'
    handler = by_host({"primary.test": overflow(body), "secondary.test": overflow(body)})
    main.UPSTREAM_TRANSPORT = httpx.MockTransport(handler)
    try:
        resp = await client.post("/api/chat", json={
            "backend": "primary", "model": "primary-big", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
        frames = [json.loads(line[len("data: "):]) for line in resp.text.split("\n\n")
                  if line.startswith("data: ")]
    finally:
        main.UPSTREAM_TRANSPORT = None

    assert frames[-1]["error_kind"] == main.OVERFLOW
    assert "too long" in frames[-1]["error"]


async def test_streaming_chat_still_falls_over_to_a_bigger_backend(client):
    import json
    stream_body = sse({"choices": [{"delta": {"content": "fits here"}, "finish_reason": None}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}]})
    handler = by_host({
        "primary.test": overflow('{"error":"maximum context length is 8192 tokens"}'),
        # A real streaming response, not a buffered one: this path forwards the
        # provider's bytes through with aiter_raw, which a Response built from
        # text= has already consumed.
        "secondary.test": lambda: httpx.Response(
            200, stream=httpx.ByteStream(stream_body.encode()),
            headers={"content-type": "text/event-stream"}),
    })
    main.UPSTREAM_TRANSPORT = httpx.MockTransport(handler)
    try:
        resp = await client.post("/api/chat", json={
            "backend": "primary", "model": "primary-big", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
    finally:
        main.UPSTREAM_TRANSPORT = None

    assert "fits here" in resp.text
    assert "error" not in resp.text


async def test_an_agent_run_reports_overflow_and_keeps_its_work(app):
    """An agent run is the likeliest thing to outgrow a window -- every tool
    result is appended verbatim -- so the work it already did must survive the
    failure."""
    import agent_tools
    import json

    body = '{"error":"maximum context length is 8192 tokens"}'
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"choices": [{"message": {
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "list_dir", "arguments": "{}"}}]}}]})
        return httpx.Response(400, text=body)

    main.UPSTREAM_TRANSPORT = httpx.MockTransport(handler)
    try:
        ctx = agent_tools.build_agent_context(None, True, ["files"])
        chain = main.chat_fallback_chain("primary", "primary-big")
        events = []
        async for frame in main.agent_loop_gen(
                chain, [{"role": "user", "content": "go"}], "primary", ctx, max_rounds=5):
            for line in frame.split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: "):]))
    finally:
        main.UPSTREAM_TRANSPORT = None

    error = next(e for e in events if "error" in e)
    assert error["error_kind"] == main.OVERFLOW

    history = next(e["agent_history"] for e in events if "agent_history" in e)
    # The tool call and its result are both still there.
    assert any(m.get("role") == "tool" for m in history)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in history)


async def test_streaming_v1_messages_says_prompt_is_too_long(app):
    """Claude Code decides to compact and retry from the error *type* and
    wording. "api_error: all backends exhausted" tells it nothing usable."""
    from conftest import anthropic_events, mock_client

    body = '{"error":"maximum context length is 8192 tokens"}'
    handler = by_host({"primary.test": overflow(body), "secondary.test": overflow(body)})
    async with mock_client(handler) as client:
        raw = ""
        async for event in main.stream_openai_as_anthropic(
                client, {"model": "m", "messages": []}, "m"):
            raw += event

    name, payload = anthropic_events(raw)[0]
    assert name == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert "too long" in payload["error"]["message"]


async def test_streaming_v1_messages_still_reports_other_failures_plainly(app):
    from conftest import anthropic_events, mock_client

    handler = by_host({"primary.test": httpx.Response(429), "secondary.test": httpx.Response(503)})
    async with mock_client(handler) as client:
        raw = ""
        async for event in main.stream_openai_as_anthropic(
                client, {"model": "m", "messages": []}, "m"):
            raw += event

    name, payload = anthropic_events(raw)[0]
    assert payload["error"]["type"] == "api_error"
    assert "exhausted" in payload["error"]["message"]
