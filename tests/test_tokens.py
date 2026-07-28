"""Token estimation.

There is no tokenizer here on purpose -- the dependency list is four packages
and stays that way -- so these tests pin the properties that matter rather than
exact numbers: everything in a request is counted, the estimate never goes
down when the request grows, and it biases high rather than low.
"""

import json

import main


def tool_schema(name):
    """Roughly the shape and size Claude Code actually sends."""
    return {
        "name": name,
        "description": f"The {name} tool. " + ("Describes what it does. " * 12),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "A file path, relative or absolute"},
                "content": {"type": "string", "description": "Full text to write"},
                "limit": {"type": "integer", "description": "How many lines"},
            },
            "required": ["path"],
        },
    }


def claude_code_body(with_tools=True):
    """A realistic agentic turn: a system prompt, tool schemas, and a
    conversation carrying a large tool_result."""
    return {
        "system": "You are Claude Code. " + ("Follow the instructions carefully. " * 40),
        "tools": [tool_schema(f"tool_{i}") for i in range(15)] if with_tools else None,
        "messages": [
            {"role": "user", "content": "review main.py"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Reading it now."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                 "input": {"path": "main.py", "limit": 2000}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "x" * 20000},
            ]},
        ],
    }


# ---------------------------------------------------------------------------
# The regression this exists for
# ---------------------------------------------------------------------------

def text_only_estimate(body):
    """What the old implementation counted: type == "text" blocks, nothing
    else, no tools."""
    text = main.system_to_text(body.get("system")) or ""
    for msg in body["messages"]:
        c = msg.get("content")
        if isinstance(c, str):
            text += c
        else:
            for block in c or []:
                if block.get("type") == "text":
                    text += block.get("text", "")
    return max(1, len(text) // 4)


def test_an_agentic_turn_is_no_longer_wildly_understated():
    body = claude_code_body()
    old = text_only_estimate(body)
    new = main.estimate_tokens(body["system"], body["messages"], body["tools"])
    # The 20k tool_result and 15 schemas are the whole request; the old count
    # saw a system prompt and one sentence.
    assert new > old * 20, f"old={old} new={new}"
    assert new > 8000


def test_tool_schemas_are_counted():
    body = claude_code_body()
    with_tools = main.estimate_tokens(body["system"], body["messages"], body["tools"])
    without = main.estimate_tokens(body["system"], body["messages"], None)
    assert with_tools > without
    # 15 realistic schemas are not a rounding error.
    assert with_tools - without > 500


def test_tool_results_are_counted():
    """The single biggest thing in an agent conversation, and it lived in a
    block type the old code skipped."""
    small = main.estimate_tokens(None, [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}])
    large = main.estimate_tokens(None, [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x" * 40000}]}])
    assert large - small > 10000


def test_openai_shaped_tool_calls_and_results_are_counted():
    """The /api/chat side speaks OpenAI, where calls are out of band on the
    assistant message and results are role:"tool"."""
    plain = main.estimate_tokens(None, [{"role": "assistant", "content": "done"}])
    with_calls = main.estimate_tokens(None, [{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": "ls " * 500})}}]}])
    assert with_calls > plain * 5

    result = main.estimate_tokens(None, [
        {"role": "tool", "tool_call_id": "c1", "content": "y" * 8000}])
    assert result > 2000


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_system_as_a_string_and_as_blocks_agree():
    text = "be brief. " * 50
    as_string = main.estimate_tokens(text, [])
    as_blocks = main.estimate_tokens([{"type": "text", "text": text}], [])
    assert as_string == as_blocks


def test_growing_a_request_never_lowers_the_estimate():
    messages = []
    previous = main.estimate_tokens(None, messages)
    for i in range(10):
        messages.append({"role": "user", "content": f"message {i} " * 20})
        current = main.estimate_tokens(None, messages)
        assert current >= previous
        previous = current


def test_empty_and_missing_inputs_are_zero_not_a_crash():
    assert main.estimate_tokens() == 0
    assert main.estimate_tokens(None, [], []) == 0
    assert main.estimate_tokens(None, [None, "nonsense", 42]) == 0


def test_unknown_block_types_still_contribute():
    """A shape this has never seen must count its characters rather than
    vanish -- silently dropping content is how the old version got it wrong."""
    known = main.estimate_tokens(None, [{"role": "user", "content": [
        {"type": "text", "text": "hello"}]}])
    unknown = main.estimate_tokens(None, [{"role": "user", "content": [
        {"type": "some_future_block", "payload": "z" * 4000}]}])
    assert unknown > known + 500


def test_json_content_is_charged_more_densely_than_prose():
    """Schemas and payloads tokenise denser than English, and the estimator
    has to lean that way or it under-reports exactly the agentic requests it
    most needs to get right."""
    n = 3000
    prose = main.estimate_tokens(None, [{"role": "user", "content": "a" * n}])
    structured = main.estimate_tokens(None, [{"role": "user", "content": [{"t": "a" * n}]}])
    assert structured > prose


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

async def test_count_tokens_endpoint_counts_tools(app):
    import httpx
    transport = httpx.ASGITransport(app=app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = claude_code_body()
        with_tools = (await client.post("/v1/messages/count_tokens", json=body)).json()
        del body["tools"]
        without = (await client.post("/v1/messages/count_tokens", json=body)).json()

    assert with_tools["input_tokens"] > without["input_tokens"]
    assert with_tools["input_tokens"] > 8000


async def test_count_tokens_is_never_zero(app):
    import httpx
    transport = httpx.ASGITransport(app=app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = (await client.post("/v1/messages/count_tokens", json={"messages": []})).json()
    assert body["input_tokens"] >= 1


async def test_message_start_reports_an_input_estimate(app):
    """It used to be hardcoded 0, which told Claude Code every request was
    free."""
    import httpx
    from conftest import anthropic_events, by_host, mock_client, sse

    handler = by_host({"primary.test": httpx.Response(200, text=sse(
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}))})
    body = claude_code_body()
    async with mock_client(handler) as client:
        raw = ""
        async for event in main.stream_openai_as_anthropic(client, body, "claude-sonnet-4"):
            raw += event

    start = next(p for n, p in anthropic_events(raw) if n == "message_start")
    assert start["message"]["usage"]["input_tokens"] > 8000
