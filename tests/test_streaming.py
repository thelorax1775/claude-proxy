"""OpenAI SSE -> Anthropic SSE.

This is the path Claude Code always uses, and the one where a bug is a 500 in
the middle of a response the client is already reading. Every test drives
`stream_openai_as_anthropic` against a stubbed upstream; nothing touches the
network.
"""

import httpx
import pytest

import main
from conftest import anthropic_events, by_host, mock_client, sse


def delta(content=None, tool_calls=None, finish=None):
    d = {}
    if content is not None:
        d["content"] = content
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    return {"choices": [{"delta": d, "finish_reason": finish}]}


async def run_stream(app, body_chunks, model="claude-sonnet-4"):
    """Feed `body_chunks` to the translator as if the first backend sent them."""
    handler = by_host({"primary.test": httpx.Response(200, text=body_chunks)})
    async with mock_client(handler) as client:
        raw = ""
        async for event in main.stream_openai_as_anthropic(
                client, {"model": model, "messages": [{"role": "user", "content": "hi"}]}, model):
            raw += event
    return anthropic_events(raw)


async def test_text_stream_shape(app):
    events = await run_stream(app, sse(delta(content="Hel"), delta(content="lo"),
                                       delta(finish="stop")))
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    assert "content_block_start" in names and "content_block_stop" in names

    text = "".join(p["delta"]["text"] for n, p in events
                   if n == "content_block_delta" and p["delta"]["type"] == "text_delta")
    assert text == "Hello"

    stop = next(p for n, p in events if n == "message_delta")
    assert stop["delta"]["stop_reason"] == "end_turn"


async def test_tool_call_stream_shape(app):
    events = await run_stream(app, sse(
        delta(tool_calls=[{"index": 0, "id": "call_1",
                           "function": {"name": "bash", "arguments": '{"comm'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": 'and": "ls"}'}}]),
        delta(finish="tool_calls"),
    ))
    start = next(p for n, p in events if n == "content_block_start")
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["name"] == "bash"
    assert start["content_block"]["id"] == "call_1"

    args = "".join(p["delta"]["partial_json"] for n, p in events
                   if n == "content_block_delta" and p["delta"]["type"] == "input_json_delta")
    assert args == '{"command": "ls"}'

    assert next(p for n, p in events if n == "message_delta")["delta"]["stop_reason"] == "tool_use"


async def test_tool_call_without_index_does_not_crash(app):
    """Several OpenAI-compatible providers omit `index` when streaming one
    call at a time. Reading it with [] was a KeyError, i.e. a 500 mid-stream."""
    events = await run_stream(app, sse(
        delta(tool_calls=[{"id": "call_1", "function": {"name": "bash", "arguments": '{"a"'}}]),
        delta(tool_calls=[{"function": {"arguments": ': 1}'}}]),
        delta(finish="tool_calls"),
    ))
    starts = [p for n, p in events if n == "content_block_start"]
    assert len(starts) == 1
    assert starts[0]["content_block"]["name"] == "bash"

    args = "".join(p["delta"]["partial_json"] for n, p in events
                   if n == "content_block_delta" and p["delta"]["type"] == "input_json_delta")
    assert args == '{"a": 1}'


async def test_name_arriving_after_the_first_chunk(app):
    """The name used to be read only from the chunk that opened the block, so
    a provider that sent it later produced a tool_use named ""."""
    events = await run_stream(app, sse(
        delta(tool_calls=[{"index": 0, "id": "call_1", "function": {"arguments": ""}}]),
        delta(tool_calls=[{"index": 0, "function": {"name": "read_file"}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"path": "a"}'}}]),
        delta(finish="tool_calls"),
    ))
    start = next(p for n, p in events if n == "content_block_start")
    assert start["content_block"]["name"] == "read_file"
    args = "".join(p["delta"]["partial_json"] for n, p in events
                   if n == "content_block_delta" and p["delta"]["type"] == "input_json_delta")
    assert args == '{"path": "a"}'


async def test_arguments_buffered_before_the_name_are_not_lost(app):
    """Arguments can arrive before the name. They have to be held back, since
    content_block_start carries the name and can't be corrected afterwards."""
    events = await run_stream(app, sse(
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"path"'}}]),
        delta(tool_calls=[{"index": 0, "function": {"name": "read_file", "arguments": ': "a"}'}}]),
        delta(finish="tool_calls"),
    ))
    args = "".join(p["delta"]["partial_json"] for n, p in events
                   if n == "content_block_delta" and p["delta"]["type"] == "input_json_delta")
    assert args == '{"path": "a"}'


async def test_two_tool_calls_get_separate_blocks(app):
    events = await run_stream(app, sse(
        delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "one", "arguments": "{}"}}]),
        delta(tool_calls=[{"index": 1, "id": "b", "function": {"name": "two", "arguments": "{}"}}]),
        delta(finish="tool_calls"),
    ))
    starts = [p for n, p in events if n == "content_block_start"]
    assert [s["content_block"]["name"] for s in starts] == ["one", "two"]
    assert [s["index"] for s in starts] == [0, 1]


async def test_object_arguments_are_serialised(app):
    """A provider sending the object rather than a JSON string used to have it
    embedded raw into partial_json, which the client cannot parse."""
    events = await run_stream(app, sse(
        delta(tool_calls=[{"index": 0, "id": "a",
                           "function": {"name": "bash", "arguments": {"command": "ls"}}}]),
        delta(finish="tool_calls"),
    ))
    args = "".join(p["delta"]["partial_json"] for n, p in events
                   if n == "content_block_delta" and p["delta"]["type"] == "input_json_delta")
    assert args == '{"command": "ls"}'


async def test_text_block_is_closed_before_a_tool_block_opens(app):
    events = await run_stream(app, sse(
        delta(content="thinking"),
        delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "bash", "arguments": "{}"}}]),
        delta(finish="tool_calls"),
    ))
    names = [n for n, _ in events]
    first_stop = names.index("content_block_stop")
    second_start = names.index("content_block_start", first_stop)
    assert first_stop < second_start


async def test_empty_choices_chunks_are_skipped(app):
    events = await run_stream(app, sse({"choices": []}, delta(content="ok"), delta(finish="stop")))
    text = "".join(p["delta"]["text"] for n, p in events
                   if n == "content_block_delta" and p["delta"]["type"] == "text_delta")
    assert text == "ok"


async def test_provider_usage_wins_over_the_estimate(app):
    """The usage chunk arrives with an empty choices list, which the old code
    dropped before ever looking at it."""
    events = await run_stream(app, sse(
        delta(content="hello there"),
        delta(finish="stop"),
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 99}},
    ))
    assert next(p for n, p in events if n == "message_delta")["usage"]["output_tokens"] == 99


async def test_usage_falls_back_to_a_character_estimate(app):
    """It used to report a count of SSE chunks, which is not a unit of
    anything -- two chunks of 500 characters reported "2 tokens"."""
    events = await run_stream(app, sse(delta(content="x" * 400), delta(finish="stop")))
    assert next(p for n, p in events if n == "message_delta")["usage"]["output_tokens"] == 100


async def test_malformed_frames_are_ignored(app):
    body = "data: not json\n\n" + sse(delta(content="ok"), delta(finish="stop"))
    events = await run_stream(app, body)
    text = "".join(p["delta"]["text"] for n, p in events
                   if n == "content_block_delta" and p["delta"]["type"] == "text_delta")
    assert text == "ok"


async def test_all_backends_exhausted_emits_an_error_event(app):
    handler = by_host({"primary.test": httpx.Response(429),
                       "secondary.test": httpx.Response(503)})
    async with mock_client(handler) as client:
        raw = ""
        async for event in main.stream_openai_as_anthropic(
                client, {"model": "m", "messages": []}, "m"):
            raw += event
    events = anthropic_events(raw)
    assert events[0][0] == "error"
    assert "exhausted" in events[0][1]["error"]["message"]
