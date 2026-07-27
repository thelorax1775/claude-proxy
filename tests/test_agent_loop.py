"""The web UI's agent loop.

The thing worth pinning down here is the conversation it builds: every
`role:"tool"` result must reference a `tool_call_id` that appears in the
assistant message before it, or every round after the first is a malformed
conversation the provider rejects.
"""

import json

import httpx
import pytest

import agent_tools
import main
from conftest import by_host


@pytest.fixture
def mock_upstream(app):
    """Programmable upstream for the agent loop, which builds its own client."""
    class Upstream:
        def __init__(self):
            self.replies = []
            self.requests = []

        def reply(self, *, text=None, tool_calls=None):
            message = {"content": text}
            if tool_calls is not None:
                message["tool_calls"] = tool_calls
            self.replies.append({"choices": [{"message": message, "finish_reason": "stop"}]})

        def handler(self, request):
            self.requests.append(json.loads(request.content))
            if not self.replies:
                return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})
            return httpx.Response(200, json=self.replies.pop(0))

    upstream = Upstream()
    main.UPSTREAM_TRANSPORT = httpx.MockTransport(upstream.handler)
    yield upstream
    main.UPSTREAM_TRANSPORT = None


async def run_agent(groups=("files",), message="go"):
    ctx = agent_tools.build_agent_context(None, True, list(groups))
    chain = main.chat_fallback_chain("primary", "primary-big")
    events = []
    async for frame in main.agent_loop_gen(
            chain, [{"role": "user", "content": message}], "primary", ctx, max_rounds=5):
        for line in frame.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


def history_of(events):
    for event in reversed(events):
        if "agent_history" in event:
            return event["agent_history"]
    return []


def assert_well_formed(history):
    """Every tool result must answer a tool call announced just before it."""
    announced = set()
    for message in history:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                assert call.get("id"), "a tool_call went into the history with no id"
                announced.add(call["id"])
        elif message.get("role") == "tool":
            assert message["tool_call_id"] in announced, (
                f"tool result references {message['tool_call_id']!r}, which no "
                f"assistant message announced (announced: {sorted(announced)})")


async def test_plain_answer_needs_no_tools(mock_upstream):
    mock_upstream.reply(text="just an answer")
    events = await run_agent()
    assert any(e.get("agent_text") == "just an answer" for e in events)
    assert any(e.get("agent_done") for e in events)


async def test_tool_call_ids_are_added_when_the_provider_omits_them(mock_upstream):
    """Ollama and several OSS-model paths return tool_calls with no id. The
    result message then referenced an id nothing had announced."""
    mock_upstream.reply(tool_calls=[{"function": {"name": "list_dir", "arguments": "{}"}}])
    mock_upstream.reply(text="had a look")

    events = await run_agent()
    history = history_of(events)
    assert_well_formed(history)

    assistant = next(m for m in history if m.get("role") == "assistant" and m.get("tool_calls"))
    assert assistant["tool_calls"][0]["id"].startswith("call_")


async def test_repeated_ids_across_rounds_are_made_unique(mock_upstream):
    """A provider that answers "call_1" every round collided in the browser's
    tool-card map and in the approval registry."""
    mock_upstream.reply(tool_calls=[{"id": "call_1", "function": {"name": "list_dir", "arguments": "{}"}}])
    mock_upstream.reply(tool_calls=[{"id": "call_1", "function": {"name": "list_dir", "arguments": "{}"}}])
    mock_upstream.reply(text="done")

    events = await run_agent()
    history = history_of(events)
    assert_well_formed(history)

    ids = [c["id"] for m in history if m.get("role") == "assistant"
           for c in (m.get("tool_calls") or [])]
    assert len(ids) == len(set(ids)), f"duplicate tool-call ids in one run: {ids}"


async def test_object_arguments_reach_the_tool(mock_upstream):
    mock_upstream.reply(tool_calls=[{"id": "c1", "function": {"name": "list_dir",
                                                              "arguments": {"path": "."}}}])
    mock_upstream.reply(text="done")

    events = await run_agent()
    call = next(e["agent_tool_call"] for e in events if "agent_tool_call" in e)
    assert call["arguments"] == {"path": "."}
    assert_well_formed(history_of(events))


async def test_empty_choices_ends_the_run_without_crashing(mock_upstream):
    """A content filter returns choices: []. This used to be an unhandled
    IndexError: the SSE simply stopped and the round's work was lost."""
    main.UPSTREAM_TRANSPORT = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": []}))
    events = await run_agent()
    assert any(e.get("agent_done") for e in events)
    assert any("agent_history" in e for e in events)


async def test_nameless_tool_calls_do_not_produce_a_broken_round(mock_upstream):
    mock_upstream.reply(text="hi", tool_calls=[{"id": "c", "function": {"arguments": "{}"}}])
    events = await run_agent()
    assert_well_formed(history_of(events))
    assert any(e.get("agent_done") for e in events)


async def test_too_many_calls_in_one_round_are_answered_not_run(mock_upstream):
    """max_rounds bounds rounds, not calls. Every dropped call still needs a
    result, or the next round is a malformed conversation."""
    many = [{"id": f"c{i}", "function": {"name": "list_dir", "arguments": "{}"}}
            for i in range(agent_tools.MAX_CALLS_PER_ROUND + 5)]
    mock_upstream.reply(tool_calls=many)
    mock_upstream.reply(text="done")

    events = await run_agent()
    history = history_of(events)
    assert_well_formed(history)

    results = [m for m in history if m.get("role") == "tool"]
    assert len(results) == len(many)
    refused = [m for m in results if "only the first" in m["content"]]
    assert len(refused) == 5


async def test_a_switched_off_group_is_refused_server_side(mock_upstream):
    """The browser decides what to ask for; the server decides what runs."""
    mock_upstream.reply(tool_calls=[{"id": "c", "function": {"name": "bash",
                                                             "arguments": '{"command": "id"}'}}])
    mock_upstream.reply(text="ok")

    events = await run_agent(groups=("files",))     # shell not ticked
    result = next(e["agent_tool_result"] for e in events if "agent_tool_result" in e)
    assert "switched off" in result["output"]


async def test_running_out_of_rounds_is_a_pause_not_a_failure(mock_upstream):
    for _ in range(10):
        mock_upstream.reply(tool_calls=[{"id": "c", "function": {"name": "list_dir",
                                                                 "arguments": "{}"}}])
    events = await run_agent()
    assert any("agent_limit" in e for e in events)
    assert any(e.get("agent_done") for e in events)
    assert_well_formed(history_of(events))
