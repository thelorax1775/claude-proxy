"""Backend failover: the ordered walk down backends.json, and what counts as
a reason to move on."""

import httpx
import pytest

import main
from conftest import by_host, mock_client, sse


def completion(text="ok"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text},
                                                  "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def hosts_tried(handler):
    return [r.url.host for r in handler.seen]


# ---------------------------------------------------------------------------
# Non-streaming (/v1/messages without stream)
# ---------------------------------------------------------------------------

async def test_first_backend_wins(app):
    handler = by_host({"primary.test": completion("from primary")})
    async with mock_client(handler) as client:
        data, err = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data["choices"][0]["message"]["content"] == "from primary"
    assert hosts_tried(handler) == ["primary.test"]
    assert [e["outcome"] for e in main.ATTEMPT_LOG] == ["ok"]


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502, 503, 504])
async def test_retryable_statuses_move_to_the_next_backend(app, status):
    handler = by_host({"primary.test": httpx.Response(status, text="nope"),
                       "secondary.test": completion("from secondary")})
    async with mock_client(handler) as client:
        data, err = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data["choices"][0]["message"]["content"] == "from secondary"
    assert hosts_tried(handler) == ["primary.test", "secondary.test"]


async def test_connection_errors_move_on_too(app):
    def handler(request):
        handler.seen.append(request)
        if request.url.host == "primary.test":
            raise httpx.ConnectError("refused", request=request)
        return completion("from secondary")
    handler.seen = []

    async with mock_client(handler) as client:
        data, err = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data["choices"][0]["message"]["content"] == "from secondary"
    assert any("connection error" in e["outcome"] for e in main.ATTEMPT_LOG)


async def test_non_retryable_status_is_passed_straight_through(app):
    """A 400 is the request's fault, not the backend's -- trying the next
    account would just produce the same 400 more slowly."""
    handler = by_host({"primary.test": httpx.Response(400, text="bad request")})
    async with mock_client(handler) as client:
        data, error_resp = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data is None
    assert error_resp.status_code == 400
    assert hosts_tried(handler) == ["primary.test"]


async def test_all_backends_exhausted_raises_502(app):
    handler = by_host({"primary.test": httpx.Response(429), "secondary.test": httpx.Response(429)})
    async with mock_client(handler) as client:
        with pytest.raises(Exception) as excinfo:
            await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert getattr(excinfo.value, "status_code", None) == 502


async def test_each_backend_is_asked_for_its_own_model(app):
    handler = by_host({"primary.test": httpx.Response(429), "secondary.test": completion()})
    async with mock_client(handler) as client:
        await main.try_backends_non_stream(client, {"model": "claude-3-5-haiku", "messages": []})
    import json
    models = [json.loads(r.content)["model"] for r in handler.seen]
    assert models == ["primary-small", "secondary-small"]


async def test_deleting_a_backend_mid_request_does_not_skip_one(app):
    """The loop spans awaits over a module-global list. Mutating it during a
    request used to silently skip the next entry."""
    def handler(request):
        handler.seen.append(request)
        if request.url.host == "primary.test":
            del main.BACKENDS[0]          # what DELETE /api/backends does
            return httpx.Response(429)
        return completion("from secondary")
    handler.seen = []

    async with mock_client(handler) as client:
        data, err = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data is not None
    assert hosts_tried(handler) == ["primary.test", "secondary.test"]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

async def test_streaming_failover_picks_the_next_backend(app):
    handler = by_host({
        "primary.test": httpx.Response(429),
        "secondary.test": httpx.Response(200, text=sse(
            {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]})),
    })
    async with mock_client(handler) as client:
        resp, backend, error = await main.find_working_stream(client, {"model": "m", "messages": []})
        assert backend["name"] == "secondary"
        await resp.aclose()
    assert hosts_tried(handler) == ["primary.test", "secondary.test"]


async def test_streaming_requests_set_stream_true(app):
    handler = by_host({"primary.test": httpx.Response(200, text=sse())})
    async with mock_client(handler) as client:
        resp, backend, error = await main.find_working_stream(client, {"model": "m", "messages": []})
        await resp.aclose()
    import json
    assert json.loads(handler.seen[0].content)["stream"] is True


# ---------------------------------------------------------------------------
# The UI chat chain
# ---------------------------------------------------------------------------

def test_chat_fallback_chain_starts_where_asked(app):
    chain = main.chat_fallback_chain("secondary", "a-model")
    assert [(b["name"], m) for b, m in chain] == [
        ("secondary", "a-model"), ("primary", "primary-big")]


def test_chat_fallback_chain_rejects_an_unknown_backend(app):
    assert main.chat_fallback_chain("nope", "m") is None


def test_reorder_chain_start_promotes_the_backend_that_worked(app):
    chain = main.chat_fallback_chain("primary", "primary-big")
    assert [b["name"] for b, _ in main.reorder_chain_start(chain, "secondary")] == [
        "secondary", "primary"]


async def test_call_chain_non_stream_reports_which_backend_answered(app):
    handler = by_host({"primary.test": httpx.Response(429), "secondary.test": completion()})
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        data, used_backend, used_model, error, hard = await main.call_chain_non_stream(
            client, chain, [{"role": "user", "content": "hi"}])
    assert used_backend == "secondary"
    assert used_model == "secondary-big"
    assert hard is None


async def test_a_200_with_a_non_json_body_moves_on(app):
    """A gateway's HTML error page, or a backend answering SSE to a
    non-streaming request. resp.json() on that used to be a 500."""
    handler = by_host({
        "primary.test": httpx.Response(200, text="<html>502 Bad Gateway</html>"),
        "secondary.test": completion("from secondary"),
    })
    async with mock_client(handler) as client:
        data, err = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data["choices"][0]["message"]["content"] == "from secondary"
    assert any("not JSON" in e["outcome"] for e in main.ATTEMPT_LOG)


async def test_the_ui_chain_also_survives_a_non_json_body(app):
    handler = by_host({
        "primary.test": httpx.Response(200, text="not json at all"),
        "secondary.test": completion("from secondary"),
    })
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        data, used_backend, used_model, error, hard = await main.call_chain_non_stream(
            client, chain, [{"role": "user", "content": "hi"}])
    assert used_backend == "secondary"
    assert hard is None
