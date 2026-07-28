"""Failover you can watch, and a timeout you can configure.

Both exist because of the same failure: a chain of backends each spending a
full timeout is minutes of a stream emitting nothing, which from the browser is
indistinguishable from the proxy having died.
"""

import httpx
import pytest

import main
from test_failover import by_host, completion, mock_client


MSG = [{"role": "user", "content": "hi"}]


async def collect(client, chain, **kw):
    """(attempt payloads, result tuple) from one pass down the chain."""
    attempts, result = [], None
    async for kind, payload in main.call_chain_events(client, chain, MSG, **kw):
        (attempts if kind == "attempt" else [])
        if kind == "attempt":
            attempts.append(payload)
        else:
            result = payload
    return attempts, result


# ---------------------------------------------------------------------------
# What the browser gets told
# ---------------------------------------------------------------------------

async def test_a_failed_backend_is_announced(app):
    handler = by_host({"primary.test": httpx.Response(429),
                       "secondary.test": completion()})
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        attempts, result = await collect(client, chain)
    assert [a["backend"] for a in attempts] == ["primary"]
    assert attempts[0]["outcome"] == "HTTP 429"
    assert result[1] == "secondary"


async def test_the_backend_that_worked_is_not_announced_as_a_failure(app):
    """Only moving on is news; succeeding is reported by the answer itself."""
    handler = by_host({"primary.test": completion()})
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        attempts, result = await collect(client, chain)
    assert attempts == []
    assert result[1] == "primary"


async def test_every_backend_failing_announces_every_one(app):
    """The case this was built for: the user sees each step instead of one
    long silence."""
    handler = by_host({"primary.test": httpx.Response(429),
                       "secondary.test": httpx.Response(500)})
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        attempts, result = await collect(client, chain)
    assert [a["backend"] for a in attempts] == ["primary", "secondary"]
    assert result[0] is None          # nothing answered


async def test_an_attempt_says_how_long_it_cost(app):
    """"Timed out" means something very different at 4 seconds and at 300."""
    handler = by_host({"primary.test": httpx.Response(429),
                       "secondary.test": completion()})
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        attempts, _ = await collect(client, chain)
    assert isinstance(attempts[0]["seconds"], float)
    assert attempts[0]["attempt"] == 1


async def test_a_timeout_says_timeout_not_connection_error(app):
    """A bare "connection error" sends you looking at the network, when the
    real answer is that the prompt outgrew the budget."""
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        attempts, _ = await collect(client, chain)
    assert attempts, "a timeout must still be announced"
    assert "timed out after" in attempts[0]["outcome"]


async def test_context_overflow_is_named_as_itself(app):
    handler = by_host({
        "primary.test": httpx.Response(400, json={
            "error": {"message": "This model's maximum context length is 8192 tokens"}}),
        "secondary.test": completion()})
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        attempts, _ = await collect(client, chain)
    assert attempts[0]["outcome"] == "context overflow"


# ---------------------------------------------------------------------------
# The wrapper still behaves, so nothing else had to change
# ---------------------------------------------------------------------------

async def test_the_old_signature_still_works(app):
    """call_chain_non_stream is now a thin wrapper; every existing caller and
    test depends on it returning the same five-tuple."""
    handler = by_host({"primary.test": httpx.Response(429),
                       "secondary.test": completion()})
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        data, backend, model, error, hard = await main.call_chain_non_stream(
            client, chain, MSG)
    assert (backend, model, error, hard) == ("secondary", "secondary-big", None, None)
    assert data is not None


async def test_a_non_retryable_error_still_stops_the_chain(app):
    handler = by_host({"primary.test": httpx.Response(422, text="nope")})
    chain = main.chat_fallback_chain("primary", "primary-big")
    async with mock_client(handler) as client:
        data, backend, model, error, hard = await main.call_chain_non_stream(
            client, chain, MSG)
    assert hard == 422 and error == "nope"


# ---------------------------------------------------------------------------
# The timeout
# ---------------------------------------------------------------------------

def test_agent_runs_get_a_longer_budget_than_ordinary_chat():
    """An agent conversation grows by a tool result every round; the round
    that first exceeds the default is where the silent failover starts."""
    assert main.AGENT_HTTP_TIMEOUT.read > main.HTTP_TIMEOUT.read


def test_both_timeouts_are_configurable():
    """The whole point -- 120s was hardcoded, and a big prompt on a free tier
    legitimately needs longer."""
    import importlib, os
    for var, attr, value in (("UPSTREAM_TIMEOUT", "HTTP_TIMEOUT", "45"),
                             ("UPSTREAM_AGENT_TIMEOUT", "AGENT_HTTP_TIMEOUT", "900")):
        old = os.environ.get(var)
        os.environ[var] = value
        try:
            reloaded = importlib.reload(main)
            assert getattr(reloaded, attr).read == float(value)
        finally:
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old
    importlib.reload(main)          # leave the module as everything else expects


def test_the_connect_timeout_stays_short():
    """A backend that isn't listening should fail fast; only the wait for an
    answer is long."""
    assert main.HTTP_TIMEOUT.connect <= 30
    assert main.AGENT_HTTP_TIMEOUT.connect <= 30
