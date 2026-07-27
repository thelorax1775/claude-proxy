"""Backend cooldown.

Without it, a backend that just answered 429 goes straight back to the front
of the chain on the next request, so every request re-pays the rate limit
before falling through to an account that can actually answer.
"""

import httpx
import pytest

import main
from conftest import by_host


def completion(text="ok"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text},
                                                  "finish_reason": "stop"}]})


def hosts_tried(handler):
    return [r.url.host for r in handler.seen]


async def test_a_rate_limited_backend_is_skipped_next_time(app):
    handler = by_host({"primary.test": httpx.Response(429), "secondary.test": completion()})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await main.try_backends_non_stream(client, {"model": "m", "messages": []})
        first_pass = list(hosts_tried(handler))
        await main.try_backends_non_stream(client, {"model": "m", "messages": []})
        second_pass = hosts_tried(handler)[len(first_pass):]

    assert first_pass == ["primary.test", "secondary.test"]
    # The whole point: the second request doesn't pay for primary's 429 again.
    assert second_pass == ["secondary.test"]


async def test_auth_failures_get_a_longer_bench(app):
    handler = by_host({"primary.test": httpx.Response(401), "secondary.test": completion()})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert main.cooling_for("primary") > main.BACKEND_COOLDOWN


async def test_rate_limits_get_the_short_bench(app):
    handler = by_host({"primary.test": httpx.Response(429), "secondary.test": completion()})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert 0 < main.cooling_for("primary") <= main.BACKEND_COOLDOWN + 1


async def test_retry_after_is_honoured(app):
    handler = by_host({"primary.test": httpx.Response(429, headers={"retry-after": "600"}),
                       "secondary.test": completion()})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert main.cooling_for("primary") > 500


async def test_cooldown_never_empties_the_chain(app):
    """An answer from an over-quota account beats no answer. When every
    backend is benched they are all tried anyway."""
    main.cool_down("primary", 300)
    main.cool_down("secondary", 300)

    handler = by_host({"primary.test": completion("still works")})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        data, err = await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert data["choices"][0]["message"]["content"] == "still works"
    assert any("trying anyway" in e["outcome"] for e in main.ATTEMPT_LOG)


async def test_success_clears_the_cooldown(app):
    main.cool_down("primary", 300)
    main.cool_down("secondary", 300)      # so primary isn't filtered out
    handler = by_host({"primary.test": completion()})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert main.cooling_for("primary") == 0


async def test_connection_errors_bench_the_backend_too(app):
    def handler(request):
        if request.url.host == "primary.test":
            raise httpx.ConnectError("refused", request=request)
        return completion()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await main.try_backends_non_stream(client, {"model": "m", "messages": []})
    assert main.cooling_for("primary") > 0


async def test_expired_cooldowns_are_forgotten(app):
    main.cool_down("primary", 0.0001)
    import time
    time.sleep(0.01)
    assert main.cooling_for("primary") == 0
    assert "primary" not in main._COOLDOWN


# ---------------------------------------------------------------------------
# Visibility -- a backend that silently vanishes from the chain is worse than
# one that fails.
# ---------------------------------------------------------------------------

@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_status_reports_what_is_benched(client):
    main.cool_down("primary", 120)
    body = (await client.get("/status")).json()
    assert body["cooling"]["primary"] > 0
    assert "secondary" not in body["cooling"]


async def test_backend_list_reports_cooling_seconds(client):
    main.cool_down("primary", 120)
    backends = (await client.get("/api/backends")).json()["backends"]
    assert backends[0]["cooling"] > 0
    assert backends[1]["cooling"] == 0


async def test_replacing_a_key_lifts_the_bench(client):
    """Someone pasting a new key has just fixed the reason it was benched."""
    main.cool_down("primary", 300)
    await client.patch("/api/backends/primary", json={"api_key": "sk-fresh-00000000000"})
    assert main.cooling_for("primary") == 0


async def test_renaming_moves_the_bench_rather_than_stranding_it(client):
    main.cool_down("primary", 300)
    await client.patch("/api/backends/primary", json={"name": "renamed"})
    assert main.cooling_for("primary") == 0
    assert main.cooling_for("renamed") == 0


async def test_deleting_forgets_the_bench(client):
    main.cool_down("primary", 300)
    await client.delete("/api/backends/primary")
    assert "primary" not in main._COOLDOWN


# ---------------------------------------------------------------------------
# probe-models against a saved backend
# ---------------------------------------------------------------------------

async def test_probe_models_uses_the_stored_key(client):
    """The browser never has the key, so detecting models for a backend you
    are editing only works if the server supplies it."""
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": [{"id": "m-2"}, {"id": "m-1"}]})

    main.UPSTREAM_TRANSPORT = httpx.MockTransport(handler)
    try:
        body = (await client.post("/api/probe-models", json={"name": "primary"})).json()
    finally:
        main.UPSTREAM_TRANSPORT = None

    assert body["models"] == ["m-1", "m-2"]
    assert seen["auth"] == "Bearer sk-primary-000000000000"
    assert seen["url"] == "https://primary.test/v1/models"


async def test_probe_models_unknown_backend_is_reported_not_raised(client):
    body = (await client.post("/api/probe-models", json={"name": "nope"})).json()
    assert body["models"] == []
    assert "not found" in body["error"]


async def test_probe_models_still_works_for_the_add_form(client):
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "new"}]})

    main.UPSTREAM_TRANSPORT = httpx.MockTransport(handler)
    try:
        body = (await client.post("/api/probe-models", json={
            "base_url": "https://fresh.test/v1", "api_key": "sk-typed-0000000000"})).json()
    finally:
        main.UPSTREAM_TRANSPORT = None
    assert body["models"] == ["new"]
