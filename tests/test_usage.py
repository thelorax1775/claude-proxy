"""Backend allowance reporting.

The point of these is that a usage lookup is a sideshow: it must never break
the page it decorates, and it must never be mistaken for zero when the answer
is actually "we don't know".
"""

import json

import httpx
import pytest

import main
import usage


@pytest.fixture(autouse=True)
def clean_cache():
    usage._CACHE.clear()
    yield
    usage._CACHE.clear()


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


OLLAMA_PAYLOAD = {
    "activity": {"cost": "0.00000", "period": {"type": "last_4_weeks"}, "models": []},
    "limits": {
        "session": {"usage": 0.25, "models": []},
        "weekly": {"usage": 0.957, "models": [
            {"name": "nemotron-3-ultra", "request_count": 292},
            {"name": "nemotron-3-super", "request_count": 138},
        ]},
    },
}


def backend(name="ollama 1", base="https://ollama.com/v1", key="k"):
    return {"name": name, "base_url": base, "api_key": key,
            "big_model": "m", "small_model": "m"}


def client_returning(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://ollama.com/v1", "ollama"),
    ("https://api.ollama.com/v1", "ollama"),
    ("https://api.groq.com/openai/v1", ""),
    ("http://localhost:11434/v1", ""),
    ("not a url", ""),
])
def test_provider_detection(url, expected):
    assert usage.provider_for(url) == expected


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_weekly_usage_becomes_a_percentage_and_a_remainder():
    row = usage.parse_ollama(OLLAMA_PAYLOAD)
    assert row["used_pct"] == 95.7
    assert row["remaining_pct"] == pytest.approx(4.3)
    assert row["session_pct"] == 25.0
    assert row["window"] == "weekly"


def test_models_are_summed_and_ordered_by_traffic():
    row = usage.parse_ollama(OLLAMA_PAYLOAD)
    assert row["requests"] == 430
    assert [m["name"] for m in row["models"]] == ["nemotron-3-ultra", "nemotron-3-super"]


def test_a_missing_usage_figure_is_none_not_zero():
    """'We don't know' and 'nothing used yet' must not render the same -- one
    is a full bar's worth of headroom, the other is no information at all."""
    row = usage.parse_ollama({"limits": {"weekly": {}}})
    assert row["used_pct"] is None
    assert row["remaining_pct"] is None


def test_an_out_of_range_fraction_is_clamped():
    """A meter drawn past its own track is a rendering bug sourced from data."""
    assert usage.parse_ollama({"limits": {"weekly": {"usage": 1.4}}})["used_pct"] == 100.0
    assert usage.parse_ollama({"limits": {"weekly": {"usage": -3}}})["used_pct"] == 0.0


def test_a_nonsense_fraction_is_none():
    assert usage.parse_ollama({"limits": {"weekly": {"usage": "lots"}}})["used_pct"] is None


def test_a_junk_payload_does_not_raise():
    row = usage.parse_ollama({"limits": {"weekly": {"models": ["not a dict", None]}}})
    assert row["models"] == [] and row["requests"] == 0


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

async def test_it_asks_the_right_url_with_the_key():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=OLLAMA_PAYLOAD)

    async with client_returning(handler) as c:
        row = await usage.for_backend(c, backend(key="secret-key"))

    # A sibling of /v1, not a child of it.
    assert seen["url"] == "https://ollama.com/api/usage"
    assert seen["auth"] == "Bearer secret-key"
    assert row["used_pct"] == 95.7


async def test_an_unsupported_provider_says_so_rather_than_guessing():
    async with client_returning(lambda r: httpx.Response(200)) as c:
        row = await usage.for_backend(c, backend(base="https://api.groq.com/openai/v1"))
    assert row["supported"] is False
    assert "groq" in row["reason"]


async def test_a_backend_with_no_key_is_reported_not_probed():
    called = False

    def handler(request):
        nonlocal called
        called = True
        return httpx.Response(200, json=OLLAMA_PAYLOAD)

    async with client_returning(handler) as c:
        row = await usage.for_backend(c, backend(key=""))
    assert called is False
    assert "no key" in row["error"]


@pytest.mark.parametrize("status,fragment", [
    (401, "rejected"), (403, "rejected"), (500, "500"),
])
async def test_an_http_failure_is_reported_on_the_card(status, fragment):
    async with client_returning(lambda r: httpx.Response(status, text="nope")) as c:
        row = await usage.for_backend(c, backend())
    assert fragment in row["error"]
    assert "used_pct" not in row


async def test_a_transport_failure_is_caught():
    """A provider that is down must not take the Backends tab with it."""
    def boom(request):
        raise httpx.ConnectError("no route to host")

    async with client_returning(boom) as c:
        row = await usage.for_backend(c, backend())
    assert "could not reach" in row["error"]


async def test_non_json_is_caught():
    async with client_returning(lambda r: httpx.Response(200, text="<html>")) as c:
        row = await usage.for_backend(c, backend())
    assert "wasn't JSON" in row["error"]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

async def test_a_second_look_is_served_from_cache():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=OLLAMA_PAYLOAD)

    async with client_returning(handler) as c:
        first = await usage.for_backend(c, backend())
        second = await usage.for_backend(c, backend())

    assert calls == 1
    assert first["cached"] is False and second["cached"] is True


async def test_force_bypasses_the_cache():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=OLLAMA_PAYLOAD)

    async with client_returning(handler) as c:
        await usage.for_backend(c, backend())
        await usage.for_backend(c, backend(), force=True)
    assert calls == 2


async def test_forget_drops_a_stale_account():
    """A replaced key is a different account; its old numbers must not linger."""
    async with client_returning(lambda r: httpx.Response(200, json=OLLAMA_PAYLOAD)) as c:
        await usage.for_backend(c, backend())
        usage.forget("ollama 1")
        row = await usage.for_backend(c, backend())
    assert row["cached"] is False


async def test_for_all_covers_every_backend():
    async with client_returning(lambda r: httpx.Response(200, json=OLLAMA_PAYLOAD)) as c:
        rows = await usage.for_all(c, [backend("a"), backend("b"), backend("c")])
    assert [r["name"] for r in rows] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

async def test_the_endpoint_reports_every_backend(client, monkeypatch):
    monkeypatch.setattr(main, "BACKENDS", [backend("ollama 1"), backend("ollama 2")])
    main.UPSTREAM_TRANSPORT = httpx.MockTransport(
        lambda r: httpx.Response(200, json=OLLAMA_PAYLOAD))
    try:
        body = (await client.get("/api/usage")).json()
    finally:
        main.UPSTREAM_TRANSPORT = None

    assert [r["name"] for r in body["backends"]] == ["ollama 1", "ollama 2"]
    assert body["backends"][0]["used_pct"] == 95.7
    assert body["ttl"] == usage.TTL_SECONDS


# ---------------------------------------------------------------------------
# Routing -- reading the cache from the request path
# ---------------------------------------------------------------------------

def remember(name, used_pct, *, age=0.0):
    """Put a figure straight into the cache, as a fetch would have."""
    import time as _t
    usage._CACHE[name] = (_t.monotonic() - age, {"supported": True, "used_pct": used_pct})


def test_is_spent_needs_evidence():
    remember("spent", 100.0)
    remember("nearly", 99.4)
    assert usage.is_spent("spent") is True
    assert usage.is_spent("nearly") is False
    # Never seen: unknown is not spent, or an unreachable usage API would
    # bench a perfectly good backend.
    assert usage.is_spent("never heard of it") is False


def test_a_stale_figure_stops_counting():
    """A weekly allowance resets. Last week's reading must not pin a backend
    to the back of the chain forever."""
    remember("old", 100.0, age=usage.ROUTING_MAX_AGE + 1)
    assert usage.cached_used_pct("old") is None
    assert usage.is_spent("old") is False


def test_the_ui_and_the_router_agree_on_spent():
    """SPENT_AT is duplicated in static/js/backends.js; if one moves the meter
    and the routing disagree about the same backend."""
    assert usage.SPENT_AT == 99.5


def test_needs_refresh_ignores_providers_it_cannot_read():
    """No usage provider configured means nothing to refresh -- this must stay
    completely inert rather than waking up to re-read nothing."""
    groq = [{"name": "g", "base_url": "https://api.groq.com/openai/v1"}]
    assert usage.needs_refresh(groq) is False

    ollama = [{"name": "o", "base_url": "https://ollama.com/v1"}]
    assert usage.needs_refresh(ollama) is True     # never read
    remember("o", 10.0)
    assert usage.needs_refresh(ollama) is False    # fresh
    remember("o", 10.0, age=usage.TTL_SECONDS + 1)
    assert usage.needs_refresh(ollama) is True     # gone stale


def order_of(app):
    return [b["name"] for b in
            main.prioritise(list(app.BACKENDS), lambda b: b["name"])]


def test_the_emptiest_account_is_tried_first(app):
    """The point of the whole thing: a pool of accounts should drain evenly
    rather than one being flattened while the rest sit idle."""
    app.BACKENDS[:] = [backend(n) for n in ("a", "b", "c")]
    remember("a", 80.0)
    remember("b", 20.0)
    remember("c", 50.0)
    assert order_of(app) == ["b", "c", "a"]


def test_close_figures_do_not_reshuffle_the_chain(app):
    """Ordering on the raw percentage would change the leader every refresh --
    the front of the chain is the end that gets used, so it would never settle.
    Within a band the configured order stands."""
    app.BACKENDS[:] = [backend("a"), backend("b")]
    remember("a", 74.9)
    remember("b", 74.1)          # emptier, but the same ten-point band
    assert order_of(app) == ["a", "b"]

    remember("b", 64.9)          # a band clear now
    assert order_of(app) == ["b", "a"]


def test_a_backend_we_cannot_measure_keeps_its_slot(app):
    """A paid endpoint with no usage API, deliberately configured first, must
    not be shuffled behind a pool of free accounts."""
    app.BACKENDS[:] = [backend(n) for n in ("paid", "a", "b")]
    remember("a", 80.0)
    remember("b", 20.0)
    assert order_of(app) == ["paid", "b", "a"]


def test_spent_backends_go_to_the_back(app):
    app.BACKENDS[:] = [backend("a"), backend("b"), backend("c")]
    remember("a", 100.0)
    assert order_of(app) == ["b", "c", "a"]


def test_spent_outranks_the_bands_even_with_nothing_to_compare(app):
    """"Cannot answer at all" beats "fuller than that one", and has to hold
    when the others have no figure to be compared against."""
    app.BACKENDS[:] = [backend(n) for n in ("a", "b", "c", "d")]
    remember("a", 100.0)
    remember("c", 100.0)
    assert order_of(app) == ["b", "d", "a", "c"]


def test_a_spent_backend_is_demoted_not_dropped(app):
    """Losing it entirely could turn a slow request into a failed one."""
    app.BACKENDS[:] = [backend("a"), backend("b")]
    remember("a", 100.0)
    kept = main.prioritise(list(app.BACKENDS), lambda b: b["name"])
    assert {b["name"] for b in kept} == {"a", "b"}


def test_all_spent_leaves_the_order_alone(app):
    """Reordering buys nothing when every option is equally bad, and the
    configured order is the better tiebreak."""
    app.BACKENDS[:] = [backend("a"), backend("b")]
    remember("a", 100.0)
    remember("b", 100.0)
    assert order_of(app) == ["a", "b"]


def test_stale_figures_fall_back_to_the_configured_order(app):
    """Nothing is reordered on a guess."""
    app.BACKENDS[:] = [backend("a"), backend("b")]
    remember("a", 90.0, age=usage.ROUTING_MAX_AGE + 1)
    remember("b", 10.0, age=usage.ROUTING_MAX_AGE + 1)
    assert order_of(app) == ["a", "b"]


def test_reordering_can_be_switched_off(app, monkeypatch):
    monkeypatch.setattr(main, "REORDERS", False)
    app.BACKENDS[:] = [backend("a"), backend("b")]
    remember("a", 100.0)
    remember("b", 10.0)
    assert order_of(app) == ["a", "b"]


def test_cooling_still_wins_over_ordering(app):
    """A cooling backend leaves the chain outright; this only reorders what is
    left."""
    app.BACKENDS[:] = [backend("a"), backend("b"), backend("c")]
    remember("b", 100.0)
    main.cool_down("a", 60)
    try:
        order = [x["name"] for x in
                 main.usable_backends(list(app.BACKENDS), lambda b: b["name"])]
    finally:
        main.clear_cooldown("a")
    assert order == ["c", "b"]


async def test_status_reports_the_effective_chain(client, app):
    app.BACKENDS[:] = [backend("a"), backend("b"), backend("c")]
    remember("a", 100.0)
    remember("b", 60.0)
    remember("c", 10.0)
    body = (await client.get("/status")).json()
    assert body["chain"] == ["c", "b", "a"]
    assert body["demoted"] == ["a"]
    assert body["order"] == "least-used"


async def test_the_endpoint_never_returns_a_key(client, monkeypatch):
    monkeypatch.setattr(main, "BACKENDS", [backend(key="super-secret")])
    main.UPSTREAM_TRANSPORT = httpx.MockTransport(
        lambda r: httpx.Response(200, json=OLLAMA_PAYLOAD))
    try:
        raw = (await client.get("/api/usage")).text
    finally:
        main.UPSTREAM_TRANSPORT = None
    assert "super-secret" not in raw
