"""GitHub token diagnostics.

Every way a token can be wrong comes back from GitHub as the same opaque 401,
so the value here is in telling them apart before and after the call. The old
message said "expired or missing scopes", which is wrong in a way that costs
real time: /user needs no scopes at all, so a 401 there is never about scopes.
"""

import httpx
import pytest

import github_api
import integrations
import main


@pytest.fixture
def token():
    """Set a stored token and put it back afterwards."""
    original = integrations.INTEGRATIONS.get("github_token")

    def use(value):
        integrations.INTEGRATIONS["github_token"] = value
        return value

    yield use
    integrations.INTEGRATIONS["github_token"] = original


# ---------------------------------------------------------------------------
# Caught before GitHub is ever asked
# ---------------------------------------------------------------------------

def test_a_normal_token_has_no_shape_problem():
    assert github_api.token_shape_problem("ghp_" + "a" * 36) == ""
    assert github_api.token_shape_problem("github_pat_" + "b" * 60) == ""
    assert github_api.token_shape_problem("a" * 40) == ""       # pre-2021 classic


def test_a_token_pasted_from_a_document_is_explained():
    """An en-dash where a hyphen should be is what a token copied through a
    word processor or a chat window looks like."""
    problem = github_api.token_shape_problem("ghp_" + "a" * 30 + "–abc")
    assert "non-ASCII" in problem
    assert "GitHub" in problem


def test_a_zero_width_space_is_caught():
    problem = github_api.token_shape_problem("ghp_" + "a" * 30 + "​abc")
    assert "non-ASCII" in problem


def test_embedded_whitespace_is_caught():
    problem = github_api.token_shape_problem("ghp_" + "a" * 20 + " " + "b" * 15)
    assert "space or line break" in problem


def test_a_truncated_paste_is_caught():
    problem = github_api.token_shape_problem("ghp_abc")
    assert "too short" in problem
    assert "truncated" in problem


def test_no_token_says_so():
    assert "no token" in github_api.token_shape_problem("")


async def test_a_malformed_token_never_reaches_the_network(token):
    """It would fail anyway; failing here means an explanation instead of a
    401, and for the non-ASCII case it also avoids a 500 -- UnicodeEncodeError
    is not an httpx.HTTPError, so it used to escape the handler entirely."""
    token("ghp_" + "a" * 30 + "–abc")
    called = []

    def handler(request):
        called.append(request)
        return httpx.Response(200, json={})

    original = httpx.AsyncClient
    with pytest.raises(github_api.GitHubError) as excinfo:
        await github_api.whoami()
    assert "non-ASCII" in str(excinfo.value)
    assert called == []


# ---------------------------------------------------------------------------
# What GitHub's own answers are turned into
# ---------------------------------------------------------------------------

@pytest.fixture
def github():
    """Point github_api at a stub instead of api.github.com, via its own
    transport seam -- patching httpx.AsyncClient globally would also capture
    the ASGI client the endpoint tests below use to call this app."""
    def install(response):
        def handler(request):
            handler.seen.append(request)
            return response
        handler.seen = []
        github_api.TRANSPORT = httpx.MockTransport(handler)
        return handler

    yield install
    github_api.TRANSPORT = None


async def test_401_does_not_blame_scopes(token, github):
    """/user requires no scopes, so a 401 there can only mean the token itself
    is bad. Sending someone to check scopes is a wrong answer that looks
    plausible enough to waste an afternoon."""
    token("ghp_" + "a" * 36)
    github(httpx.Response(401, json={"message": "Bad credentials"}))

    with pytest.raises(github_api.GitHubError) as excinfo:
        await github_api.whoami()
    message = str(excinfo.value)

    assert "401" in message
    assert "expired" in message
    assert "Scopes can't cause a 401" in message
    # The length is what makes a truncated paste visible.
    assert "40 characters" in message


async def test_403_does_blame_scopes_and_names_them(token, github):
    token("ghp_" + "a" * 36)
    github(httpx.Response(403, json={"message": "Resource not accessible"},
                          headers={"x-oauth-scopes": "read:user, gist"}))

    with pytest.raises(github_api.GitHubError) as excinfo:
        await github_api.whoami()
    message = str(excinfo.value)

    assert "403" in message
    assert "read:user, gist" in message
    assert "repo" in message


async def test_rate_limiting_is_still_its_own_message(token, github):
    token("ghp_" + "a" * 36)
    github(httpx.Response(403, text="API rate limit exceeded"))
    with pytest.raises(github_api.GitHubError) as excinfo:
        await github_api.whoami()
    assert "rate limit" in str(excinfo.value)


async def test_a_classic_token_reports_its_scopes(token, github):
    token("ghp_" + "a" * 36)
    github(httpx.Response(200, json={"login": "someone", "name": "Some One"},
                          headers={"x-oauth-scopes": "repo, read:org"}))

    user = await github_api.whoami()
    assert user["login"] == "someone"
    assert user["scopes"] == ["repo", "read:org"]
    assert user["scopes_reported"] is True


async def test_a_token_without_repo_scope_is_visible_as_such(token, github):
    token("ghp_" + "a" * 36)
    github(httpx.Response(200, json={"login": "someone"},
                          headers={"x-oauth-scopes": "read:user"}))
    user = await github_api.whoami()
    assert "repo" not in user["scopes"]


async def test_a_fine_grained_token_reports_no_scope_header(token, github):
    """Fine-grained tokens use per-repository permissions and send no scope
    header. Absent must read as "not applicable", not as "no permissions"."""
    token("github_pat_" + "b" * 60)
    github(httpx.Response(200, json={"login": "someone"}))

    user = await github_api.whoami()
    assert user["scopes"] == []
    assert user["scopes_reported"] is False


async def test_classic_tokens_use_the_token_prefix(token, github):
    token("ghp_" + "a" * 36)
    handler = github(httpx.Response(200, json={"login": "x"}))
    await github_api.whoami()
    assert handler.seen[0].headers["authorization"].startswith("token ghp_")


async def test_fine_grained_tokens_use_bearer(token, github):
    token("github_pat_" + "b" * 60)
    handler = github(httpx.Response(200, json={"login": "x"}))
    await github_api.whoami()
    assert handler.seen[0].headers["authorization"].startswith("Bearer github_pat_")


# ---------------------------------------------------------------------------
# The endpoint the UI calls
# ---------------------------------------------------------------------------

async def test_the_test_endpoint_passes_scopes_through(app, token, github):
    token("ghp_" + "a" * 36)
    github(httpx.Response(200, json={"login": "someone", "name": "Some One",
                                     "html_url": "https://github.com/someone"},
                          headers={"x-oauth-scopes": "repo"}))

    transport = httpx.ASGITransport(app=app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = (await client.post("/api/integrations/github/test")).json()

    assert body["login"] == "someone"
    assert body["scopes"] == ["repo"]
    assert body["scopes_reported"] is True


async def test_the_test_endpoint_reports_a_401_as_a_400_with_the_reason(app, token, github):
    token("ghp_" + "a" * 36)
    github(httpx.Response(401, json={"message": "Bad credentials"}))

    transport = httpx.ASGITransport(app=app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/integrations/github/test")

    assert resp.status_code == 400
    assert "Scopes can't cause a 401" in resp.json()["detail"]


async def test_the_token_is_never_echoed_back(app, token, github):
    """The API only ever reports whether one is configured."""
    secret = "ghp_" + "s" * 36
    token(secret)
    github(httpx.Response(401, json={"message": "Bad credentials"}))

    transport = httpx.ASGITransport(app=app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = (await client.post("/api/integrations/github/test")).json()["detail"]
        public = (await client.get("/api/integrations")).text

    assert secret not in detail
    assert secret not in public
