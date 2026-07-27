"""Minimal GitHub REST client built on httpx.

Only what the agent tools and the UI need: repos, pull requests, issues,
comments. The token comes from the integrations store, so it's set once in
the UI and never travels through a chat transcript.
"""

import re

import httpx

import integrations

TIMEOUT = httpx.Timeout(30.0, connect=10.0)
REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


class GitHubError(Exception):
    pass


def require_token() -> str:
    token = integrations.get("github_token")
    if not token:
        raise GitHubError("no GitHub token configured -- add one in the UI's Agent tab, under Integrations")
    return token


def check_repo(repo: str) -> str:
    repo = (repo or "").strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
    if not REPO_RE.match(repo):
        raise GitHubError(f"'{repo}' is not a valid owner/repo")
    return repo


def scrub(text: str) -> str:
    """Never let a token reach a chat bubble or a log line."""
    token = integrations.get("github_token")
    return text.replace(token, "***") if token else text


async def request(method: str, path: str, *, params=None, json_body=None, accept: str | None = None):
    token = require_token()
    base = integrations.get("github_api")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": accept or "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "claude-proxy",
    }
    url = path if path.startswith("http") else f"{base}{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.request(method, url, headers=headers, params=params, json=json_body)
    except httpx.HTTPError as e:
        raise GitHubError(f"could not reach GitHub: {scrub(str(e))}") from e

    if resp.status_code == 401:
        raise GitHubError("GitHub rejected the token (401) -- it may be expired or missing scopes")
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise GitHubError("GitHub rate limit hit -- wait a bit and retry")
    if resp.status_code == 404:
        raise GitHubError(f"not found (404): {path}")
    if resp.status_code >= 400:
        detail = ""
        try:
            detail = resp.json().get("message", "")
        except ValueError:
            detail = resp.text[:200]
        raise GitHubError(f"GitHub returned {resp.status_code}: {scrub(detail)}")

    if accept and "json" not in accept:
        return resp.text
    try:
        return resp.json()
    except ValueError:
        return resp.text


# ---------------------------------------------------------------------------
# The specific calls the tools and UI make
# ---------------------------------------------------------------------------

async def whoami() -> dict:
    return await request("GET", "/user")


async def list_repos(limit: int = 30, query: str = "") -> list[dict]:
    if query:
        data = await request("GET", "/search/repositories",
                             params={"q": query, "per_page": min(limit, 50)})
        return data.get("items", [])
    return await request("GET", "/user/repos",
                         params={"per_page": min(limit, 50), "sort": "pushed", "affiliation": "owner,collaborator,organization_member"})


async def list_pulls(repo: str, state: str = "open", limit: int = 20) -> list[dict]:
    return await request("GET", f"/repos/{check_repo(repo)}/pulls",
                         params={"state": state, "per_page": min(limit, 50), "sort": "updated", "direction": "desc"})


async def get_pull(repo: str, number: int) -> dict:
    return await request("GET", f"/repos/{check_repo(repo)}/pulls/{int(number)}")


async def get_pull_diff(repo: str, number: int) -> str:
    return await request("GET", f"/repos/{check_repo(repo)}/pulls/{int(number)}",
                         accept="application/vnd.github.diff")


async def list_issues(repo: str, state: str = "open", limit: int = 20) -> list[dict]:
    issues = await request("GET", f"/repos/{check_repo(repo)}/issues",
                           params={"state": state, "per_page": min(limit, 50)})
    # /issues returns PRs too; the tools have separate calls for those.
    return [i for i in issues if "pull_request" not in i]


async def get_issue(repo: str, number: int) -> dict:
    return await request("GET", f"/repos/{check_repo(repo)}/issues/{int(number)}")


async def list_comments(repo: str, number: int, limit: int = 30) -> list[dict]:
    return await request("GET", f"/repos/{check_repo(repo)}/issues/{int(number)}/comments",
                         params={"per_page": min(limit, 50)})


async def create_issue(repo: str, title: str, body: str = "") -> dict:
    return await request("POST", f"/repos/{check_repo(repo)}/issues",
                         json_body={"title": title, "body": body})


async def add_comment(repo: str, number: int, body: str) -> dict:
    return await request("POST", f"/repos/{check_repo(repo)}/issues/{int(number)}/comments",
                         json_body={"body": body})


async def create_pull(repo: str, title: str, head: str, base: str, body: str = "") -> dict:
    return await request("POST", f"/repos/{check_repo(repo)}/pulls",
                         json_body={"title": title, "head": head, "base": base, "body": body})


def clone_url(repo: str) -> str:
    """Where to git-clone from, derived from the API base so a GitHub
    Enterprise host (or a test double) works without extra configuration."""
    base = integrations.get("github_api")
    scheme, _, rest = base.partition("://")
    host = rest.split("/")[0]
    if host == "api.github.com":
        host = "github.com"
    return f"{scheme}://{host}/{check_repo(repo)}.git"


async def default_branch(repo: str) -> str:
    data = await request("GET", f"/repos/{check_repo(repo)}")
    return data.get("default_branch", "main")
