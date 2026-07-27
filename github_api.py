"""Minimal GitHub REST client built on httpx.

Only what the agent tools and the UI need: repos, pull requests, issues,
comments. The token comes from the integrations store, so it's set once in
the UI and never travels through a chat transcript.
"""

import base64
import re
import urllib.parse

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
    # REPO_RE allows dots, so 'owner/..' gets this far; callers join the name
    # onto a clone root, where '..' would escape it.
    if any(part in (".", "..") for part in repo.split("/")):
        raise GitHubError(f"'{repo}' is not a valid owner/repo")
    return repo


def scrub(text: str) -> str:
    """Never let a stored secret reach a chat bubble or a log line."""
    return integrations.redact(text)


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


def check_path(path: str) -> str:
    """A repository-relative file path, safe to interpolate into a URL. The
    model supplies these, so '..' and absolute paths are refused rather than
    quietly resolved somewhere else, and the value is percent-encoded."""
    path = (path or "").strip().lstrip("/")
    if not path:
        raise GitHubError("a file path is required")
    if any(part in ("..", ".") for part in path.split("/")):
        raise GitHubError(f"'{path}' is not a valid repository path")
    return urllib.parse.quote(path)


# ---------------------------------------------------------------------------
# Reading: CI, file contents, branches, code search, commits
# ---------------------------------------------------------------------------

async def list_workflow_runs(repo: str, branch: str = "", limit: int = 10) -> list[dict]:
    params = {"per_page": min(int(limit or 10), 50)}
    if branch:
        params["branch"] = branch
    data = await request("GET", f"/repos/{check_repo(repo)}/actions/runs", params=params)
    return data.get("workflow_runs", []) if isinstance(data, dict) else []


async def get_content(repo: str, path: str, ref: str = "") -> str:
    """A file's raw text at a ref. Uses the raw media type so the answer is
    the file itself rather than base64 wrapped in JSON."""
    params = {"ref": ref} if ref else None
    return await request("GET", f"/repos/{check_repo(repo)}/contents/{check_path(path)}",
                         params=params, accept="application/vnd.github.raw")


async def get_content_sha(repo: str, path: str, ref: str = "") -> str:
    """The blob sha of an existing file, or "" if it doesn't exist yet.
    Updating a file through the API requires the sha it's replacing; creating
    one requires that sha to be absent."""
    params = {"ref": ref} if ref else None
    try:
        data = await request("GET", f"/repos/{check_repo(repo)}/contents/{check_path(path)}",
                             params=params)
    except GitHubError as e:
        if "404" in str(e):
            return ""
        raise
    return data.get("sha", "") if isinstance(data, dict) else ""


async def list_branches(repo: str, limit: int = 30) -> list[dict]:
    return await request("GET", f"/repos/{check_repo(repo)}/branches",
                         params={"per_page": min(int(limit or 30), 100)})


async def search_code(query: str, repo: str = "", limit: int = 20) -> list[dict]:
    q = (query or "").strip()
    if not q:
        raise GitHubError("a search query is required")
    if repo:
        q = f"{q} repo:{check_repo(repo)}"
    data = await request("GET", "/search/code", params={"q": q, "per_page": min(int(limit or 20), 50)})
    return data.get("items", []) if isinstance(data, dict) else []


async def get_commit(repo: str, sha: str) -> dict:
    sha = (sha or "").strip()
    if not sha:
        raise GitHubError("a commit sha or ref is required")
    return await request("GET", f"/repos/{check_repo(repo)}/commits/{urllib.parse.quote(sha, safe='')}")


# ---------------------------------------------------------------------------
# Writing: merge, review, close, commit a file
# ---------------------------------------------------------------------------

MERGE_METHODS = ("merge", "squash", "rebase")


async def merge_pull(repo: str, number: int, method: str = "merge", title: str = "") -> dict:
    method = (method or "merge").strip().lower()
    if method not in MERGE_METHODS:
        raise GitHubError(f"merge method must be one of: {', '.join(MERGE_METHODS)}")
    body = {"merge_method": method}
    if title:
        body["commit_title"] = title
    return await request("PUT", f"/repos/{check_repo(repo)}/pulls/{int(number)}/merge", json_body=body)


REVIEW_EVENTS = ("APPROVE", "REQUEST_CHANGES", "COMMENT")


async def create_review(repo: str, number: int, event: str, body: str = "") -> dict:
    event = (event or "").strip().upper()
    if event not in REVIEW_EVENTS:
        raise GitHubError(f"review event must be one of: {', '.join(REVIEW_EVENTS)}")
    # GitHub rejects these two without a body, which is reasonable: a bare
    # "changes requested" tells the author nothing.
    if event in ("REQUEST_CHANGES", "COMMENT") and not (body or "").strip():
        raise GitHubError(f"a body is required for a {event} review")
    return await request("POST", f"/repos/{check_repo(repo)}/pulls/{int(number)}/reviews",
                         json_body={"event": event, "body": body})


async def set_issue_state(repo: str, number: int, state: str = "closed",
                          reason: str = "") -> dict:
    state = (state or "closed").strip().lower()
    if state not in ("open", "closed"):
        raise GitHubError("issue state must be 'open' or 'closed'")
    body = {"state": state}
    if reason and state == "closed":
        if reason not in ("completed", "not_planned"):
            raise GitHubError("close reason must be 'completed' or 'not_planned'")
        body["state_reason"] = reason
    return await request("PATCH", f"/repos/{check_repo(repo)}/issues/{int(number)}", json_body=body)


async def put_content(repo: str, path: str, content: str, message: str,
                      branch: str = "") -> dict:
    """Create or update one file with a commit, without a clone. The existing
    blob sha is looked up first: GitHub needs it to update, and rejects it
    when creating, so getting this wrong is the usual 422."""
    if not (message or "").strip():
        raise GitHubError("a commit message is required")
    body = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if branch:
        body["branch"] = branch
    sha = await get_content_sha(repo, path, ref=branch)
    if sha:
        body["sha"] = sha
    return await request("PUT", f"/repos/{check_repo(repo)}/contents/{check_path(path)}",
                         json_body=body)
