"""Agent mode: the working-directory registry, the tool implementations, and
the registry that decides which tools a given run may call.

Tools are grouped (shell / files / git / github / github_write / web) so a
session can hand the model a shell without also handing it the ability to
open pull requests. Groups the UI doesn't ask for are simply not sent to the
model, and are refused server-side if it invents a call anyway.

Nothing here is wired into /v1/messages -- Claude Code's own tools are
untouched by any of this.
"""

import asyncio
import json
import logging
import os
import re
import sys

import git_tools
import github_api
import integrations
import web_tools

log = logging.getLogger("claude-proxy")


def norm_path(p: str) -> str:
    """Absolute, ~-expanded, symlinks resolved. Resolving symlinks matters:
    otherwise a link inside a working directory would defeat the confinement
    check by pointing anywhere."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(p)))


AGENT_WORKSPACE = norm_path(os.getenv("AGENT_WORKSPACE", "workspace"))
os.makedirs(AGENT_WORKSPACE, exist_ok=True)

# Where github_clone puts repositories. Inside the workspace by default, so a
# fresh install can clone without anyone configuring a thing.
CLONE_ROOT = norm_path(os.getenv("AGENT_CLONE_ROOT", os.path.join(AGENT_WORKSPACE, "repos")))

# Extra working directories added through the UI live here, so they survive a
# restart the same way backends.json does.
WORKDIRS_PATH = os.getenv("AGENT_DIRS_CONFIG", "workdirs.json")

AGENT_TOOL_TIMEOUT = int(os.getenv("AGENT_TOOL_TIMEOUT", "30"))
AGENT_OUTPUT_LIMIT = int(os.getenv("AGENT_OUTPUT_LIMIT", "8000"))
# Reviewing a real codebase is dozens of reads, so the default budget is
# deliberately high -- the loop stops on its own the moment the model answers.
# The UI can raise or lower it per message, up to ROUND_CEILING.
MAX_AGENT_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "100"))
ROUND_CEILING = int(os.getenv("AGENT_ROUND_CEILING", "500"))


def clamp_rounds(requested) -> int:
    """What the browser asked for, kept inside something sane."""
    try:
        value = int(requested)
    except (TypeError, ValueError):
        return MAX_AGENT_ITERATIONS
    return max(1, min(value, ROUND_CEILING))

# search_files is the one file tool that can run long -- the model picks both
# the tree and the regex -- so it runs as a child process on the same budget
# as a shell command. What it walks and how many hits it returns is
# file_search.py's business.
SEARCH_TIMEOUT = AGENT_TOOL_TIMEOUT
SEARCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_search.py")

GROUPS = ["shell", "files", "git", "github", "github_write", "web"]
# Everything except the group that can post to other people's repositories.
DEFAULT_GROUPS = ["shell", "files", "git", "github", "web"]


def _truncate(text: str) -> str:
    if len(text) <= AGENT_OUTPUT_LIMIT:
        return text
    return text[:AGENT_OUTPUT_LIMIT] + f"\n...[truncated, {len(text)} bytes total]"


def path_within(target: str, root: str) -> bool:
    return target == root or target.startswith(root.rstrip(os.sep) + os.sep)


# ---------------------------------------------------------------------------
# Working directories -- the set of places agent mode is allowed to operate
# in. The AGENT_WORKSPACE entry is always present and can't be removed;
# everything else is added/removed from the UI and persisted to WORKDIRS_PATH.
# ---------------------------------------------------------------------------

def load_workdirs() -> list[dict]:
    saved = []
    if os.path.exists(WORKDIRS_PATH):
        try:
            with open(WORKDIRS_PATH) as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:
            log.warning("could not read %s (%s) -- starting with the default workspace only", WORKDIRS_PATH, e)
            raw = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, str):
                item = {"path": item}
            if not isinstance(item, dict) or not item.get("path"):
                continue
            path = norm_path(item["path"])
            if path == AGENT_WORKSPACE or any(d["path"] == path for d in saved):
                continue
            saved.append({
                "path": path,
                "label": item.get("label") or os.path.basename(path) or path,
                "default": False,
            })
    return [{"path": AGENT_WORKSPACE, "label": "workspace", "default": True}] + saved


WORKDIRS = load_workdirs()


def save_workdirs():
    payload = [{"path": d["path"], "label": d["label"]} for d in WORKDIRS if not d["default"]]
    tmp = f"{WORKDIRS_PATH}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, WORKDIRS_PATH)


def register_workdir(path: str, label: str = "") -> dict:
    """Add a directory to the registry if it isn't there yet. Used by the API
    and by github_clone, which registers what it just cloned."""
    path = norm_path(path)
    for d in WORKDIRS:
        if d["path"] == path:
            return d
    entry = {"path": path, "label": label or os.path.basename(path) or path, "default": False}
    WORKDIRS.append(entry)
    save_workdirs()
    return entry


def public_workdir(d: dict) -> dict:
    return {
        "path": d["path"],
        "label": d["label"],
        "default": d["default"],
        "exists": os.path.isdir(d["path"]),
        "writable": os.access(d["path"], os.W_OK),
    }


# ---------------------------------------------------------------------------
# Per-run context
# ---------------------------------------------------------------------------

class AgentContext:
    """What one agent-mode message may touch: an ordered list of working
    directories (the first is where bash starts), whether path confinement is
    on, which tool groups are in play, and whether web_fetch may reach the
    local network."""

    def __init__(self, roots: list[str], sandbox: bool, groups: list[str], lan: bool = False):
        self.roots = roots
        self.sandbox = sandbox
        self.groups = groups
        self.lan = lan

    @property
    def cwd(self) -> str:
        for root in self.roots:
            if os.path.isdir(root):
                return root
        return AGENT_WORKSPACE

    def allows(self, group: str) -> bool:
        return group in self.groups


def available_groups() -> dict:
    """Which groups are usable right now. GitHub needs a token; web search
    needs a provider key, though web_fetch works without one."""
    has_github = bool(integrations.get("github_token"))
    return {
        "shell": True,
        "files": True,
        "git": True,
        "github": has_github,
        "github_write": has_github,
        "web": True,
    }


def build_agent_context(requested_paths, sandbox_requested: bool, requested_groups=None,
                        lan_requested: bool = False) -> AgentContext:
    """Map what the browser asked for onto directories and groups that are
    actually available. Unknown paths and unusable groups are dropped rather
    than trusted -- a session picks from the registry, it doesn't invent
    entries mid-request."""
    registered = [d["path"] for d in WORKDIRS]
    roots = []
    for p in requested_paths or []:
        if not isinstance(p, str):
            continue
        path = norm_path(p)
        if path in registered and path not in roots:
            roots.append(path)
    if not roots:
        roots = [AGENT_WORKSPACE]

    usable = available_groups()
    # Only an absent/malformed selection falls back to the defaults. An
    # explicit empty list means "no tools", and a selection whose groups are
    # all unavailable means "nothing usable" -- neither may quietly turn into
    # a shell. The run just has no tools to offer, which is a plain chat.
    asked = requested_groups if isinstance(requested_groups, list) else DEFAULT_GROUPS
    groups = [g for g in GROUPS if g in asked and usable.get(g)]

    # lan_allowed clamps the browser's request against AGENT_WEB_ALLOW_PRIVATE,
    # so a request asking for the LAN when the operator hasn't allowed it just
    # gets the ordinary refusal.
    return AgentContext(roots, bool(sandbox_requested), groups,
                        web_tools.lan_allowed(lan_requested))


def resolve_agent_path(ctx: AgentContext, path: str) -> str:
    raw = os.path.expanduser(path or ".")
    target = norm_path(raw if os.path.isabs(raw) else os.path.join(ctx.cwd, raw))
    if ctx.sandbox and not any(path_within(target, root) for root in ctx.roots):
        raise ValueError(
            f"'{path}' is outside this session's working directories "
            f"({', '.join(ctx.roots)}) and the sandbox is on"
        )
    return target


# ---------------------------------------------------------------------------
# shell + files
# ---------------------------------------------------------------------------

async def _stop_process(proc):
    """Kill a child that is still running and reap it. Used on both exits that
    aren't a clean finish -- the timeout, and the run being cancelled because
    the user pressed Stop -- so neither leaves the command executing behind a
    UI that says it stopped, nor a zombie behind the loop."""
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass  # it exited between the check and the signal
    try:
        # Shielded: on the cancellation path the await would otherwise be
        # cancelled again immediately, before the child is collected.
        await asyncio.shield(proc.wait())
    except BaseException:
        pass


async def run_bash(ctx, command: str, cwd: str = "") -> str:
    try:
        workdir = resolve_agent_path(ctx, cwd) if cwd else ctx.cwd
        if not os.path.isdir(workdir):
            return f"[error] working directory '{workdir}' does not exist"
        proc = await asyncio.create_subprocess_shell(
            command, cwd=workdir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=AGENT_TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            await _stop_process(proc)
            return f"[error] command timed out after {AGENT_TOOL_TIMEOUT}s"
        except BaseException:
            # CancelledError is a BaseException, so a Stop press lands here
            # rather than in either except clause below.
            await _stop_process(proc)
            raise
        text = out.decode("utf-8", "replace")
        status = "" if proc.returncode == 0 else f"[exit {proc.returncode}]\n"
        return github_api.scrub(_truncate(status + (text or "(no output)")))
    except Exception as e:
        return f"[error] {e}"


async def run_read_file(ctx, path: str) -> str:
    try:
        target = resolve_agent_path(ctx, path)
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            # Scrubbed like every other tool's output: the file being read may
            # well be this proxy's own integrations.json.
            return github_api.scrub(_truncate(f.read()))
    except Exception as e:
        return f"[error] {e}"


async def run_write_file(ctx, path: str, content: str) -> str:
    try:
        target = resolve_agent_path(ctx, path)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content)} bytes to {target}"
    except Exception as e:
        return f"[error] {e}"


async def run_edit_file(ctx, path: str, old_text: str, new_text: str) -> str:
    try:
        target = resolve_agent_path(ctx, path)
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
        if not old_text:
            return "[error] old_text is required -- use write_file to create a file from scratch"
        occurrences = content.count(old_text)
        if occurrences == 0:
            return "[error] old_text not found in the file -- read it again and quote the exact text"
        if occurrences > 1:
            return f"[error] old_text appears {occurrences} times -- include more surrounding context so it's unique"
        with open(target, "w", encoding="utf-8") as f:
            f.write(content.replace(old_text, new_text, 1))
        return f"edited {target} (replaced {len(old_text)} bytes with {len(new_text)})"
    except Exception as e:
        return f"[error] {e}"


async def run_list_dir(ctx, path: str = ".") -> str:
    try:
        target = resolve_agent_path(ctx, path)
        entries = []
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            entries.append(f"{name}/" if os.path.isdir(full) else name)
        return github_api.scrub(_truncate("\n".join(entries))) if entries else "(empty)"
    except Exception as e:
        return f"[error] {e}"


async def run_search_files(ctx, pattern: str, path: str = ".", file_glob: str = "") -> str:
    """Runs file_search.py as a child process. See that module for why it
    isn't just a function call: a model-supplied regex that backtracks badly
    can't be stopped from inside this process, but a child can be killed."""
    try:
        root = resolve_agent_path(ctx, path)
        re.compile(pattern)  # a bad pattern is a tool error, not a dead child
    except Exception as e:
        return f"[error] {e}"

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, SEARCH_SCRIPT, root, pattern, file_glob, str(SEARCH_TIMEOUT),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as e:
        return f"[error] could not start the search: {e}"

    try:
        # The child stops itself at SEARCH_TIMEOUT; this is the backstop for
        # the one case it can't -- a single re.search that never returns.
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=SEARCH_TIMEOUT + 5)
    except asyncio.TimeoutError:
        await _stop_process(proc)
        return (f"[error] the search did not finish within {SEARCH_TIMEOUT}s and was stopped -- "
                "the pattern is probably backtracking badly, try a simpler one")
    except BaseException:
        await _stop_process(proc)  # the run was cancelled; don't leave it scanning
        raise

    hits = out.decode("utf-8", "replace").strip()
    return github_api.scrub(_truncate(hits)) if hits else "(no matches)"


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def _git_auth_args() -> list[str]:
    """Authenticate git over HTTPS with a per-command header instead of a
    credential embedded in the remote URL, so the token never lands in
    .git/config on disk."""
    token = integrations.get("github_token")
    return ["-c", f"http.extraHeader=Authorization: Bearer {token}"] if token else []


def _repo_dir(ctx, path: str) -> str:
    return resolve_agent_path(ctx, path or ".")


async def run_git_status(ctx, path: str = "") -> str:
    try:
        return github_api.scrub(await git_tools.git_status(_repo_dir(ctx, path)))
    except Exception as e:
        return f"[error] {e}"


async def run_git_diff(ctx, path: str = "", staged: bool = False, file: str = "") -> str:
    try:
        return github_api.scrub(await git_tools.git_diff(_repo_dir(ctx, path), staged=bool(staged), path=file or ""))
    except Exception as e:
        return f"[error] {e}"


async def run_git_log(ctx, path: str = "", limit: int = 10) -> str:
    try:
        return github_api.scrub(await git_tools.git_log(_repo_dir(ctx, path), limit=int(limit or 10)))
    except Exception as e:
        return f"[error] {e}"


async def run_git_branch(ctx, path: str = "") -> str:
    try:
        return github_api.scrub(await git_tools.git_branch(_repo_dir(ctx, path)))
    except Exception as e:
        return f"[error] {e}"


async def run_git_commit(ctx, message: str, path: str = "", add_all: bool = True) -> str:
    try:
        return github_api.scrub(await git_tools.git_commit(_repo_dir(ctx, path), message, add_all=bool(add_all)))
    except Exception as e:
        return f"[error] {e}"


async def run_git_checkout(ctx, branch: str, path: str = "", create: bool = False) -> str:
    try:
        return github_api.scrub(await git_tools.git_checkout(_repo_dir(ctx, path), branch, create=bool(create)))
    except Exception as e:
        return f"[error] {e}"


async def run_git_push(ctx, path: str = "", remote: str = "origin", branch: str = "") -> str:
    # Through git_tools.git_push like every other git tool, so the model can't
    # smuggle an option (--receive-pack=...) in as a remote or branch name.
    try:
        return github_api.scrub(await git_tools.git_push(
            _repo_dir(ctx, path), remote=remote, branch=branch, config_args=_git_auth_args()))
    except Exception as e:
        return f"[error] {e}"


# ---------------------------------------------------------------------------
# github
# ---------------------------------------------------------------------------

def _login(item: dict) -> str:
    """GitHub sends "user": null for anything by a deleted account, so one
    such PR or issue must not blow up the whole listing."""
    return ((item.get("user") or {}).get("login") or "(unknown)")


def _ref(item: dict, side: str) -> str:
    return ((item.get(side) or {}).get("ref") or "?")


def _short(text: str, limit: int = 400) -> str:
    text = (text or "").strip().replace("\r", "")
    return text if len(text) <= limit else text[:limit] + "…"


async def run_github_list_repos(ctx, query: str = "", limit: int = 20) -> str:
    try:
        repos = await github_api.list_repos(limit=int(limit or 20), query=query or "")
        if not repos:
            return "(no repositories)"
        lines = []
        for r in repos:
            visibility = "private" if r.get("private") else "public"
            lines.append(f"{r.get('full_name')} [{visibility}] {r.get('description') or ''}".rstrip())
        return _truncate("\n".join(lines))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_list_prs(ctx, repo: str, state: str = "open", limit: int = 20) -> str:
    try:
        pulls = await github_api.list_pulls(repo, state=state or "open", limit=int(limit or 20))
        if not pulls:
            return f"(no {state} pull requests)"
        lines = [
            f"#{p.get('number')} {p.get('title')} [{p.get('state')}] by {_login(p)} "
            f"({_ref(p, 'head')} -> {_ref(p, 'base')})"
            for p in pulls
        ]
        return _truncate("\n".join(lines))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_read_pr(ctx, repo: str, number: int, include_diff: bool = True) -> str:
    try:
        pr = await github_api.get_pull(repo, number)
        parts = [
            f"#{pr.get('number')} {pr.get('title')} [{pr.get('state')}{', merged' if pr.get('merged') else ''}]",
            f"by {_login(pr)} · {_ref(pr, 'head')} -> {_ref(pr, 'base')} · "
            f"+{pr.get('additions', 0)}/-{pr.get('deletions', 0)} across {pr.get('changed_files', 0)} file(s)",
            f"mergeable: {pr.get('mergeable_state', 'unknown')}",
            "",
            _short(pr.get("body") or "(no description)", 1500),
        ]
        comments = await github_api.list_comments(repo, number)
        if comments:
            parts.append("\n--- comments ---")
            parts += [f"{_login(c)}: {_short(c.get('body'), 500)}" for c in comments]
        if include_diff:
            diff = await github_api.get_pull_diff(repo, number)
            parts.append("\n--- diff ---")
            parts.append(diff)
        return _truncate("\n".join(parts))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_list_issues(ctx, repo: str, state: str = "open", limit: int = 20) -> str:
    try:
        issues = await github_api.list_issues(repo, state=state or "open", limit=int(limit or 20))
        if not issues:
            return f"(no {state} issues)"
        lines = [
            f"#{i.get('number')} {i.get('title')} [{i.get('state')}] by {_login(i)} "
            f"({i.get('comments', 0)} comment(s))"
            for i in issues
        ]
        return _truncate("\n".join(lines))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_read_issue(ctx, repo: str, number: int) -> str:
    try:
        issue = await github_api.get_issue(repo, number)
        parts = [
            f"#{issue.get('number')} {issue.get('title')} [{issue.get('state')}] by {_login(issue)}",
            "",
            _short(issue.get("body") or "(no description)", 1500),
        ]
        comments = await github_api.list_comments(repo, number)
        if comments:
            parts.append("\n--- comments ---")
            parts += [f"{_login(c)}: {_short(c.get('body'), 500)}" for c in comments]
        return _truncate("\n".join(parts))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_clone(ctx, repo: str, branch: str = "") -> str:
    """Clone into CLONE_ROOT and register the result as a working directory,
    then add it to this run's roots so the model can keep working in it
    without waiting for the user to tick a box."""
    try:
        repo = github_api.check_repo(repo)
        problem = git_tools.check_name("branch", branch) if branch else ""
        if problem:
            return problem
        os.makedirs(CLONE_ROOT, exist_ok=True)
        target = os.path.join(CLONE_ROOT, repo.split("/")[1])
        # check_repo already refuses '.' and '..' as a repository name; this is
        # the belt to that braces, since the answer decides what becomes a
        # working directory.
        if not path_within(norm_path(target), norm_path(CLONE_ROOT)) or norm_path(target) == norm_path(CLONE_ROOT):
            return f"[error] '{repo}' does not name a directory inside {CLONE_ROOT}"

        if os.path.isdir(os.path.join(target, ".git")):
            entry = register_workdir(target, repo.split("/")[1])
            if entry["path"] not in ctx.roots:
                ctx.roots.append(entry["path"])
            return f"already cloned at {target} (registered as a working directory)"

        url = github_api.clone_url(repo)
        branch_args = ["--branch", branch] if branch else []

        def clone_args(depth):
            return _git_auth_args() + ["clone"] + (["--depth", str(depth)] if depth else []) + branch_args + [url, target]

        # Shallow first so a big repository lands in seconds; fall back to a
        # full clone, since some hosts and mirrors reject shallow capabilities.
        out = await git_tools.run_git(CLONE_ROOT, clone_args(50), timeout=300, require_repo=False)
        if out.startswith("[error]") and "shallow" in out:
            out = await git_tools.run_git(CLONE_ROOT, clone_args(0), timeout=300, require_repo=False)
        if out.startswith("[error]") or not os.path.isdir(target):
            return github_api.scrub(out or f"[error] clone failed for {repo}")

        entry = register_workdir(target, repo.split("/")[1])
        if entry["path"] not in ctx.roots:
            ctx.roots.append(entry["path"])
        return f"cloned {repo} to {target} and registered it as a working directory"
    except (github_api.GitHubError, OSError) as e:
        return f"[error] {github_api.scrub(str(e))}"


async def run_github_create_issue(ctx, repo: str, title: str, body: str = "") -> str:
    try:
        if not title.strip():
            return "[error] title is required"
        issue = await github_api.create_issue(repo, title, body)
        return f"opened issue #{issue['number']}: {issue['html_url']}"
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_comment(ctx, repo: str, number: int, body: str) -> str:
    try:
        if not body.strip():
            return "[error] body is required"
        comment = await github_api.add_comment(repo, number, body)
        return f"commented on #{number}: {comment['html_url']}"
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_create_pr(ctx, repo: str, title: str, head: str, base: str = "", body: str = "") -> str:
    try:
        if not title.strip() or not head.strip():
            return "[error] title and head branch are required"
        target = base or await github_api.default_branch(repo)
        pr = await github_api.create_pull(repo, title, head, target, body)
        return f"opened PR #{pr['number']} ({head} -> {target}): {pr['html_url']}"
    except github_api.GitHubError as e:
        return f"[error] {e}"


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------

async def run_web_fetch(ctx, url: str) -> str:
    return await web_tools.web_fetch(url, max_chars=AGENT_OUTPUT_LIMIT, allow_private=ctx.lan)


async def run_web_search(ctx, query: str, limit: int = 5) -> str:
    return await web_tools.web_search(
        query,
        provider=integrations.get("search_provider"),
        api_key=integrations.get("search_api_key"),
        limit=int(limit or 5),
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _tool(group, name, description, properties, required=(), handler=None):
    return {
        "group": group,
        "name": name,
        "handler": handler,
        "spec": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": list(required)},
            },
        },
    }


STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}

TOOLS = [
    _tool("shell", "bash",
          "Run a shell command. Runs in the session's primary working directory unless 'cwd' says otherwise. Output (stdout+stderr) is truncated if large.",
          {"command": {**STR, "description": "Shell command to run"},
           "cwd": {**STR, "description": "Directory to run in; must be inside a working directory while sandboxed"}},
          ["command"], run_bash),

    _tool("files", "read_file",
          "Read a text file. Relative paths resolve against the primary working directory.",
          {"path": {**STR, "description": "File path, relative or absolute"}},
          ["path"], run_read_file),
    _tool("files", "write_file",
          "Write (create or overwrite) a text file, creating parent directories as needed.",
          {"path": STR, "content": {**STR, "description": "Full file content to write"}},
          ["path", "content"], run_write_file),
    _tool("files", "edit_file",
          "Replace an exact snippet of text in an existing file. Fails if the snippet is missing or appears more than once, so read the file first and quote enough context to be unique.",
          {"path": STR,
           "old_text": {**STR, "description": "Exact text to replace, including indentation"},
           "new_text": {**STR, "description": "Replacement text"}},
          ["path", "old_text", "new_text"], run_edit_file),
    _tool("files", "list_dir",
          "List the files and directories inside a directory. Defaults to the primary working directory.",
          {"path": {**STR, "description": "Directory path; defaults to '.'"}},
          (), run_list_dir),
    _tool("files", "search_files",
          "Recursively search text files for a regular expression, returning 'path:line: text' hits. Skips .git, node_modules and similar.",
          {"pattern": {**STR, "description": "Python regular expression"},
           "path": {**STR, "description": "Directory to search under; defaults to '.'"},
           "file_glob": {**STR, "description": "Optional filename glob, e.g. '*.py'"}},
          ["pattern"], run_search_files),

    _tool("git", "git_status",
          "Show the working tree status of the git repository at 'path' (defaults to the primary working directory).",
          {"path": STR}, (), run_git_status),
    _tool("git", "git_diff",
          "Show uncommitted changes. Set staged=true for what's already staged; set 'file' to limit the diff to one path.",
          {"path": STR, "staged": BOOL, "file": STR}, (), run_git_diff),
    _tool("git", "git_log",
          "Show recent commits, newest first.",
          {"path": STR, "limit": {**INT, "description": "How many commits (default 10)"}}, (), run_git_log),
    _tool("git", "git_branch",
          "Show the current branch and the list of local branches.",
          {"path": STR}, (), run_git_branch),
    _tool("git", "git_commit",
          "Stage everything (unless add_all=false) and commit with the given message.",
          {"message": {**STR, "description": "Commit message"}, "path": STR, "add_all": BOOL},
          ["message"], run_git_commit),
    _tool("git", "git_checkout",
          "Switch branches, or create a branch first with create=true.",
          {"branch": STR, "path": STR, "create": BOOL}, ["branch"], run_git_checkout),
    _tool("git", "git_push",
          "Push commits to a remote. Give 'branch' to push and set upstream for a new branch.",
          {"path": STR, "remote": {**STR, "description": "Defaults to origin"}, "branch": STR},
          (), run_git_push),

    _tool("github", "github_list_repos",
          "List the authenticated user's repositories, most recently pushed first. Pass 'query' to search GitHub instead.",
          {"query": STR, "limit": INT}, (), run_github_list_repos),
    _tool("github", "github_clone",
          "Clone a GitHub repository onto this machine and register it as a working directory, so the file and git tools can work on it immediately.",
          {"repo": {**STR, "description": "owner/repo"}, "branch": STR}, ["repo"], run_github_clone),
    _tool("github", "github_list_prs",
          "List pull requests on a repository.",
          {"repo": {**STR, "description": "owner/repo"},
           "state": {**STR, "description": "open, closed or all"}, "limit": INT},
          ["repo"], run_github_list_prs),
    _tool("github", "github_read_pr",
          "Read one pull request: description, comments, and (by default) its full diff.",
          {"repo": STR, "number": INT, "include_diff": BOOL}, ["repo", "number"], run_github_read_pr),
    _tool("github", "github_list_issues",
          "List issues on a repository (pull requests excluded).",
          {"repo": STR, "state": STR, "limit": INT}, ["repo"], run_github_list_issues),
    _tool("github", "github_read_issue",
          "Read one issue with its comments.",
          {"repo": STR, "number": INT}, ["repo", "number"], run_github_read_issue),

    _tool("github_write", "github_create_issue",
          "Open a new issue on a repository. This is publicly visible -- only do it when the user asked for it.",
          {"repo": STR, "title": STR, "body": STR}, ["repo", "title"], run_github_create_issue),
    _tool("github_write", "github_comment",
          "Post a comment on an issue or pull request. This is publicly visible -- only do it when the user asked for it.",
          {"repo": STR, "number": INT, "body": STR}, ["repo", "number", "body"], run_github_comment),
    _tool("github_write", "github_create_pr",
          "Open a pull request from an already-pushed branch. Base defaults to the repository's default branch.",
          {"repo": STR, "title": STR, "head": {**STR, "description": "Branch with the changes"},
           "base": STR, "body": STR}, ["repo", "title", "head"], run_github_create_pr),

    _tool("web", "web_fetch",
          "Fetch a URL and return its readable text content.",
          {"url": STR}, ["url"], run_web_fetch),
    _tool("web", "web_search",
          "Search the web and return the top results with links and snippets.",
          {"query": STR, "limit": INT}, ["query"], run_web_search),
]

BY_NAME = {t["name"]: t for t in TOOLS}


def tools_for(ctx: AgentContext) -> list[dict]:
    """The OpenAI-format tool specs this run is allowed to use. web_search is
    dropped when no search key is set, so the model isn't offered a tool that
    can only fail."""
    specs = []
    has_search_key = bool(integrations.get("search_api_key"))
    for tool in TOOLS:
        if not ctx.allows(tool["group"]):
            continue
        if tool["name"] == "web_search" and not has_search_key:
            continue
        specs.append(tool["spec"])
    return specs


async def execute_tool_call(ctx: AgentContext, name: str, args: dict) -> str:
    tool = BY_NAME.get(name)
    if tool is None:
        return f"[error] unknown tool '{name}'"
    if not ctx.allows(tool["group"]):
        return f"[error] the '{tool['group']}' tool group is switched off for this session"
    try:
        return await tool["handler"](ctx, **{k: v for k, v in (args or {}).items() if isinstance(k, str)})
    except TypeError as e:
        # A model passing junk arguments shouldn't kill the run.
        return f"[error] bad arguments for {name}: {e}"
    except Exception as e:
        return f"[error] {name} failed: {github_api.scrub(str(e))}"


def agent_system_prompt(ctx: AgentContext) -> str:
    dirs = "\n".join(f"- {p}" + ("  (primary -- bash starts here)" if p == ctx.cwd else "") for p in ctx.roots)
    if ctx.sandbox:
        scope = ("The file tools refuse any path outside those directories. If you need "
                 "something elsewhere, say so instead of trying to work around it.")
    else:
        scope = ("The sandbox is OFF for this session: the file tools can reach anywhere "
                 "on this machine. Stay inside the directories above unless the user "
                 "clearly asks otherwise, and never touch system files casually.")

    lines = [
        "You are an agent with real tool access to the machine hosting this proxy.",
        "",
        f"Working directories:\n{dirs}",
        "",
        scope,
        "",
        "Use the tools to check things rather than guessing: read files before editing "
        "them, and verify a change landed after making it. Prefer edit_file over "
        "rewriting a whole file.",
    ]
    if ctx.allows("git"):
        lines.append(
            "For repository work use the git_* tools rather than shell git. Commit only "
            "when asked, with a message describing why the change was made."
        )
    if ctx.allows("github"):
        lines.append(
            "github_clone puts a repository on this machine and makes it a working "
            "directory, which is how you get from 'a repo on GitHub' to 'files you can edit'."
        )
    if ctx.allows("github_write"):
        lines.append(
            "You can open issues, comment, and open pull requests. Those are public "
            "actions other people will see: do them only when the user has asked for "
            "that specific thing, and never as a way of 'reporting' your own progress."
        )
    elif ctx.allows("github"):
        lines.append(
            "You can read GitHub but not write to it -- no issues, comments, or pull "
            "requests. Say so if the user asks for one."
        )
    if ctx.allows("web"):
        lines.append("Use web_fetch/web_search for anything you'd otherwise be guessing about.")
        if ctx.lan:
            lines.append(
                "The LAN switch is ON for this message: web_fetch may also reach private "
                "addresses on this machine's own network -- routers, other containers, "
                "internal dashboards. Only go there when the user's request is actually "
                "about their network, and never scan or enumerate hosts they didn't name."
            )
        else:
            lines.append(
                "web_fetch reaches the public internet only; private and loopback "
                "addresses are refused. Say so rather than trying to work around it."
            )

    lines.append(
        "When you're done, reply with a short plain-text summary of what you did and "
        "what you found."
    )
    # The "" entries above are deliberate blank separators between the intro,
    # the directory list, the scope rule and the instructions -- joined as-is.
    return "\n".join(lines)
