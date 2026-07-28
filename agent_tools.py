"""Agent mode: the working-directory registry, the tool implementations, and
the registry that decides which tools a given run may call.

Tools are grouped (shell / files / files_write / git / github / github_write /
web) so a session can hand the model a shell without also handing it the
ability to delete files or open pull requests. Groups the UI doesn't ask for
are simply not sent to the model, and are refused server-side if it invents a
call anyway. The two groups that do something you can't take back --
files_write and github_write -- are off by default.

Nothing here is wired into /v1/messages -- Claude Code's own tools are
untouched by any of this.
"""

import asyncio
import fnmatch
import json
import logging
import os
import re
import shlex
import shutil
import sys

import git_tools
import github_api
import integrations
import ssh_tools
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
# Ceiling for bash's optional per-call timeout. The default above stays 30s;
# this only bounds what a single call may ask for.
BASH_MAX_TIMEOUT = int(os.getenv("AGENT_BASH_MAX_TIMEOUT", "300"))
AGENT_OUTPUT_LIMIT = int(os.getenv("AGENT_OUTPUT_LIMIT", "8000"))
# Reviewing a real codebase is dozens of reads, so the default budget is
# deliberately high -- the loop stops on its own the moment the model answers.
# The UI can raise or lower it per message, up to ROUND_CEILING.
MAX_AGENT_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "100"))
ROUND_CEILING = int(os.getenv("AGENT_ROUND_CEILING", "500"))
# The round budget bounds rounds, not calls, and a model is free to ask for any
# number of tools in one answer. Without a cap, one confused round can spend an
# unbounded amount of time before the loop gets to check anything.
MAX_CALLS_PER_ROUND = int(os.getenv("AGENT_MAX_CALLS_PER_ROUND", "25"))


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

GROUPS = ["shell", "files", "files_write", "git", "github", "github_write", "web", "ssh"]
# Everything except the three groups that do something you can't take back:
# post to other people's repositories, delete files, and act on a machine this
# proxy's sandbox has no authority over.
DEFAULT_GROUPS = ["shell", "files", "git", "github", "web"]

# How tool calls are gated in agent mode:
#   manual -- the run pauses before each call and waits for the browser
#   auto   -- calls run as they come, the Stop button is the only brake
#   skip   -- same as auto; named separately so the UI can say what it means
# Only "manual" changes behaviour; the other two are the same code path with
# different labels, which is worth keeping so "auto" can grow a policy later
# without the UI having to change again.
APPROVAL_MODES = ("manual", "auto", "skip")

# Directories find_files never walks into, matching file_search.py -- a hit
# inside .git or node_modules is noise in both tools.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}
MAX_FIND_RESULTS = 300


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

    def __init__(self, roots: list[str], sandbox: bool, groups: list[str], lan: bool = False,
                 approval: str = "skip"):
        self.roots = roots
        self.sandbox = sandbox
        self.groups = groups
        self.lan = lan
        self.approval = approval if approval in APPROVAL_MODES else "skip"

    @property
    def cwd(self) -> str:
        for root in self.roots:
            if os.path.isdir(root):
                return root
        return AGENT_WORKSPACE

    def allows(self, group: str) -> bool:
        return group in self.groups


# There is deliberately no notion of a group being "unavailable" any more.
#
# Availability was a second gate in front of the checkbox: GitHub needed a
# token, ssh needed a registered host, and the Agent tab disabled the control
# until you satisfied it. That put the answer in the wrong place. Whether a
# credential exists is something the tool discovers when it runs, and every
# tool here already reports it well -- the GitHub tools raise "no GitHub token
# configured -- add one in the UI's Agent tab", ssh_list_hosts says the
# registry is empty, web_search names the missing provider key. A greyed-out
# box, by contrast, tells you nothing about what to do and disables the very
# control you came to use while you set the credential up.
#
# So the tick is the only gate. What a group can actually do when you tick it
# is between you and the tool's own error message.


def build_agent_context(requested_paths, sandbox_requested: bool, requested_groups=None,
                        lan_requested: bool = False, approval: str = "skip") -> AgentContext:
    """Map what the browser asked for onto directories and groups this build
    actually has. Unknown paths and unknown group names are dropped rather than
    trusted -- a session picks from the registry, it doesn't invent entries
    mid-request."""
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

    # Only an absent/malformed selection falls back to the defaults. An
    # explicit empty list means "no tools" and must stay that way rather than
    # quietly turning back into a shell: the run simply has nothing to offer,
    # which is a plain chat. Filtering against GROUPS drops names this build
    # doesn't have, which is all the filtering left to do.
    asked = requested_groups if isinstance(requested_groups, list) else DEFAULT_GROUPS
    groups = [g for g in GROUPS if g in asked]

    # lan_allowed clamps the browser's request against AGENT_WEB_ALLOW_PRIVATE,
    # so a request asking for the LAN when the operator hasn't allowed it just
    # gets the ordinary refusal.
    return AgentContext(roots, bool(sandbox_requested), groups,
                        web_tools.lan_allowed(lan_requested),
                        approval if approval in APPROVAL_MODES else "skip")


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

# One implementation for every subprocess tool in the project; it lives in
# git_tools (which this module already imports) so the git wrappers get the
# same cancellation safety.
_stop_process = git_tools.stop_process


def clamp_tool_timeout(requested) -> int:
    """Per-call timeout for bash, bounded. The 30s default is right for the
    commands agent mode mostly runs, but a test suite or a build is a normal
    thing to ask for and simply cannot finish inside it."""
    try:
        value = int(requested)
    except (TypeError, ValueError):
        return AGENT_TOOL_TIMEOUT
    return max(1, min(value, BASH_MAX_TIMEOUT))


async def run_bash(ctx, command: str, cwd: str = "", timeout=None) -> str:
    seconds = clamp_tool_timeout(timeout)
    try:
        workdir = resolve_agent_path(ctx, cwd) if cwd else ctx.cwd
        if not os.path.isdir(workdir):
            return f"[error] working directory '{workdir}' does not exist"
        proc = await asyncio.create_subprocess_shell(
            command, cwd=workdir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=seconds)
        except asyncio.TimeoutError:
            await _stop_process(proc)
            return (f"[error] command timed out after {seconds}s. Pass a larger "
                    f"'timeout' if it genuinely needs longer (max {BASH_MAX_TIMEOUT}s).")
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


async def run_read_file(ctx, path: str, offset: int = 0, limit: int = 0) -> str:
    """Whole file by default, or a line range. The range matters for files
    bigger than the output budget: without it, everything past the cut is
    simply unreachable, and the model has no way to ask for the rest."""
    try:
        target = resolve_agent_path(ctx, path)
        try:
            start = max(1, int(offset or 1))
        except (TypeError, ValueError):
            return f"[error] offset '{offset}' is not a line number"
        try:
            count = int(limit or 0)
        except (TypeError, ValueError):
            return f"[error] limit '{limit}' is not a number of lines"

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        if total and start > total:
            return f"[error] offset {start} is past the end of the file ({total} lines)"
        end = total if count <= 0 else min(total, start - 1 + count)

        # Scrubbed like every other tool's output: the file being read may
        # well be this proxy's own integrations.json.
        body = github_api.scrub(_truncate("".join(lines[start - 1:end])))
        # The header only appears when the answer isn't the whole file, so an
        # ordinary read still returns content the model can quote verbatim
        # into edit_file without stripping anything first.
        if start > 1 or end < total:
            return f"[lines {start}-{end} of {total}]\n{body}"
        return body
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


async def run_find_files(ctx, pattern: str, path: str = ".") -> str:
    """Match filenames, where search_files matches file contents. Between them
    they cover both halves of "where is that thing" without dropping to bash.

    No child process needed: this only reads directory entries and compares
    names with fnmatch, so there's no model-supplied regex to backtrack. It
    still runs in a thread, since a deep tree is a lot of syscalls."""
    if not (pattern or "").strip():
        return "[error] pattern is required, e.g. '*.py' or 'docker-compose*'"
    try:
        root = resolve_agent_path(ctx, path)
    except Exception as e:
        return f"[error] {e}"
    if not os.path.isdir(root):
        return f"[error] '{root}' is not a directory"

    hits = []
    capped = False

    def walk():
        nonlocal capped
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                if not fnmatch.fnmatch(name, pattern):
                    continue
                hits.append(os.path.relpath(os.path.join(dirpath, name), root))
                if len(hits) >= MAX_FIND_RESULTS:
                    capped = True
                    return

    try:
        await asyncio.to_thread(walk)
    except OSError as e:
        return f"[error] {e}"

    if not hits:
        return f"(no files matching '{pattern}' under {root})"
    body = "\n".join(hits)
    if capped:
        body += f"\n...[stopped at {MAX_FIND_RESULTS} matches -- narrow the pattern or the path]"
    return github_api.scrub(_truncate(body))


async def run_move_file(ctx, source: str, destination: str) -> str:
    try:
        src = resolve_agent_path(ctx, source)
        dst = resolve_agent_path(ctx, destination)
        if not os.path.exists(src) and not os.path.islink(src):
            return f"[error] '{src}' does not exist"
        if os.path.exists(dst):
            return f"[error] '{dst}' already exists -- moving onto it would overwrite it"
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # shutil.move rather than os.replace: a working directory and the
        # clone root can be on different filesystems, where rename(2) fails.
        await asyncio.to_thread(shutil.move, src, dst)
        return f"moved {src} -> {dst}"
    except Exception as e:
        return f"[error] {e}"


async def run_delete_file(ctx, path: str) -> str:
    """One file at a time, never a directory. A model that has misunderstood
    the task can delete a file and be corrected; a recursive tree delete is
    not something anyone recovers from mid-conversation."""
    try:
        target = resolve_agent_path(ctx, path)
        if os.path.isdir(target) and not os.path.islink(target):
            return ("[error] that's a directory, and this tool only deletes single files. "
                    "Delete the files individually if that's really what's wanted.")
        if not os.path.exists(target) and not os.path.islink(target):
            return f"[error] '{target}' does not exist"
        if any(target == norm_path(root) for root in ctx.roots):
            return f"[error] '{target}' is one of this session's working directories"
        os.remove(target)
        return f"deleted {target}"
    except Exception as e:
        return f"[error] {e}"


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def _git_auth_args() -> list[str]:
    """Authenticate git over HTTPS with a per-command header instead of a
    credential embedded in the remote URL, so the token never lands in
    .git/config on disk."""
    token = integrations.get("github_token")
    if not token:
        return []
    # Classic PATs (ghp_*) use "token" prefix; fine-grained PATs (github_pat_*)
    # and GitHub App tokens use "Bearer".
    prefix = "token" if token.startswith("ghp_") else "Bearer"
    return ["-c", f"http.extraHeader=Authorization: {prefix} {token}"]


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


async def run_git_fetch(ctx, path: str = "", remote: str = "origin") -> str:
    try:
        return github_api.scrub(await git_tools.git_fetch(
            _repo_dir(ctx, path), remote=remote, config_args=_git_auth_args()))
    except Exception as e:
        return f"[error] {e}"


async def run_git_pull(ctx, path: str = "", remote: str = "origin", branch: str = "") -> str:
    try:
        return github_api.scrub(await git_tools.git_pull(
            _repo_dir(ctx, path), remote=remote, branch=branch, config_args=_git_auth_args()))
    except Exception as e:
        return f"[error] {e}"


async def run_git_show(ctx, ref: str = "HEAD", path: str = "", stat_only: bool = False) -> str:
    try:
        return github_api.scrub(await git_tools.git_show(
            _repo_dir(ctx, path), ref=ref or "HEAD", stat_only=bool(stat_only)))
    except Exception as e:
        return f"[error] {e}"


async def run_git_stash(ctx, action: str = "list", path: str = "", message: str = "") -> str:
    try:
        return github_api.scrub(await git_tools.git_stash(
            _repo_dir(ctx, path), action=action or "list", message=message or ""))
    except Exception as e:
        return f"[error] {e}"


async def run_git_restore(ctx, file: str, path: str = "", staged: bool = False) -> str:
    try:
        return github_api.scrub(await git_tools.git_restore(
            _repo_dir(ctx, path), path=file, staged=bool(staged)))
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


async def run_github_ci(ctx, repo: str, branch: str = "", limit: int = 10) -> str:
    """Recent workflow runs. 'conclusion' is the answer to "is it green" --
    'status' only says whether it has finished."""
    try:
        runs = await github_api.list_workflow_runs(repo, branch=branch or "", limit=int(limit or 10))
        if not runs:
            return "(no workflow runs)" + (f" on branch {branch}" if branch else "")
        lines = []
        for r in runs:
            outcome = r.get("conclusion") or r.get("status") or "unknown"
            lines.append(
                f"{outcome:<12} {r.get('name') or '(unnamed)'} · {r.get('head_branch') or '?'} · "
                f"{(r.get('head_sha') or '')[:7]} · {r.get('created_at') or ''}"
            )
        return _truncate("\n".join(lines))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_read_file(ctx, repo: str, path: str, ref: str = "") -> str:
    """Read one file straight from GitHub. Cloning a whole repository to look
    at a single file is the slow way round, especially for a repo the user
    only wants an opinion on."""
    try:
        content = await github_api.get_content(repo, path, ref=ref or "")
        if not isinstance(content, str):
            return "[error] that path is a directory or a binary file, not readable text"
        header = f"{repo}:{path}" + (f"@{ref}" if ref else "")
        return github_api.scrub(_truncate(f"{header}\n\n{content}"))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_list_branches(ctx, repo: str, limit: int = 30) -> str:
    try:
        branches = await github_api.list_branches(repo, limit=int(limit or 30))
        if not branches:
            return "(no branches)"
        default = await github_api.default_branch(repo)
        lines = []
        for b in branches:
            name = b.get("name") or "?"
            mark = "  (default)" if name == default else ""
            protected = " [protected]" if b.get("protected") else ""
            lines.append(f"{name}{mark}{protected}")
        return _truncate("\n".join(lines))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_search_code(ctx, query: str, repo: str = "", limit: int = 20) -> str:
    try:
        items = await github_api.search_code(query, repo=repo or "", limit=int(limit or 20))
        if not items:
            return "(no code matches)"
        lines = [
            f"{(i.get('repository') or {}).get('full_name') or '?'}: {i.get('path') or '?'}"
            for i in items
        ]
        return _truncate("\n".join(lines))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_read_commit(ctx, repo: str, sha: str) -> str:
    try:
        c = await github_api.get_commit(repo, sha)
        commit = c.get("commit") or {}
        author = (commit.get("author") or {}).get("name") or _login(c)
        stats = c.get("stats") or {}
        files = c.get("files") or []
        parts = [
            f"{(c.get('sha') or '')[:10]} {(commit.get('message') or '').splitlines()[0] if commit.get('message') else ''}",
            f"by {author} · {(commit.get('author') or {}).get('date') or ''} · "
            f"+{stats.get('additions', 0)}/-{stats.get('deletions', 0)} across {len(files)} file(s)",
            "",
            _short(commit.get("message") or "", 1500),
            "",
            "--- files ---",
        ]
        parts += [f"{f.get('status', '?'):<9} {f.get('filename')}" for f in files]
        return github_api.scrub(_truncate("\n".join(parts)))
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_merge_pr(ctx, repo: str, number: int, method: str = "merge", title: str = "") -> str:
    try:
        result = await github_api.merge_pull(repo, number, method=method or "merge", title=title or "")
        if not result.get("merged"):
            return f"[error] GitHub did not merge #{number}: {result.get('message') or 'unknown reason'}"
        return f"merged #{number} ({method or 'merge'}): {result.get('sha', '')[:10]}"
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_review_pr(ctx, repo: str, number: int, event: str, body: str = "") -> str:
    try:
        review = await github_api.create_review(repo, number, event, body or "")
        return f"submitted a {event.upper()} review on #{number}: {review.get('html_url', '')}"
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_close_issue(ctx, repo: str, number: int, reason: str = "completed") -> str:
    try:
        issue = await github_api.set_issue_state(repo, number, state="closed", reason=reason or "completed")
        return f"closed #{number} ({reason or 'completed'}): {issue.get('html_url', '')}"
    except github_api.GitHubError as e:
        return f"[error] {e}"


async def run_github_write_file(ctx, repo: str, path: str, content: str, message: str,
                                branch: str = "") -> str:
    """Commit one file through the API, no clone involved. Creates or updates
    depending on whether the path already exists on that branch."""
    try:
        result = await github_api.put_content(repo, path, content, message, branch=branch or "")
        commit = (result.get("commit") or {})
        return (f"committed {path} to {repo}"
                + (f" on {branch}" if branch else "")
                + f": {(commit.get('sha') or '')[:10]} {commit.get('html_url', '')}")
    except github_api.GitHubError as e:
        return f"[error] {e}"


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------

async def run_web_fetch(ctx, url: str) -> str:
    return await web_tools.web_fetch(url, max_chars=AGENT_OUTPUT_LIMIT, allow_private=ctx.lan)


async def run_http_request(ctx, method: str, url: str, headers=None, body: str = "") -> str:
    # Scrubbed on the way out: an API's error body can echo back the very
    # token that was sent to it.
    return github_api.scrub(await web_tools.http_request(
        method, url, headers=headers, body=body or "",
        max_chars=AGENT_OUTPUT_LIMIT, allow_private=ctx.lan))


async def run_web_search(ctx, query: str, limit: int = 5) -> str:
    return await web_tools.web_search(
        query,
        provider=integrations.get("search_provider"),
        api_key=integrations.get("search_api_key"),
        limit=int(limit or 5),
    )


# ---------------------------------------------------------------------------
# ssh
#
# Every handler resolves a registered host *name* first. There is deliberately
# no way to give these tools a connection string: the registry is the list of
# machines this proxy may reach, and it is edited by a person, not chosen by a
# model mid-conversation.
# ---------------------------------------------------------------------------

def _ssh_host(name: str):
    """(host, "") or (None, error). Kept separate so every handler refuses an
    unknown name the same way, and points at the tool that lists them."""
    host = ssh_tools.find_host((name or "").strip())
    if host is None:
        known = ", ".join(h["name"] for h in ssh_tools.HOSTS) or "none registered"
        return None, (f"[error] no SSH host called '{name}' is registered. "
                      f"Known hosts: {known}. Use ssh_list_hosts to see them.")
    return host, ""


async def run_ssh_list_hosts(ctx) -> str:
    if not ssh_tools.HOSTS:
        return ("No SSH hosts are registered. They are added in the Agent tab of "
                "the web UI; you cannot add one yourself.")
    lines = []
    for h in ssh_tools.HOSTS:
        key = "configured key" if h["key_path"] else "default keys/agent"
        label = f"  ({h['label']})" if h["label"] != h["name"] else ""
        lines.append(f"{h['name']}: {ssh_tools.describe(h)}  [{key}]{label}")
    return "\n".join(lines)


async def run_ssh_run(ctx, host: str, command: str, cwd: str = "", timeout=None) -> str:
    entry, error = _ssh_host(host)
    if error:
        return error
    if cwd:
        problem = ssh_tools.check_remote_path(cwd)
        if problem:
            return problem
        # The far side of ssh is a shell by definition, so this is quoting for
        # that shell, not an attempt to avoid one.
        command = f"cd {shlex.quote(cwd)} && {command}"
    out = await ssh_tools.run_command(entry, command, ssh_tools.clamp_timeout(timeout))
    return github_api.scrub(out)


async def run_ssh_upload(ctx, host: str, local_path: str, remote_path: str, timeout=None) -> str:
    entry, error = _ssh_host(host)
    if error:
        return error
    problem = ssh_tools.check_remote_path(remote_path)
    if problem:
        return problem
    try:
        # The local side goes through the same confinement as every other file
        # tool: the sandbox switch decides what may leave this machine.
        local = resolve_agent_path(ctx, local_path)
    except ValueError as e:
        return f"[error] {e}"
    if not os.path.isfile(local):
        return f"[error] '{local}' is not a file"
    return github_api.scrub(await ssh_tools.copy_file(
        entry, local, remote_path, upload=True, timeout=ssh_tools.clamp_timeout(timeout)))


async def run_ssh_download(ctx, host: str, remote_path: str, local_path: str, timeout=None) -> str:
    entry, error = _ssh_host(host)
    if error:
        return error
    problem = ssh_tools.check_remote_path(remote_path)
    if problem:
        return problem
    try:
        local = resolve_agent_path(ctx, local_path)
    except ValueError as e:
        return f"[error] {e}"
    if os.path.exists(local):
        # Same rule as move_file: silently replacing a local file with a
        # remote one is not something to do without being asked.
        return f"[error] '{local}' already exists; choose another name"
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    return github_api.scrub(await ssh_tools.copy_file(
        entry, local, remote_path, upload=False, timeout=ssh_tools.clamp_timeout(timeout)))


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
OBJ = {"type": "object"}

TOOLS = [
    _tool("shell", "bash",
          "Run a shell command. Runs in the session's primary working directory unless 'cwd' says otherwise. Output (stdout+stderr) is truncated if large.",
          {"command": {**STR, "description": "Shell command to run"},
           "cwd": {**STR, "description": "Directory to run in; must be inside a working directory while sandboxed"},
           "timeout": {**INT, "description": f"Seconds to allow; defaults to {AGENT_TOOL_TIMEOUT}, max {BASH_MAX_TIMEOUT}. Raise it for builds and test suites."}},
          ["command"], run_bash),

    _tool("files", "read_file",
          "Read a text file. Relative paths resolve against the primary working directory. For a file too big to return at once, page through it with offset/limit.",
          {"path": {**STR, "description": "File path, relative or absolute"},
           "offset": {**INT, "description": "First line to return, 1-based; omit for the start"},
           "limit": {**INT, "description": "How many lines to return; omit for the whole file"}},
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
    _tool("files", "find_files",
          "Find files by NAME using a glob, recursively. Use this to locate a file ('*.env', 'docker-compose*'); use search_files to find text inside files.",
          {"pattern": {**STR, "description": "Filename glob, e.g. '*.py' or 'Dockerfile*'"},
           "path": {**STR, "description": "Directory to search under; defaults to '.'"}},
          ["pattern"], run_find_files),

    _tool("files_write", "move_file",
          "Move or rename a file or directory. Refuses to overwrite an existing destination.",
          {"source": STR, "destination": STR}, ["source", "destination"], run_move_file),
    _tool("files_write", "delete_file",
          "Delete a single file. Cannot delete directories. This is not reversible -- only do it when the user asked for it.",
          {"path": STR}, ["path"], run_delete_file),

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
    _tool("git", "git_fetch",
          "Update remote-tracking branches without changing the working tree. Safe to run any time; do this before judging whether a clone is out of date.",
          {"path": STR, "remote": {**STR, "description": "Defaults to origin"}}, (), run_git_fetch),
    _tool("git", "git_pull",
          "Fast-forward the current branch from a remote. Refuses to merge or rebase: if the branches have diverged it reports that instead.",
          {"path": STR, "remote": STR, "branch": STR}, (), run_git_pull),
    _tool("git", "git_show",
          "Show one commit: message, author and its diff. Set stat_only=true for just the changed-file list, which is much shorter for a big commit.",
          {"ref": {**STR, "description": "Commit sha, tag or ref; defaults to HEAD"},
           "path": STR, "stat_only": BOOL}, (), run_git_show),
    _tool("git", "git_stash",
          "Save, restore or list stashed changes. action is one of push, pop, list, drop.",
          {"action": {**STR, "description": "push, pop, list or drop; defaults to list"},
           "path": STR, "message": STR}, (), run_git_stash),
    _tool("git", "git_restore",
          "Discard uncommitted changes to a named file, or unstage it with staged=true. This throws away work that was never committed, so name the file explicitly.",
          {"file": {**STR, "description": "File to restore -- required"},
           "path": STR, "staged": BOOL}, ["file"], run_git_restore),

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
    _tool("github", "github_ci",
          "Recent GitHub Actions workflow runs for a repository -- this is how to answer 'is the build passing'. Read the 'conclusion' column (success/failure), not 'status'.",
          {"repo": STR, "branch": {**STR, "description": "Limit to one branch"}, "limit": INT},
          ["repo"], run_github_ci),
    _tool("github", "github_read_file",
          "Read a single file's contents straight from a GitHub repository, without cloning it.",
          {"repo": STR, "path": {**STR, "description": "Path within the repository"},
           "ref": {**STR, "description": "Branch, tag or sha; defaults to the default branch"}},
          ["repo", "path"], run_github_read_file),
    _tool("github", "github_list_branches",
          "List a repository's branches, marking the default one and any that are protected.",
          {"repo": STR, "limit": INT}, ["repo"], run_github_list_branches),
    _tool("github", "github_search_code",
          "Search code on GitHub, optionally limited to one repository. Returns matching file paths.",
          {"query": STR, "repo": {**STR, "description": "Limit to owner/repo"}, "limit": INT},
          ["query"], run_github_search_code),
    _tool("github", "github_read_commit",
          "Read one commit from GitHub: message, author, stats and the list of changed files.",
          {"repo": STR, "sha": {**STR, "description": "Commit sha, branch or tag"}},
          ["repo", "sha"], run_github_read_commit),

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
    _tool("github_write", "github_merge_pr",
          "Merge a pull request. Irreversible and visible to everyone on the repository -- only when the user asked for this specific PR to be merged.",
          {"repo": STR, "number": INT,
           "method": {**STR, "description": "merge, squash or rebase; defaults to merge"},
           "title": STR}, ["repo", "number"], run_github_merge_pr),
    _tool("github_write", "github_review_pr",
          "Submit a review on a pull request: APPROVE, REQUEST_CHANGES or COMMENT. Publicly visible, and an approval may be what lets it merge -- only do it when asked.",
          {"repo": STR, "number": INT,
           "event": {**STR, "description": "APPROVE, REQUEST_CHANGES or COMMENT"},
           "body": {**STR, "description": "Review text; required except for APPROVE"}},
          ["repo", "number", "event"], run_github_review_pr),
    _tool("github_write", "github_close_issue",
          "Close an issue, optionally as not_planned. Publicly visible.",
          {"repo": STR, "number": INT,
           "reason": {**STR, "description": "completed or not_planned"}},
          ["repo", "number"], run_github_close_issue),
    _tool("github_write", "github_write_file",
          "Create or update one file in a GitHub repository with a commit, without cloning it. Commits directly to the named branch.",
          {"repo": STR, "path": STR, "content": {**STR, "description": "Full new file content"},
           "message": {**STR, "description": "Commit message"},
           "branch": {**STR, "description": "Defaults to the repository's default branch"}},
          ["repo", "path", "content", "message"], run_github_write_file),

    _tool("web", "web_fetch",
          "Fetch a URL and return its readable text content, with markup stripped. For reading pages.",
          {"url": STR}, ["url"], run_web_fetch),
    _tool("web", "web_search",
          "Search the web and return the top results with links and snippets.",
          {"query": STR, "limit": INT}, ["query"], run_web_search),
    _tool("web", "http_request",
          "Make one HTTP request with an explicit method, headers and body, returning the raw response. For talking to APIs, where web_fetch's GET-and-strip-markup is the wrong shape.",
          {"method": {**STR, "description": "GET, POST, PUT, PATCH, DELETE, HEAD or OPTIONS"},
           "url": STR,
           "headers": {**OBJ, "description": "Header name -> value, e.g. {\"Authorization\": \"Bearer ...\"}"},
           "body": {**STR, "description": "Request body, already serialised (e.g. a JSON string)"}},
          ["method", "url"], run_http_request),

    _tool("ssh", "ssh_list_hosts",
          "List the SSH hosts this proxy is allowed to connect to. Commands run on a host by NAME from this list; there is no way to connect to anything else.",
          {}, (), run_ssh_list_hosts),
    _tool("ssh", "ssh_run",
          "Run a shell command on a registered remote host and return its output. Non-interactive: a command that waits for input will time out rather than prompt.",
          {"host": {**STR, "description": "Registered host name, from ssh_list_hosts"},
           "command": {**STR, "description": "Shell command to run on that host"},
           "cwd": {**STR, "description": "Directory on the remote host to run in"},
           "timeout": {**INT, "description": f"Seconds to allow; defaults to {ssh_tools.SSH_TIMEOUT}, max {ssh_tools.SSH_MAX_TIMEOUT}"}},
          ["host", "command"], run_ssh_run),
    _tool("ssh", "ssh_upload",
          "Copy a local file to a registered remote host. The local path obeys this session's working directories exactly as the file tools do.",
          {"host": STR,
           "local_path": {**STR, "description": "File on this machine, relative or absolute"},
           "remote_path": {**STR, "description": "Destination path on the remote host"},
           "timeout": INT},
          ["host", "local_path", "remote_path"], run_ssh_upload),
    _tool("ssh", "ssh_download",
          "Copy a file from a registered remote host to this machine. Refuses to overwrite an existing local file.",
          {"host": STR,
           "remote_path": {**STR, "description": "File on the remote host"},
           "local_path": {**STR, "description": "Destination on this machine; must not already exist"},
           "timeout": INT},
          ["host", "remote_path", "local_path"], run_ssh_download),
]

BY_NAME = {t["name"]: t for t in TOOLS}


def group_of(name: str) -> str:
    """Which group a tool belongs to, for the UI's approval card -- 'bash' and
    'read_file' deserve different amounts of hesitation."""
    tool = BY_NAME.get(name)
    return tool["group"] if tool else "unknown"


def tools_for(ctx: AgentContext) -> list[dict]:
    """The OpenAI-format tool specs this run is allowed to use -- which is
    exactly what the ticked groups say, and nothing else.

    web_search used to be dropped here when no search key was set. It is
    offered now: web_tools.web_search answers "no search API key configured --
    add one in the UI's Agent tab", which tells the model something true it
    can pass on, where a silently absent tool just made it guess why searching
    wasn't working."""
    return [tool["spec"] for tool in TOOLS if ctx.allows(tool["group"])]


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
        "find_files locates files by name, search_files finds text inside them. For a "
        "file too large to return at once, read_file takes offset and limit so you can "
        "page through it instead of working from a truncated copy.",
    ]
    if ctx.allows("files_write"):
        lines.append(
            "You can also move and delete files. delete_file cannot be undone: do it only "
            "when the user asked for that specific file to go, and never to 'clean up' "
            "something you created while working."
        )
    if ctx.allows("git"):
        lines.append(
            "For repository work use the git_* tools rather than shell git. Commit only "
            "when asked, with a message describing why the change was made. git_pull is "
            "fast-forward only -- if it reports the branches have diverged, say so rather "
            "than resolving it with shell git."
        )
    if ctx.allows("github"):
        lines.append(
            "github_clone puts a repository on this machine and makes it a working "
            "directory, which is how you get from 'a repo on GitHub' to 'files you can edit'."
        )
    if ctx.allows("github_write"):
        lines.append(
            "You can open issues, comment, open and merge pull requests, submit reviews, "
            "close issues, and commit files. Those are public actions other people will "
            "see: do them only when the user has asked for that specific thing, and never "
            "as a way of 'reporting' your own progress. Merging and approving are the two "
            "worst to get wrong -- confirm you have the right PR number first."
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

    if ctx.allows("ssh"):
        names = ", ".join(h["name"] for h in ssh_tools.HOSTS)
        if not names:
            # The group can be switched on before any host exists, so say what
            # to do about it rather than letting the model discover an empty
            # list mid-task and guess at a connection string.
            names = ("none yet -- tell the user to add one in the Agent tab; "
                     "do not try to reach a machine another way")
        lines.append(
            f"You can also reach other machines over SSH. Registered hosts: {names}. "
            "Refer to a host by that name -- there is no way to connect to anything "
            "not on the list, and you cannot add to it. "
            "Commands there run on a DIFFERENT machine: the working directories and "
            "the sandbox above say nothing about what you can touch on it, and there "
            "is no undo. Read before you change, and only run something destructive if "
            "the user asked for that specific thing on that specific host. "
            "Sessions are non-interactive, so a command that waits for input will time "
            "out instead of prompting -- prefer non-interactive flags (-y and the like). "
            f"Long jobs take a 'timeout' argument, up to {ssh_tools.SSH_MAX_TIMEOUT}s. "
            "Move files with ssh_upload/ssh_download rather than pasting them through a "
            "shell command; their local side obeys the working-directory rules above."
        )

    lines.append(
        "When you're done, reply with a short plain-text summary of what you did and "
        "what you found."
    )
    # The "" entries above are deliberate blank separators between the intro,
    # the directory list, the scope rule and the instructions -- joined as-is.
    return "\n".join(lines)
