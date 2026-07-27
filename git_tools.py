"""Structured git operations for agent mode.

The agent can already drive git through the bash tool, but that means it has
to compose shell strings and then parse whatever falls out. These wrappers
run git with an argument list (no shell, so nothing to inject through), and
turn every failure into a readable "[error] ..." string instead of an
exception -- the agent loop feeds the return value straight back to the model
as a tool result, so raising would just abort the turn.

Standard library only, on purpose: nothing here should drag the rest of the
proxy in.
"""

import asyncio
import os
import re
import shlex

OUTPUT_LIMIT = 8000
GIT_TIMEOUT = 30

# https://user:token@host/... -- push/fetch/remote output happily echoes back
# whatever credentials are baked into a remote URL. Blank the userinfo before
# any of it reaches a chat bubble. [^\s/@] keeps the match inside one URL's
# authority section rather than swallowing a path that happens to hold an @.
CREDENTIAL_URL_RE = re.compile(r"(https?://)[^\s/@]+@")

# Anything starting with "-" would be read as an option, not a branch/remote.
# git's own "--" separator doesn't help for these positions, so refuse instead.
def _check_name(kind: str, value: str) -> str:
    if not value:
        return f"[error] {kind} is required"
    if value.startswith("-"):
        return f"[error] '{value}' is not a valid {kind} name"
    return ""


def _redact(text: str) -> str:
    return CREDENTIAL_URL_RE.sub(r"\1***@", text)


def _truncate(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    return text[:OUTPUT_LIMIT] + f"\n...[truncated, {len(text)} bytes total]"


def _clean(text: str) -> str:
    # Redact before truncating, so a cut can't leave half a token behind.
    return _truncate(_redact(text))


async def _exec(cwd: str, args: list, timeout: int):
    """One `git ...` run. Returns (returncode, combined stdout+stderr), or
    (None, "") if it had to be killed for running past the timeout."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()  # reap it rather than leaving a zombie behind
        return None, ""
    return proc.returncode, out.decode("utf-8", "replace")


async def _repo_error(cwd: str, timeout: int) -> str:
    """Empty string if cwd is inside a work tree, an [error] line if not."""
    rc, out = await _exec(cwd, ["rev-parse", "--is-inside-work-tree"], timeout)
    if rc is None:
        return f"[error] git timed out after {timeout}s"
    if rc != 0 or out.strip() != "true":
        detail = out.strip().splitlines()
        suffix = f" ({detail[0]})" if detail else ""
        return f"[error] '{cwd}' is not inside a git work tree{suffix}"
    return ""


async def run_git(cwd: str, args: list, timeout: int = GIT_TIMEOUT, require_repo: bool = True) -> str:
    """Run git in cwd and return its output. Successful commands that print
    nothing come back as an empty string -- the wrappers below decide what to
    say about that, since "no output" means something different per command.

    require_repo=False is for the commands that create a repository rather
    than act on one (clone, init), where cwd is deliberately not one yet."""
    try:
        if not args:
            return "[error] no git arguments given"
        if not cwd or not os.path.isdir(cwd):
            return f"[error] working directory '{cwd}' does not exist"
        if require_repo:
            problem = await _repo_error(cwd, timeout)
            if problem:
                return problem
        rc, out = await _exec(cwd, args, timeout)
        if rc is None:
            return f"[error] git timed out after {timeout}s"
        out = _clean(out).strip()
        if rc != 0:
            command = shlex.join(["git"] + [str(a) for a in args])
            return f"[error] {command} failed [exit {rc}]" + (f"\n{out}" if out else "")
        return out
    except OSError as e:
        return f"[error] could not run git: {e}"
    except Exception as e:
        return f"[error] {e}"


def _failed(text: str) -> bool:
    return text.startswith("[error]")


async def git_status(cwd: str) -> str:
    out = await run_git(cwd, ["status", "--short", "--branch"])
    if _failed(out):
        return out
    return out or "(clean working tree)"


async def git_diff(cwd: str, staged: bool = False, path: str = "") -> str:
    args = ["diff"]
    if staged:
        args.append("--staged")
    if path:
        args += ["--", path]  # -- so a path can never be read as an option
    out = await run_git(cwd, args)
    if _failed(out):
        return out
    return out or "(no changes)"


async def git_log(cwd: str, limit: int = 10) -> str:
    try:
        limit = max(1, min(100, int(limit)))
    except (TypeError, ValueError):
        return f"[error] limit '{limit}' is not a number"
    out = await run_git(cwd, ["log", "--oneline", "--decorate", "-n", str(limit)])
    if _failed(out):
        return out
    return out or "(no commits yet)"


async def git_branch(cwd: str) -> str:
    current = await run_git(cwd, ["branch", "--show-current"])
    if _failed(current):
        return current
    listing = await run_git(cwd, ["branch", "--list"])
    if _failed(listing):
        return listing
    # --show-current is empty on a detached HEAD and in a repo with no commits.
    head = current or "(detached or unborn HEAD)"
    return f"current branch: {head}\n\nlocal branches:\n{listing or '(none yet)'}"


async def git_commit(cwd: str, message: str, add_all: bool = True) -> str:
    if not (message or "").strip():
        return "[error] commit message is required"
    if add_all:
        staged = await run_git(cwd, ["add", "-A"])
        if _failed(staged):
            return staged

    # Check before committing so "nothing to commit" reads as a clear answer
    # instead of git's exit-1 wall of status output.
    pending = await run_git(cwd, ["diff", "--cached", "--name-only"])
    if _failed(pending):
        return pending
    if not pending:
        return "[error] nothing to commit" + ("" if add_all else " -- nothing is staged (try add_all)")

    out = await run_git(cwd, ["commit", "-m", message])
    if _failed(out):
        return out
    committed = await run_git(cwd, ["log", "-1", "--pretty=format:%h %s"])
    if _failed(committed):
        return f"committed\n{out}"
    files = len(pending.splitlines())
    return f"committed {committed} ({files} file{'s' if files != 1 else ''})"


async def git_checkout(cwd: str, branch: str, create: bool = False) -> str:
    problem = _check_name("branch", branch)
    if problem:
        return problem
    args = ["checkout", "-b", branch] if create else ["checkout", branch]
    out = await run_git(cwd, args)
    if _failed(out):
        return out
    verb = "created and switched to" if create else "switched to"
    return f"{verb} branch '{branch}'" + (f"\n{out}" if out else "")


async def git_push(cwd: str, remote: str = "origin", branch: str = "") -> str:
    remote = remote or "origin"
    problem = _check_name("remote", remote)
    if problem:
        return problem
    if branch:
        problem = _check_name("branch", branch)
        if problem:
            return problem
        # An explicit branch is usually a first push of a new branch, so track it.
        args = ["push", "--set-upstream", remote, branch]
        target = f"{remote}/{branch}"
    else:
        args = ["push", remote]
        target = f"{remote} (current branch)"
    out = await run_git(cwd, args)
    if _failed(out):
        return out
    return f"pushed to {target}" + (f"\n{out}" if out else "")
