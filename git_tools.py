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
# Public because callers outside this module (github_clone's --branch) need
# the same guard on the same kind of model-supplied value.
def check_name(kind: str, value: str) -> str:
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


async def stop_process(proc):
    """Kill a child that is still running and reap it. Used on both exits that
    aren't a clean finish -- the timeout, and the run being cancelled because
    the user pressed Stop -- so neither leaves the command executing behind a
    UI that says it stopped, nor a zombie behind the loop.

    Lives here rather than in agent_tools because agent_tools imports this
    module, not the other way round, and every subprocess tool in the project
    needs it."""
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
        await stop_process(proc)
        return None, ""
    except BaseException:
        # CancelledError is a BaseException, so pressing Stop lands here. Left
        # unhandled it walked out of this function with the child still
        # running -- a `git clone` that keeps writing after the UI says the run
        # stopped.
        await stop_process(proc)
        raise
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
    problem = check_name("branch", branch)
    if problem:
        return problem
    args = ["checkout", "-b", branch] if create else ["checkout", branch]
    out = await run_git(cwd, args)
    if _failed(out):
        return out
    verb = "created and switched to" if create else "switched to"
    return f"{verb} branch '{branch}'" + (f"\n{out}" if out else "")


async def git_push(cwd: str, remote: str = "origin", branch: str = "", config_args: list | None = None) -> str:
    """config_args is prepended to the git command line -- it's how the caller
    passes `-c http.extraHeader=...` so a push can authenticate without the
    credential ever touching .git/config. It comes from the proxy, never from
    the model; remote and branch do, which is what check_name is for."""
    remote = remote or "origin"
    problem = check_name("remote", remote)
    if problem:
        return problem
    if branch:
        problem = check_name("branch", branch)
        if problem:
            return problem
        # An explicit branch is usually a first push of a new branch, so track it.
        args = ["push", "--set-upstream", remote, branch]
        target = f"{remote}/{branch}"
    else:
        args = ["push", remote]
        target = f"{remote} (current branch)"
    out = await run_git(cwd, list(config_args or []) + args)
    if _failed(out):
        return out
    return f"pushed to {target}" + (f"\n{out}" if out else "")


async def git_fetch(cwd: str, remote: str = "origin", config_args: list | None = None) -> str:
    """Update remote-tracking refs without touching the working tree. Safe to
    run at any time, which is why it's the one to reach for before deciding
    whether a pull is even needed."""
    remote = remote or "origin"
    problem = check_name("remote", remote)
    if problem:
        return problem
    out = await run_git(cwd, list(config_args or []) + ["fetch", "--prune", remote], timeout=120)
    if _failed(out):
        return out
    # A fetch that found nothing prints nothing, which reads as a failure if
    # it's handed back bare.
    return f"fetched {remote}" + (f"\n{out}" if out else " (already up to date)")


async def git_pull(cwd: str, remote: str = "origin", branch: str = "",
                   config_args: list | None = None) -> str:
    """Fast-forward only. A merge or rebase started by a model is how you end
    up with a half-finished conflict state and no one watching, so a pull that
    can't fast-forward is reported rather than attempted."""
    remote = remote or "origin"
    problem = check_name("remote", remote)
    if problem:
        return problem
    args = ["pull", "--ff-only", remote]
    if branch:
        problem = check_name("branch", branch)
        if problem:
            return problem
        args.append(branch)

    out = await run_git(cwd, list(config_args or []) + args, timeout=120)
    if _failed(out):
        if "not possible to fast-forward" in out or "diverging" in out.lower():
            return (out + "\n\n[note] the local branch has commits the remote doesn't, so this "
                    "would need a merge or rebase. That's a decision for the user -- report it "
                    "rather than forcing it.")
        return out
    return out or "already up to date"


async def git_show(cwd: str, ref: str = "HEAD", stat_only: bool = False) -> str:
    """One commit: message, author, and either its diff or just the file
    list. stat_only exists because a big commit's full diff can eat an entire
    tool-output budget on its own."""
    ref = (ref or "HEAD").strip()
    if ref.startswith("-"):
        return f"[error] '{ref}' is not a valid ref"
    args = ["show", "--stat" if stat_only else "--patch", ref]
    out = await run_git(cwd, args)
    if _failed(out):
        return out
    return out or f"(nothing to show for {ref})"


async def git_stash(cwd: str, action: str = "list", message: str = "") -> str:
    """push / pop / list / drop. Deliberately not 'clear' -- that discards
    every stash at once with no undo."""
    action = (action or "list").strip().lower()
    if action not in ("push", "pop", "list", "drop"):
        return f"[error] stash action '{action}' must be one of: push, pop, list, drop"

    if action == "push":
        args = ["stash", "push", "--include-untracked"]
        if message:
            args += ["-m", message]
    else:
        args = ["stash", action]

    out = await run_git(cwd, args)
    if _failed(out):
        return out
    if action == "list":
        return out or "(no stashes)"
    return out or f"stash {action} done"


async def git_restore(cwd: str, path: str = "", staged: bool = False) -> str:
    """Discard uncommitted changes. This destroys work that was never
    committed, so it refuses to run without an explicit path -- 'restore
    everything' should be a decision someone types out, not a default."""
    if not (path or "").strip():
        return ("[error] path is required -- restoring every file at once would discard "
                "uncommitted work with no way back. Name the file(s) to restore.")
    args = ["restore"]
    if staged:
        args.append("--staged")
    args += ["--", path]
    out = await run_git(cwd, args)
    if _failed(out):
        return out
    what = "unstaged" if staged else "discarded local changes in"
    return f"{what} {path}" + (f"\n{out}" if out else "")
