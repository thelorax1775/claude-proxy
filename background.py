"""Shells that outlive the tool call that started them.

An agent round is a request/response: the model asks for a command, waits, and
gets the output. That works until the command is a build, a test suite, an apt
upgrade or a dev server -- things that either take longer than any sensible
timeout or never finish at all. The old answer was "raise the timeout", which
turns a ten-minute build into ten minutes of a model sitting idle and burning a
round, and has no answer at all for a server you want left running.

So a command can be started and left. The call returns an id immediately, the
process keeps going, and its output accumulates here to be collected later or
watched a chunk at a time. Three rules make that safe to hand to a model:

  * Output is drained continuously into a bounded buffer, not left in a pipe.
    A process whose pipe fills up blocks forever, and a "background" job that
    silently wedges at 64KB is worse than no feature.

  * Everything is capped -- how many shells, how much output each keeps. A
    model that starts a hundred servers should hit a refusal, not an OOM.

  * A finished shell keeps its output until it is read or reaped. Exiting is
    not the same as being collected, and a background job whose result vanished
    because nobody asked in time is a bug that only shows up under load.

Reaping is by age and only for finished shells, so nothing running is ever
collected out from under a model that is about to ask about it.
"""

import asyncio
import logging
import os
import time
from collections import deque

log = logging.getLogger("claude-proxy")

# Per shell. Enough for a build log's interesting parts; small enough that a
# runaway `yes` is a bounded nuisance rather than a memory leak.
OUTPUT_LIMIT = int(os.getenv("AGENT_BG_OUTPUT_LIMIT", "200000"))

# Concurrent live shells. A model that wants more than this is confused, and
# the refusal is more useful to it than the resources.
MAX_SHELLS = int(os.getenv("AGENT_BG_MAX_SHELLS", "8"))

# How long a *finished* shell's output is kept before it can be reaped. Only
# ever applied to exited shells, so a long-running job is never collected.
KEEP_FINISHED = int(os.getenv("AGENT_BG_KEEP_FINISHED", "1800"))

_SHELLS: dict[str, "Shell"] = {}
_counter = 0


class Shell:
    """One backgrounded command and everything known about it."""

    def __init__(self, shell_id: str, command: str, cwd: str, proc):
        self.id = shell_id
        self.command = command
        self.cwd = cwd
        self.proc = proc
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.returncode: int | None = None
        # Kept as chunks so reading is "everything since last time" without
        # re-scanning, and the cap can drop whole chunks off the front.
        self._chunks: deque[str] = deque()
        self._size = 0
        self._dropped = 0
        self._read_to = 0          # chunks already handed over
        self._pump: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self.returncode is None

    def append(self, text: str) -> None:
        self._chunks.append(text)
        self._size += len(text)
        # Drop from the front rather than refusing to record: for a build log
        # the tail is the part that says what went wrong.
        while self._size > OUTPUT_LIMIT and len(self._chunks) > 1:
            gone = self._chunks.popleft()
            self._size -= len(gone)
            self._dropped += len(gone)
            if self._read_to > 0:
                self._read_to -= 1

    def peek(self) -> str:
        """Everything so far, without moving the read cursor. For anything that
        wants to look without consuming -- read() advances what counts as
        'new', which is right for a caller and wrong for an observer."""
        return "".join(self._chunks)

    def read(self, *, only_new: bool) -> str:
        chunks = list(self._chunks)
        text = "".join(chunks[self._read_to:] if only_new else chunks)
        self._read_to = len(chunks)
        if self._dropped and not only_new:
            text = f"...[{self._dropped} earlier bytes dropped]\n" + text
        return text

    def status(self) -> str:
        if self.running:
            return f"running for {int(time.time() - self.started_at)}s"
        return f"exited {self.returncode} after {int((self.finished_at or 0) - self.started_at)}s"


async def _pump(shell: Shell) -> None:
    """Drain the process's output for as long as it lives.

    This is the whole reason a background shell doesn't wedge: nothing else is
    reading that pipe, and a full pipe stops the process dead. Running it as a
    task also means the exit code is recorded the moment it happens rather than
    whenever somebody next asks.
    """
    try:
        while True:
            block = await shell.proc.stdout.read(8192)
            if not block:
                break
            shell.append(block.decode("utf-8", "replace"))
    except asyncio.CancelledError:
        raise
    except Exception as e:
        shell.append(f"\n[error reading output: {e}]\n")
    finally:
        try:
            shell.returncode = await shell.proc.wait()
        except Exception:
            shell.returncode = -1
        shell.finished_at = time.time()


def reap(now: float | None = None) -> None:
    """Forget finished shells nobody collected. Never touches a running one."""
    now = now or time.time()
    for shell_id, shell in list(_SHELLS.items()):
        if not shell.running and shell.finished_at and now - shell.finished_at > KEEP_FINISHED:
            _SHELLS.pop(shell_id, None)


def live_count() -> int:
    return sum(1 for s in _SHELLS.values() if s.running)


async def start(command: str, cwd: str) -> "Shell | str":
    """Launch a command and return its Shell, or an [error] string."""
    global _counter
    reap()
    if live_count() >= MAX_SHELLS:
        return (f"[error] {MAX_SHELLS} background shells are already running. "
                "Collect or kill one with kill_shell before starting another.")
    try:
        proc = await asyncio.create_subprocess_shell(
            command, cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,      # so kill_shell can take the whole tree
        )
    except Exception as e:
        return f"[error] could not start the command: {e}"

    _counter += 1
    shell = Shell(f"bg_{_counter}", command, cwd, proc)
    _SHELLS[shell.id] = shell
    shell._pump = asyncio.create_task(_pump(shell))
    return shell


def get(shell_id: str) -> "Shell | None":
    return _SHELLS.get((shell_id or "").strip())


def listing() -> list["Shell"]:
    reap()
    return list(_SHELLS.values())


async def kill(shell: Shell) -> str:
    """Stop a shell and everything it started.

    The process group, not the process: a shell that ran `npm start` is a shell
    whose *child* is holding the port, and killing only the shell leaves that
    orphan running with nothing left pointing at it.
    """
    if not shell.running:
        return f"{shell.id} had already {shell.status()}."
    try:
        os.killpg(os.getpgid(shell.proc.pid), 15)     # SIGTERM to the group
    except (ProcessLookupError, PermissionError, OSError):
        try:
            shell.proc.terminate()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(asyncio.shield(shell._pump), timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
        try:
            os.killpg(os.getpgid(shell.proc.pid), 9)  # SIGKILL, it had its chance
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return f"Killed {shell.id} ({shell.command[:60]})."


async def shutdown() -> None:
    """Kill everything still running. Called when the process is going down, so
    a restart doesn't leave orphaned builds and dev servers behind."""
    for shell in list(_SHELLS.values()):
        if shell.running:
            await kill(shell)
    _SHELLS.clear()
