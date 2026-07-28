"""Background shells: started and left, collected later.

Real processes, because everything worth testing here is about what an OS
process does -- filling a pipe, ignoring a signal, leaving a child behind --
and none of that survives being mocked.
"""

import asyncio
import os
import time

import pytest

import agent_tools
import background


@pytest.fixture(autouse=True)
async def clean_registry():
    """No shell outlives its test. A leaked `sleep 300` would still be running
    when the suite finishes."""
    await background.shutdown()
    yield
    await background.shutdown()


@pytest.fixture
def ctx(app, tmp_path):
    agent_tools.WORKDIRS[:] = [{"path": str(tmp_path), "label": "tmp"}]
    return agent_tools.build_agent_context(
        requested_paths=[str(tmp_path)], sandbox_requested=True,
        requested_groups=["shell"])


async def wait_for(predicate, timeout=10.0):
    """Poll until true. Beats a fixed sleep, which is either flaky or slow."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# The basic contract
# ---------------------------------------------------------------------------

async def test_background_returns_immediately(ctx):
    """The point of the feature: a slow command must not cost a slow tool
    call."""
    started = time.monotonic()
    out = await agent_tools.run_bash(ctx, "sleep 5", run_in_background=True)
    assert time.monotonic() - started < 2
    assert "bg_" in out


async def test_output_is_collected_after_the_call_returned(ctx):
    out = await agent_tools.run_bash(ctx, "echo hello-bg", run_in_background=True)
    shell_id = out.split()[1]
    assert await wait_for(lambda: not background.get(shell_id).running)
    assert "hello-bg" in await agent_tools.run_bash_output(ctx, shell_id)


async def test_the_exit_code_is_recorded(ctx):
    out = await agent_tools.run_bash(ctx, "exit 3", run_in_background=True)
    shell_id = out.split()[1]
    assert await wait_for(lambda: not background.get(shell_id).running)
    assert background.get(shell_id).returncode == 3
    assert "exited 3" in await agent_tools.run_bash_output(ctx, shell_id)


async def test_only_new_output_by_default(ctx):
    """Re-reading a growing log every time costs a round's context to learn one
    line."""
    out = await agent_tools.run_bash(
        ctx, "echo one; sleep 0.4; echo two", run_in_background=True)
    shell_id = out.split()[1]

    assert await wait_for(lambda: "one" in background.get(shell_id).peek())
    first = await agent_tools.run_bash_output(ctx, shell_id)
    assert "one" in first

    assert await wait_for(lambda: not background.get(shell_id).running)
    second = await agent_tools.run_bash_output(ctx, shell_id)
    assert "two" in second
    assert "one" not in second          # already delivered

    everything = await agent_tools.run_bash_output(ctx, shell_id, only_new=False)
    assert "one" in everything and "two" in everything


async def test_a_finished_shell_keeps_its_output_until_read(ctx):
    """Exiting is not the same as being collected."""
    out = await agent_tools.run_bash(ctx, "echo done-already", run_in_background=True)
    shell_id = out.split()[1]
    assert await wait_for(lambda: not background.get(shell_id).running)
    await asyncio.sleep(0.3)
    assert "done-already" in await agent_tools.run_bash_output(ctx, shell_id)


# ---------------------------------------------------------------------------
# The things that make it safe to hand to a model
# ---------------------------------------------------------------------------

async def test_a_noisy_command_does_not_wedge_on_a_full_pipe(ctx):
    """Nothing else reads that pipe. A process whose pipe fills up blocks
    forever, and a background job that silently stops at 64KB is worse than no
    feature at all."""
    out = await agent_tools.run_bash(
        ctx, "for i in $(seq 1 20000); do echo 'filling the pipe with output'; done",
        run_in_background=True)
    shell_id = out.split()[1]
    # ~640KB, far past any pipe buffer: if the pump weren't draining, this
    # would still be running when the wait expires.
    assert await wait_for(lambda: not background.get(shell_id).running, timeout=30)
    assert background.get(shell_id).returncode == 0


async def test_output_is_capped_and_says_so(ctx, monkeypatch):
    monkeypatch.setattr(background, "OUTPUT_LIMIT", 2000)
    out = await agent_tools.run_bash(
        ctx, "for i in $(seq 1 5000); do echo padding-line-$i; done",
        run_in_background=True)
    shell_id = out.split()[1]
    assert await wait_for(lambda: not background.get(shell_id).running, timeout=30)
    shell = background.get(shell_id)
    everything = shell.read(only_new=False)
    assert "earlier bytes dropped" in everything
    # The tail is kept, because that's the part that says what went wrong.
    assert "padding-line-5000" in everything


async def test_too_many_shells_is_refused_not_absorbed(ctx, monkeypatch):
    monkeypatch.setattr(background, "MAX_SHELLS", 2)
    for _ in range(2):
        assert "bg_" in await agent_tools.run_bash(ctx, "sleep 30", run_in_background=True)
    refused = await agent_tools.run_bash(ctx, "sleep 30", run_in_background=True)
    assert refused.startswith("[error]")
    assert "kill_shell" in refused


async def test_kill_stops_it(ctx):
    out = await agent_tools.run_bash(ctx, "sleep 300", run_in_background=True)
    shell_id = out.split()[1]
    assert "Killed" in await agent_tools.run_kill_shell(ctx, shell_id)
    assert await wait_for(lambda: not background.get(shell_id).running)


async def test_kill_takes_the_children_too(ctx, tmp_path):
    """A shell that ran `npm start` is a shell whose *child* holds the port.
    Killing only the shell leaves that orphan running with nothing pointing at
    it."""
    marker = tmp_path / "child.pid"
    out = await agent_tools.run_bash(
        ctx, f"sh -c 'echo $$ > {marker}; sleep 300' & wait", run_in_background=True)
    shell_id = out.split()[1]
    assert await wait_for(lambda: marker.exists())
    child_pid = int(marker.read_text().strip())

    await agent_tools.run_kill_shell(ctx, shell_id)
    assert await wait_for(lambda: not background.get(shell_id).running)

    def gone():
        try:
            os.kill(child_pid, 0)
            return False
        except OSError:
            return True
    assert await wait_for(gone), "the grandchild survived the kill"


async def test_reaping_never_collects_a_running_shell(ctx):
    out = await agent_tools.run_bash(ctx, "sleep 300", run_in_background=True)
    shell_id = out.split()[1]
    background.reap(now=time.time() + background.KEEP_FINISHED * 10)
    assert background.get(shell_id) is not None


async def test_reaping_forgets_old_finished_shells(ctx):
    out = await agent_tools.run_bash(ctx, "true", run_in_background=True)
    shell_id = out.split()[1]
    assert await wait_for(lambda: not background.get(shell_id).running)
    background.reap(now=time.time() + background.KEEP_FINISHED + 60)
    assert background.get(shell_id) is None


async def test_shutdown_leaves_nothing_running(ctx):
    out = await agent_tools.run_bash(ctx, "sleep 300", run_in_background=True)
    shell_id = out.split()[1]
    pid = background.get(shell_id).proc.pid
    await background.shutdown()

    def gone():
        try:
            os.kill(pid, 0)
            return False
        except OSError:
            return True
    assert await wait_for(gone)


# ---------------------------------------------------------------------------
# Talking to a model
# ---------------------------------------------------------------------------

async def test_an_unknown_id_lists_what_does_exist(ctx):
    out = await agent_tools.run_bash_output(ctx, "bg_nope")
    assert out.startswith("[error]")
    assert "Known:" in out


async def test_no_id_lists_every_shell(ctx):
    await agent_tools.run_bash(ctx, "sleep 30", run_in_background=True)
    listing = await agent_tools.run_bash_output(ctx, "")
    assert "bg_1" in listing and "running" in listing


async def test_the_foreground_path_is_unchanged(ctx):
    """Background is opt-in; the ordinary call must still block and answer."""
    assert "plain-old-bash" in await agent_tools.run_bash(ctx, "echo plain-old-bash")


async def test_the_tools_are_in_the_shell_group():
    names = {t["name"] for t in agent_tools.TOOLS if t["group"] == "shell"}
    assert {"bash", "bash_output", "kill_shell"} <= names
