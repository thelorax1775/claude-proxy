"""The run's checklist.

Most of these are about honesty rather than bookkeeping: the list only earns
its place if "done" means the thing works and "in progress" means one thing.
"""

import pytest

import agent_tools
import tasklist


@pytest.fixture
def ctx(app, tmp_path):
    agent_tools.WORKDIRS[:] = [{"path": str(tmp_path), "label": "tmp"}]
    return agent_tools.build_agent_context(
        requested_paths=[str(tmp_path)], sandbox_requested=True,
        requested_groups=["tasks"])


def rows(*pairs):
    return [{"content": c, "status": s} for c, s in pairs]


async def test_a_plan_round_trips(ctx):
    out = await agent_tools.run_write_todos(ctx, rows(
        ("read the config", "completed"),
        ("add the parser", "in_progress"),
        ("write the tests", "pending")))
    assert "[x] read the config" in out
    assert "[>] add the parser" in out
    assert "[ ] write the tests" in out
    assert "(1/3 done)" in out


async def test_the_whole_list_is_replaced(ctx):
    """Whole-list rather than patch-by-index: a model editing item 3 of a list
    it cannot see is guessing."""
    await agent_tools.run_write_todos(ctx, rows(("first", "pending")))
    await agent_tools.run_write_todos(ctx, rows(("second", "pending")))
    assert [i["content"] for i in ctx.tasks.public()] == ["second"]


async def test_only_one_task_may_be_in_progress(ctx):
    """Marking six items started describes an intention, not a state, and the
    list stops meaning anything."""
    out = await agent_tools.run_write_todos(ctx, rows(
        ("one", "in_progress"), ("two", "in_progress")))
    assert out.startswith("[error]")
    assert ctx.tasks.public() == []          # rejected outright, not half-applied


async def test_zero_in_progress_is_fine(ctx):
    """Between steps is a real state -- everything pending, or everything
    done."""
    assert not (await agent_tools.run_write_todos(ctx, rows(
        ("one", "completed"), ("two", "pending")))).startswith("[error]")


async def test_a_bad_status_is_named(ctx):
    out = await agent_tools.run_write_todos(ctx, rows(("one", "almost")))
    assert out.startswith("[error]")
    assert "pending" in out and "in_progress" in out and "completed" in out


async def test_empty_content_is_refused(ctx):
    assert (await agent_tools.run_write_todos(ctx, rows(("   ", "pending")))).startswith("[error]")


async def test_a_wrong_shape_is_refused_rather_than_crashing(ctx):
    """This arrives from a model, so it will eventually be the wrong shape."""
    for bad in ("not a list", [1, 2, 3], [None], {"content": "x"}):
        assert (await agent_tools.run_write_todos(ctx, bad)).startswith("[error]")


async def test_too_many_tasks_is_refused(ctx):
    out = await agent_tools.run_write_todos(
        ctx, rows(*[(f"step {i}", "pending") for i in range(tasklist.MAX_ITEMS + 1)]))
    assert out.startswith("[error]")
    assert "Group the small steps" in out


async def test_long_content_is_truncated_not_rejected(ctx):
    """A rambling task is still a usable task; losing the whole update over it
    is not."""
    await agent_tools.run_write_todos(ctx, rows(("x" * 500, "pending")))
    assert len(ctx.tasks.public()[0]["content"]) <= tasklist.MAX_LEN + 3


async def test_unfinished_is_what_the_run_reports(ctx):
    """A run that stops with items open is the failure this exists to catch."""
    await agent_tools.run_write_todos(ctx, rows(
        ("done thing", "completed"),
        ("half-done thing", "in_progress"),
        ("untouched thing", "pending")))
    assert ctx.tasks.unfinished() == ["half-done thing", "untouched thing"]


async def test_an_all_done_run_reports_nothing_outstanding(ctx):
    await agent_tools.run_write_todos(ctx, rows(("a", "completed"), ("b", "completed")))
    assert ctx.tasks.unfinished() == []


async def test_the_list_is_per_run(app, tmp_path):
    """It's working state for one task, not a project tracker. A checklist that
    outlives its reason is worse than none."""
    agent_tools.WORKDIRS[:] = [{"path": str(tmp_path), "label": "tmp"}]
    make = lambda: agent_tools.build_agent_context(
        requested_paths=[str(tmp_path)], sandbox_requested=True,
        requested_groups=["tasks"])
    first = make()
    await agent_tools.run_write_todos(first, rows(("something", "pending")))
    assert make().tasks.public() == []


async def test_public_is_a_copy(ctx):
    """The UI must not be able to edit the run's state by holding its list."""
    await agent_tools.run_write_todos(ctx, rows(("one", "pending")))
    ctx.tasks.public()[0]["status"] = "completed"
    assert ctx.tasks.public()[0]["status"] == "pending"


def test_the_group_is_on_by_default():
    assert "tasks" in agent_tools.DEFAULT_GROUPS
    assert "write_todos" in {t["name"] for t in agent_tools.TOOLS if t["group"] == "tasks"}


def test_the_prompt_says_when_to_skip_it(app, tmp_path):
    """Without this a model writes a one-item checklist for 'what time is
    it', which is pure noise."""
    agent_tools.WORKDIRS[:] = [{"path": str(tmp_path), "label": "tmp"}]
    prompt = agent_tools.agent_system_prompt(agent_tools.build_agent_context(
        requested_paths=[str(tmp_path)], sandbox_requested=True,
        requested_groups=["tasks"]))
    assert "write_todos" in prompt
    assert "single-step" in prompt
