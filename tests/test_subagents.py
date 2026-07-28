"""Sub-agents: budgets, permissions, and what comes back.

The tool itself is mostly refusals, so most of this is about whether the
refusals happen before a child exists rather than after. The nested loop is
exercised against a fake runner -- what matters here is the contract between
parent and child, not that a model answers.
"""

import pytest

import agent_tools


@pytest.fixture
def ctx(app, tmp_path):
    agent_tools.WORKDIRS[:] = [{"path": str(tmp_path), "label": "tmp"}]
    context = agent_tools.build_agent_context(
        requested_paths=[str(tmp_path)], sandbox_requested=True,
        requested_groups=["agents", "files", "shell", "web"])
    context.spawn = None
    return context


def recorder(ctx):
    """Stand in for the real runner and record what it was asked for."""
    calls = []

    async def spawn(**kw):
        calls.append(kw)
        return "the answer"

    ctx.spawn = spawn
    return calls


BRIEF = "Find every file that defines a retry policy and report the file names."


# ---------------------------------------------------------------------------
# Budgets, checked before a child exists
# ---------------------------------------------------------------------------

async def test_a_child_runs_and_only_its_answer_comes_back(ctx):
    calls = recorder(ctx)
    assert await agent_tools.run_spawn_agent(ctx, task=BRIEF) == "the answer"
    assert len(calls) == 1


async def test_a_subagent_cannot_spawn_its_own(ctx):
    """Each level multiplies the cost of one request and makes it harder to say
    what is actually running."""
    recorder(ctx)
    ctx.depth = agent_tools.MAX_AGENT_DEPTH
    out = await agent_tools.run_spawn_agent(ctx, task=BRIEF)
    assert out.startswith("[error]")
    assert "yourself" in out


async def test_the_number_of_children_is_capped(ctx, monkeypatch):
    monkeypatch.setattr(agent_tools, "MAX_SPAWNS", 2)
    calls = recorder(ctx)
    for _ in range(2):
        assert await agent_tools.run_spawn_agent(ctx, task=BRIEF) == "the answer"
    out = await agent_tools.run_spawn_agent(ctx, task=BRIEF)
    assert out.startswith("[error]")
    assert len(calls) == 2          # refused before a third child existed


async def test_it_is_unavailable_outside_an_agent_run(ctx):
    """No runner means no model to drive; saying so beats failing oddly."""
    out = await agent_tools.run_spawn_agent(ctx, task=BRIEF)
    assert out.startswith("[error]")
    assert "aren't available here" in out


# ---------------------------------------------------------------------------
# Permissions can only narrow
# ---------------------------------------------------------------------------

async def test_a_child_inherits_the_parents_groups(ctx):
    calls = recorder(ctx)
    await agent_tools.run_spawn_agent(ctx, task=BRIEF)
    assert set(calls[0]["groups"]) == {"files", "shell", "web"}


async def test_a_child_never_gets_the_spawn_tool_itself(ctx):
    """Depth is the real limit, but not handing it the tool is the cheaper one:
    it can't try."""
    calls = recorder(ctx)
    await agent_tools.run_spawn_agent(ctx, task=BRIEF)
    assert "agents" not in calls[0]["groups"]


async def test_groups_can_be_narrowed(ctx):
    calls = recorder(ctx)
    await agent_tools.run_spawn_agent(ctx, task=BRIEF, groups=["files"])
    assert calls[0]["groups"] == ["files"]


async def test_groups_cannot_be_widened(ctx):
    """A child holding a permission its parent doesn't is a way to launder one."""
    calls = recorder(ctx)
    await agent_tools.run_spawn_agent(ctx, task=BRIEF, groups=["files", "github_write", "ssh"])
    assert calls[0]["groups"] == ["files"]


async def test_asking_only_for_groups_you_lack_is_refused(ctx):
    """Silently running with nothing would look like the tools were broken."""
    recorder(ctx)
    out = await agent_tools.run_spawn_agent(ctx, task=BRIEF, groups=["github_write"])
    assert out.startswith("[error]")
    assert "files" in out          # says what it could have had


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------

async def test_a_thin_brief_is_refused(ctx):
    """The child sees only this text. A vague brief produces a confident answer
    to a question nobody asked, and the parent can't tell."""
    recorder(ctx)
    out = await agent_tools.run_spawn_agent(ctx, task="look at the code")
    assert out.startswith("[error]")
    assert "self-contained" in out


async def test_an_empty_brief_is_refused(ctx):
    recorder(ctx)
    assert (await agent_tools.run_spawn_agent(ctx, task="   ")).startswith("[error]")


async def test_a_refused_spawn_does_not_spend_the_budget(ctx):
    recorder(ctx)
    await agent_tools.run_spawn_agent(ctx, task="too short")
    assert ctx.spawns_used == 0


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_the_group_is_off_by_default():
    """It multiplies what one message costs, so it's opt-in like the other
    groups that spend something."""
    assert "agents" in agent_tools.GROUPS
    assert "agents" not in agent_tools.DEFAULT_GROUPS


def test_the_prompt_says_the_child_is_blind_to_the_conversation(app, tmp_path):
    agent_tools.WORKDIRS[:] = [{"path": str(tmp_path), "label": "tmp"}]
    prompt = agent_tools.agent_system_prompt(agent_tools.build_agent_context(
        requested_paths=[str(tmp_path)], sandbox_requested=True,
        requested_groups=["agents"]))
    assert "only the brief" in prompt
    assert "cannot ask you anything" in prompt
