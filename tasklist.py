"""The model's checklist for a run.

A multi-step request drifts. Round eight is a long way from the sentence that
asked for six things, most of the context in between is tool output, and the
usual failure is not a wrong answer but a quiet one -- three of the six done,
reported as if that were all of it. Writing the list down turns "what was I
doing" into something the model can re-read instead of reconstruct, and turns
"did it finish" into something you can see rather than infer.

Deliberately per-run and in memory. This is working state for one task, not a
project tracker: persisting it would mean deciding when a task is stale, and
a list that outlives its reason is worse than no list. It dies with the run.

The rules that matter are about honesty rather than bookkeeping:

  * Exactly one thing in progress. A model that marks six items started has
    described an intention, not a state, and the list stops meaning anything.

  * Completing is a claim about the world. "Done" must mean the thing works,
    not that it was attempted -- so the tool says so, and a step that failed
    stays open where it can be seen.
"""

import logging

log = logging.getLogger("claude-proxy")

STATES = ("pending", "in_progress", "completed")

# Enough for a genuinely large piece of work; small enough that a model
# decomposing a task into fifty micro-steps gets told to stop.
MAX_ITEMS = 40

MAX_LEN = 200


class TaskList:
    """One run's checklist. Owned by the AgentContext, dies with it."""

    def __init__(self):
        self.items: list[dict] = []

    def replace(self, rows) -> str:
        """Set the whole list at once. Returns "" or an [error] line.

        Whole-list rather than add/update/remove because a model editing item
        3 of a list it cannot see is guessing at an index. Sending the list
        back means what it believes is always what is stored -- and any
        disagreement is visible immediately rather than compounding.
        """
        if not isinstance(rows, list):
            return "[error] tasks must be a list of {content, status} objects"
        if len(rows) > MAX_ITEMS:
            return (f"[error] {len(rows)} tasks is too many (max {MAX_ITEMS}). "
                    "Group the small steps together.")

        cleaned, in_progress = [], 0
        for row in rows:
            if not isinstance(row, dict):
                return "[error] each task must be an object with 'content' and 'status'"
            content = str(row.get("content") or "").strip()
            if not content:
                return "[error] every task needs a non-empty 'content'"
            if len(content) > MAX_LEN:
                content = content[:MAX_LEN] + "..."
            status = str(row.get("status") or "pending").strip().lower()
            if status not in STATES:
                return f"[error] '{status}' is not a status; use {', '.join(STATES)}"
            if status == "in_progress":
                in_progress += 1
            cleaned.append({"content": content, "status": status})

        if in_progress > 1:
            return ("[error] only one task may be in_progress at a time -- "
                    "marking several started describes an intention, not a state")

        self.items = cleaned
        return ""

    def counts(self) -> tuple[int, int]:
        done = sum(1 for i in self.items if i["status"] == "completed")
        return done, len(self.items)

    def render(self) -> str:
        """What the model gets back. Shows the whole list every time: the value
        is in re-reading it, and a diff of a list you can't see is no use."""
        if not self.items:
            return "The task list is empty."
        mark = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
        done, total = self.counts()
        lines = [f"{mark[i['status']]} {i['content']}" for i in self.items]
        return "\n".join(lines) + f"\n({done}/{total} done)"

    def public(self) -> list[dict]:
        """For the UI. A copy, so nothing downstream can edit the run's state."""
        return [dict(i) for i in self.items]

    def unfinished(self) -> list[str]:
        return [i["content"] for i in self.items if i["status"] != "completed"]
