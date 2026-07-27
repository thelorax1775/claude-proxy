"""Pending tool-call approvals.

In manual mode the agent loop stops before running a tool and waits for the
browser to say yes or no. The loop is inside a streaming response, so it can't
read the answer from its own request -- the decision arrives on a *separate*
HTTP call (POST /api/approvals/{id}). This module is the meeting point: the
loop parks on a future, the endpoint resolves it.

Kept deliberately small and in-process. Approvals are meaningful only while
the run that created them is still streaming, so there's nothing worth
persisting -- a proxy restart ends the run anyway.
"""

import asyncio

APPROVE = "approve"
APPROVE_ALL = "approve_all"
DENY = "deny"
DECISIONS = (APPROVE, APPROVE_ALL, DENY)

# call_id -> Future[str]. One entry per tool call currently awaiting an answer.
_PENDING: dict[str, asyncio.Future] = {}


def create(call_id: str) -> asyncio.Future:
    """Register a call as awaiting a decision. Replacing an existing entry
    would strand whoever was waiting on it, so the old one is cancelled."""
    old = _PENDING.pop(call_id, None)
    if old is not None and not old.done():
        old.cancel()
    future = asyncio.get_running_loop().create_future()
    _PENDING[call_id] = future
    return future


def resolve(call_id: str, decision: str) -> bool:
    """Answer a pending call. False means there was nothing waiting -- a stale
    click, or a run that has already moved on."""
    future = _PENDING.pop(call_id, None)
    if future is None or future.done():
        return False
    future.set_result(decision)
    return True


def cancel(call_id: str) -> None:
    """Give up on a call: it timed out, or the run was stopped. Safe to call
    for an id that was already resolved."""
    future = _PENDING.pop(call_id, None)
    if future is not None and not future.done():
        future.cancel()


def is_pending(call_id: str) -> bool:
    future = _PENDING.get(call_id)
    return future is not None and not future.done()


def count() -> int:
    return len(_PENDING)
