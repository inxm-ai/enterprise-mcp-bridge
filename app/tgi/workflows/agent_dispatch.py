"""Agent dispatch logic for workflow execution.

Pure functions for agent selection, status processing, and completion
handling within a workflow run.  Extracted from ``engine._run_agents``
to reduce its size and improve testability.
"""

import logging
from typing import Any, Optional

from app.tgi.workflows import state_management
from app.tgi.workflows.models import (
    WorkflowAgentDef,
    WorkflowDefinition,
    WorkflowExecutionState,
)

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Agent selection (pure)
# ---------------------------------------------------------------------------


def find_runnable_agents(
    agents: list[WorkflowAgentDef],
    completed_agents: set[str],
) -> list[WorkflowAgentDef]:
    """Return agents whose dependencies are satisfied and not yet completed."""
    return [
        a
        for a in agents
        if a.agent not in completed_agents
        and (not a.depends_on or set(a.depends_on).issubset(completed_agents))
    ]


def select_parallel_batch(
    runnable_agents: list[WorkflowAgentDef],
    max_parallel: int,
) -> list[WorkflowAgentDef]:
    """Return up to *max_parallel* parallelizable agents.

    Returns an empty list if fewer than 2 candidates qualify.
    """
    candidates = [
        a for a in runnable_agents if state_management.can_parallelize_agent(a)
    ]
    if len(candidates) > 1:
        return candidates[:max_parallel]
    return []


# ---------------------------------------------------------------------------
# Status processing (async – uses callbacks for side effects)
# ---------------------------------------------------------------------------


async def process_agent_status(
    result: dict[str, Any],
    workflow_def: WorkflowDefinition,
    state: WorkflowExecutionState,
    *,
    save_fn,
    record_event_fn,
    handle_workflow_handoff_fn,
) -> dict[str, Any]:
    """Interpret a single agent status result.

    *handle_workflow_handoff_fn* is an ``async (result) -> (events, should_return)``
    callback that the caller provides (typically a closure capturing the
    session / request context).

    Returns a dict with control signals:

    - ``events``: SSE chunks to yield
    - ``should_return``, ``should_break``: control flow signals
    - ``last_visible``: update ``last_visible_output`` when present
    - ``forced_next``: new forced agent (key present → update)
    - ``clear_forced``: set ``forced_next`` to ``None``
    - ``retry``: set ``retry_triggered`` + ``progress_made``
    """
    action: dict[str, Any] = {"events": []}
    status = result.get("status")
    logger.debug("[_run_agents] status=%s", status)

    if status == "done" and result.get("content") and not result.get("pass_through"):
        action["last_visible"] = result["content"]
    if status == "done" and workflow_def.loop:
        text = (result.get("content") or "").strip()
        if result.get("pass_through") and text:
            state_management.append_assistant_message(state, text)
            save_fn(state)

    if status == "feedback":
        action["should_return"] = True
    elif status == "reroute":
        target = result.get("target")
        action["forced_next"] = target
        if target:
            action["events"].append(
                record_event_fn(state, f"\nRerouting to {target}\n", status="reroute")
            )
    elif status == "workflow_reroute":
        handoff_events, should_return = await handle_workflow_handoff_fn(result)
        action["events"].extend(handoff_events)
        if should_return:
            action["should_return"] = True
    elif status == "retry":
        action["retry"] = True
        action["should_break"] = True
    elif status == "abort":
        action["should_return"] = True
    else:
        action["clear_forced"] = True
    return action


def apply_status_action(
    action: dict[str, Any],
    forced_next: Optional[str],
    last_visible_output: Optional[str],
) -> tuple[Optional[str], Optional[str], bool, bool]:
    """Apply a status action dict to tracking variables.

    Returns ``(forced_next, last_visible_output, retry, progress_made_delta)``
    where *progress_made_delta* should be OR-ed into the caller's flag.

    The caller is still responsible for yielding ``action["events"]`` and
    honoring ``action["should_return"]`` / ``action["should_break"]``.
    """
    if "last_visible" in action:
        last_visible_output = action["last_visible"]
    if "forced_next" in action:
        forced_next = action["forced_next"]
    elif action.get("clear_forced"):
        forced_next = None
    retry = bool(action.get("retry"))
    return forced_next, last_visible_output, retry, retry


# ---------------------------------------------------------------------------
# Post-execution helpers (use callbacks for side effects)
# ---------------------------------------------------------------------------


def check_stop_point(
    agent_def: WorkflowAgentDef,
    state: WorkflowExecutionState,
    save_fn,
    record_event_fn,
) -> list[str]:
    """Mark workflow as completed if *agent_def* is a stop point.

    Returns a list of SSE event strings to yield.
    """
    if not agent_def.stop_point:
        return []
    state.completed = True
    save_fn(state)
    return [
        record_event_fn(
            state,
            f"Stop point reached at {agent_def.agent}; halting workflow execution.",
            status="stop_point",
        )
    ]


def build_workflow_completion(
    state: WorkflowExecutionState,
    completed_agents: set[str],
    workflow_def: WorkflowDefinition,
    last_visible_output: Optional[str],
    record_event_fn,
    save_fn,
) -> list[str]:
    """Build final completion events when all agents are done.

    Returns a list of SSE event strings to yield.
    """
    if state.awaiting_feedback or len(completed_agents) != len(workflow_def.agents):
        return []
    events: list[str] = []
    if last_visible_output:
        events.append(record_event_fn(state, f"Result: {last_visible_output.strip()}"))
    if workflow_def.loop:
        save_fn(state)
        return events
    state.completed = True
    events.append(
        record_event_fn(
            state,
            "\nWorkflow complete\n",
            status="completed",
            finish_reason="stop",
        )
    )
    save_fn(state)
    return events
