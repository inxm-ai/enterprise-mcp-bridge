"""
State management helpers for workflow execution.

Pure functions and simple state-mutating helpers that operate on
``WorkflowExecutionState`` and related structures.  Extracted from
``engine.py`` to reduce its size and improve testability.
"""

import logging
from typing import Any, Optional

from app.tgi.models import ChatCompletionRequest, MessageRole
from app.tgi.workflows.models import (
    WorkflowAgentDef,
    WorkflowDefinition,
    WorkflowExecutionState,
)

logger = logging.getLogger("uvicorn.error")

# Constants (mirrored from engine for decoupling)
WORKFLOW_OWNER_KEY = "_workflow_owner_id"
CONTINUE_PLACEHOLDER = "[continue]"
END_PLACEHOLDER = "[end]"
FEEDBACK_PAUSE_NOTICE_KEY = "_feedback_pause_notice_sent"


def find_feedback_agent(state: WorkflowExecutionState) -> Optional[str]:
    """Identify an agent that requested feedback so we can resume it first."""
    agents_ctx = state.context.get("agents", {}) or {}
    for name, ctx in agents_ctx.items():
        if isinstance(ctx, dict) and ctx.get("awaiting_feedback"):
            logger.info("[state_management] Found feedback agent: '%s'", name)
            return name
    return None


def mark_workflow_success(
    state: WorkflowExecutionState,
    save_fn,
    reason: Optional[str] = None,
) -> None:
    """Mark a workflow as successfully completed."""
    state.completed = True
    state.awaiting_feedback = False
    state.context["_workflow_outcome"] = "success"
    if reason:
        state.context["_workflow_end_reason"] = reason
    save_fn(state)


def reset_feedback_pause_notice(state: WorkflowExecutionState) -> None:
    """Clear the feedback-pause notice flag."""
    state.context.pop(FEEDBACK_PAUSE_NOTICE_KEY, None)


def set_awaiting_feedback(state: WorkflowExecutionState) -> None:
    """Put the workflow into awaiting-feedback mode."""
    state.awaiting_feedback = True
    reset_feedback_pause_notice(state)


def append_user_message(
    state: WorkflowExecutionState,
    message: Optional[str],
    *,
    update_user_query: bool = True,
) -> None:
    """Persist user messages so resumed runs have full history."""
    if not message:
        return
    history = state.context.setdefault("user_messages", [])
    if not history or history[-1] != message:
        history.append(message)
    if update_user_query and not state.awaiting_feedback:
        state.context["user_query"] = message


def append_assistant_message(
    state: WorkflowExecutionState, message: Optional[str]
) -> None:
    """Persist assistant messages for looping workflows."""
    if not message:
        return
    history = state.context.setdefault("assistant_messages", [])
    if not history or history[-1] != message:
        history.append(message)


def get_original_user_prompt(state: WorkflowExecutionState) -> Optional[str]:
    """Fetch the first user message captured for this workflow."""
    messages = state.context.get("user_messages") or []
    if not isinstance(messages, list) or not messages:
        return None
    first = messages[0]
    return str(first) if first is not None else None


def should_persist_inner_thinking(
    request_persist: Optional[bool],
    state: WorkflowExecutionState,
) -> bool:
    """Decide whether to keep full agent content in context.

    Args:
        request_persist: The ``persist_inner_thinking`` flag from the request,
            or ``None`` if not set.
        state: The workflow execution state.
    """
    if request_persist is not None:
        persist = bool(request_persist)
    else:
        persist = bool(state.context.get("_persist_inner_thinking"))
    state.context["_persist_inner_thinking"] = persist
    return persist


def prune_inner_thinking(
    state: WorkflowExecutionState, workflow_def: WorkflowDefinition
) -> None:
    """Strip stored inner thinking for non-pass-through agents to reduce context."""
    agents_ctx = state.context.get("agents", {}) or {}
    pass_through_map = {
        agent_def.agent: bool(agent_def.pass_through)
        for agent_def in workflow_def.agents
    }
    for agent_name, ctx in agents_ctx.items():
        if not isinstance(ctx, dict):
            continue
        should_keep = pass_through_map.get(agent_name, bool(ctx.get("pass_through")))
        if should_keep:
            continue
        ctx["content"] = ""


def get_completed_agents(
    workflow_def: WorkflowDefinition, state: WorkflowExecutionState
) -> set[str]:
    """Determine which agents are complete based on stored context.

    Agents explicitly marked as awaiting feedback or with ``completed=False``
    are treated as incomplete so they can resume when the user responds.
    """
    completed: set[str] = set()
    agents_ctx = state.context.get("agents", {}) or {}
    for agent_def in workflow_def.agents:
        ctx = agents_ctx.get(agent_def.agent) or {}
        if not isinstance(ctx, dict):
            continue
        if ctx.get("awaiting_feedback"):
            logger.info(
                "[state_management] Agent '%s' awaiting feedback, treating as incomplete",
                agent_def.agent,
            )
            continue
        explicitly_completed = ctx.get("completed")
        if explicitly_completed is False:
            logger.info(
                "[state_management] Agent '%s' explicitly marked incomplete",
                agent_def.agent,
            )
            continue
        if explicitly_completed is True:
            completed.add(agent_def.agent)
            continue
        # Legacy check: consider presence of content (non-empty) as completed
        content = ctx.get("content")
        if content:
            completed.add(agent_def.agent)
    logger.info("[state_management] Completed agents: %s", completed)
    return completed


def apply_start_with(
    state: WorkflowExecutionState,
    workflow_def: WorkflowDefinition,
    payload: dict,
    save_fn,
    complete_deps_fn,
) -> None:
    """Prefill workflow context and optionally force a starting agent.

    Expected payload shape::

        {"args": {...}, "agent": "agent_name"}
    """
    if not isinstance(payload, dict):
        return

    args = payload.get("args")
    if isinstance(args, dict):
        for key, value in args.items():
            state.context[key] = value

    target_agent = payload.get("agent")
    if target_agent:
        complete_deps_fn(state, workflow_def, target_agent)
        state.context["_resume_agent"] = target_agent

    state.context["_start_with_applied"] = True
    save_fn(state)


def complete_dependencies_for_agent(
    state: WorkflowExecutionState,
    workflow_def: WorkflowDefinition,
    agent_name: str,
) -> None:
    """Mark dependent agents as completed so a forced start agent can run.

    This lets a forced start agent execute even if it normally depends on
    earlier steps.
    """
    agents_map = {a.agent: a for a in workflow_def.agents}
    visited: set[str] = set()

    def _mark(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        agent_def = agents_map.get(name)
        if not agent_def:
            return
        for dep in agent_def.depends_on or []:
            _mark(dep)
            ctx = state.context.setdefault("agents", {}).setdefault(
                dep, {"content": "", "pass_through": False}
            )
            ctx.setdefault("reason", "start_with_prefill")
            ctx["completed"] = True

    _mark(agent_name)


def reset_state_for_handoff(
    state: WorkflowExecutionState,
    new_flow_id: str,
    save_fn,
) -> None:
    """Reset workflow state for an in-place handoff to a new workflow.

    Preserves execution history and user ownership while clearing agent context.
    """
    preserved: dict[str, Any] = {}
    for key in (
        WORKFLOW_OWNER_KEY,
        "user_query",
        "user_messages",
        "_persist_inner_thinking",
    ):
        if key in state.context:
            preserved[key] = state.context.get(key)

    state.flow_id = new_flow_id
    state.context = {"agents": {}}
    state.context.update(preserved)
    state.current_agent = None
    state.completed = False
    state.awaiting_feedback = False
    save_fn(state)


def reset_state_for_loop_turn(
    state: WorkflowExecutionState,
    ensure_task_id_fn,
) -> None:
    """Reset per-turn workflow state for looping workflows while preserving history."""
    state.completed = False
    state.awaiting_feedback = False
    state.current_agent = None
    state.context.pop("_resume_agent", None)
    agents_ctx = state.context.get("agents", {}) or {}
    for ctx in agents_ctx.values():
        if not isinstance(ctx, dict):
            continue
        ctx["completed"] = False
        ctx.pop("awaiting_feedback", None)
        ctx.pop("had_feedback", None)
        ctx.pop("pending_user_reroute", None)
        ctx.pop("reroute_reason", None)
        ctx.pop("reroute_start_with", None)
        ctx.pop("feedback_prompt", None)
        ctx.pop("elicitation_spec", None)
        ctx.pop("elicitation_choices", None)
        ctx.pop("feedback_spec", None)
        ctx.pop("feedback_choices", None)
        ctx.pop("tool_errors", None)
        ctx.pop("return_attempts", None)
        ctx.pop("skipped", None)
        ctx.pop("reason", None)
    ensure_task_id_fn(state, reset=True)


def can_parallelize_agent(agent_def: WorkflowAgentDef) -> bool:
    """Conservative safety gate for parallel execution.

    Parallel execution is limited to agents with disabled tools and no reroute
    or stop semantics, to avoid side effects and control-flow races.
    """
    if agent_def.stop_point:
        return False
    if agent_def.reroute or agent_def.on_tool_error:
        return False
    if agent_def.pass_through:
        return False
    return agent_def.tools == []


def is_routing_only_agent(agent_def: WorkflowAgentDef) -> bool:
    """Heuristic to detect routing-only agents that should not ask for feedback."""
    desc = (agent_def.description or "").lower()
    if "routing-only" in desc or "routing only" in desc:
        return True
    if "respond only with" in desc and "<reroute>" in desc:
        return True
    return False


def strip_continue_placeholder(
    request: ChatCompletionRequest,
    placeholder: str = CONTINUE_PLACEHOLDER,
) -> bool:
    """Drop user messages when last user message is the resume placeholder.

    Returns ``True`` if the placeholder was found and removed.
    """
    for idx in range(len(request.messages) - 1, -1, -1):
        message = request.messages[idx]
        if message.role != MessageRole.USER:
            continue
        if not (message.content and message.content.strip().lower() == placeholder):
            return False
        del request.messages[idx]
        return True
    return False


def skip_agent(
    agent_def: WorkflowAgentDef,
    state: WorkflowExecutionState,
    completed_agents: set[str],
    save_fn,
) -> None:
    """Mark *agent_def* as skipped (condition not met) and persist the state."""
    state.context["agents"][agent_def.agent] = {
        "content": "",
        "pass_through": agent_def.pass_through,
        "skipped": True,
        "reason": "condition_not_met",
        "completed": True,
    }
    save_fn(state)
    completed_agents.add(agent_def.agent)


def resolve_request_user_id(user_token: str, user_info_extractor) -> str:
    """Extract and return the user ID from *user_token*.

    Raises ``PermissionError`` when extraction fails or yields no ID.
    """
    try:
        user_info = user_info_extractor.extract_user_info(user_token)
    except Exception as exc:
        logger.warning(
            "[WorkflowEngine] Failed to extract user info for workflow access: %s",
            exc,
        )
        raise PermissionError(
            "Invalid access token; unable to identify workflow owner."
        ) from exc
    user_id = user_info.get("user_id")
    if not user_id:
        raise PermissionError(
            "Access token did not include a user identifier for workflow access."
        )
    return str(user_id)


def enforce_workflow_owner(
    state: WorkflowExecutionState,
    user_token: str,
    user_info_extractor,
    owner_key: str = WORKFLOW_OWNER_KEY,
) -> None:
    """Ensure *user_token* matches the stored owner, or set it for new executions.

    Raises ``PermissionError`` on mismatch or missing token.
    """
    stored_owner = state.context.get(owner_key)
    if not user_token:
        if stored_owner:
            raise PermissionError(
                "Access token required to resume this workflow execution."
            )
        return

    current_user_id = resolve_request_user_id(user_token, user_info_extractor)
    if stored_owner:
        if current_user_id != stored_owner:
            raise PermissionError(
                f"Workflow execution '{state.execution_id}' belongs to a different user."
            )
        state.owner_id = stored_owner
        return
    if current_user_id:
        state.context[owner_key] = current_user_id
        state.owner_id = current_user_id
