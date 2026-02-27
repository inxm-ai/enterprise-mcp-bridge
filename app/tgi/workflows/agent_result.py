"""
Post-streaming analysis for a completed agent execution.

After the LLM streaming loop finishes, ``finalize_agent_result`` parses the
accumulated content for reroute, feedback and return tags and yields the
appropriate status events back to ``WorkflowEngine._execute_agent``.

All I/O is performed through callback parameters so the module stays
free of direct dependencies on ``WorkflowEngine``.
"""

import copy
import logging
from typing import Any, AsyncGenerator, Callable, Optional

from app.elicitation import canonicalize_elicitation_payload
from app.tgi.workflows import (
    context_builder,
    dict_utils,
    error_analysis,
    reroute,
    state_management,
    tag_parser,
)
from app.tgi.workflows import feedback as feedback_mod
from app.tgi.workflows.arg_injector import ToolResultCapture
from app.tgi.workflows.models import (
    WorkflowAgentDef,
    WorkflowDefinition,
    WorkflowExecutionState,
)

logger = logging.getLogger("uvicorn.error")


async def finalize_agent_result(
    *,
    agent_def: WorkflowAgentDef,
    workflow_def: WorkflowDefinition,
    state: WorkflowExecutionState,
    agent_context: dict[str, Any],
    content_text: str,
    tool_errors: list[dict[str, Any]],
    tool_outcomes: list[dict[str, str]],
    passthrough_history: list[str],
    persist_inner_thinking: bool,
    was_awaiting_feedback: bool,
    had_feedback: bool,
    result_capture: Optional[ToolResultCapture],
    no_reroute: bool,
    request: Any,
    access_token: Optional[str],
    span: Any,
    # callbacks
    record_event_fn: Callable[..., str],
    save_fn: Callable[[WorkflowExecutionState], None],
    render_feedback_question_fn: Any,
    routing_decide_fn: Any,
    max_return_retries: int,
) -> AsyncGenerator[Any, None]:
    """Analyse accumulated LLM output after streaming and yield status events.

    This is the second half of ``_execute_agent``, factored out so that
    ``engine.py`` stays focused on streaming orchestration.
    """

    reroute_tag = tag_parser.extract_tag_with_attrs(content_text, "reroute")
    reroute_reason = reroute_tag.content
    reroute_source: Optional[str] = "llm" if reroute_reason else None
    if not reroute_reason and tool_outcomes:
        tool_reroute_reason = reroute.match_tool_reroute_reason(
            agent_def.reroute, tool_outcomes
        )
        if tool_reroute_reason:
            reroute_reason = tool_reroute_reason
            reroute_source = "tool"
    reroute_start_with = reroute.parse_start_with(reroute_tag.attrs.get("start_with"))

    feedback_tag = tag_parser.extract_tag_with_attrs(
        content_text, "user_feedback_needed"
    )
    feedback_needed = bool(feedback_tag.content)
    inline_returns = tag_parser.extract_return_values(content_text)
    cleaned_content = tag_parser.strip_tags(content_text)
    content_for_context = cleaned_content.strip()

    if not persist_inner_thinking:
        content_for_context = context_builder.visible_agent_content(
            agent_def, cleaned_content, passthrough_history
        )
        if (
            not agent_def.pass_through
            and not agent_def.returns
            and not inline_returns
            and not agent_context.get("awaiting_feedback")
            and not was_awaiting_feedback
            and not had_feedback
        ):
            content_for_context = ""

    # Update agent context, preserving any captured returns
    agent_context["content"] = content_for_context or ""
    agent_context["pass_through"] = agent_def.pass_through

    # Store tool errors in agent context for visibility
    if tool_errors:
        agent_context["tool_errors"] = tool_errors

    if reroute_start_with is not None:
        agent_context["reroute_start_with"] = reroute_start_with

    # Capture inline <return name="...">value</return> tags into agent context
    for name, value in inline_returns:
        dict_utils.set_path_value(agent_context, name, value)

    if feedback_needed and state_management.is_routing_only_agent(agent_def):
        if not reroute_reason and reroute.match_reroute_entry(
            agent_def.reroute, "ASK_USER"
        ):
            reroute_reason = "ASK_USER"
            reroute_source = "engine"
        if reroute_reason:
            logger.warning(
                "[WorkflowEngine] Routing-only agent '%s' emitted user_feedback_needed; "
                "ignoring feedback and forcing reroute '%s'.",
                agent_def.agent,
                reroute_reason,
            )
            feedback_needed = False

    if reroute_reason:
        agent_context["reroute_reason"] = reroute_reason

    # ---- Feedback handling ----
    if feedback_needed:
        result = _handle_feedback(
            feedback_tag,
            agent_def,
            state,
            agent_context,
            cleaned_content,
            record_event_fn,
            save_fn,
        )
        if result is not None:
            for event in result:
                yield event
            return

    agent_context.pop("awaiting_feedback", None)

    # ---- Reroute handling ----
    if reroute_reason and not no_reroute:
        async for event in _handle_reroute(
            agent_def=agent_def,
            workflow_def=workflow_def,
            state=state,
            agent_context=agent_context,
            reroute_reason=reroute_reason,
            reroute_source=reroute_source,
            reroute_start_with=reroute_start_with,
            request=request,
            access_token=access_token,
            span=span,
            record_event_fn=record_event_fn,
            save_fn=save_fn,
            render_feedback_question_fn=render_feedback_question_fn,
            routing_decide_fn=routing_decide_fn,
        ):
            yield event
        # If _handle_reroute yielded a terminal status, it set completed/etc.
        # Check if the generator signalled completion via agent_context marker.
        if agent_context.get("_finalized"):
            agent_context.pop("_finalized", None)
            return

    # ---- Tool error reroute ----
    if tool_errors and agent_def.on_tool_error and not no_reroute:
        logger.info(
            "[WorkflowEngine] Tool errors detected, triggering on_tool_error reroute to '%s'",
            agent_def.on_tool_error,
        )
        agent_context["reroute_reason"] = "TOOL_ERROR"
        agent_context["completed"] = True
        save_fn(state)
        yield record_event_fn(
            state,
            f"\nTool execution failed, rerouting to {agent_def.on_tool_error}...\n",
            status="reroute",
        )
        yield {"status": "reroute", "target": agent_def.on_tool_error}
        return

    # ---- Missing returns ----
    missing_returns = error_analysis.get_missing_returns(
        agent_def, result_capture, agent_context
    )
    if missing_returns:
        action, events = error_analysis.handle_missing_returns(
            agent_def=agent_def,
            state=state,
            agent_context=agent_context,
            missing_returns=missing_returns,
            content=cleaned_content,
            tool_errors=tool_errors,
            return_specs=result_capture.return_specs if result_capture else None,
            record_event_fn=record_event_fn,
            save_fn=save_fn,
            max_retries=max_return_retries,
        )
        for event in events:
            yield event
        if action == "abort":
            yield {"status": "abort"}
            return
        if action == "retry":
            yield {"status": "retry"}
            return

    # ---- Success ----
    agent_context["completed"] = True
    agent_context.pop("had_feedback", None)
    agent_context.pop("return_attempts", None)
    save_fn(state)
    yield {
        "status": "done",
        "content": cleaned_content,
        "pass_through": agent_def.pass_through,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _handle_feedback(
    feedback_tag: Any,
    agent_def: WorkflowAgentDef,
    state: WorkflowExecutionState,
    agent_context: dict[str, Any],
    cleaned_content: str,
    record_event_fn: Callable[..., str],
    save_fn: Callable[[WorkflowExecutionState], None],
) -> Optional[list[Any]]:
    """Process user_feedback_needed and return events list, or None to skip."""
    feedback_payload = feedback_mod.parse_feedback_payload(feedback_tag.content)
    if feedback_payload:
        raw_payload = feedback_payload
        feedback_choices = feedback_mod.build_feedback_choices_from_payload(raw_payload)
        resolved_payload = feedback_mod.resolve_external_with_in_feedback_payload(
            copy.deepcopy(raw_payload), agent_context, state.context
        )
        return _setup_feedback(
            state,
            agent_context,
            resolved_payload,
            feedback_choices,
            resolved_payload.get("message") or cleaned_content,
            record_event_fn,
            save_fn,
        )

    if cleaned_content:
        agent_context["feedback_prompt"] = cleaned_content

    fallback_payload = canonicalize_elicitation_payload(
        {
            "message": cleaned_content or "Please provide feedback.",
            "requestedSchema": {
                "type": "object",
                "properties": {"response": {"type": "string"}},
                "required": ["response"],
                "additionalProperties": False,
            },
            "meta": {},
        }
    )
    return _setup_feedback(
        state,
        agent_context,
        fallback_payload,
        None,
        cleaned_content,
        record_event_fn,
        save_fn,
    )


def _setup_feedback(
    state: WorkflowExecutionState,
    agent_context: dict[str, Any],
    payload: dict[str, Any],
    choices: Optional[list[dict[str, Any]]],
    prompt_text: Optional[str],
    record_event_fn: Callable[..., str],
    save_fn: Callable[[WorkflowExecutionState], None],
) -> list[Any]:
    """Set up agent context for feedback and return events to yield."""
    feedback_block = feedback_mod.format_feedback_block(payload)
    agent_context["feedback_prompt"] = prompt_text or ""
    agent_context["elicitation_spec"] = payload
    if choices is not None:
        agent_context["elicitation_choices"] = choices
        agent_context["feedback_choices"] = choices
    agent_context["feedback_spec"] = feedback_mod.legacy_feedback_spec_view(payload)
    agent_context["awaiting_feedback"] = True
    agent_context["completed"] = False
    state_management.set_awaiting_feedback(state)
    save_fn(state)
    event = record_event_fn(
        state,
        feedback_block,
        status="waiting_for_feedback",
    )
    return [event, {"status": "feedback", "content": feedback_block}]


async def _handle_reroute(
    *,
    agent_def: WorkflowAgentDef,
    workflow_def: WorkflowDefinition,
    state: WorkflowExecutionState,
    agent_context: dict[str, Any],
    reroute_reason: str,
    reroute_source: Optional[str],
    reroute_start_with: Optional[dict[str, Any]],
    request: Any,
    access_token: Optional[str],
    span: Any,
    record_event_fn: Callable[..., str],
    save_fn: Callable[[WorkflowExecutionState], None],
    render_feedback_question_fn: Any,
    routing_decide_fn: Any,
) -> AsyncGenerator[Any, None]:
    """Handle reroute logic, yielding events.

    Sets ``agent_context["_finalized"] = True`` when the reroute is
    terminal (i.e. the caller should ``return`` after consuming events).
    """
    matched_cfg = reroute.match_reroute_entry(agent_def.reroute, reroute_reason)
    ask_cfg = matched_cfg.get("ask") if isinstance(matched_cfg, dict) else None

    if isinstance(ask_cfg, dict):
        question = await render_feedback_question_fn(
            ask_cfg,
            agent_context,
            state.context,
            request,
            access_token,
            span,
        )
        choices = feedback_mod.build_feedback_choices(
            ask_cfg, agent_context, state.context
        )
        feedback_payload = feedback_mod.build_feedback_payload(
            question, choices, agent_context, state.context
        )
        events = _setup_feedback(
            state,
            agent_context,
            feedback_payload,
            choices,
            feedback_payload.get("message") or question,
            record_event_fn,
            save_fn,
        )
        for event in events:
            yield event
        agent_context["_finalized"] = True
        return

    agent_context["completed"] = True
    target, with_fields = reroute.match_reroute_target(
        agent_def.reroute, reroute_reason
    )
    workflow_target = reroute.parse_workflow_reroute_target(reroute_reason)
    if not workflow_target and target:
        workflow_target = reroute.parse_workflow_reroute_target(target)

    if workflow_target:
        if with_fields:
            reroute.apply_reroute_with(state.context, agent_context, with_fields)
            reroute_start_with = reroute.merge_start_with(
                reroute_start_with, agent_context, state.context, with_fields
            )
        save_fn(state)
        yield {
            "status": "workflow_reroute",
            "target_workflow": workflow_target,
            "start_with": reroute_start_with,
        }
        agent_context["_finalized"] = True
        return

    if target:
        if with_fields:
            reroute.apply_reroute_with(state.context, agent_context, with_fields)
        save_fn(state)
        yield {"status": "reroute", "target": target}
        agent_context["_finalized"] = True
        return

    if reroute_source != "tool":
        dynamic_target = await routing_decide_fn(
            agent_def,
            reroute_reason,
            state.context,
            request,
            access_token,
            span,
            execution_id=state.execution_id,
        )
        if dynamic_target:
            save_fn(state)
            yield {"status": "reroute", "target": dynamic_target}
            agent_context["_finalized"] = True
            return
