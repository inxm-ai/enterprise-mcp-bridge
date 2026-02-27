"""
Pure functions for detecting, classifying and summarising errors that
occur during agent execution or tool invocation.

All functions are stateless.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.tgi.workflows import dict_utils

logger = logging.getLogger("uvicorn.error")


@dataclass
class ReturnFailureAnalysis:
    """Categorised result of analysing missing-return errors."""

    fatal: bool
    recoverable: bool
    messages: list[str]


# ---------------------------------------------------------------------------
# Individual error detectors
# ---------------------------------------------------------------------------


def tool_result_has_error(content: str) -> bool:
    """
    Detect if a tool-result *content* string indicates an error.

    Mirrors the logic in ``ToolService._result_has_error`` but works on the
    raw content string from tool_result events.
    """
    if not content:
        return False

    def _try_parse(value: Any) -> Any:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    def _parse_text_entries(value: Any) -> Any:
        if not isinstance(value, list):
            return None
        texts: list[str] = []
        for item in value:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
        if not texts:
            return None
        if len(texts) == 1:
            return _try_parse(texts[0])
        return _try_parse("".join(texts).strip())

    def _has_error(payload: Any) -> bool:
        if isinstance(payload, dict):
            if payload.get("error"):
                return True
            if payload.get("isError") is True:
                return True
            if payload.get("success") is False:
                return True
            if payload.get("errors"):
                return True
            nested_json = _parse_text_entries(payload.get("content"))
            if nested_json is not None and _has_error(nested_json):
                return True
            return False
        if isinstance(payload, list):
            if any(_has_error(item) for item in payload):
                return True
            nested_json = _parse_text_entries(payload)
            if nested_json is not None and _has_error(nested_json):
                return True
            return False
        if isinstance(payload, str):
            parsed = _try_parse(payload)
            if parsed is not None:
                return _has_error(parsed)
            lowered = payload.lower()
            if '"error":' in lowered or "'error':" in lowered:
                return True
            if "bad request" in lowered or "400" in payload:
                return True
            if "internal server error" in lowered or "500" in payload:
                return True
            if "not found" in lowered and "404" in payload:
                return True
            if "unauthorized" in lowered or "401" in payload:
                return True
            if "forbidden" in lowered or "403" in payload:
                return True
        return False

    parsed_content = _try_parse(content)
    if parsed_content is not None:
        return _has_error(parsed_content)
    return _has_error(content)


def looks_like_error_text(text: str) -> bool:
    """Heuristic check for common error keywords."""
    lowered = (text or "").lower()
    indicators = [
        "error",
        "exception",
        "fail",
        "unable",
        "denied",
        "invalid",
        "timeout",
        "unavailable",
    ]
    return any(indicator in lowered for indicator in indicators)


def shorten_error_text(text: str, max_length: int = 200) -> str:
    """Collapse whitespace and truncate long error text."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) > max_length:
        return cleaned[: max_length - 3] + "..."
    return cleaned


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def collect_error_messages(
    content: Optional[str],
    tool_errors: Optional[list[dict[str, Any]]],
) -> list[str]:
    """Collect human-readable error snippets from tool errors and response content."""
    messages: list[str] = []

    for err in tool_errors or []:
        snippet = shorten_error_text(err.get("content", ""))
        if not snippet:
            continue
        name = err.get("name")
        if name:
            snippet = f"{name}: {snippet}"
        messages.append(snippet)

    if content and looks_like_error_text(content):
        messages.append(shorten_error_text(content))

    return messages


def analyze_return_failure(
    content: Optional[str],
    tool_errors: Optional[list[dict[str, Any]]],
) -> ReturnFailureAnalysis:
    """Classify error messages into *fatal* vs *recoverable*."""
    error_messages = collect_error_messages(content, tool_errors)
    fatal_markers = [
        "unauthorized",
        "forbidden",
        "permission denied",
        "policy",
        "bad request",
        "invalid",
        "not found",
        "internal server error",
        "traceback",
        "exception",
        "fatal",
    ]
    recoverable_markers = [
        "timeout",
        "temporarily",
        "temporary",
        "rate limit",
        "retry",
        "try again",
        "unavailable",
        "busy",
        "conflict",
        "locked",
        "overloaded",
        "429",
        "503",
    ]

    fatal = False
    for message in error_messages:
        lower = message.lower()
        if any(marker in lower for marker in fatal_markers):
            fatal = True
            break

    recoverable = bool(error_messages) and not fatal
    recoverable_hint = any(
        marker in message.lower()
        for message in error_messages
        for marker in recoverable_markers
    )
    if recoverable_hint and not fatal:
        recoverable = True

    return ReturnFailureAnalysis(
        fatal=fatal,
        recoverable=recoverable,
        messages=error_messages,
    )


def format_error_summary(messages: list[str]) -> str:
    """Join error messages with ``|``."""
    if not messages:
        return ""
    return " | ".join(messages)


def response_has_bad_error(
    content: Optional[str],
    tool_errors: Optional[list[dict[str, Any]]] = None,
) -> bool:
    """Detect whether the agent response is non-recoverable."""
    analysis = analyze_return_failure(content, tool_errors)
    return analysis.fatal


# ---------------------------------------------------------------------------
# Missing returns helpers
# ---------------------------------------------------------------------------

MAX_RETURN_RETRIES = 3


def get_missing_returns(
    agent_def: Any,
    result_capture: Any,
    agent_context: dict,
) -> list[str]:
    """Identify which declared return values are missing from agent context."""
    if not agent_def.returns or not result_capture:
        return []

    missing: list[str] = []
    for spec in result_capture.return_specs:
        store_path = spec.as_name or spec.field
        if not store_path:
            continue
        if not dict_utils.has_value(agent_context, store_path):
            missing.append(store_path)
    return missing


def clear_return_paths(agent_context: dict, return_specs: Optional[list[Any]]) -> None:
    """Clear previously captured return data so retries start fresh."""
    if not return_specs:
        return
    for spec in return_specs:
        store_path = getattr(spec, "as_name", None) or getattr(spec, "field", None)
        if not store_path:
            continue
        if "." in store_path:
            dict_utils.delete_path(agent_context, store_path)
        else:
            agent_context.pop(store_path, None)


def handle_missing_returns(
    agent_def: Any,
    state: Any,
    agent_context: dict,
    missing_returns: list[str],
    content: str,
    tool_errors: Optional[list[dict[str, Any]]],
    return_specs: Optional[list[Any]],
    record_event_fn,
    save_fn,
    max_retries: int = MAX_RETURN_RETRIES,
) -> tuple[str, list[Any]]:
    """Decide whether to retry or abort when required returns are missing.

    Returns:
        Tuple of (action, events)
        action: ``"retry"`` or ``"abort"``
        events: list of streamable events already formatted
    """
    events: list[Any] = []
    attempts = int(agent_context.get("return_attempts", 0)) + 1
    agent_context["return_attempts"] = attempts

    missing_list = ", ".join(missing_returns)
    failure = analyze_return_failure(content, tool_errors)
    summary = format_error_summary(failure.messages)

    if failure.fatal:
        agent_context["completed"] = True
        state.completed = True
        save_fn(state)
        message = (
            f"Agent '{agent_def.agent}' encountered a non-recoverable error "
            f"while missing required data ({missing_list}). "
            "This agent needs to be fixed."
        )
        if summary:
            message += f" Errors: {summary}"
        events.append(
            record_event_fn(
                state,
                message,
                status="error",
                metadata={
                    "missing_returns": missing_returns,
                    "attempts": attempts,
                    "reason": "non_recoverable_error",
                    "errors": failure.messages or None,
                },
            )
        )
        events.append("data: [DONE]\n\n")
        return "abort", events

    if attempts >= max_retries:
        agent_context["completed"] = True
        state.completed = True
        save_fn(state)
        message = (
            f"Agent '{agent_def.agent}' did not provide required data "
            f"after {attempts} attempts ({missing_list}). "
            "This agent needs to be fixed."
        )
        if summary:
            message += f" Errors: {summary}"
        events.append(
            record_event_fn(
                state,
                message,
                status="error",
                metadata={
                    "missing_returns": missing_returns,
                    "attempts": attempts,
                    "reason": "max_retries_exceeded",
                    "errors": failure.messages or None,
                },
            )
        )
        events.append("data: [DONE]\n\n")
        return "abort", events

    # Retry path
    clear_return_paths(agent_context, return_specs)
    agent_context["completed"] = False
    save_fn(state)
    if failure.recoverable and summary:
        retry_message = (
            f"Agent '{agent_def.agent}' returned an error while missing required data "
            f"({missing_list}): {summary}. "
            f"Retrying (attempt {attempts}/{max_retries})."
        )
        reason = "recoverable_error"
    else:
        retry_message = (
            f"Expected data from agent '{agent_def.agent}' was missing "
            f"({missing_list}). Retrying (attempt {attempts}/{max_retries})."
        )
        reason = "missing_returns"
    events.append(
        record_event_fn(
            state,
            retry_message,
            status="retry",
            metadata={
                "missing_returns": missing_returns,
                "attempts": attempts,
                "reason": reason,
                "errors": failure.messages or None,
            },
        )
    )
    return "retry", events


def build_deadlock_error(
    workflow_agents: list,
    completed_agents: set[str],
    forced_next: Optional[str],
    state_context: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Determine *why* the workflow is stuck and return ``(message, metadata)``.

    Pure function — no side-effects.
    """
    remaining_agents = [
        a.agent for a in workflow_agents if a.agent not in completed_agents
    ]
    metadata: dict[str, Any] = {"remaining_agents": remaining_agents}
    message = "\nNo runnable agents remain; workflow cannot make progress.\n"

    if forced_next:
        metadata["forced_next"] = forced_next
        target_agent = next(
            (a for a in workflow_agents if a.agent == forced_next), None
        )
        if target_agent:
            missing_deps = sorted(
                set(target_agent.depends_on or []).difference(completed_agents)
            )
            if missing_deps:
                metadata["missing_dependencies"] = missing_deps
                message = (
                    f"\nReroute target '{forced_next}' is not runnable; "
                    f"missing dependencies: {', '.join(missing_deps)}.\n"
                )
            elif forced_next in completed_agents:
                agent_ctx = state_context.get("agents", {}).get(forced_next, {})
                reason = agent_ctx.get("reason") or agent_ctx.get("reroute_reason")
                metadata["reason"] = "already_completed"
                if reason:
                    metadata["prior_reason"] = reason
                    message = (
                        f"\nReroute target '{forced_next}' was already "
                        f"completed (reason: {reason}); reroutes cannot "
                        f"re-run completed agents.\n"
                    )
                else:
                    message = (
                        f"\nReroute target '{forced_next}' was already "
                        "completed; reroutes cannot re-run completed agents.\n"
                    )
            else:
                message = f"\nReroute target '{forced_next}' could not be executed.\n"
        else:
            message = (
                f"\nReroute target '{forced_next}' is not defined in this workflow.\n"
            )

    return message, metadata
