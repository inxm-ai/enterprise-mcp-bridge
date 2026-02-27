"""
Pure functions and a small ``RoutingService`` for reroute matching,
tool-reroute triggers, workflow handoffs and the LLM-based routing agent.

Pure functions have no side-effects and no ``self`` parameter.
``RoutingService`` encapsulates the async LLM interactions for routing
decisions while keeping all config-matching logic as importable functions.
"""

import json
import logging
import re
from typing import Any, Optional

from app.tgi.workflows.context_builder import (
    create_lazy_context_tool,
    handle_lazy_context_tool,
    summarize_tool_result,
)
from app.tgi.workflows.dict_utils import (
    get_path_value,
    path_exists,
    set_path_value,
)
from app.tgi.workflows.error_analysis import tool_result_has_error
from app.tgi.workflows.lazy_context import LazyContextProvider
from app.tgi.workflows.tag_parser import (
    extract_next_agent,
    extract_run_tag,
    extract_tag,
)

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Pure reroute-config matching
# ---------------------------------------------------------------------------


def parse_start_with(value: Optional[str]) -> Optional[dict]:
    """Parse a JSON ``start_with`` payload."""
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:  # pragma: no cover
        logger.debug("[reroute] Failed to parse start_with payload: %s", exc)
    return None


def parse_workflow_reroute_target(reroute_reason: Optional[str]) -> Optional[str]:
    """Extract a workflow name from ``workflows[name]``."""
    if not reroute_reason:
        return None
    normalized = reroute_reason.strip()
    match = re.match(
        r"^workflows\[\s*(?P<name>[^\]]+)\s*\]\s*(?:\(\s*\))?$",
        normalized,
        re.IGNORECASE,
    )
    if match:
        return match.group("name").strip()
    return None


def is_external_reroute_target(target: Optional[str]) -> bool:
    """Check for ``external[name]`` reroute targets."""
    if not target:
        return False
    normalized = str(target).strip()
    return bool(
        re.match(r"^external\[\s*(?P<name>[^\]]+)\s*\]$", normalized, re.IGNORECASE)
    )


def parse_tool_reroute_trigger(value: Optional[str]) -> Optional[tuple[str, str]]:
    """Parse ``tool:tool_name:success|error``."""
    if not value:
        return None
    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 3:
        return None
    prefix, tool_name, status = parts
    if prefix.lower() != "tool" or not tool_name:
        return None
    status_norm = status.lower()
    if status_norm not in ("success", "error"):
        return None
    return tool_name, status_norm


def match_tool_reroute_reason(
    reroute_config: Any, tool_outcomes: list[dict[str, str]]
) -> Optional[str]:
    """
    Return the first matching reroute reason from config order,
    e.g. ``"tool:plan:success"``.
    """
    if not reroute_config or not tool_outcomes:
        return None
    tool_status: dict[str, str] = {}
    for entry in tool_outcomes:
        name = entry.get("name")
        status = entry.get("status")
        if name and status:
            tool_status[name] = status
    available = {(name, status) for name, status in tool_status.items()}
    if not available:
        return None
    configs = reroute_config if isinstance(reroute_config, list) else [reroute_config]
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        reroute_on = cfg.get("on") or []
        if isinstance(reroute_on, str):
            reroute_on = [reroute_on]
        for candidate in reroute_on:
            if not isinstance(candidate, str):
                continue
            parsed = parse_tool_reroute_trigger(candidate)
            if not parsed:
                continue
            tool_name, status = parsed
            if (tool_name, status) in available:
                return candidate
    return None


def has_tool_reroute(reroute_config: Any) -> bool:
    """Check whether a reroute config has any tool-based triggers."""
    if not reroute_config:
        return False
    configs = reroute_config if isinstance(reroute_config, list) else [reroute_config]
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        reroute_on = cfg.get("on") or []
        if isinstance(reroute_on, str):
            reroute_on = [reroute_on]
        for candidate in reroute_on:
            if not isinstance(candidate, str):
                continue
            if parse_tool_reroute_trigger(candidate):
                return True
    return False


def build_tool_outcomes_from_results(
    raw_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Map raw tool results to ``[{"name": ..., "status": ...}]``."""
    tool_outcomes: list[dict[str, str]] = []
    for raw_result in raw_results or []:
        tool_name = raw_result.get("name")
        if not tool_name:
            continue
        content = raw_result.get("content", "")
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                content = str(content)
        status = "error" if tool_result_has_error(content) else "success"
        tool_outcomes.append({"name": tool_name, "status": status})
    return tool_outcomes


def should_stop_after_tool_results(
    reroute_config: Any, raw_results: list[dict[str, Any]]
) -> bool:
    """Whether any tool result matches a reroute trigger."""
    if not reroute_config or not raw_results:
        return False
    tool_outcomes = build_tool_outcomes_from_results(raw_results)
    return bool(match_tool_reroute_reason(reroute_config, tool_outcomes))


def match_reroute_entry(
    reroute_config: Any, reroute_reason: Optional[str]
) -> Optional[dict[str, Any]]:
    """Return the first matching reroute entry for a given reason string."""
    if not reroute_config or not reroute_reason:
        return None
    configs = reroute_config if isinstance(reroute_config, list) else [reroute_config]
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        reroute_on = cfg.get("on") or []
        if isinstance(reroute_on, str):
            reroute_on = [reroute_on]
        if reroute_reason in reroute_on:
            return cfg
    return None


def match_reroute_target(
    reroute_config: Any, reroute_reason: Optional[str]
) -> tuple[Optional[str], list[str]]:
    """Match a reroute config, returning ``(target, with_fields)``."""
    cfg = match_reroute_entry(reroute_config, reroute_reason)
    if not cfg:
        return None, []
    with_fields = cfg.get("with") or []
    if isinstance(with_fields, str):
        with_fields = [with_fields]
    return cfg.get("to"), [w for w in with_fields if isinstance(w, str)]


# ---------------------------------------------------------------------------
# Context propagation helpers (use dict_utils under the hood)
# ---------------------------------------------------------------------------


def apply_reroute_with(
    state_context: dict, agent_context: dict, fields: list[str]
) -> None:
    """Copy selected fields from agent context to shared workflow context."""
    if not fields:
        return
    for field in fields:
        value = get_path_value(agent_context, field)
        if value is None:
            value = get_path_value(state_context, field)
            if value is not None:
                logger.info(
                    "[reroute] Field '%s' missing from agent context; using shared context value: %s",
                    field,
                    str(value)[:200],
                )
        logger.info(
            "[reroute] Copying field '%s' into workflow context, value: %s",
            field,
            str(value)[:200] if value else None,
        )
        if value is None:
            logger.warning("[reroute] Field '%s' is None, not copying", field)
            continue
        set_path_value(state_context, field, value)
        logger.info("[reroute] Successfully set '%s' in workflow context", field)


def resolve_external_with_values(
    with_fields: list[str],
    agent_context: dict[str, Any],
    shared_context: dict[str, Any],
) -> list[Any]:
    """Resolve ``with`` field values for an external reroute target."""
    resolved: list[Any] = []
    for field in with_fields or []:
        if not isinstance(field, str):
            continue
        value = get_path_value(agent_context, field)
        if isinstance(value, str) and value == field:
            value = None
        if value is None:
            value = get_path_value(shared_context, field)
            if isinstance(value, str) and value == field:
                value = None
            if value is not None:
                logger.info(
                    "[reroute] Field '%s' missing from agent context; using shared context value: %s",
                    field,
                    str(value)[:200],
                )
        if value is None:
            logger.warning(
                "[reroute] Field '%s' is None; external payload will include null",
                field,
            )
        resolved.append(value)
    return resolved


def merge_start_with(
    start_with: Optional[dict[str, Any]],
    agent_context: dict,
    state_context: dict,
    fields: list[str],
) -> Optional[dict[str, Any]]:
    """Merge reroute ``with`` fields into a ``start_with`` payload for workflow handoff."""
    if not fields:
        return start_with

    payload: dict[str, Any] = dict(start_with) if isinstance(start_with, dict) else {}
    args = payload.get("args")
    if not isinstance(args, dict):
        args = {}
        payload["args"] = args

    for field in fields:
        if path_exists(args, field):
            continue
        value = get_path_value(agent_context, field)
        if value is None:
            value = get_path_value(state_context, field)
            if value is not None:
                logger.info(
                    "[reroute] Field '%s' missing from agent context; using shared context value: %s",
                    field,
                    str(value)[:200],
                )
        logger.info(
            "[reroute] Copying field '%s' into start_with args, value: %s",
            field,
            str(value)[:200] if value else None,
        )
        if value is None:
            logger.warning("[reroute] Field '%s' is None, not copying", field)
            continue
        set_path_value(args, field, value)

    return payload


def augment_workflow_handoff_start_with(
    start_with: Optional[dict[str, Any]],
    state: Any,
) -> Optional[dict[str, Any]]:
    """Ensure workflow handoffs keep the user's original request context."""
    has_start_with = isinstance(start_with, dict)
    payload: dict[str, Any] = dict(start_with) if has_start_with else {}
    args = payload.get("args")
    if not isinstance(args, dict):
        args = {}
        payload["args"] = args

    added = False
    user_query = state.context.get("user_query")
    if "user_query" not in args and isinstance(user_query, str) and user_query:
        args["user_query"] = user_query
        added = True

    user_messages = state.context.get("user_messages")
    if (
        "user_messages" not in args
        and isinstance(user_messages, list)
        and user_messages
    ):
        args["user_messages"] = list(user_messages)
        added = True

    if not has_start_with and not added:
        return None
    return payload


def process_pending_user_reroute(
    pending: dict[str, Any],
    agent_context: dict[str, Any],
    state_context: dict[str, Any],
    *,
    append_user_message_fn,
) -> Optional[dict[str, Any]]:
    """
    Apply a pending user-reroute to *agent_context* and return a status dict.

    Returns one of ``{"status": "workflow_reroute", ...}``,
    ``{"status": "reroute", ...}``, or ``{"status": "done", ...}``.
    The caller should yield the result and then ``return``.
    """
    assignments = pending.get("assignments") or {}
    if isinstance(assignments, dict):
        for key, value in assignments.items():
            set_path_value(agent_context, key, value)

    with_fields = pending.get("with_fields") or []
    if isinstance(with_fields, str):
        with_fields = [with_fields]
    with_fields = [w for w in with_fields if isinstance(w, str)]

    if with_fields and isinstance(assignments, dict):
        for field in with_fields:
            if field in assignments:
                set_path_value(state_context, field, assignments[field])
    if with_fields:
        apply_reroute_with(state_context, agent_context, with_fields)

    # Append per-choice input field values as user messages
    input_fields = pending.get("input_fields")
    if isinstance(input_fields, dict) and isinstance(assignments, dict):
        input_parts = [
            str(assignments[fname])
            for fname in input_fields
            if fname in assignments and assignments[fname] is not None
        ]
        if input_parts:
            append_user_message_fn(" ".join(input_parts))

    agent_context["completed"] = True
    agent_context.pop("had_feedback", None)
    agent_context.pop("awaiting_feedback", None)

    workflow_target = pending.get("workflow_target")
    if workflow_target:
        return {
            "status": "workflow_reroute",
            "target_workflow": workflow_target,
            "start_with": pending.get("start_with"),
        }
    target = pending.get("target")
    if target:
        return {"status": "reroute", "target": target}
    return {"status": "done", "content": "", "pass_through": False}


def build_workflow_reroute_request(
    base_request,
    target_workflow: str,
    start_with: Optional[dict[str, Any]],
    execution_id: Optional[str] = None,
):
    """Construct a new :class:`ChatCompletionRequest` for workflow handoff."""
    copier = getattr(base_request, "model_copy", None)
    new_request = (
        copier(deep=True) if callable(copier) else base_request.copy(deep=True)
    )
    new_request.use_workflow = target_workflow
    new_request.workflow_execution_id = execution_id
    new_request.return_full_state = False
    new_request.start_with = start_with if isinstance(start_with, dict) else None
    new_request.stream = True
    return new_request


# ---------------------------------------------------------------------------
# RoutingService — async LLM interactions for routing decisions
# ---------------------------------------------------------------------------


class RoutingService:
    """
    Encapsulates the LLM calls used for routing decisions
    (intent checks, ``when`` conditions, reroute-target resolution).
    """

    def __init__(
        self,
        llm_client: Any,
        prompt_service: Any,
        state_store: Any,
    ):
        self.llm_client = llm_client
        self.prompt_service = prompt_service
        self.state_store = state_store

    async def routing_intent_check(
        self,
        session: Any,
        workflow_def: Any,
        user_message: Optional[str],
        request: Any,
        access_token: Optional[str],
        span,
        routing_tools: Optional[list] = None,
        execution_id: Optional[str] = None,
    ) -> Optional[dict]:
        payload = (
            "ROUTING_INTENT_CHECK\n"
            f"root_intent={workflow_def.root_intent}\n"
            f"user_message={user_message or ''}\n"
            f"agents={[a.agent for a in workflow_def.agents]}\n"
            f"Use get_workflow_context tool if you need to inspect detailed workflow context."
        )
        response = await self._call_routing_agent(
            session,
            request,
            access_token,
            span,
            payload,
            workflow_def,
            routing_tools=routing_tools,
            execution_id=execution_id,
        )
        reroute_reason = extract_tag(response, "reroute")
        if reroute_reason:
            return {"reroute": reroute_reason}
        return None

    async def routing_when_check(
        self,
        session: Any,
        agent_def: Any,
        context: dict,
        request: Any,
        access_token: Optional[str],
        span,
        workflow_def: Any,
        extract_user_message_fn: Any = None,
        execution_id: Optional[str] = None,
    ) -> Optional[bool]:
        context_summary = {
            "available_keys": list(context.keys()),
            "use_get_workflow_context_tool": "for detailed context inspection",
        }
        user_msg = ""
        if extract_user_message_fn:
            user_msg = extract_user_message_fn(request) or ""
        payload = (
            "ROUTING_WHEN_CHECK\n"
            f"agent={agent_def.agent}\n"
            f"when={agent_def.when}\n"
            f"root_intent={workflow_def.root_intent}\n"
            f"context_summary={json.dumps(context_summary, ensure_ascii=False)}\n"
            f"user_message={user_msg}\n"
            f"Use get_workflow_context tool to retrieve specific context details if needed."
        )
        response = await self._call_routing_agent(
            session,
            request,
            access_token,
            span,
            payload,
            workflow_def,
            execution_id=execution_id,
        )
        return extract_run_tag(response)

    async def routing_decide_next_agent(
        self,
        session: Any,
        workflow_def: Any,
        agent_def: Any,
        reroute_reason: str,
        context: dict,
        request: Any,
        access_token: Optional[str],
        span,
        execution_id: Optional[str] = None,
    ) -> Optional[str]:
        context_summary = {
            "available_keys": list(context.keys()),
            "use_get_workflow_context_tool": "for detailed context inspection",
        }
        payload = (
            "ROUTING_REROUTE_DECISION\n"
            f"agent={agent_def.agent}\n"
            f"reason={reroute_reason}\n"
            f"available_agents={[a.agent for a in workflow_def.agents]}\n"
            f"context_summary={json.dumps(context_summary, ensure_ascii=False)}\n"
            f"Use get_workflow_context tool to retrieve specific context if needed."
        )
        response = await self._call_routing_agent(
            session,
            request,
            access_token,
            span,
            payload,
            workflow_def,
            execution_id=execution_id,
        )
        return extract_next_agent(response)

    async def resolve_routing_prompt(self, session: Any, workflow_def: Any) -> str:
        default_prompt = (
            "You are the routing_agent. Decide whether a workflow matches and which agent to run next. "
            "Respond only with tags: <run>true/false</run>, <reroute>REASON</reroute>, "
            "and optional <next_agent>agent_name</next_agent>. "
            "Use provided context, root_intent, when expression, and user message."
        )
        try:
            prompt = await self.prompt_service.find_prompt_by_name_or_role(
                session, workflow_def.flow_id
            )
            if prompt:
                custom = await self.prompt_service.get_prompt_content(session, prompt)
                return f"{custom}\n\n{default_prompt}"
        except Exception as exc:  # pragma: no cover
            logger.debug("[RoutingService] Using default routing prompt: %s", exc)
        return default_prompt

    # ------------------------------------------------------------------
    # Internal LLM call (multi-turn with lazy context tool)
    # ------------------------------------------------------------------

    async def _call_routing_agent(
        self,
        session: Any,
        request: Any,
        access_token: Optional[str],
        span,
        payload: str,
        workflow_def: Any,
        routing_tools: Optional[list] = None,
        execution_id: Optional[str] = None,
    ) -> str:
        # Avoid circular imports – these models are lightweight dataclasses
        from app.tgi.models import ChatCompletionRequest, Message, MessageRole
        from app.tgi.protocols.chunk_reader import chunk_reader
        from app.vars import TGI_MODEL_NAME

        routing_prompt = await self.resolve_routing_prompt(session, workflow_def)

        tools_for_routing = list(routing_tools or [])
        lazy_context_tool = create_lazy_context_tool()
        tools_for_routing.append(lazy_context_tool)

        context_provider = (
            LazyContextProvider(self.state_store, execution_id, logger)
            if execution_id
            else None
        )

        routing_request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=routing_prompt),
                Message(role=MessageRole.USER, content=payload),
            ],
            model=request.model or TGI_MODEL_NAME,
            stream=True,
            tools=tools_for_routing,
        )
        text = ""
        stream = self.llm_client.stream_completion(
            routing_request, access_token or "", span
        )

        messages_to_append: list[dict[str, Any]] = []
        tool_call_id_overrides: dict[int, str] = {}
        tool_call_chunks: dict[int, dict] = {}

        async with chunk_reader(stream, enable_tracing=False) as reader:
            async for parsed in reader.as_parsed():
                if parsed.is_done:
                    break
                if parsed.content:
                    text += parsed.content

                if context_provider and parsed.tool_calls:
                    for tool_call in parsed.tool_calls:
                        if (
                            tool_call.get("function", {}).get("name")
                            == "get_workflow_context"
                        ):
                            try:
                                tool_call_index = tool_call.get("index")
                                if tool_call_index is not None:
                                    try:
                                        tool_call_index = int(tool_call_index)
                                    except (TypeError, ValueError):
                                        tool_call_index = None
                                tool_call_id = tool_call.get("id")
                                if not tool_call_id:
                                    tool_call_id = f"call_{len(messages_to_append)}"
                                    if isinstance(tool_call_index, int):
                                        tool_call_id_overrides[tool_call_index] = (
                                            tool_call_id
                                        )
                                tool_input = json.loads(
                                    tool_call.get("function", {}).get("arguments", "{}")
                                )
                                result = await handle_lazy_context_tool(
                                    context_provider, tool_input
                                )
                                logger.debug(
                                    "[RoutingService] Routing agent used lazy context tool: "
                                    "%s",
                                    tool_input.get("operation"),
                                )
                                summary = summarize_tool_result(result)
                                messages_to_append.append(
                                    {
                                        "tool_call_id": tool_call_id,
                                        "tool_name": "get_workflow_context",
                                        "result": summary,
                                        "full_result": result,
                                    }
                                )
                            except Exception as exc:
                                logger.debug(
                                    "[RoutingService] Error processing tool call: %s",
                                    exc,
                                )
            tool_call_chunks = reader.get_accumulated_tool_calls()

        if messages_to_append and text.strip():
            tool_calls_for_message = []
            for index in sorted(tool_call_chunks):
                chunk_data = tool_call_chunks.get(index) or {}
                name = chunk_data.get("name")
                if not name:
                    continue
                args = chunk_data.get("arguments")
                if not isinstance(args, str):
                    try:
                        args = json.dumps(
                            args, ensure_ascii=False, separators=(",", ":")
                        )
                    except Exception:
                        args = "" if args is None else str(args)
                tool_call_id = chunk_data.get("id") or tool_call_id_overrides.get(index)
                if not tool_call_id:
                    tool_call_id = f"call_{index}"
                tool_calls_for_message.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args or ""},
                    }
                )
            routing_request.messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=text,
                    tool_calls=tool_calls_for_message or None,
                )
            )

            for tool_result in messages_to_append:
                routing_request.messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=tool_result["result"],
                        name=tool_result["tool_name"],
                        tool_call_id=tool_result.get("tool_call_id"),
                    )
                )

            text = ""
            stream = self.llm_client.stream_completion(
                routing_request, access_token or "", span
            )
            async with chunk_reader(stream, enable_tracing=False) as reader:
                async for parsed in reader.as_parsed():
                    if parsed.is_done:
                        break
                    if parsed.content:
                        text += parsed.content

        return text.strip()


async def evaluate_condition(
    agent_def,
    context: dict,
    session,
    request,
    access_token,
    span,
    workflow_def,
    routing_service: "RoutingService",
    *,
    execution_id=None,
) -> bool:
    """Evaluate the ``when`` guard on *agent_def*.

    First attempts a simple ``eval()`` of the expression.  Falls back to
    an LLM-based routing check via *routing_service*.
    """
    if not agent_def.when:
        return True
    try:
        evaluated = eval(agent_def.when, {"__builtins__": {}}, {"context": context})
        return bool(evaluated)
    except Exception:
        pass
    decision = await routing_service.routing_when_check(
        session,
        agent_def,
        context,
        request,
        access_token,
        span,
        workflow_def,
        execution_id=execution_id,
    )
    if decision is not None:
        return decision
    try:
        return bool(eval(agent_def.when, {"__builtins__": {}}, {"context": context}))
    except Exception:
        return False
