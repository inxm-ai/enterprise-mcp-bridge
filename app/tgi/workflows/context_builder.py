"""
Functions for building, selecting and summarising the context payload
that is passed to each workflow agent.

Most functions are pure; the async ``handle_lazy_context_tool`` is a
thin wrapper around ``LazyContextProvider``.
"""

import copy
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.tgi.workflows.dict_utils import get_path_value, set_nested_value
from app.tgi.workflows.lazy_context import LazyContextProvider
from app.tgi.workflows.tag_parser import extract_passthrough_content, strip_tags

logger = logging.getLogger("uvicorn.error")

NESTED_PLACEHOLDER = "<nested too deep>"


@dataclass
class ContextCompressionReport:
    """Report for deciding whether helper calls should use lazy context loading."""

    serialized_size: int
    placeholder_count: int
    compaction_marker_count: int
    lossy: bool
    should_use_lazy_context: bool


# ---------------------------------------------------------------------------
# Agent context building
# ---------------------------------------------------------------------------


def build_agent_context(
    agent_def: Any,
    state: Any,
    *,
    resolve_arg_reference: Any = None,
    get_original_user_prompt: Any = None,
) -> dict:
    """
    Build context payload for an agent based on its ``context`` setting.

    Args:
        agent_def: ``WorkflowAgentDef`` instance.
        state: ``WorkflowExecutionState`` instance.
        resolve_arg_reference: Callable ``(ref, context) -> value`` from
            ``AgentExecutor.resolve_arg_reference``.
        get_original_user_prompt: Callable ``(state) -> Optional[str]`` for
            the ``"user_prompt"`` context mode.
    """
    context_setting = getattr(agent_def, "context", True)
    if context_setting is False:
        return {}
    if context_setting is True or context_setting is None:
        return state.context
    if isinstance(context_setting, str):
        if context_setting == "user_prompt":
            if get_original_user_prompt:
                original_prompt = get_original_user_prompt(state)
            else:
                messages = state.context.get("user_messages") or []
                original_prompt = (
                    str(messages[0]) if messages and messages[0] is not None else None
                )
            if original_prompt is None:
                return {}
            return {"user_prompt": original_prompt}
        return state.context
    if isinstance(context_setting, list):
        return select_context_references(
            context_setting,
            state.context,
            resolve_arg_reference=resolve_arg_reference,
        )
    return state.context


def select_context_references(
    references: list[Any],
    full_context: dict,
    *,
    resolve_arg_reference: Any = None,
) -> dict:
    """
    Extract only the requested references from prior agent contexts.
    """
    selected: dict[str, Any] = {"agents": {}}
    logger.info(
        "[context_builder] Selecting references: %s",
        references,
    )
    for ref in references:
        if not isinstance(ref, str):
            continue
        value = None
        if "." not in ref:
            value = full_context.get(ref)
        else:
            if resolve_arg_reference:
                value = resolve_arg_reference(ref, full_context)
            if value is None:
                value = get_path_value(full_context, ref)
        logger.info(
            "[context_builder] Reference '%s' resolved to: %s",
            ref,
            str(value)[:200] if value else None,
        )
        if value is None:
            logger.warning("[context_builder] Reference '%s' is None", ref)
            continue
        parts = ref.split(".")
        if len(parts) < 2:
            selected[ref] = value
            continue
        agent_name, path_parts = parts[0], parts[1:]
        if agent_name in (full_context.get("agents") or {}):
            agent_ctx = selected["agents"].setdefault(agent_name, {})
            set_nested_value(agent_ctx, path_parts, value)
        else:
            set_nested_value(selected, parts, value)

    if selected.get("agents"):
        return selected
    selected.pop("agents", None)
    return selected if selected else {}


# ---------------------------------------------------------------------------
# Context summarisation
# ---------------------------------------------------------------------------


def create_context_summary(
    context: dict, max_full_size: int = 2000, scoped: bool = False
) -> str:
    """
    Create a compact summary of *context* instead of serialising everything.

    For small or explicitly scoped contexts returns the full JSON;
    for large accumulated contexts returns metadata only.
    """
    if scoped:
        max_full_size = max(max_full_size, 10000)
    context_json = json.dumps(context, ensure_ascii=False, default=str)
    if len(context_json) <= max_full_size:
        return context_json

    summary: dict[str, Any] = {}
    summary["available_keys"] = list(context.keys())

    if "agents" in context and isinstance(context["agents"], dict):
        agents_summary: dict[str, Any] = {}
        for agent_name, agent_data in context["agents"].items():
            if not isinstance(agent_data, dict):
                continue
            agents_summary[agent_name] = {
                "completed": agent_data.get("completed", False),
                "has_content": bool(agent_data.get("content")),
                "has_result": "result" in agent_data,
                "content_length": (
                    len(str(agent_data.get("content", "")))
                    if "content" in agent_data
                    else 0
                ),
            }
            for key in agent_data.keys():
                if key not in [
                    "completed",
                    "content",
                    "result",
                    "pass_through",
                    "awaiting_feedback",
                    "had_feedback",
                    "skipped",
                    "reason",
                    "reroute_reason",
                    "tool_errors",
                    "return_attempts",
                ]:
                    agents_summary[agent_name][f"has_{key}"] = True
        summary["agents"] = agents_summary

    if "user_messages" in context:
        messages = context.get("user_messages", [])
        summary["user_messages_count"] = (
            len(messages) if isinstance(messages, list) else 0
        )

    for key, value in context.items():
        if key in ["agents", "user_messages", "_persist_inner_thinking"]:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value

    return json.dumps(summary, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Data compaction
# ---------------------------------------------------------------------------


def compact_large_structure(
    obj: Any, max_items: int = 3, max_depth: int = 3, current_depth: int = 0
) -> Any:
    """
    Recursively compact large data structures while preserving type/schema.
    """
    if current_depth >= max_depth:
        if isinstance(obj, str):
            if len(obj) > 200:
                return obj[:200] + f"... ({len(obj)} chars total)"
            return obj
        if isinstance(obj, (int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            scalar_keys = [
                key
                for key, value in obj.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            ]
            sampled = scalar_keys[:max_items]
            scalar_sample = {
                key: compact_large_structure(
                    obj[key], max_items, max_depth, current_depth + 1
                )
                for key in sampled
            }
            payload: dict[str, Any] = {
                "_summary": f"Dict with {len(obj)} keys (depth-limited)"
            }
            if scalar_sample:
                payload["_scalar_sample"] = scalar_sample
            omitted = len(obj) - len(sampled)
            if omitted > 0:
                payload["_omitted_keys"] = omitted
            return payload
        if isinstance(obj, list):
            scalar_sample = [
                compact_large_structure(item, max_items, max_depth, current_depth + 1)
                for item in obj
                if isinstance(item, (str, int, float, bool)) or item is None
            ][:max_items]
            payload = {"_summary": f"Array with {len(obj)} items (depth-limited)"}
            if scalar_sample:
                payload["_scalar_sample"] = scalar_sample
            return payload
        return NESTED_PLACEHOLDER

    if isinstance(obj, dict):
        if len(obj) <= max_items:
            return {
                k: compact_large_structure(v, max_items, max_depth, current_depth + 1)
                for k, v in obj.items()
            }
        sample_keys = list(obj.keys())[:max_items]
        return {
            "_summary": f"Dict with {len(obj)} keys",
            "_sample_keys": sample_keys,
            **{
                k: compact_large_structure(
                    obj[k], max_items, max_depth, current_depth + 1
                )
                for k in sample_keys
            },
        }

    if isinstance(obj, list):
        if len(obj) <= max_items:
            return [
                compact_large_structure(item, max_items, max_depth, current_depth + 1)
                for item in obj
            ]
        return {
            "_summary": f"Array with {len(obj)} items",
            "_sample": [
                compact_large_structure(obj[i], max_items, max_depth, current_depth + 1)
                for i in range(min(max_items, len(obj)))
            ],
        }

    if isinstance(obj, str):
        if len(obj) > 200:
            return obj[:200] + f"... ({len(obj)} chars total)"
        return obj

    return obj


def _count_placeholders(obj: Any) -> int:
    if isinstance(obj, str):
        return 1 if obj == NESTED_PLACEHOLDER else 0
    if isinstance(obj, list):
        return sum(_count_placeholders(item) for item in obj)
    if isinstance(obj, dict):
        return sum(_count_placeholders(value) for value in obj.values())
    return 0


def _count_compaction_markers(obj: Any) -> int:
    if isinstance(obj, list):
        return sum(_count_compaction_markers(item) for item in obj)
    if isinstance(obj, dict):
        local = 0
        if "_summary" in obj:
            local += 1
        if obj.get("_compacted") is True:
            local += 1
        return local + sum(_count_compaction_markers(value) for value in obj.values())
    return 0


def analyze_context_compression(
    original_context: Any,
    compacted_context: Any,
    *,
    size_threshold: int = 12000,
    placeholder_threshold: int = 6,
    marker_threshold: int = 4,
) -> ContextCompressionReport:
    """Compute size/loss metrics for deciding lazy-context helper usage."""
    try:
        serialized_size = len(
            json.dumps(original_context, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
        )
    except Exception:
        serialized_size = 0
    placeholder_count = _count_placeholders(compacted_context)
    marker_count = _count_compaction_markers(compacted_context)
    lossy = placeholder_count > 0 or marker_count > 0
    should_use_lazy = (
        serialized_size > size_threshold
        or placeholder_count >= placeholder_threshold
        or marker_count >= marker_threshold
    )
    return ContextCompressionReport(
        serialized_size=serialized_size,
        placeholder_count=placeholder_count,
        compaction_marker_count=marker_count,
        lossy=lossy,
        should_use_lazy_context=should_use_lazy,
    )


def summarize_tool_result(result_json: str, max_size: int = 500) -> str:
    """
    Summarise a large tool result to avoid bloating routing-agent messages.
    """
    if len(result_json) <= max_size:
        return result_json

    try:
        data = json.loads(result_json)
        compacted = compact_large_structure(data, max_items=3, max_depth=3)

        summary: dict[str, Any] = {
            "_compacted": True,
            "_original_size_bytes": len(result_json),
            "_note": (
                "Large result compacted. Use get_workflow_context tool "
                "with specific filters to retrieve full data."
            ),
        }

        if isinstance(compacted, dict):
            summary.update(compacted)
        else:
            summary["data"] = compacted

        result = json.dumps(summary, ensure_ascii=False, default=str)

        if len(result) > max_size * 2:
            return json.dumps(
                {
                    "_compacted": True,
                    "_original_size_bytes": len(result_json),
                    "_summary": "Very large result. Structure too complex to summarize inline.",
                    "_note": "Use get_workflow_context tool to query specific fields.",
                },
                ensure_ascii=False,
                default=str,
            )
        return result

    except Exception as exc:
        logger.debug("[context_builder] Error compacting tool result: %s", exc)
        return json.dumps(
            {
                "_compacted": True,
                "_original_size_bytes": len(result_json),
                "_summary": "Large tool result. Data omitted to reduce message size.",
                "_note": "Refine your query with specific path or field parameters.",
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Pass-through / visible content
# ---------------------------------------------------------------------------


def visible_agent_content(
    agent_def: Any,
    raw_content: str,
    passthrough_history: Optional[list[str]] = None,
) -> str:
    """
    Reduce stored agent content to user-visible data to keep context small.
    """
    if agent_def.pass_through:
        if passthrough_history:
            joined = "\n\n".join(
                [entry.strip() for entry in passthrough_history if entry.strip()]
            )
            return joined.strip()
        extracted = extract_passthrough_content(raw_content or "")
        if extracted:
            return strip_tags(extracted).strip()
        return strip_tags(raw_content).strip()
    return strip_tags(raw_content or "").strip()


# ---------------------------------------------------------------------------
# Lazy-context tool
# ---------------------------------------------------------------------------


def create_lazy_context_tool() -> dict:
    """Create the lazy-context retrieval tool definition."""
    return LazyContextProvider.create_tool_definition()


async def handle_lazy_context_tool(
    context_provider: LazyContextProvider,
    tool_input: dict,
) -> str:
    """
    Handle ``get_workflow_context`` tool calls from agents.
    """
    operation = tool_input.get("operation", "summary")

    try:
        if operation == "summary":
            result = context_provider.get_context_summary()
        elif operation == "get_value":
            path = tool_input.get("path", "")
            max_depth = tool_input.get("max_depth", 2)
            max_size = tool_input.get("max_size_bytes")
            query_result = context_provider.get_context_value(
                path, max_depth=max_depth, max_size_bytes=max_size
            )
            result = query_result.to_dict()
        elif operation == "get_agent":
            agent_name = tool_input.get("agent_name", "")
            fields = tool_input.get("fields")
            query_result = context_provider.get_agent_context(agent_name, fields)
            result = query_result.to_dict()
        elif operation == "get_messages":
            limit = tool_input.get("limit")
            query_result = context_provider.get_user_messages(limit=limit)
            result = query_result.to_dict()
        else:
            result = {"error": f"Unknown operation: {operation}"}

        return json.dumps(result, ensure_ascii=False, default=str)

    except Exception as exc:
        logger.debug("[context_builder] Error handling lazy context tool: %s", exc)
        return json.dumps(
            {
                "success": False,
                "error": str(exc),
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

WORKFLOW_DESCRIPTION_TOOL_NAMES = {"plan"}


def needs_workflow_description(tools: Optional[list]) -> bool:
    """Check if any tool in the list requires a workflow description argument."""
    if not tools:
        return False
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("function", {}).get("name")
        else:
            func = getattr(tool, "function", None)
            name = getattr(func, "name", None) if func else getattr(tool, "name", None)
        if name in WORKFLOW_DESCRIPTION_TOOL_NAMES:
            return True
    return False


def workflow_description_instruction() -> str:
    """Return the instruction string for the plan tool description argument."""
    return (
        "If you call the 'plan' tool to create a new workflow, generate a short "
        "(3-5 words max) identification derived from the user's original request "
        "and include it as the 'description' argument."
    )


def short_request_identifier(text: str, max_words: int = 5) -> Optional[str]:
    """Generate a short identifier from a user request text."""
    if not text:
        return None
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", cleaned)
    if not words:
        return None
    selected = words[:max_words]
    return " ".join(selected)


def ensure_workflow_description_arg(
    tool_name: str, args: dict, user_prompt: Optional[str]
) -> dict:
    """Inject a *description* arg for workflow-description tools if missing."""
    if tool_name not in WORKFLOW_DESCRIPTION_TOOL_NAMES:
        return args
    if isinstance(args, dict) and args.get("description"):
        return args
    if not user_prompt:
        return args
    description = short_request_identifier(user_prompt)
    if not description:
        return args
    updated = dict(args or {})
    updated["description"] = description
    return updated


def tool_name(tool: Any) -> Optional[str]:
    """Extract the name from a tool definition (dict or object)."""
    if isinstance(tool, dict):
        func = (
            tool.get("function", {}) if isinstance(tool.get("function"), dict) else {}
        )
        return func.get("name") or tool.get("name")
    func = getattr(tool, "function", None)
    if func and hasattr(func, "name"):
        return getattr(func, "name")
    return getattr(tool, "name", None)


def normalize_tools(tools: Optional[list]) -> list:
    """Normalise tool definitions to OpenAI-style dicts.

    Downstream comparisons and tool execution work regardless of upstream
    shape after normalisation.
    """
    normalized: list = []
    for tool in tools or []:
        if isinstance(tool, dict):
            normalized.append(tool)
            continue
        func = getattr(tool, "function", None)
        normalized.append(
            {
                "type": getattr(tool, "type", "function"),
                "function": {
                    "name": getattr(func, "name", None) if func else None,
                    "description": (
                        getattr(func, "description", None)
                        if func
                        else getattr(tool, "description", None)
                    ),
                    "parameters": (
                        getattr(func, "parameters", None)
                        if func
                        else getattr(tool, "inputSchema", None)
                    ),
                },
            }
        )
    return normalized


def get_tool_names_from_config(agent_def: Any) -> Optional[set]:
    """Extract tool names from an agent definition.

    Returns:
        Set of tool names, empty set for disabled tools, or ``None`` for all tools.
    """
    tools = getattr(agent_def, "tools", None)
    if tools is None:
        return None
    if isinstance(tools, list) and len(tools) == 0:
        return set()

    names = set()
    for tool_def in tools or []:
        if isinstance(tool_def, str):
            names.add(tool_def)
        elif isinstance(tool_def, dict) and len(tool_def) == 1:
            names.add(next(iter(tool_def.keys())))

    return names if names else None


async def resolve_tools(
    session: Any,
    agent_def: Any,
    tool_service: Any,
    agent_executor: Any,
) -> tuple[Optional[list], Optional[list]]:
    """Resolve tools for an agent, returning both modified and original versions.

    Returns:
        Tuple of (modified_tools, original_tools):
        - modified_tools: Schema with pre-mapped args removed for LLM
        - original_tools: Original schema for tool_argument_fixer validation
    """
    if not tool_service or not hasattr(session, "list_tools"):
        return None, None
    try:
        all_tools = await tool_service.get_all_mcp_tools(session)
    except Exception:
        return None, None

    # Drop helper tools so agents see only user-available MCP tools
    all_tools = [t for t in all_tools or [] if tool_name(t) != "describe_tool"]

    names = get_tool_names_from_config(agent_def)
    if names is None:
        return all_tools, all_tools
    if len(names) == 0:
        return [], []

    filtered = [
        tool
        for tool in all_tools
        if (
            tool.get("function", {}).get("name")
            if isinstance(tool, dict)
            else getattr(getattr(tool, "function", None), "name", None)
        )
        in names
    ]
    if not filtered:
        logger.warning(
            "[context_builder] No matching tools for agent '%s'; using all tools",
            agent_def.agent,
        )
        return all_tools, all_tools

    # Apply tool schema modifications for pre-mapped arguments
    tool_configs = agent_executor.get_tool_configs_for_agent(agent_def)
    config_map = {tc.name: tc for tc in tool_configs if tc.args_mapping}

    if not config_map:
        return filtered, filtered

    original_tools = [copy.deepcopy(t) for t in filtered]
    modified_tools = []
    for tool in filtered:
        tn = tool_name(tool)
        if tn and tn in config_map:
            modified = agent_executor.modify_tool_for_agent(tool, config_map[tn])
            modified_tools.append(modified)
        else:
            modified_tools.append(tool)

    return modified_tools, original_tools


def extract_user_message(request: Any) -> Optional[str]:
    """Return the content of the last user message in *request*, or ``None``."""
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content
    return None


def build_user_content(
    agent_name: str,
    root_intent: str,
    user_prompt: str,
    context_summary: str,
    state_context: dict[str, Any],
    *,
    is_loop: bool = False,
) -> str:
    """
    Build the user-content string sent to the agent LLM.

    Pure function — no side-effects.
    """
    history_messages = state_context.get("user_messages", [])
    history_text = "\n".join(history_messages[-10:]) if history_messages else "<none>"
    lines = [
        f"<agent:{agent_name}> Goal: {root_intent}",
        f"Latest user input: {user_prompt}",
        f"User message history:\n{history_text}",
    ]
    if is_loop:
        assistant_messages = state_context.get("assistant_messages", [])
        assistant_text = (
            "\n".join(assistant_messages[-10:]) if assistant_messages else "<none>"
        )
        lines.append(f"Assistant message history:\n{assistant_text}")
    lines.append(f"Context summary: {context_summary}")
    lines.append("Note: Full context available via tools if you have them configured.")
    return "\n".join(lines)
