"""Shared helper for context-aware LLM calls with optional lazy context retrieval."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.protocols.chunk_reader import chunk_reader
from app.tgi.workflows.context_builder import (
    create_lazy_context_tool,
    handle_lazy_context_tool,
    summarize_tool_result,
)
from app.tgi.workflows.lazy_context import LazyContextProvider
from app.vars import TGI_MODEL_NAME

logger = logging.getLogger("uvicorn.error")


@dataclass
class ContextAwareLLMResult:
    """Result metadata for a context-aware helper execution."""

    text: str
    used_tools: bool = False
    turns: int = 1
    stopped_by_max_turns: bool = False


async def run_context_aware_llm_helper(
    *,
    llm_client: Any,
    base_request: Any,
    access_token: Optional[str],
    span: Any,
    system_prompt: str,
    user_payload: str,
    state_store: Any = None,
    execution_id: Optional[str] = None,
    additional_tools: Optional[list] = None,
    max_turns: int = 2,
) -> ContextAwareLLMResult:
    """Run an LLM helper call with optional lazy workflow context retrieval.

    The helper performs up to ``max_turns`` model calls.
    When ``execution_id`` and ``state_store`` are provided, the helper exposes
    ``get_workflow_context`` and resolves it locally through ``LazyContextProvider``.
    """

    tools_for_call = list(additional_tools or [])
    context_provider = None
    if state_store is not None and execution_id:
        tools_for_call.append(create_lazy_context_tool())
        context_provider = LazyContextProvider(state_store, execution_id, logger)

    messages = [
        Message(role=MessageRole.SYSTEM, content=system_prompt),
        Message(role=MessageRole.USER, content=user_payload),
    ]

    used_tools = False
    turns = 0
    stopped_by_max_turns = False
    final_text = ""

    while turns < max_turns:
        turns += 1
        llm_request = ChatCompletionRequest(
            model=(getattr(base_request, "model", None) or TGI_MODEL_NAME),
            messages=messages,
            stream=True,
            tools=tools_for_call if tools_for_call else None,
        )

        text = ""
        tool_call_chunks: dict[int, dict[str, Any]] = {}

        stream = llm_client.stream_completion(llm_request, access_token or "", span)
        async with chunk_reader(stream, enable_tracing=False) as reader:
            async for parsed in reader.as_parsed():
                if parsed.is_done:
                    break
                if parsed.content:
                    text += parsed.content
            tool_call_chunks = reader.get_accumulated_tool_calls() or {}

        final_text = text.strip()
        if not tool_call_chunks:
            return ContextAwareLLMResult(
                text=final_text,
                used_tools=used_tools,
                turns=turns,
                stopped_by_max_turns=False,
            )

        if not context_provider:
            logger.debug(
                "[contextual_llm_helper] Tool calls emitted but lazy context is unavailable. "
                "Returning first-pass text."
            )
            return ContextAwareLLMResult(
                text=final_text,
                used_tools=used_tools,
                turns=turns,
                stopped_by_max_turns=False,
            )

        assistant_tool_calls: list[dict[str, Any]] = []
        tool_messages: list[Message] = []

        for index in sorted(tool_call_chunks):
            chunk_data = tool_call_chunks.get(index) or {}
            tool_name = chunk_data.get("name")
            if not tool_name:
                continue

            args = chunk_data.get("arguments")
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    args = "" if args is None else str(args)
            args = args or "{}"

            tool_call_id = chunk_data.get("id") or f"call_{turns}_{index}"
            assistant_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": args},
                }
            )

            if tool_name != "get_workflow_context":
                logger.debug(
                    "[contextual_llm_helper] Ignoring unsupported helper tool call: %s",
                    tool_name,
                )
                continue

            try:
                tool_input = json.loads(args)
            except Exception:
                tool_input = {"operation": "summary"}

            raw_result = await handle_lazy_context_tool(context_provider, tool_input)
            summarized_result = summarize_tool_result(raw_result)
            tool_messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=summarized_result,
                    name="get_workflow_context",
                    tool_call_id=tool_call_id,
                )
            )
            used_tools = True

        if not tool_messages:
            return ContextAwareLLMResult(
                text=final_text,
                used_tools=used_tools,
                turns=turns,
                stopped_by_max_turns=False,
            )

        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=text if text else None,
                tool_calls=assistant_tool_calls or None,
            )
        )
        messages.extend(tool_messages)

    stopped_by_max_turns = True
    return ContextAwareLLMResult(
        text=final_text,
        used_tools=used_tools,
        turns=turns,
        stopped_by_max_turns=stopped_by_max_turns,
    )
