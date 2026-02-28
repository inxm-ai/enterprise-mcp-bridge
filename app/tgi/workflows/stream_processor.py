"""
Stream processing for workflow agent execution.

Handles the main streaming loop: passthrough tag parsing, tool result
capture, progress events, and heartbeat emission.  All functions are
pure (no ``self``) and receive callbacks for side-effects.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

from app.tgi.workflows import context_builder, error_analysis, tag_parser
from app.tgi.workflows.arg_injector import ToolResultCapture
from app.tgi.workflows.models import WorkflowAgentDef, WorkflowExecutionState

logger = logging.getLogger("uvicorn.error")

START_TAG = "<passthrough>"
END_TAG = "</passthrough>"


# ---------------------------------------------------------------------------
# Accumulated results from a single agent stream
# ---------------------------------------------------------------------------
@dataclass
class StreamResult:
    """Mutable accumulator populated by *process_agent_stream*."""

    content_text: str = ""
    tool_errors: list[dict[str, Any]] = field(default_factory=list)
    tool_outcomes: list[dict[str, str]] = field(default_factory=list)
    passthrough_history: list[str] = field(default_factory=list)
    emitted_passthrough: bool = False


# ---------------------------------------------------------------------------
# Main stream processing
# ---------------------------------------------------------------------------
async def process_agent_stream(
    *,
    runner_stream: Any,
    chunk_reader_fn: Callable,
    agent_def: WorkflowAgentDef,
    state: WorkflowExecutionState,
    agent_context: dict[str, Any],
    result_capture: Optional[ToolResultCapture],
    result: StreamResult,
    record_event_fn: Callable[..., str],
    run_progress_handler_fn: Callable[..., Coroutine[Any, Any, list[str]]],
) -> AsyncGenerator[str, None]:
    """
    Consume *runner_stream*, yield SSE events, and populate *result* in place.

    ``run_progress_handler_fn(payload, passthrough_history=...)`` is called for
    each progress event and must return ``list[str]`` of SSE events.
    """
    passthrough_buffer = ""
    use_passthrough_tags = agent_def.pass_through_guideline is not None
    passthrough_pending = ""
    passthrough_emitted_len = 0
    inside_passthrough = False
    progress_task: Optional[asyncio.Task[list[str]]] = None

    # -- helper closures ---------------------------------------------------
    async def _flush_progress(wait: bool = False) -> list[str]:
        nonlocal progress_task
        if not progress_task:
            return []
        if not wait and not progress_task.done():
            return []
        try:
            return await progress_task
        except asyncio.CancelledError:
            return []
        except Exception as exc:  # pragma: no cover
            logger.debug("[stream_processor] Progress task failed: %s", exc)
            return []
        finally:
            progress_task = None

    async def _cancel_progress() -> None:
        nonlocal progress_task
        if progress_task and not progress_task.done():
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover
                logger.debug("[stream_processor] Progress cancel error: %s", exc)
            progress_task = None

    def _emit_passthrough_delta(
        delta: str, add_to_history: bool = False
    ) -> Optional[str]:
        if not delta:
            return None
        event = record_event_fn(state, delta)
        if add_to_history:
            result.passthrough_history.append(delta.strip())
        return event

    # -- main loop ---------------------------------------------------------
    try:
        async with chunk_reader_fn(runner_stream, enable_tracing=False) as reader:
            async for parsed in reader.as_parsed():
                if parsed.is_done:
                    break

                parsed_payload = (
                    parsed.parsed if isinstance(parsed.parsed, dict) else None
                )

                # --- progress events ---
                if parsed_payload and parsed_payload.get("type") == "progress":
                    if progress_task:
                        if not progress_task.done():
                            await _cancel_progress()
                        progress_task = None
                    progress_task = asyncio.create_task(
                        run_progress_handler_fn(
                            parsed_payload,
                            passthrough_history=list(result.passthrough_history),
                        )
                    )
                    for event in await _flush_progress(wait=False):
                        yield event
                    continue

                # --- tool results ---
                if parsed.tool_result:
                    _process_tool_result(
                        parsed.tool_result,
                        agent_context,
                        state.context,
                        result,
                        result_capture,
                    )

                # --- content / passthrough ---
                if parsed.content:
                    result.content_text += parsed.content
                    if agent_def.pass_through:
                        if use_passthrough_tags:
                            passthrough_buffer += parsed.content
                            while passthrough_buffer:
                                if inside_passthrough:
                                    end_idx = passthrough_buffer.find(END_TAG)
                                    if end_idx == -1:
                                        safe_len = max(
                                            0,
                                            len(passthrough_buffer)
                                            - (len(END_TAG) - 1),
                                        )
                                        if safe_len:
                                            passthrough_pending += passthrough_buffer[
                                                :safe_len
                                            ]
                                            passthrough_buffer = passthrough_buffer[
                                                safe_len:
                                            ]
                                            delta = passthrough_pending[
                                                passthrough_emitted_len:
                                            ]
                                            event = _emit_passthrough_delta(delta)
                                            if event:
                                                result.emitted_passthrough = True
                                                yield event
                                            passthrough_emitted_len = len(
                                                passthrough_pending
                                            )
                                        break

                                    passthrough_pending += passthrough_buffer[:end_idx]
                                    passthrough_buffer = passthrough_buffer[
                                        end_idx + len(END_TAG) :
                                    ]
                                    delta = passthrough_pending[
                                        passthrough_emitted_len:
                                    ]
                                    event = _emit_passthrough_delta(delta)
                                    if event:
                                        result.emitted_passthrough = True
                                        yield event
                                    if passthrough_pending.strip():
                                        result.passthrough_history.append(
                                            passthrough_pending.strip()
                                        )
                                    passthrough_pending = ""
                                    passthrough_emitted_len = 0
                                    inside_passthrough = False
                                    continue

                                start_idx = passthrough_buffer.find(START_TAG)
                                if start_idx == -1:
                                    if len(passthrough_buffer) > (len(START_TAG) - 1):
                                        passthrough_buffer = passthrough_buffer[
                                            -(len(START_TAG) - 1) :
                                        ]
                                    break
                                passthrough_buffer = passthrough_buffer[
                                    start_idx + len(START_TAG) :
                                ]
                                inside_passthrough = True
                                passthrough_pending = "\n"
                                passthrough_emitted_len = 0
                        else:
                            yield record_event_fn(state, parsed.content)
                            result.emitted_passthrough = True

                for event in await _flush_progress(wait=False):
                    yield event

        # Flush any outstanding progress task at end of stream
        for event in await _flush_progress(wait=True):
            yield event

        # Fallback: ensure pass-through content is emitted at least once
        if agent_def.pass_through and not result.emitted_passthrough:
            fallback_visible = tag_parser.extract_passthrough_content(
                result.content_text
            )
            if fallback_visible:
                fallback_visible = tag_parser.strip_tags(fallback_visible)
            else:
                fallback_visible = tag_parser.strip_tags(result.content_text)
            if fallback_visible:
                yield record_event_fn(state, fallback_visible)
                result.emitted_passthrough = True

    except Exception:
        # Re-raise; the caller handles ArgResolutionError etc.
        raise
    finally:
        await _cancel_progress()


# ---------------------------------------------------------------------------
# Tool result processing
# ---------------------------------------------------------------------------
def _process_tool_result(
    tool_result: dict[str, Any],
    agent_context: dict[str, Any],
    state_context: dict[str, Any],
    result: StreamResult,
    result_capture: Optional[ToolResultCapture],
) -> None:
    """Extract returns, detect errors, compact + store tool result."""
    tool_result_content = tool_result.get("content", "")
    tool_result_name = tool_result.get("name")

    tool_error = error_analysis.tool_result_has_error(tool_result_content)
    if tool_result_name:
        result.tool_outcomes.append(
            {
                "name": tool_result_name,
                "status": "error" if tool_error else "success",
            }
        )
    if tool_error:
        result.tool_errors.append(
            {"name": tool_result_name, "content": tool_result_content}
        )
        logger.warning(
            "[stream_processor] Tool error detected for %s: %s",
            tool_result_name,
            (
                tool_result_content[:200]
                if len(tool_result_content) > 200
                else tool_result_content
            ),
        )

    if result_capture:
        result_capture.capture(
            tool_result_content,
            state_context,
            tool_name=tool_result_name,
        )

    try:
        full_result = json.loads(tool_result_content)
        compacted_json = context_builder.summarize_tool_result(
            tool_result_content, max_size=2000
        )
        agent_context["result"] = json.loads(compacted_json)
        full_results = agent_context.setdefault("_full_tool_results", {})
        if tool_result_name:
            full_results[tool_result_name] = full_result
    except Exception:
        agent_context["result"] = tool_result_content


# ---------------------------------------------------------------------------
# Tool-progress handling
# ---------------------------------------------------------------------------
async def handle_tool_progress(
    *,
    state: WorkflowExecutionState,
    agent_def: WorkflowAgentDef,
    progress_payload: dict[str, Any],
    user_message: str,
    model_name: str,
    passthrough_history: Optional[list[str]] = None,
    llm_stream_fn: Callable[..., Any],
    chunk_reader_fn: Callable,
    record_event_fn: Callable[..., str],
    format_chunk_fn: Callable[..., str],
) -> AsyncGenerator[str, None]:
    """
    Surface tool progress to the user.

    When pass-through is enabled, asks the LLM for a brief update.
    Otherwise emits a heartbeat chunk with progress metadata.
    """
    progress_value = progress_payload.get("progress")
    total_value = progress_payload.get("total")
    progress_message = progress_payload.get("message")
    tool_name = progress_payload.get("tool")

    metadata: dict[str, Any] = {
        "type": "tool_progress",
        "progress": progress_value,
        "total": total_value,
        "message": progress_message,
        "tool": tool_name,
    }

    if agent_def.pass_through:
        try:
            system_prompt = (
                "You are providing brief user-visible updates while a tool runs.\n"
                "Decide whether the user needs to see this progress. "
                "Respond with a short, reassuring update only when helpful. "
                "If no update is needed, respond exactly with <no_update/>.\n"
                "Be concise and avoid repeating prior details."
            )
            if agent_def.pass_through_guideline:
                system_prompt += (
                    "\nPass-through hint: "
                    f"{agent_def.pass_through_guideline}\n"
                    "Wrap any user-visible text in <passthrough></passthrough> tags."
                )

            progress_label = (
                "Progress: " + str(progress_value)
                if progress_value is not None
                else "Progress update"
            )
            progress_line = progress_label + (
                f"/{total_value}" if total_value is not None else ""
            )
            if progress_message:
                progress_line += f" - {progress_message}"
            if tool_name:
                progress_line = f"[{tool_name}] {progress_line}"

            user_content = f"{progress_line}\nLatest user message: {user_message}"
            if passthrough_history and agent_def.pass_through_guideline:
                history_text = "\n".join(f"- {msg}" for msg in passthrough_history)
                user_content = (
                    f"{progress_line}\n"
                    f"Latest user message: {user_message}\n\n"
                    f"Previous messages you've shown to the user (do not repeat):\n"
                    f"{history_text}"
                )

            from app.tgi.models import ChatCompletionRequest, Message, MessageRole

            progress_request = ChatCompletionRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content=system_prompt),
                    Message(role=MessageRole.USER, content=user_content),
                ],
                model=model_name,
                stream=True,
            )

            stream = llm_stream_fn(progress_request)

            reply_text = ""
            async with chunk_reader_fn(stream, enable_tracing=False) as reader:
                async for progress_parsed in reader.as_parsed():
                    if progress_parsed.is_done:
                        break
                    if progress_parsed.content:
                        reply_text += progress_parsed.content

            cleaned = reply_text.strip()
            if "<no_update" in cleaned.lower():
                cleaned = ""

            visible = cleaned
            if agent_def.pass_through_guideline:
                passthrough_only = tag_parser.extract_passthrough_content(reply_text)
                if passthrough_only:
                    visible = tag_parser.strip_tags(passthrough_only)
            else:
                visible = tag_parser.strip_tags(visible).strip()

            if visible:
                if not visible.endswith("\n"):
                    visible += "\n"
                yield record_event_fn(
                    state,
                    visible,
                    status="in_progress",
                    metadata=metadata,
                )
                return
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "[stream_processor] Progress update failed, sending heartbeat: %s",
                exc,
            )

    yield format_chunk_fn(
        state=state,
        content="",
        status="in_progress",
        metadata=metadata,
    )


async def run_progress_handler(
    *,
    state: WorkflowExecutionState,
    agent_def: WorkflowAgentDef,
    progress_payload: dict[str, Any],
    user_message: str,
    model_name: str,
    passthrough_history: Optional[list[str]] = None,
    llm_stream_fn: Callable[..., Any],
    chunk_reader_fn: Callable,
    record_event_fn: Callable[..., str],
    format_chunk_fn: Callable[..., str],
) -> list[str]:
    """
    Consume *handle_tool_progress* into a list for use as a cancellable task.
    """
    events: list[str] = []
    gen = handle_tool_progress(
        state=state,
        agent_def=agent_def,
        progress_payload=progress_payload,
        user_message=user_message,
        model_name=model_name,
        passthrough_history=passthrough_history,
        llm_stream_fn=llm_stream_fn,
        chunk_reader_fn=chunk_reader_fn,
        record_event_fn=record_event_fn,
        format_chunk_fn=format_chunk_fn,
    )
    try:
        async with contextlib.aclosing(gen):
            async for event in gen:
                events.append(event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover
        logger.debug("[stream_processor] Progress handler failed: %s", exc)
    return events
