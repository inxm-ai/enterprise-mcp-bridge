import asyncio
import json
import re
import inspect
from typing import Any, AsyncGenerator, Callable, List, Optional

from opentelemetry import trace

from app.json.schema_validation import validate_schema
from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
)
from app.tgi.services.tools.tool_resolution import ToolCallFormat, ParsedToolCall
from app.tgi.protocols.chunk_reader import chunk_reader
from app.vars import TGI_MODEL_NAME

logger = None  # Will be injected by callers
tracer = trace.get_tracer(__name__)


class ToolChatRunner:
    """
    Shared streaming chat runner with tool-call handling.

    This is extracted from ProxiedTGIService._stream_chat_with_tools so both the
    well-planned orchestrator and workflow engine can share the same, tested
    behavior.
    """

    def __init__(
        self,
        llm_client,
        tool_service,
        tool_resolution,
        message_summarization_service=None,
        logger_obj=None,
        max_iterations: int = 10,
    ) -> None:
        global logger
        logger = logger_obj
        self.llm_client = llm_client
        self.tool_service = tool_service
        self.tool_resolution = tool_resolution
        self.message_summarization_service = message_summarization_service
        self.max_iterations = max_iterations

    async def stream_chat_with_tools(
        self,
        session: Any,
        messages: List[Message],
        available_tools: List[dict],
        chat_request: ChatCompletionRequest,
        access_token: Optional[str],
        parent_span,
        emit_think_messages: bool = True,
        arg_injector: Optional[Callable[[str, dict], dict]] = None,
        tools_for_validation: Optional[List[dict]] = None,
        streaming_tools: Optional[set[str]] = None,
        stop_after_tool_results: Optional[Callable[[List[dict]], bool]] = None,
    ) -> AsyncGenerator[str, None]:
        messages_history = messages.copy()
        if chat_request.messages and chat_request.messages is not messages:
            # Prefer any messages already on the request, but preserve explicit ones too
            messages_history = list(chat_request.messages) + messages_history
        if messages_history and messages_history[0].role != MessageRole.SYSTEM:
            # Ensure a leading system slot so downstream history matches expectations
            messages_history.insert(0, Message(role=MessageRole.SYSTEM, content=None))
        iteration = 0
        outer_span = parent_span
        announced_tool_calls = set()

        class _DummySpan:
            def set_attribute(self, *_args, **_kwargs):
                return None

        if outer_span is None:
            outer_span = _DummySpan()

        outer_span.set_attribute("chat.max_iterations", self.max_iterations)
        outer_span.set_attribute("chat.has_available_tools", bool(available_tools))

        response_format = chat_request.response_format
        response_type = self._response_format_type(response_format)
        enforce_json = response_type in ("json_schema", "json_object")
        emit_tool_events = not enforce_json
        emit_think = emit_think_messages and not enforce_json
        final_content_message = None

        while iteration < self.max_iterations:
            iteration += 1

            llm_request = ChatCompletionRequest(
                messages=messages_history,
                model=chat_request.model or TGI_MODEL_NAME,
                tools=available_tools if available_tools else chat_request.tools,
                tool_choice=chat_request.tool_choice,
                response_format=chat_request.response_format,
                stop=chat_request.stop,
                stream=True,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                top_p=chat_request.top_p,
            )

            llm_stream_generator = self.llm_client.stream_completion(
                llm_request, access_token, outer_span
            )

            content_message = ""

            async with chunk_reader(llm_stream_generator) as reader:
                async for parsed in reader.as_parsed():
                    if parsed.is_done:
                        break

                    if parsed.tool_calls and emit_think:
                        for tc in parsed.tool_calls:
                            tc_index = tc.get("index")
                            tc_func = tc.get("function", {})
                            if tc_index not in announced_tool_calls:
                                content = "<think>I need to call a tool. Preparing tools to call...</think>\n\n"
                                yield f"data: {json.dumps({'choices':[{'delta':{'content':content},'index':tc_index}]})}\n\n"
                                announced_tool_calls.add(tc_index)
                            if "name" in tc_func and tc_func["name"]:
                                status_str = f"<think>{tc_index + 1}. I will run <code>{tc_func['name']}</code></think>\n\n"
                                yield f"data: {json.dumps({'choices':[{'delta':{'content': status_str},'index': tc_index}]})}\n\n"

                    if parsed.content:
                        content_message += parsed.content
                        if not enforce_json:
                            raw_chunk = (
                                parsed.raw
                                if parsed.raw.endswith("\n\n")
                                else f"{parsed.raw}\n\n"
                            )
                            yield raw_chunk

                    if not enforce_json:
                        yield "\n\n"

            with tracer.start_as_current_span("execute_tool_calls") as tool_span:
                tool_call_chunks = reader.get_accumulated_tool_calls()

                parsed_tool_calls, _ = self.tool_resolution.resolve_tool_calls(
                    content_message, tool_call_chunks
                )

                if not parsed_tool_calls and not tool_call_chunks:
                    fallback_call = self._infer_single_tool_call(
                        content_message, available_tools
                    )
                    if fallback_call:
                        parsed_tool_calls = [fallback_call]

                if parsed_tool_calls:
                    tool_span.set_attribute(
                        "tool_calls.resolved_count", len(parsed_tool_calls)
                    )
                    tool_calls_to_execute = []

                    available_tool_names = set()
                    for tool in available_tools or []:
                        if isinstance(tool, dict):
                            name = tool.get("function", {}).get("name") or tool.get(
                                "name"
                            )
                        else:
                            func = getattr(tool, "function", None)
                            name = getattr(func, "name", None) if func else None
                            if not name:
                                name = getattr(tool, "name", None)
                        if name:
                            available_tool_names.add(name)

                    content_message = self._strip_fabricated_tool_results(
                        content_message, tool_calls_to_execute or parsed_tool_calls
                    )

                    for parsed_call in parsed_tool_calls:
                        if parsed_call.name not in available_tool_names:
                            if logger:
                                logger.warning(
                                    "[ToolChatRunner] Skipping tool call '%s' "
                                    "(not in available tools: %s)",
                                    parsed_call.name,
                                    (
                                        ", ".join(sorted(available_tool_names))
                                        if available_tool_names
                                        else "none"
                                    ),
                                )
                            continue

                        # Inject additional arguments if arg_injector is provided
                        call_arguments = parsed_call.arguments
                        if arg_injector:
                            call_arguments = arg_injector(
                                parsed_call.name, call_arguments
                            )

                        tool_call = ToolCall(
                            id=parsed_call.id,
                            function=ToolCallFunction(
                                name=parsed_call.name,
                                arguments=json.dumps(
                                    call_arguments,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            ),
                        )
                        tool_calls_to_execute.append((tool_call, parsed_call.format))

                    messages_history.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=(
                                content_message if content_message.strip() else None
                            ),
                            tool_calls=[tc for tc, _ in tool_calls_to_execute],
                        )
                    )

                    tool_span.set_attribute(
                        "tool_calls.execute_count", len(tool_calls_to_execute)
                    )

                    if emit_think:

                        def tool_arguments(tc):
                            try:
                                args = json.loads(tc.function.arguments)
                                return ",".join(f"{k}='{v}'" for k, v in args.items())
                            except Exception:
                                return ""

                        tools_summary = "\n" + "\n".join(
                            [
                                f"- {tc.function.name}({tool_arguments(tc)})"
                                for tc, _ in tool_calls_to_execute
                            ]
                        )
                        execution_msg = f"<think>Executing the following tools{tools_summary}</think>\n\n"
                        yield f"data: {json.dumps({'choices':[{'delta':{'content': execution_msg},'index':0}]})}\n\n"

                    # Use original tools for validation (includes pre-mapped args in schema)
                    # but available_tools for LLM (has those args removed)
                    validation_tools = tools_for_validation or available_tools
                    progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

                    async def _handle_progress(
                        progress: float,
                        total: Optional[float] = None,
                        message: Optional[str] = None,
                        tool_name: Optional[str] = None,
                    ):
                        try:
                            if logger:
                                progress_label = (
                                    f"{progress}/{total}"
                                    if total is not None
                                    else str(progress)
                                )
                                log_msg = (
                                    f"[ToolProgress] {tool_name or 'tool'} "
                                    f"{progress_label}"
                                )
                                if message:
                                    log_msg += f" - {message}"
                                logger.info(log_msg)
                            progress_queue.put_nowait(
                                {
                                    "type": "progress",
                                    "progress": progress,
                                    "total": total,
                                    "message": message,
                                    "tool": tool_name,
                                }
                            )
                        except Exception:
                            if logger:
                                logger.debug("[ToolChatRunner] Dropped progress update")

                    async def _handle_log(
                        level: str, data: Any, logger_name: Optional[str] = None
                    ):
                        try:
                            if logger:
                                log_fn = getattr(logger, (level or "").lower(), None)
                                if not callable(log_fn):
                                    log_fn = logger.info
                                log_fn(f"[ToolLog][{logger_name or 'MCP'}] {data}")
                            progress_queue.put_nowait(
                                {
                                    "type": "log",
                                    "level": level,
                                    "data": data,
                                    "logger_name": logger_name,
                                }
                            )
                        except Exception:
                            if logger:
                                logger.debug("[ToolChatRunner] Dropped log update")

                    async def _run_tools():
                        stream_set = streaming_tools or set()
                        normal_calls = []
                        streaming_calls = []

                        for tc, fmt in tool_calls_to_execute:
                            if tc.function.name in stream_set:
                                streaming_calls.append((tc, fmt))
                            else:
                                normal_calls.append((tc, fmt))

                        tool_results: list = []
                        raw_results: list = []
                        success = True

                        if normal_calls:
                            execute_fn = getattr(
                                self.tool_service, "execute_tool_calls"
                            )
                            sig = inspect.signature(execute_fn)
                            kwargs = {
                                "available_tools": validation_tools,
                                "return_raw_results": True,
                            }
                            if "progress_callback" in sig.parameters:
                                kwargs["progress_callback"] = _handle_progress
                            if "log_callback" in sig.parameters:
                                kwargs["log_callback"] = _handle_log
                            if (
                                stop_after_tool_results
                                and "summarize_tool_results" in sig.parameters
                            ):
                                kwargs["summarize_tool_results"] = False
                            if (
                                stop_after_tool_results
                                and "build_messages" in sig.parameters
                            ):
                                kwargs["build_messages"] = False
                            normal_results, normal_success, normal_raw = (
                                await execute_fn(
                                    session,
                                    normal_calls,
                                    access_token,
                                    parent_span,
                                    **kwargs,
                                )
                            )
                            tool_results.extend(normal_results)
                            raw_results.extend(normal_raw)
                            success = success and normal_success

                        for streaming_call, fmt in streaming_calls:
                            (
                                streaming_results,
                                streaming_success,
                                streaming_raw,
                            ) = await self._run_streaming_tool(
                                session=session,
                                tool_call=streaming_call,
                                tool_call_format=fmt,
                                access_token=access_token,
                                progress_callback=_handle_progress,
                                log_callback=_handle_log,
                                summarize_tool_results=(
                                    False if stop_after_tool_results else True
                                ),
                                build_messages=(
                                    False if stop_after_tool_results else True
                                ),
                            )
                            tool_results.extend(streaming_results)
                            raw_results.extend(streaming_raw)
                            success = success and streaming_success

                        return tool_results, success, raw_results

                    tool_task = asyncio.create_task(_run_tools())
                    queue_get = asyncio.create_task(progress_queue.get())
                    tool_results: list = []
                    success: bool = True
                    raw_results: list = []

                    try:
                        while True:
                            done, _pending = await asyncio.wait(
                                {tool_task, queue_get},
                                return_when=asyncio.FIRST_COMPLETED,
                            )

                            if queue_get in done:
                                try:
                                    event = queue_get.result()
                                except asyncio.CancelledError:
                                    event = None
                                except Exception:
                                    event = None
                                queue_get = asyncio.create_task(progress_queue.get())
                                if emit_tool_events and isinstance(event, dict) and event:
                                    yield f"data: {json.dumps(event)}\n\n"
                                    yield "\n\n"
                                continue

                            if tool_task in done:
                                try:
                                    queue_get.cancel()
                                    await queue_get
                                except asyncio.CancelledError:
                                    pass
                                except Exception:
                                    pass
                                try:
                                    tool_results, success, raw_results = (
                                        tool_task.result()
                                    )
                                except Exception as exc:  # pragma: no cover - defensive
                                    if logger:
                                        logger.error(
                                            "[ToolChatRunner] Tool execution task failed: %s",
                                            exc,
                                        )
                                    tool_results, success, raw_results = [], False, []

                                # Flush any remaining queued events
                                while not progress_queue.empty():
                                    try:
                                        leftover = progress_queue.get_nowait()
                                        if emit_tool_events:
                                            yield (
                                                f"data: {json.dumps(leftover)}\n\n"
                                            )
                                            yield "\n\n"
                                    except Exception:
                                        break
                                break
                    except asyncio.CancelledError:
                        tool_task.cancel()
                        queue_get.cancel()
                        raise
                    finally:
                        if not tool_task.done():
                            tool_task.cancel()
                            try:
                                await tool_task
                            except asyncio.CancelledError:
                                pass
                            except Exception:
                                pass
                        if not queue_get.done():
                            queue_get.cancel()
                            try:
                                await queue_get
                            except asyncio.CancelledError:
                                pass
                            except Exception:
                                pass

                    tool_span.set_attribute(
                        "tool_calls.executed_count", len(tool_results)
                    )

                    stop_after_tools = False
                    if stop_after_tool_results:
                        try:
                            stop_after_tools = stop_after_tool_results(raw_results)
                        except Exception as exc:
                            if logger:
                                logger.debug(
                                    "[ToolChatRunner] stop_after_tool_results failed: %s",
                                    exc,
                                )

                    if stop_after_tool_results and not stop_after_tools:
                        format_by_id = {tc.id: fmt for tc, fmt in tool_calls_to_execute}
                        format_by_name = {
                            tc.function.name: fmt for tc, fmt in tool_calls_to_execute
                        }
                        tool_results = []
                        for raw_result in raw_results:
                            fmt = format_by_id.get(raw_result.get("tool_call_id"))
                            if not fmt:
                                fmt = format_by_name.get(
                                    raw_result.get("name"), ToolCallFormat.OPENAI_JSON
                                )
                            tool_results.append(
                                await self.tool_service.create_result_message(
                                    fmt, raw_result
                                )
                            )

                    # Emit tool result events for workflow engines to capture
                    # Use raw_results to get unsummarized content for structured data extraction
                    if emit_tool_events:
                        for raw_result in raw_results:
                            tool_name = raw_result.get("name", "unknown")
                            raw_content = raw_result.get("content", "")
                            # Ensure content is a string for the event
                            if not isinstance(raw_content, str):
                                try:
                                    raw_content = json.dumps(
                                        raw_content,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                except Exception:
                                    raw_content = str(raw_content)
                            tool_result_event = {
                                "choices": [
                                    {
                                        "delta": {
                                            "tool_result": {
                                                "name": tool_name,
                                                "content": raw_content,
                                            }
                                        },
                                        "index": 0,
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(tool_result_event)}\n\n"

                    if not stop_after_tools:
                        messages_history.extend(tool_results)
                        self._deduplicate_retry_hints(messages_history)

                    if success:
                        if emit_think and not stop_after_tools:
                            success_msg = "<think>I executed the tools successfully, resuming response generation</think>\n\n"
                            yield f"data: {json.dumps({'choices':[{'delta':{'content': success_msg},'index':0}]})}\n\n"
                        tool_span.set_attribute("tool_calls.success", True)
                    else:
                        if emit_think and not stop_after_tools:
                            failure_report = "<think>The tool call failed. I will try to adjust my approach</think>\n\n"
                            yield f"data: {json.dumps({'choices':[{'delta':{'content': failure_report},'index': 0}]})}\n\n"
                        tool_span.set_attribute("tool_calls.success", False)

                    if stop_after_tools:
                        break

                    should_continue = (
                        bool(tool_results) or len(tool_calls_to_execute) == 1
                    )
                    if should_continue:
                        continue
                    break

                tool_span.set_attribute("tool_calls.resolved_count", 0)
                if content_message and content_message.strip():
                    final_content_message = content_message
                break

        if enforce_json and final_content_message:
            formatted_json = await self._coerce_response_format(
                final_content_message,
                response_format,
                access_token,
                outer_span,
                chat_request.model or TGI_MODEL_NAME,
            )
            if formatted_json:
                yield self._build_content_chunk(formatted_json)

        yield "data: [DONE]\n\n"

    @staticmethod
    def _response_format_type(response_format: Optional[dict]) -> Optional[str]:
        if not isinstance(response_format, dict):
            return None
        return response_format.get("type")

    @staticmethod
    def _response_format_schema(response_format: Optional[dict]) -> Optional[dict]:
        if not isinstance(response_format, dict):
            return None
        if response_format.get("type") != "json_schema":
            return None
        json_schema = response_format.get("json_schema") or {}
        schema = json_schema.get("schema")
        return schema if isinstance(schema, dict) else None

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        start = None
        stack: List[str] = []
        in_string = False
        escape = False

        for idx, char in enumerate(text):
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in "{[":
                if not stack:
                    start = idx
                stack.append(char)
                continue
            if char in "}]":
                if not stack:
                    continue
                opening = stack.pop()
                if (opening == "{" and char != "}") or (
                    opening == "[" and char != "]"
                ):
                    stack.clear()
                    start = None
                    continue
                if not stack and start is not None:
                    return text[start : idx + 1]
        return None

    @classmethod
    def _parse_json_from_text(cls, text: str) -> tuple[Optional[Any], Optional[str]]:
        if not text or not text.strip():
            return None, "empty response"
        payload = text.strip()
        decoder = json.JSONDecoder()
        try:
            parsed, end_idx = decoder.raw_decode(payload)
            remaining = payload[end_idx:].strip()
            if remaining and logger:
                logger.debug(
                    "[ToolChatRunner] Trailing data after JSON response (len=%s)",
                    len(remaining),
                )
            return parsed, None
        except Exception as exc:
            candidate = cls._extract_json_block(payload)
            if candidate:
                try:
                    return json.loads(candidate), None
                except Exception as inner_exc:
                    return None, f"failed to parse extracted JSON: {inner_exc}"
            return None, f"failed to parse JSON: {exc}"

    @staticmethod
    def _format_json_payload(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _build_error_payload(error_message: str, raw_text: str) -> dict:
        snippet = raw_text if len(raw_text) <= 2000 else raw_text[:2000]
        return {"error": error_message, "raw": snippet}

    @staticmethod
    def _build_content_chunk(content: str) -> str:
        return (
            f"data: {json.dumps({'choices':[{'delta':{'content': content},'index':0}]})}\n\n"
        )

    async def _coerce_response_format(
        self,
        content: str,
        response_format: Optional[dict],
        access_token: Optional[str],
        parent_span,
        model_name: str,
    ) -> Optional[str]:
        response_type = self._response_format_type(response_format)
        if response_type not in ("json_schema", "json_object"):
            return content.strip()

        schema = self._response_format_schema(response_format)
        parsed, parse_error = self._parse_json_from_text(content)
        validation_error = None

        if parsed is not None:
            if schema:
                try:
                    validate_schema(schema, parsed)
                except Exception as exc:
                    validation_error = str(exc)
            if not validation_error:
                return self._format_json_payload(parsed)

        repair_attempt = await self._repair_response_format(
            content,
            response_format,
            access_token,
            parent_span,
            model_name,
        )
        if repair_attempt:
            repaired, repair_error = self._parse_json_from_text(repair_attempt)
            if repaired is not None:
                if schema:
                    try:
                        validate_schema(schema, repaired)
                        return self._format_json_payload(repaired)
                    except Exception as exc:
                        validation_error = str(exc)
                else:
                    return self._format_json_payload(repaired)
            parse_error = repair_error or parse_error

        if parsed is not None:
            if logger and validation_error:
                logger.warning(
                    "[ToolChatRunner] JSON output failed schema validation: %s",
                    validation_error,
                )
            return self._format_json_payload(parsed)

        error_msg = validation_error or parse_error or "failed to parse response as JSON"
        if logger:
            logger.warning("[ToolChatRunner] %s", error_msg)
        return self._format_json_payload(self._build_error_payload(error_msg, content))

    async def _repair_response_format(
        self,
        content: str,
        response_format: Optional[dict],
        access_token: Optional[str],
        parent_span,
        model_name: str,
    ) -> Optional[str]:
        response_type = self._response_format_type(response_format)
        if response_type not in ("json_schema", "json_object"):
            return None

        if not hasattr(self.llm_client, "non_stream_completion"):
            return None

        schema = self._response_format_schema(response_format)
        schema_text = (
            json.dumps(schema, ensure_ascii=False) if schema else "a JSON object"
        )
        system_prompt = (
            "You are a JSON repair assistant. Return only JSON that matches the "
            "provided schema. Do not add commentary or markdown."
        )
        user_prompt = (
            "Schema:\n"
            f"{schema_text}\n\n"
            "Source content (convert this into valid JSON that matches the schema):\n"
            f"{content}"
        )

        repair_request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_prompt),
            ],
            model=model_name,
            stream=False,
            temperature=0,
        )
        repair_request.response_format = response_format

        try:
            response = await self.llm_client.non_stream_completion(
                repair_request,
                access_token or "",
                parent_span,
            )
        except Exception as exc:
            if logger:
                logger.warning("[ToolChatRunner] JSON repair failed: %s", exc)
            return None

        if not response or not response.choices:
            return None

        message = response.choices[0].message if response.choices else None
        content = message.content if message else None
        return content.strip() if content else None

    @staticmethod
    def _infer_single_tool_call(
        content: str, available_tools: Optional[List[dict]]
    ) -> Optional[ParsedToolCall]:
        if not available_tools or len(available_tools) != 1:
            return None

        if not content or not content.strip():
            return None

        tool = available_tools[0]
        tool_name = None
        tool_params = None
        if isinstance(tool, dict):
            tool_name = tool.get("function", {}).get("name") or tool.get("name")
            tool_params = tool.get("function", {}).get("parameters")
        else:
            func = getattr(tool, "function", None)
            tool_name = (
                getattr(func, "name", None) if func else getattr(tool, "name", None)
            )
            tool_params = getattr(func, "parameters", None) if func else None
        if not tool_name:
            return None

        content_str = content.strip()
        decoder = json.JSONDecoder()
        try:
            parsed_args, end_idx = decoder.raw_decode(content_str)
        except json.JSONDecodeError:
            return None
        if content_str[end_idx:].strip():
            return None

        if isinstance(parsed_args, dict):
            props = None
            if isinstance(tool_params, dict):
                props = tool_params.get("properties")
            if isinstance(props, dict) and props:
                arg_keys = set(parsed_args.keys())
                prop_keys = set(props.keys())
                if arg_keys and not arg_keys.issubset(prop_keys):
                    return None
        else:
            parsed_args = {"raw": parsed_args}

        return ParsedToolCall(
            id=f"content_{tool_name}",
            index=0,
            name=tool_name,
            arguments=parsed_args,
            format=ToolCallFormat.OPENAI_JSON,
            raw_content=content_str,
        )

    async def _run_streaming_tool(
        self,
        session: Any,
        tool_call: ToolCall,
        tool_call_format: Any,
        access_token: Optional[str],
        progress_callback: Optional[
            Callable[[float, Optional[float], Optional[str], Optional[str]], Any]
        ] = None,
        log_callback: Optional[Callable[[str, Any, Optional[str]], Any]] = None,
        summarize_tool_results: bool = True,
        build_messages: bool = True,
    ) -> tuple[list, bool, list]:
        """
        Execute a single tool via the streaming endpoint and normalize results.
        """
        tool_results: list = []
        raw_results: list = []
        success = True

        if not hasattr(session, "call_tool_streaming"):
            execute_fn = getattr(self.tool_service, "execute_tool_calls")
            fallback_results, fallback_success, fallback_raw = await execute_fn(
                session,
                [(tool_call, tool_call_format)],
                access_token,
                None,
                available_tools=None,
                return_raw_results=True,
                summarize_tool_results=summarize_tool_results,
                build_messages=build_messages,
            )
            return fallback_results, fallback_success, fallback_raw

        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except Exception:
            args = {}

        try:
            maybe_stream = session.call_tool_streaming(
                tool_call.function.name, args, access_token
            )
            stream = (
                await maybe_stream
                if inspect.isawaitable(maybe_stream)
                else maybe_stream
            )
        except Exception as exc:  # pragma: no cover - defensive
            error_payload = {
                "name": tool_call.function.name,
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    {"error": f"Streaming call failed: {exc}"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            if build_messages:
                msg = await self.tool_service.create_result_message(
                    tool_call_format,
                    error_payload,
                    summarize=summarize_tool_results,
                )
                return [msg], False, [error_payload]
            return [], False, [error_payload]

        try:
            async for event in stream:
                if not isinstance(event, dict):
                    continue

                etype = event.get("type")
                if etype == "progress":
                    if progress_callback:
                        try:
                            await progress_callback(
                                event.get("progress"),
                                event.get("total"),
                                event.get("message"),
                                tool_call.function.name,
                            )
                        except Exception:
                            if logger:
                                logger.debug(
                                    "[ToolChatRunner] Dropped streaming progress event"
                                )
                    continue

                if etype == "log":
                    if log_callback:
                        try:
                            await log_callback(
                                event.get("level", "info"),
                                event.get("data"),
                                event.get("logger_name"),
                            )
                        except Exception:
                            if logger:
                                logger.debug(
                                    "[ToolChatRunner] Dropped streaming log event"
                                )
                    continue

                if etype != "result":
                    continue

                result_data = event.get("data")
                raw_results.append(
                    {
                        "name": tool_call.function.name,
                        "tool_call_id": tool_call.id,
                        "content": result_data,
                    }
                )

                try:
                    content_str = (
                        result_data
                        if isinstance(result_data, str)
                        else json.dumps(
                            result_data,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                except Exception:
                    content_str = str(result_data)

                result_payload = {
                    "name": tool_call.function.name,
                    "tool_call_id": tool_call.id,
                    "content": content_str,
                }
                if build_messages:
                    msg = await self.tool_service.create_result_message(
                        tool_call_format,
                        result_payload,
                        summarize=summarize_tool_results,
                    )
                    tool_results.append(msg)
                try:
                    if self.tool_service._result_has_error(result_data):
                        success = False
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover - defensive
            if logger:
                logger.error("[ToolChatRunner] Streaming tool failed: %s", exc)
            success = False

        return tool_results, success, raw_results

    def _deduplicate_retry_hints(self, messages: List[Message]) -> None:
        if not self.message_summarization_service:
            return
        seen_hint = False
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if getattr(msg, "name", None) == "mcp_tool_retry_hint":
                if seen_hint:
                    del messages[idx]
                else:
                    seen_hint = True

    @staticmethod
    def _strip_fabricated_tool_results(content: str, tool_calls: List[Any]) -> str:
        """
        Remove hallucinated inline tool result blocks (e.g. <tool_result>...</tool_result>)
        from the assistant content before it is fed back to the model.
        """
        if not content or not tool_calls:
            return content

        names = []
        for call in tool_calls:
            if hasattr(call, "name"):
                names.append(getattr(call, "name"))
                continue
            if isinstance(call, tuple) and call:
                fn = getattr(getattr(call[0], "function", None), "name", None)
                if fn:
                    names.append(fn)

        if not names:
            return content

        cleaned = content
        for name in set(filter(None, names)):
            escaped = re.escape(name)
            pattern = rf"<{escaped}_result\b[^>]*>.*?</{escaped}_result\s*>"
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
            open_tag = rf"<{escaped}_result\b[^>]*>"
            close_tag = rf"</{escaped}_result\s*>"
            cleaned = re.sub(open_tag, "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(close_tag, "", cleaned, flags=re.IGNORECASE)

        return cleaned
