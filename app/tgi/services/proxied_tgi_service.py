import json
import logging
import time
from typing import List, Optional, Union, AsyncGenerator
from opentelemetry import trace


from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
    Tool,
    ToolCall,
    ToolCallFunction,
)
from app.session import MCPSessionBase
from app.tgi.services.prompt_service import PromptService
from app.tgi.services.tool_service import (
    ToolService,
)
from app.tgi.clients.llm_client import LLMClient
from app.tgi.models.model_formats import BaseModelFormat, get_model_format_for
from app.tgi.protocols.chunk_reader import chunk_reader
from app.vars import TGI_MODEL_NAME
from app.tgi.behaviours.well_planned_orchestrator import WellPlannedOrchestrator

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class ProxiedTGIService:
    """Service that orchestrates chat completions with tool support."""

    def __init__(self, model_format: Optional[BaseModelFormat] = None):
        self.logger = logger
        self.model_format = model_format or get_model_format_for()
        self.prompt_service = PromptService()
        self.llm_client = LLMClient(self.model_format)
        self.tool_service = ToolService(
            model_format=self.model_format, llm_client=self.llm_client
        )
        self.tool_resolution = self.model_format.create_tool_resolution_strategy()
        # pass a lambda that looks up the current _non_stream_chat_with_tools
        # attribute at call time so tests can monkeypatch it after
        # construction.
        self.well_planned_orchestrator = WellPlannedOrchestrator(
            llm_client=self.llm_client,
            prompt_service=self.prompt_service,
            tool_service=self.tool_service,
            non_stream_chat_with_tools_callable=(
                lambda *a, **k: self._non_stream_chat_with_tools(*a, **k)
            ),
            stream_chat_with_tools_callable=(
                lambda *a, **k: self._stream_chat_with_tools(*a, **k)
            ),
            tool_resolution=self.tool_resolution,
            logger_obj=self.logger,
            model_name=TGI_MODEL_NAME,
        )

    async def one_off_chat_completion(
        self,
        session: MCPSessionBase,
        request: ChatCompletionRequest,
        access_token: Optional[str] = None,
        prompt: Optional[str] = None,
        span: trace.Span = None,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """Handle chat completion requests with optional tool support."""
        span.set_attribute("chat.strategy", "one-off")

        # Prepare messages including system prompt if provided
        messages = await self.prompt_service.prepare_messages(
            session, request.messages, prompt, span
        )

        # Get available tools from the session
        available_tools = await self.tool_service.get_all_mcp_tools(session, span)

        if request.stream:
            # Return streaming async generator
            return self._stream_chat_with_tools(
                session, messages, available_tools, request, access_token, span
            )
        else:
            # Return complete response
            return await self._non_stream_chat_with_tools(
                session, messages, available_tools, request, access_token, span
            )

    async def well_planned_chat_completion(
        self,
        session: MCPSessionBase,
        request: ChatCompletionRequest,
        access_token: Optional[str] = None,
        prompt: Optional[str] = None,
        span: trace.Span = None,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        # Delegate the well-planned flow to the orchestrator. This keeps the
        # public API unchanged while moving the implementation elsewhere.
        return await self.well_planned_orchestrator.well_planned_chat_completion(
            session, request, access_token, prompt, span
        )

    async def chat_completion(
        self,
        session: MCPSessionBase,
        request: ChatCompletionRequest,
        access_token: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """Handle chat completion requests with optional tool support."""
        with tracer.start_as_current_span("chat_completion") as span:
            span.set_attribute("chat.model", request.model or "unknown")
            span.set_attribute("chat.streaming", request.stream or False)
            span.set_attribute("chat.messages_count", len(request.messages))
            span.set_attribute("chat.has_tools", bool(request.tools))
            span.set_attribute("chat.tool_choice", bool(request.tool_choice))

            if request.tools:
                span.set_attribute("chat.tools_count", len(request.tools))

            # Route to well-planned orchestrator when tool_choice is explicitly set to
            # "auto". This is the orchestrator trigger used by the well-planned flow.
            use_well_planned = request.tool_choice and request.tool_choice == "auto"

            # Always await the chosen handler so that it performs any
            # necessary synchronous setup (prepare messages, tool discovery)
            # and then returns either a ChatCompletionResponse or an
            # async-generator object for streaming results.
            if use_well_planned:
                res = await self.well_planned_chat_completion(
                    session, request, access_token, prompt, span
                )
            else:
                res = await self.one_off_chat_completion(
                    session, request, access_token, prompt, span
                )

            # The awaited call returns either a response object or an
            # async-generator. Return it directly; do not await the result
            # again (awaiting an async-generator raises TypeError).
            return res

    async def _stream_chat_with_tools(
        self,
        session: MCPSessionBase,
        messages: List[Message],
        available_tools: List[dict],
        chat_request: ChatCompletionRequest,
        access_token: Optional[str],
        parent_span,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion with tool handling, supporting tool calls."""
        messages_history = messages.copy()
        max_iterations = 10
        iteration = 0
        # Ensure we have an outer span object for downstream calls and attributes
        outer_span = parent_span

        class _DummySpan:
            def set_attribute(self, *_args, **_kwargs):
                return None

        if outer_span is None:
            outer_span = _DummySpan()

        outer_span.set_attribute("chat.max_iterations", max_iterations)
        outer_span.set_attribute("chat.has_available_tools", bool(available_tools))

        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(
                f"[ProxiedTGI] Starting stream chat iteration {iteration}/{max_iterations}"
            )

            llm_request = ChatCompletionRequest(
                messages=messages_history,
                model=chat_request.model or TGI_MODEL_NAME,
                tools=available_tools if available_tools else chat_request.tools,
                tool_choice=chat_request.tool_choice,
                stream=True,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                top_p=chat_request.top_p,
            )

            # Log the ChatCompletionRequest, but replace tools with their count
            llm_request_log = llm_request.model_dump(exclude_none=True)
            if "tools" in llm_request_log and llm_request_log["tools"] is not None:
                llm_request_log["tools"] = f"[{len(llm_request_log['tools'])} tools]"
            self.logger.debug(f"ChatCompletionRequest: {llm_request_log}")

            # Tool call chunk accumulator: {id: {index, name, arguments}}
            # Now handled by chunk_reader
            self.logger.debug(
                f"[ProxiedTGI] Creating LLM stream for iteration {iteration}"
            )
            llm_stream_generator = self.llm_client.stream_completion(
                llm_request, access_token, outer_span
            )

            self.logger.debug(
                f"[ProxiedTGI] Starting to consume LLM stream for iteration {iteration}"
            )
            finish_reason = "no reason given"
            content_message = ""

            async with chunk_reader(llm_stream_generator) as reader:
                async for parsed in reader.as_parsed():
                    if parsed.is_done:
                        if finish_reason:
                            self.logger.debug(
                                f"[ProxiedTGI] Finish reason: {finish_reason}"
                            )
                        self.logger.debug(
                            f"[ProxiedTGI] Received [DONE] chunk, breaking stream for iteration {iteration}"
                        )
                        break

                    # Handle chunks with no valid parsed data
                    if not parsed.parsed:
                        # Ensure proper SSE format with \n\n terminator
                        raw_chunk = (
                            parsed.raw
                            if parsed.raw.endswith("\n\n")
                            else f"{parsed.raw}\n\n"
                        )
                        yield raw_chunk
                        continue

                    chunk = parsed.parsed
                    choices = chunk.get("choices", [])
                    if not choices:
                        raw_chunk = (
                            parsed.raw
                            if parsed.raw.endswith("\n\n")
                            else f"{parsed.raw}\n\n"
                        )
                        yield raw_chunk
                        continue

                    choice = choices[0]
                    finish_reason = choice.get("finish_reason")

                    # Handle tool calls - now using chunk_reader's accumulation
                    if parsed.tool_calls:
                        for tc in parsed.tool_calls:
                            tc_index = tc.get("index")
                            tc_func = tc.get("function", {})
                            logger.debug(
                                f"[ProxiedTGI] Tool call chunk for index {tc_index} detected: {str(tc)}"
                            )

                            # Check if this is a new tool call
                            if (
                                tc_index not in parsed.accumulated_tool_calls
                                or not parsed.accumulated_tool_calls[tc_index].get(
                                    "name"
                                )
                            ):
                                content = "<think>I need to call a tool. Preparing tools to call...</think>\n\n"
                                yield f"data: {json.dumps({'choices':[{'delta':{'content':content},'index':tc_index}]})}\n\n"

                            # If we just got the name, announce it
                            if "name" in tc_func and tc_func["name"]:
                                status_str = f"<think>{tc_index + 1}. I will run <code>{tc_func['name']}</code></think>\n\n"
                                yield f"data: {json.dumps({'choices':[{'delta':{'content': status_str},'index': tc_index}]})}\n\n"

                    # Handle content
                    if parsed.content:
                        content_message += parsed.content
                        raw_chunk = (
                            parsed.raw
                            if parsed.raw.endswith("\n\n")
                            else f"{parsed.raw}\n\n"
                        )
                        yield raw_chunk

                    # yield empty chunks to keep the connection alive
                    yield "\n\n"

            with tracer.start_as_current_span("execute_tool_calls") as tool_span:
                # Get accumulated tool calls from the reader
                tool_call_chunks = reader.get_accumulated_tool_calls()

                parsed_tool_calls, resolution_success = (
                    self.tool_resolution.resolve_tool_calls(
                        content_message, tool_call_chunks
                    )
                )

                if parsed_tool_calls:
                    tool_span.set_attribute(
                        "tool_calls.resolved_count", len(parsed_tool_calls)
                    )
                    tool_calls_to_execute = []

                    for parsed_call in parsed_tool_calls:
                        if parsed_call.name not in map(
                            lambda tool: tool["function"]["name"], available_tools
                        ):
                            self.logger.info(
                                f"[ProxiedTGI] Tool {parsed_call.name} is not in the available tools, but that might be the LLM hallucinating"
                            )
                            continue
                        tool_call = ToolCall(
                            id=parsed_call.id,
                            function=ToolCallFunction(
                                name=parsed_call.name,
                                arguments=json.dumps(
                                    parsed_call.arguments,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            ),
                        )
                        tool_calls_to_execute.append((tool_call, parsed_call.format))

                        # Log the format detected for each tool call
                        self.logger.debug(
                            f"[ProxiedTGI] Resolved {parsed_call.format.value} tool call: {parsed_call.name}"
                        )

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

                    self.logger.debug(
                        f"[ProxiedTGI] Executing {len(tool_calls_to_execute)} tool calls"
                    )
                    tool_results, success = await self.tool_service.execute_tool_calls(
                        session,
                        tool_calls_to_execute,
                        access_token,
                        parent_span,
                        available_tools=available_tools,
                    )

                    tool_span.set_attribute(
                        "tool_calls.executed_count", len(tool_results)
                    )

                    messages_history.extend(tool_results)
                    self._deduplicate_retry_hints(messages_history)
                    # TODO summarize the first few messages to avoid hitting token limits

                    # If we didn't fail, this should repeat the cycle
                    if success:
                        yield f"data: {json.dumps({'choices':[{'delta':{'content': '<think>I executed the tools successfully, resuming response generation...</think>'},'index':0}]})}\n\n"
                        self.logger.debug(
                            "[ProxiedTGI] Tool execution successful, continuing to next iteration"
                        )
                        tool_span.set_attribute("tool_calls.success", True)
                    else:
                        failure_report = "<think>The tool call failed. I will try to adjust my approach</think>"
                        self.logger.info(
                            "[ProxiedTGI] Tool execution failed, asking the llm to fix the tool call"
                        )
                        yield f"data: {json.dumps({'choices':[{'delta':{'content': failure_report},'index': 0}]})}\n\n"
                        tool_span.set_attribute("tool_calls.success", False)

                    # Continue to next iteration (both success and error paths)
                    continue

                # No tool calls resolved, break loop, we are probably done
                tool_span.set_attribute("tool_calls.resolved_count", 0)
                self.logger.debug(
                    f"[ProxiedTGI] No tool calls resolved in iteration {iteration}, breaking loop"
                )
                break

        self.logger.debug(
            f"[ProxiedTGI] Stream chat completed after {iteration} iterations"
        )

        yield "data: [DONE]\n\n"

    def _deduplicate_retry_hints(self, messages: List[Message]) -> None:
        """Keep only the most recent retry instruction to prevent payload bloat."""
        seen_hint = False
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if getattr(msg, "name", None) == "mcp_tool_retry_hint":
                if seen_hint:
                    del messages[idx]
                else:
                    seen_hint = True

    async def _non_stream_chat_with_tools(
        self,
        session: MCPSessionBase,
        messages: List[Message],
        available_tools: List[Tool],
        chat_request: ChatCompletionRequest,
        access_token: Optional[str],
        parent_span,
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion with tool handling."""
        from app.tgi.models import ChatCompletionResponse

        messages_history = messages.copy()
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(f"[ProxiedTGI] Non-stream chat iteration {iteration}")

            # Create request for LLM
            llm_request = ChatCompletionRequest(
                messages=messages_history,
                model=chat_request.model or TGI_MODEL_NAME,
                tools=available_tools if available_tools else None,
                tool_choice=chat_request.tool_choice,
                stream=False,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                top_p=chat_request.top_p,
            )

            # Call actual LLM
            response = await self.llm_client.non_stream_completion(
                llm_request, access_token, parent_span
            )

            # Check if response contains tool calls
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.tool_calls
            ):

                # Add assistant message with tool calls to history
                messages_history.append(response.choices[0].message)

                tool_results, success = await self.tool_service.execute_tool_calls(
                    session,
                    response.choices[0].message.tool_calls,
                    access_token,
                    parent_span,
                    available_tools=available_tools,
                )

                # Add tool results to history
                messages_history.extend(tool_results)
            else:
                # No tool calls, return final response
                return response

        # If we reach here, we hit max iterations
        return ChatCompletionResponse(
            id=self.llm_client.create_completion_id(),
            object="chat.completion",
            created=int(time.time()),
            model=chat_request.model or TGI_MODEL_NAME,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content="Maximum conversation iterations reached.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=self.llm_client.create_usage_stats(0, 0),
        )
