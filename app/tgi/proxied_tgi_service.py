import json
import logging
from typing import List, Optional, Union, AsyncGenerator
from opentelemetry import trace

from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    MessageRole,
    Tool,
    ToolCall,
    ToolCallFunction,
)
from app.session import MCPSessionBase
from app.tgi.prompt_service import PromptService
from app.tgi.tool_service import (
    ToolService,
)
from app.tgi.llm_client import LLMClient
from app.tgi.model_formats import BaseModelFormat, get_model_format_for

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

            if request.tools:
                span.set_attribute("chat.tools_count", len(request.tools))

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
        with tracer.start_as_current_span("stream_chat_with_tools") as outer_span:
            outer_span.set_attribute("chat.max_iterations", max_iterations)
            outer_span.set_attribute("chat.has_available_tools", bool(available_tools))

            while iteration < max_iterations:
                iteration += 1
                self.logger.debug(
                    f"[ProxiedTGI] Starting stream chat iteration {iteration}/{max_iterations}"
                )

                llm_request = ChatCompletionRequest(
                    messages=messages_history,
                    model=chat_request.model,
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
                    llm_request_log["tools"] = (
                        f"[{len(llm_request_log['tools'])} tools]"
                    )
                self.logger.debug(f"ChatCompletionRequest: {llm_request_log}")

                # Tool call chunk accumulator: {id: {index, name, arguments}}
                tool_call_chunks = {}
                tool_call_ready = set()

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
                async for raw_chunk in llm_stream_generator:
                    # Remove chunk span to avoid context issues
                    with tracer.start_as_current_span(
                        "process_stream_chunk"
                    ) as chunk_span:
                        # Parse chunk JSON
                        try:
                            if raw_chunk.startswith("data: "):
                                chunk_data = raw_chunk[len("data: ") :].strip()
                                if chunk_data == "[DONE]":
                                    if finish_reason:
                                        self.logger.debug(
                                            f"[ProxiedTGI] Finish reason: {finish_reason}"
                                        )
                                    chunk_span.set_attribute("stream_chat.done", True)
                                    # yield raw_chunk
                                    self.logger.debug(
                                        f"[ProxiedTGI] Received [DONE] chunk, breaking stream for iteration {iteration}"
                                    )
                                    break
                                chunk_span.set_attribute("stream_chat.done", False)
                                chunk = json.loads(chunk_data)
                            else:
                                yield raw_chunk
                                continue
                        except Exception as e:
                            self.logger.error(
                                f"[ProxiedTGI] Error parsing streamed chunk: {e}"
                            )
                            yield raw_chunk
                            continue

                        # OpenAI compatible chunk: check for choices[0].delta.tool_calls or content
                        choices = chunk.get("choices", [])
                        if not choices:
                            yield raw_chunk
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")
                        # We need to build this over multiple chunks, jieij...
                        if "tool_calls" in delta and delta["tool_calls"]:
                            chunk_span.set_attribute(
                                "stream_chat.tool_call_chunk", True
                            )
                            for tc in delta["tool_calls"]:
                                tc_id = tc.get("id")
                                tc_index = tc.get("index")
                                tc_func = tc.get("function", {})
                                logger.debug(
                                    f"[ProxiedTGI] Tool call chunk for index {tc_index} detected: {str(tc)}"
                                )
                                chunk_span.set_attribute(
                                    "tool_call.index", tc_index or "unknown"
                                )
                                if tc_index not in tool_call_chunks:
                                    tool_call_chunks[tc_index] = {
                                        "index": tc_index,
                                        "name": "",
                                        "arguments": "",
                                    }
                                    content = "<think>I need to call a tool. Preparing tools to call...</think>\n\n"
                                    # Be nice, inform them we are still awake
                                    yield f"data: {json.dumps({'choices':[{'delta':{'content':content},'index':tc_index}]})}\n\n"
                                if "id" in tc and tc["id"]:
                                    tool_call_chunks[tc_index]["id"] = tc["id"]
                                if "name" in tc_func and tc_func["name"]:
                                    tool_call_chunks[tc_index]["name"] = tc_func["name"]
                                    status_str = f"<think>{tc_index + 1}. I will run <code>{tc_func['name']}</code></think>\n\n"
                                    # still nice, yielding more spam
                                    yield f"data: {json.dumps({'choices':[{'delta':{'content': status_str},'index': tc_index}]})}\n\n"
                                if "arguments" in tc_func and tc_func["arguments"]:
                                    tool_call_chunks[tc_index]["arguments"] += tc_func[
                                        "arguments"
                                    ]

                                chunk_span.set_attribute(
                                    "tool_call.id", tc_id or "unknown"
                                )
                                # If both name and arguments are present, mark as ready
                                if (
                                    tool_call_chunks[tc_index]["name"]
                                    and tool_call_chunks[tc_index]["arguments"]
                                    and tc_index not in tool_call_ready
                                ):
                                    chunk_span.set_attribute("tool_call.ready", True)
                                    tool_call_ready.add(tc_index)
                                else:
                                    chunk_span.set_attribute("tool_call.ready", False)
                        if "content" in delta and delta["content"]:
                            # If content is present, just yield
                            chunk_span.set_attribute("stream_chat.content_chunk", True)
                            content_message += delta["content"]
                            yield raw_chunk

                with tracer.start_as_current_span("execute_tool_calls") as tool_span:
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
                            tool_calls_to_execute.append(
                                (tool_call, parsed_call.format)
                            )

                            # Log the format detected for each tool call
                            self.logger.debug(
                                f"[ProxiedTGI] Resolved {parsed_call.format.value} tool call: {parsed_call.name}"
                            )

                        # Add content message
                        if content_message.strip():
                            messages_history.append(
                                Message(
                                    role=MessageRole.ASSISTANT,
                                    content=content_message,
                                    tool_calls=[tc for tc, _ in tool_calls_to_execute],
                                )
                            )

                        tool_span.set_attribute(
                            "tool_calls.execute_count", len(tool_calls_to_execute)
                        )

                        self.logger.debug(
                            f"[ProxiedTGI] Executing {len(tool_calls_to_execute)} tool calls"
                        )
                        tool_results, success = (
                            await self.tool_service.execute_tool_calls(
                                session,
                                tool_calls_to_execute,
                                access_token,
                                parent_span,
                                available_tools=available_tools,
                            )
                        )

                        tool_span.set_attribute(
                            "tool_calls.executed_count", len(tool_results)
                        )

                        messages_history.extend(tool_results)
                        self._deduplicate_retry_hints(messages_history)

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
                    else:
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
        from app.tgi.models import ChatCompletionResponse, Choice
        import time

        messages_history = messages.copy()
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(f"[ProxiedTGI] Non-stream chat iteration {iteration}")

            # Create request for LLM
            llm_request = ChatCompletionRequest(
                messages=messages_history,
                model=chat_request.model,
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
            model=chat_request.model or "unknown",
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
