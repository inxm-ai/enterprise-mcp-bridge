import json
import re
from typing import Any, AsyncGenerator, List, Optional

from opentelemetry import trace

from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
)
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

        while iteration < self.max_iterations:
            iteration += 1

            llm_request = ChatCompletionRequest(
                messages=messages_history,
                model=chat_request.model or TGI_MODEL_NAME,
                tools=available_tools if available_tools else chat_request.tools,
                tool_choice=chat_request.tool_choice,
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

                    if parsed.tool_calls and emit_think_messages:
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
                        raw_chunk = (
                            parsed.raw
                            if parsed.raw.endswith("\n\n")
                            else f"{parsed.raw}\n\n"
                        )
                        yield raw_chunk

                    yield "\n\n"

            with tracer.start_as_current_span("execute_tool_calls") as tool_span:
                tool_call_chunks = reader.get_accumulated_tool_calls()

                parsed_tool_calls, _ = self.tool_resolution.resolve_tool_calls(
                    content_message, tool_call_chunks
                )

                if parsed_tool_calls:
                    tool_span.set_attribute(
                        "tool_calls.resolved_count", len(parsed_tool_calls)
                    )
                    tool_calls_to_execute = []

                    content_message = self._strip_fabricated_tool_results(
                        content_message, tool_calls_to_execute or parsed_tool_calls
                    )

                    for parsed_call in parsed_tool_calls:
                        if parsed_call.name not in map(
                            lambda tool: tool["function"]["name"], available_tools
                        ):
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

                    if emit_think_messages:

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

                    if success:
                        if emit_think_messages:
                            success_msg = "<think>I executed the tools successfully, resuming response generation</think>\n\n"
                            yield f"data: {json.dumps({'choices':[{'delta':{'content': success_msg},'index':0}]})}\n\n"
                        tool_span.set_attribute("tool_calls.success", True)
                    else:
                        if emit_think_messages:
                            failure_report = "<think>The tool call failed. I will try to adjust my approach</think>\n\n"
                            yield f"data: {json.dumps({'choices':[{'delta':{'content': failure_report},'index': 0}]})}\n\n"
                        tool_span.set_attribute("tool_calls.success", False)

                    should_continue = (
                        bool(tool_results) or len(tool_calls_to_execute) == 1
                    )
                    if should_continue:
                        continue
                    break

                tool_span.set_attribute("tool_calls.resolved_count", 0)
                break

        yield "data: [DONE]\n\n"

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
