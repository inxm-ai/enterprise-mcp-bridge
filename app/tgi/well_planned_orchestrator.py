import json
import logging
import time
import uuid
from typing import Any, List, Optional, AsyncGenerator, Tuple
from app.tgi.think_helper import ThinkExtractor
from opentelemetry import trace

from app.tgi.todo_manager import TodoItem, TodoManager
from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
)
from app.tgi.chunk_reader import chunk_reader, ParsedChunk, create_response_chunk
from app.vars import TGI_MODEL_NAME

tracer = trace.get_tracer(__name__)


async def read_todo_stream(stream_gen) -> AsyncGenerator[ParsedChunk, None]:
    """
    Lightweight helper that consumes an async stream of raw SSE/TGI chunks
    and yields the same ParsedChunk objects that `chunk_reader.as_parsed()`
    would produce. This is a small, well-tested utility used by the
    orchestrator tests.

    The function accepts any async generator yielding raw strings/bytes/dicts
    and yields ParsedChunk instances. It stops when a DONE marker is seen.
    """
    # Reuse chunk_reader to get normalized ParsedChunk objects
    async with chunk_reader(stream_gen, enable_tracing=False) as reader:
        async for parsed in reader.as_parsed():
            # Forward parsed chunks directly
            yield parsed

class WellPlannedOrchestrator:
    """Encapsulates the well-planned orchestration for chat completions.

    This class is intentionally dependency-injected: the caller should provide
    the llm_client, prompt_service, tool_service, and a callable implementing
    the non-stream chat-with-tools behavior (usually ProxiedTGIService._non_stream_chat_with_tools).
    """

    def __init__(
        self,
        llm_client,
        prompt_service,
        tool_service,
        non_stream_chat_with_tools_callable,
        stream_chat_with_tools_callable,
        tool_resolution,
        logger_obj=None,
        model_name: Optional[str] = None,
    ) -> None:
        global logger
        logger = logger_obj
        self.llm_client = llm_client
        self.prompt_service = prompt_service
        self.tool_service = tool_service
        self._non_stream_chat_with_tools = non_stream_chat_with_tools_callable
        self._stream_chat_with_tools = stream_chat_with_tools_callable
        self.tool_resolution = tool_resolution
        self.model_name = model_name or TGI_MODEL_NAME

    async def _stream_todo_plan_json(
        self,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
    ) -> Tuple[Optional[list], Optional[str]]:
        plan_text = ""
        try:
            plan_stream = self.llm_client.stream_completion(request, access_token or "", span)
            async with chunk_reader(plan_stream) as reader:
                async for parsed in reader.as_parsed():
                    if parsed.is_done:
                        break
                    if parsed.content:
                        plan_text += parsed.content
        except Exception as exc:
            if logger:
                logger.error(f"[WellPlanned] Error streaming todo plan: {exc}")
            return None, str(exc)

        plan_text = plan_text.strip()
        if span:
            span.set_attribute("chat.todo_plan_chars", len(plan_text))

        if not plan_text:
            return None, "empty todo plan"

        try:
            decoder = json.JSONDecoder()
            todos_json_obj, idx = decoder.raw_decode(plan_text)
            remaining = plan_text[idx:].strip()
            if remaining and logger:
                logger.debug(
                    f"[WellPlanned] Trailing data after todos JSON (len={len(remaining)}), ignoring."
                )
            todos_json = todos_json_obj
        except Exception as exc:
            if logger:
                logger.error(f"[WellPlanned] Failed to parse todo plan JSON: {exc}")
            return None, str(exc)

        if not isinstance(todos_json, list):
            return None, "todo plan must be a JSON array"

        return todos_json, None

    def _todo_plan_error_generator(self, detail: str) -> AsyncGenerator[str, None]:
        payload = {"error": "failed to parse todos", "detail": detail}

        async def _generator():
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return _generator()

    def _todo_plan_error_response(
        self,
        detail: str,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        payload = {"error": "failed to parse todos", "detail": detail}
        content = json.dumps(payload, ensure_ascii=False)
        return ChatCompletionResponse(
            id=self.llm_client.create_completion_id(),
            created=int(time.time()),
            model=request.model or self.model_name,
            choices=[
                Choice(
                    index=0,
                    message=Message(role=MessageRole.ASSISTANT, content=content),
                    finish_reason="stop",
                )
            ],
            usage=self.llm_client.create_usage_stats(),
        )

    def _create_todo_manager(self, todos_json: list) -> TodoManager:
        todo_manager = TodoManager()
        todo_items: List[TodoItem] = []
        for entry in todos_json:
            if not isinstance(entry, dict):
                continue

            tid = entry.get("id") or str(uuid.uuid4())
            name = entry.get("name") or entry.get("title") or f"todo-{tid}"
            goal = entry.get("goal") or entry.get("description") or ""
            needed_info = entry.get("needed_info") or entry.get("neededInfo")
            tools = entry.get("tools") or []

            todo_items.append(
                TodoItem(
                    id=tid,
                    name=name,
                    goal=goal,
                    needed_info=needed_info,
                    tools=tools,
                )
            )

        if todo_items:
            todo_manager.add_todos(todo_items)

        return todo_manager

    def _select_tools_for_todo(
        self,
        todo: TodoItem,
        available_tools: Optional[List[Any]],
    ) -> List[Any]:
        if not todo.tools:
            return []

        selected: List[Any] = []
        for tool in available_tools or []:
            name = None
            function = getattr(tool, "function", None)
            if function and getattr(function, "name", None):
                name = function.name
            elif isinstance(tool, dict):
                name = tool.get("function", {}).get("name")

            if name and name in todo.tools:
                selected.append(tool)

        return selected

    def _build_focused_request(
        self,
        todo_manager: TodoManager,
        todo: TodoItem,
        base_request: ChatCompletionRequest,
    ):
        focused_messages = [
            Message(role=MessageRole.SYSTEM, content=f"Goal: {todo.goal}")
        ]

        if todo.needed_info:
            focused_messages.append(
                Message(
                    role=MessageRole.USER,
                    content=f"Needed info: {todo.needed_info}",
                )
            )

        for hist in todo_manager.history():
            if hist.get("event") == "finish":
                focused_messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=str(hist.get("result")),
                    )
                )

        focused_request = ChatCompletionRequest(
            messages=focused_messages,
            model=base_request.model or self.model_name,
            stream=False,
            temperature=base_request.temperature,
            max_tokens=base_request.max_tokens,
            top_p=base_request.top_p,
        )

        return focused_messages, focused_request

    def _stringify_result(self, result: Any) -> str:
        if isinstance(result, ChatCompletionResponse):
            choice = result.choices[0] if result.choices else None
            message = choice.message if choice else None
            if message and message.content is not None:
                return message.content
            return result.model_dump_json()

        if isinstance(result, dict):
            try:
                return json.dumps(result, ensure_ascii=False)
            except TypeError:
                return str(result)

        return str(result)

    def _well_planned_streaming(
        self,
        todo_manager: TodoManager,
        session,
        request: ChatCompletionRequest,
        available_tools: List[Any],
        access_token: Optional[str],
        span,
    ) -> AsyncGenerator[str, None]:
        async def _generator():
            id = f"mcp-{str(uuid.uuid4())}"
            names = '\n - '.join([t.name for t in todo_manager.list_todos()])
            message = f"<think>I have planned the following todos:\n - {names}\n</think>"

            yield create_response_chunk(id, message)

            for todo in todo_manager.list_todos():
                todo_manager.start_todo(todo.id)
                yield create_response_chunk(f"mcp-{str(uuid.uuid4())}", f"<think>I marked '{todo.name}' as current todo, and will work on the following goal: {todo.goal}</think>")

                focused_messages, focused_request = self._build_focused_request(
                    todo_manager, todo, request
                )
                filtered_tools = self._select_tools_for_todo(
                    todo, available_tools
                )
                if len(filtered_tools) == 0 and len(todo.tools) > 0:
                    logger.warning(
                        f"[WellPlanned] No matching tools found for todo {todo.id} with requested tools {todo.tools} in the available tools {available_tools}. Passing all tools as fallback."
                    )
                    filtered_tools = available_tools

                try:
                    logger.info(f"[WellPlanned] Processing todo {todo.id}")

                    stream_gen = self._stream_chat_with_tools(
                        session,
                        focused_messages,
                        filtered_tools,
                        focused_request,
                        access_token,
                        span,
                    )

                    aggregated = ""
                    think_extractor = ThinkExtractor()

                    async for parsed in read_todo_stream(stream_gen):
                        yield think_extractor.feed(parsed.content or "")
                        aggregated += parsed.content or ""

                    # Once the stream completes, turn the aggregated content into
                    # a stored result. If no content was aggregated, mark an error.
                    if aggregated:
                        try:
                            # Attempt to interpret aggregated as a JSON
                            try:
                                result = json.loads(aggregated)
                            except Exception:
                                result = aggregated
                        except Exception as exc:
                            result = {"error": str(exc)}
                except Exception as exc:
                    result = {"error": str(exc)}

                todo_manager.finish_todo(todo.id, result)
                summary = f"<think>I have completed the todo '{todo.name}'.</think>"
                yield create_response_chunk(f"mcp-{str(uuid.uuid4())}", summary)

            # yield the result of the final todo as a last chunk
            final_result = todo_manager.list_todos()[-1].result if todo_manager.list_todos() else "No todos were processed."
            yield create_response_chunk(f"mcp-{str(uuid.uuid4())}", final_result)
            yield create_response_chunk(f"mcp-{str(uuid.uuid4())}", "[DONE]")

        return _generator()

    async def _well_planned_non_streaming(
        self,
        todo_manager: TodoManager,
        session,
        request: ChatCompletionRequest,
        available_tools: List[Any],
        access_token: Optional[str],
        span,
    ) -> ChatCompletionResponse:
        names = [t.name for t in todo_manager.list_todos()]
        results_payload: List[dict] = []

        for todo in todo_manager.list_todos():
            todo_manager.start_todo(todo.id)

            focused_messages, focused_request = self._build_focused_request(
                todo_manager, todo, request
            )
            filtered_tools = self._select_tools_for_todo(todo, available_tools)

            try:
                result = await self._non_stream_chat_with_tools(
                    session,
                    focused_messages,
                    filtered_tools,
                    focused_request,
                    access_token,
                    span,
                )
            except Exception as exc:
                result = {"error": str(exc)}

            todo_manager.finish_todo(todo.id, result)

            results_payload.append(
                {
                    "id": todo.id,
                    "name": todo.name,
                    "goal": todo.goal,
                    "result": self._stringify_result(result),
                }
            )

        content = json.dumps(
            {"todos": names, "results": results_payload},
            ensure_ascii=False,
        )

        return ChatCompletionResponse(
            id=self.llm_client.create_completion_id(),
            created=int(time.time()),
            model=request.model or self.model_name,
            choices=[
                Choice(
                    index=0,
                    message=Message(role=MessageRole.ASSISTANT, content=content),
                    finish_reason="stop",
                )
            ],
            usage=self.llm_client.create_usage_stats(),
        )

    async def well_planned_chat_completion(
        self,
        session,
        request: ChatCompletionRequest,
        access_token: Optional[str] = None,
        prompt: Optional[str] = None,
        span=None,
    ) -> Any:
        if span:
            span.set_attribute("chat.strategy", "well-planned")
            span.set_attribute("chat.well_planned", True)

        messages = await self.prompt_service.prepare_messages(
            session, request.messages, prompt, span
        )

        available_tools = await self.tool_service.get_all_mcp_tools(session, span)

        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "goal": {"type": "string"},
                    "needed_info": {"type": ["string", "null"]},
                    "tools": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "name", "goal", "tools"],
                "additionalProperties": False,
            },
        }

        todo_prompt = (
            "You are an assistant that turns the user's conversation into a short, "
            "ordered todo list. Make sure that at the end of all todos, the user's "
            "original goal is achieved, and that the todos are as specific as possible. "
            "Return only JSON that conforms to the following JSON Schema (response_format):\n\n"
            + json.dumps(response_schema, ensure_ascii=False)
            + "\n\nConversation:\n"
            + "\n".join([f"{m.role}: {m.content}" for m in messages])
            + "\n\nReturn only the JSON array that matches the schema above."
        )

        plan_request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.SYSTEM, content=todo_prompt)],
            model=request.model or self.model_name,
            stream=True,
        )

        todos_json, plan_error = await self._stream_todo_plan_json(
            plan_request, access_token, span
        )

        if plan_error:
            if request.stream:
                return self._todo_plan_error_generator(plan_error)
            return self._todo_plan_error_response(plan_error, request)

        todo_manager = self._create_todo_manager(todos_json)

        if span:
            span.set_attribute(
                "chat.todos_count", len(todo_manager.list_todos())
            )

        if request.stream:
            return self._well_planned_streaming(
                todo_manager,
                session,
                request,
                available_tools,
                access_token,
                span,
            )

        return await self._well_planned_non_streaming(
            todo_manager,
            session,
            request,
            available_tools,
            access_token,
            span,
        )
