import copy
import json
import re
import time
import uuid
from typing import Any, List, Optional, AsyncGenerator, Tuple
from app.tgi.protocols.think_helper import ThinkExtractor
from opentelemetry import trace

from app.tgi.behaviours.todos.todo_manager import TodoItem, TodoManager
from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
)
from app.tgi.protocols.chunk_reader import (
    chunk_reader,
    ParsedChunk,
    create_response_chunk,
)
from app.vars import TGI_MODEL_NAME

tracer = trace.get_tracer(__name__)

todo_response_schema = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "goal": {"type": "string"},
                    "needed_info": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "required_from_user": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "context": {"type": ["string", "null"]},
                                },
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "expected_result": {"type": ["string", "null"]},
                    "tools": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "name", "goal", "tools"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["todos"],
    "additionalProperties": False,
}

intent_response_schema = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["plan", "answer", "reroute"]},
        "reason": {"type": ["string", "null"]},
        "answer": {"type": ["string", "null"]},
        "todos": todo_response_schema["properties"]["todos"],
    },
    "required": ["intent"],
    "additionalProperties": False,
}


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

    def _normalize_intent_response(self, intent_obj: Any) -> Optional[dict]:
        if isinstance(intent_obj, list):
            return {
                "intent": "plan",
                "todos": intent_obj,
                "answer": None,
                "reason": None,
            }

        if not isinstance(intent_obj, dict):
            return None

        intent = intent_obj.get("intent")
        todos = intent_obj.get("todos")

        if not intent:
            if todos is not None:
                intent = "plan"
            elif intent_obj.get("answer") is not None:
                intent = "answer"
            else:
                intent = None

        if intent not in ("plan", "answer", "reroute"):
            return None

        return {
            "intent": intent,
            "todos": todos if todos is not None else [],
            "answer": intent_obj.get("answer"),
            "reason": intent_obj.get("reason") or intent_obj.get("reroute_reason"),
        }

    async def _stream_intent_decision(
        self,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
        intent_schema: Optional[dict] = None,
    ) -> Tuple[Optional[dict], Optional[str]]:
        plan_text = ""
        schema = intent_schema or intent_response_schema
        request.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "intent_response",
                "schema": schema,
            },
        }
        try:
            plan_stream = self.llm_client.stream_completion(
                request, access_token or "", span
            )
            async with chunk_reader(plan_stream) as reader:
                async for parsed in reader.as_parsed():
                    if parsed.is_done:
                        break
                    if parsed.content:
                        plan_text += parsed.content
        except Exception as exc:
            if logger:
                logger.error(f"[WellPlanned] Error streaming intent response: {exc}")
                logger.debug(exc, exc_info=True)
            return None, str(exc)

        plan_text = plan_text.strip()
        if span:
            span.set_attribute("chat.intent_response_chars", len(plan_text))

        if not plan_text:
            return None, "empty intent response"

        try:
            decoder = json.JSONDecoder()
            intent_obj, idx = decoder.raw_decode(plan_text)
            remaining = plan_text[idx:].strip()
            if remaining and logger:
                logger.debug(
                    f"[WellPlanned] Trailing data after intent JSON (len={len(remaining)}), ignoring."
                )
        except Exception as exc:
            if logger:
                logger.error(f"[WellPlanned] Failed to parse intent JSON: {exc}")
            return None, str(exc)

        normalized = self._normalize_intent_response(intent_obj)
        if not normalized:
            return None, "intent response must include a valid intent"

        if normalized["intent"] == "plan" and not isinstance(
            normalized.get("todos"), list
        ):
            return None, "todo plan must be a JSON array"

        if normalized["intent"] != "plan":
            normalized["todos"] = []

        return normalized, None

    def _intent_error_generator(self, detail: str) -> AsyncGenerator[str, None]:
        payload = {"error": "failed to parse intent", "detail": detail}

        async def _generator():
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return _generator()

    def _intent_error_response(
        self,
        detail: str,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        payload = {"error": "failed to parse intent", "detail": detail}
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

    def _single_message_response(
        self, content: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        payload = self._stringify_payload(content)
        return ChatCompletionResponse(
            id=self.llm_client.create_completion_id(),
            created=int(time.time()),
            model=request.model or self.model_name,
            choices=[
                Choice(
                    index=0,
                    message=Message(role=MessageRole.ASSISTANT, content=payload),
                    finish_reason="stop",
                )
            ],
            usage=self.llm_client.create_usage_stats(),
        )

    def _stream_single_message(self, content: str) -> AsyncGenerator[str, None]:
        payload = self._stringify_payload(content)

        async def _generator():
            yield create_response_chunk(self.llm_client.create_completion_id(), payload)
            yield create_response_chunk(
                self.llm_client.create_completion_id(), "[DONE]"
            )

        return _generator()

    def _create_metadata_chunk(
        self,
        content: str,
        metadata: dict,
        model_name: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> str:
        chunk_dict = {
            "id": chunk_id or self.llm_client.create_completion_id(),
            "model": model_name or self.model_name,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
            "metadata": metadata,
        }
        return f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"

    def _todo_metadata(self, todo: TodoItem) -> dict:
        return {
            "id": todo.id,
            "name": todo.name,
            "goal": todo.goal,
            "needed_info": todo.needed_info,
            "expected_result": todo.expected_result,
            "tools": list(todo.tools),
            "state": getattr(todo.state, "value", str(todo.state)),
        }

    def _attach_metadata_to_chunk(self, chunk: str, metadata: dict) -> str:
        if not metadata or not chunk:
            return chunk

        chunk_str = chunk.strip()
        if chunk_str == "[DONE]" or chunk_str == "data: [DONE]":
            return chunk

        if chunk_str.startswith("data:"):
            payload = chunk_str[len("data:") :].strip()
        else:
            return chunk

        if payload == "[DONE]":
            return chunk

        try:
            parsed = json.loads(payload)
        except Exception:
            return chunk

        if not isinstance(parsed, dict):
            return chunk

        existing = parsed.get("metadata")
        if isinstance(existing, dict):
            merged = {**existing, **metadata}
        else:
            merged = dict(metadata)
        parsed["metadata"] = merged
        return f"data: {json.dumps(parsed, ensure_ascii=False)}\n\n"

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
            expected_result = entry.get("expected_result") or entry.get(
                "expectedResult"
            )
            tools = entry.get("tools") or []

            todo_items.append(
                TodoItem(
                    id=tid,
                    name=name,
                    goal=goal,
                    needed_info=needed_info,
                    expected_result=expected_result,
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

    def _extract_tool_names(self, tools: Optional[List[Any]]) -> List[str]:
        names: List[str] = []
        for tool in tools or []:
            function = getattr(tool, "function", None)
            if function and getattr(function, "name", None):
                names.append(function.name)
            elif isinstance(tool, dict):
                name = tool.get("function", {}).get("name")
                if name:
                    names.append(name)
        return names

    def _build_focused_request(
        self,
        todo_manager: TodoManager,
        todo: TodoItem,
        base_request: ChatCompletionRequest,
        original_messages: Optional[List[Message]] = None,
        is_final_multistep_todo: bool = False,
        is_final_step: bool = False,
    ):
        # Preserve any original system prompt from the base request/prepared messages
        original_system_content = None
        if original_messages:
            for msg in original_messages:
                if msg.role == MessageRole.SYSTEM:
                    original_system_content = msg.content
                    break

        # Combine original system prompt with todo-specific goal
        if original_system_content:
            combined_system_content = (
                f"{original_system_content}\n\nCurrent Goal: {todo.goal}\n"
                "You have access to the results of previous steps. Use them to achieve the current goal."
            )
        else:
            combined_system_content = (
                f"Goal: {todo.goal}\n"
                "You have access to the results of previous steps. Use them to achieve the current goal."
            )

        tool_sentence = (
            f"\nAllowed tools for this step: {', '.join(todo.tools)}."
            if todo.tools
            else "\nNo tools are available for this step; do not invent new tool calls."
        )
        combined_system_content += (
            "\nUse only the tools listed for this todo; do not add new ones."
            + tool_sentence
        )

        if is_final_multistep_todo:
            combined_system_content += (
                "\nThis is the final step of the plan. Do not start additional tool calls "
                "or new tasks—deliver the final user-facing answer directly."
            )

        if todo.expected_result:
            combined_system_content += f"\n\nExpected Result: {todo.expected_result}"

        focused_messages = [
            Message(role=MessageRole.SYSTEM, content=combined_system_content)
        ]

        for hist in todo_manager.history():
            if hist.get("event") == "finish":
                focused_messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=str(hist.get("result")),
                    )
                )

        needed_info = self._normalize_needed_info(todo.needed_info)
        if needed_info:
            focused_messages.append(
                Message(
                    role=MessageRole.USER,
                    content=f"Needed info: {needed_info}",
                )
            )
        else:
            context_info = self._normalize_needed_info_context(todo.needed_info)
            if context_info:
                focused_messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=f"Context: {context_info}",
                    )
                )
            focused_messages.append(
                Message(
                    role=MessageRole.USER,
                    content="Achieve the goal.",
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
        if is_final_step and base_request.response_format:
            focused_request.response_format = base_request.response_format

        return focused_messages, focused_request

    def _strip_think_tags(self, text: str) -> str:
        # Remove <think>...</think> blocks, including newlines that might follow them
        # The pattern should be non-greedy
        # This pattern matches <think>...</think> and optional following whitespace
        pattern = r"<think>.*?</think>\s*"
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def _normalize_needed_info(self, needed_info: Optional[Any]) -> Optional[str]:
        if not needed_info:
            return None
        if isinstance(needed_info, dict):
            required = (
                needed_info.get("required_from_user")
                or needed_info.get("required")
                or needed_info.get("from_user")
                or []
            )
            if isinstance(required, str):
                required = [required]
            if not isinstance(required, list):
                required = []
            required = [
                str(item).strip()
                for item in required
                if isinstance(item, (str, int, float)) and str(item).strip()
            ]
            if not required:
                return None
            return ". ".join(required)

        if isinstance(needed_info, list):
            required = [
                str(item).strip()
                for item in needed_info
                if isinstance(item, (str, int, float)) and str(item).strip()
            ]
            if not required:
                return None
            return ". ".join(required)

        text = str(needed_info).strip()
        if not text:
            return None
        lowered = text.lower().strip().strip(".")
        if lowered in {"none", "n/a", "na", "n.a", "null", "no", "not applicable"}:
            return None
        return None

    def _normalize_needed_info_context(
        self, needed_info: Optional[Any]
    ) -> Optional[str]:
        if isinstance(needed_info, dict):
            context = needed_info.get("context") or needed_info.get("from_previous")
        elif isinstance(needed_info, str):
            context = needed_info
        else:
            return None
        if isinstance(context, list):
            context = ". ".join(
                [str(item).strip() for item in context if str(item).strip()]
            )
        if isinstance(context, str):
            context = context.strip()
            lowered = context.lower().strip().strip(".")
            if lowered in {"none", "n/a", "na", "n.a", "null", "no", "not applicable"}:
                return None
        return context or None

    def _format_needed_info_question(self, needed_info: str) -> str:
        text = needed_info.strip()
        if not text:
            return ""
        if "?" in text:
            return text
        return f"Before I can continue, I need:\n{text}"

    def _strip_tool_call_blocks(
        self, text: str, tool_names: Optional[List[str]]
    ) -> str:
        if not text:
            return text

        default_names = {"get_tools"}
        names = [re.escape(n) for n in set(tool_names or []).union(default_names) if n]
        if not names:
            return text

        joined = "|".join(names)
        open_tag = rf"<(?:{joined})\b[^>]*?>"
        close_tag = rf"</(?:{joined})>"
        paired_pattern = rf"{open_tag}.*?{close_tag}"
        text = re.sub(paired_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(open_tag, "", text, flags=re.IGNORECASE)
        text = re.sub(close_tag, "", text, flags=re.IGNORECASE)
        return text

    def _extract_tag_content(self, text: str, tag: str) -> Optional[str]:
        match = re.search(
            rf"<{tag}[^>]*>(?P<content>.*?)</{tag}>",
            text or "",
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        content = match.group("content").strip() if match.group("content") else None
        return content or None

    def _looks_like_question(self, text: str) -> bool:
        if not text:
            return False
        stripped = text.strip()
        if "?" not in stripped:
            return False
        if stripped.startswith("{") or stripped.startswith("["):
            return False
        if len(stripped) <= 500:
            return True
        lowered = stripped.lower()
        triggers = (
            "please",
            "clarif",
            "confirm",
            "specify",
            "which",
            "what",
            "how",
            "when",
            "where",
            "do you",
            "can you",
            "could you",
            "would you",
        )
        return any(trigger in lowered for trigger in triggers)

    def _extract_feedback_from_text(
        self, text: Optional[str], tool_names: List[str]
    ) -> Optional[str]:
        if not text:
            return None
        cleaned = self._strip_tool_call_blocks(self._strip_think_tags(text), tool_names)
        tag_content = self._extract_tag_content(cleaned, "user_feedback_needed")
        if tag_content:
            return tag_content
        if self._looks_like_question(cleaned):
            return cleaned.strip()
        return None

    def _try_parse_json(self, text: str) -> Optional[Any]:
        if not isinstance(text, str):
            return None
        stripped = text.strip()
        if not stripped or stripped[0] not in "{[":
            return None
        try:
            return json.loads(stripped)
        except Exception:
            return None

    def _extract_feedback_from_result(
        self, result: Any, tool_names: List[str]
    ) -> Optional[str]:
        if result is None:
            return None

        if isinstance(result, ChatCompletionResponse):
            choice = result.choices[0] if result.choices else None
            message = choice.message if choice else None
            return self._extract_feedback_from_text(
                message.content if message else None, tool_names
            )

        if isinstance(result, str):
            return self._extract_feedback_from_text(result, tool_names)

        if isinstance(result, list):
            for item in result:
                found = self._extract_feedback_from_result(item, tool_names)
                if found:
                    return found
            return None

        if isinstance(result, dict):
            for key in ("user_feedback_needed", "question", "needed_info"):
                val = result.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

            for key in ("response", "answer", "content", "message"):
                val = result.get(key)
                if isinstance(val, str):
                    found = self._extract_feedback_from_text(val, tool_names)
                    if found:
                        return found
                    parsed = self._try_parse_json(val)
                    if parsed is not None:
                        found = self._extract_feedback_from_result(parsed, tool_names)
                        if found:
                            return found

            tool_results = result.get("tool_results")
            if isinstance(tool_results, list):
                for entry in tool_results:
                    found = self._extract_feedback_from_result(entry, tool_names)
                    if found:
                        return found

        return None

    def _visible_text_delta(
        self, chunk: str, state: dict, tool_names: List[str]
    ) -> str:
        """
        Extract non-<think> content from the chunk, accumulate it, strip tool-call
        blocks, and return only the newly added portion.
        """
        if chunk is None:
            chunk = ""

        buffer = state.get("buffer", "")
        in_think = state.get("in_think", False)

        i = 0
        while i < len(chunk):
            if not in_think:
                open_idx = chunk.find("<think>", i)
                if open_idx == -1:
                    buffer += chunk[i:]
                    break
                buffer += chunk[i:open_idx]
                i = open_idx + len("<think>")
                in_think = True
            else:
                close_idx = chunk.find("</think>", i)
                if close_idx == -1:
                    # Still inside a think block; skip the rest of this chunk
                    i = len(chunk)
                    break
                i = close_idx + len("</think>")
                in_think = False

        state["buffer"] = buffer
        state["in_think"] = in_think

        cleaned = self._strip_tool_call_blocks(
            self._strip_think_tags(buffer), tool_names
        )
        # Normalize excessive blank lines introduced by stripping think/tool blocks
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"^\n{2,}", "\n", cleaned)
        previous_clean = state.get("clean", "")

        if not cleaned.startswith(previous_clean):
            # If cleaning removed something we had not yet emitted, reset cursor safely
            emitted_len = min(state.get("emitted", 0), len(cleaned))
        else:
            emitted_len = state.get("emitted", 0)

        delta = cleaned[emitted_len:]
        state["clean"] = cleaned
        state["emitted"] = len(cleaned)

        return delta

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

    def _enforce_final_answer_step(
        self, todo_manager: TodoManager, user_message: Optional[str]
    ) -> None:
        todos = todo_manager.list_todos()
        user_goal = (user_message or "").strip() or "Answer the user's request."

        if not todos:
            todo_manager.add_todos(
                [
                    TodoItem(
                        id=str(uuid.uuid4()),
                        name="final-answer",
                        goal=user_goal,
                        needed_info=None,
                        expected_result="A clear final answer to the user.",
                        tools=[],
                    )
                ]
            )
            return

        final_todo = todos[-1]
        if final_todo and final_todo.tools:
            todo_manager.add_todos(
                [
                    TodoItem(
                        id=str(uuid.uuid4()),
                        name="final-answer",
                        goal=f"Use only the existing information to answer the user's request: {user_goal}",
                        needed_info=None,
                        expected_result="A concise final answer with no tool calls.",
                        tools=[],
                    )
                ]
            )
            return

        if final_todo and len(todos) > 1:
            final_todo.goal = f"Use only the existing information to answer the user's request: {user_goal}"
            final_todo.expected_result = final_todo.expected_result or (
                "A concise final answer with no tool calls."
            )
            final_todo.tools = []

    def _well_planned_streaming(
        self,
        todo_manager: TodoManager,
        session,
        request: ChatCompletionRequest,
        available_tools: List[Any],
        access_token: Optional[str],
        span,
        original_messages: Optional[List[Message]] = None,
    ) -> AsyncGenerator[str, None]:
        async def _generator():
            todos = todo_manager.list_todos()
            all_tool_names = self._extract_tool_names(available_tools)
            id = f"mcp-{str(uuid.uuid4())}"
            names = "\n - ".join([t.name for t in todos])
            message = (
                f"<think>I have planned the following todos:\n - {names}\n</think>"
            )

            yield create_response_chunk(id, message)

            visible_accum = ""

            for idx, todo in enumerate(todos):
                is_last_todo = idx == len(todos) - 1
                todo_manager.start_todo(todo.id)
                todo_metadata = None if is_last_todo else self._todo_metadata(todo)
                todo_meta_wrapper = (
                    {"todo": todo_metadata} if todo_metadata is not None else {}
                )
                todo_status_chunk = create_response_chunk(
                    f"mcp-{str(uuid.uuid4())}",
                    f"<think>I marked '{todo.name}' as current todo, and will work on the following goal: {todo.goal}</think>",
                )
                yield self._attach_metadata_to_chunk(
                    todo_status_chunk, todo_meta_wrapper
                )

                needed_info = self._normalize_needed_info(todo.needed_info)
                if needed_info:
                    question = self._format_needed_info_question(needed_info)
                    todo_manager.finish_todo(todo.id, question)
                    yield create_response_chunk(f"mcp-{str(uuid.uuid4())}", question)
                    yield create_response_chunk(f"mcp-{str(uuid.uuid4())}", "[DONE]")
                    return

                focused_messages, focused_request = self._build_focused_request(
                    todo_manager,
                    todo,
                    request,
                    original_messages,
                    is_final_multistep_todo=is_last_todo and len(todos) > 1,
                    is_final_step=is_last_todo,
                )
                filtered_tools = self._select_tools_for_todo(todo, available_tools)
                if len(filtered_tools) == 0 and len(todo.tools) > 0:
                    available_tool_names = [
                        t.function.name if hasattr(t, "function") else str(t)
                        for t in (available_tools or [])
                    ]
                    if logger:
                        logger.warning(
                            f"[WellPlanned] No matching tools found for todo {todo.id} with requested tools {todo.tools}. Available: {available_tool_names}. Proceeding without tools."
                        )

                visible_state = {
                    "buffer": "",
                    "in_think": False,
                    "clean": "",
                    "emitted": 0,
                }

                try:
                    if logger:
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
                    tool_results = []
                    think_extractor = ThinkExtractor()

                    async for parsed in read_todo_stream(stream_gen):
                        think_chunk = think_extractor.feed(parsed.content or "")
                        if think_chunk and think_chunk.strip():
                            yield self._attach_metadata_to_chunk(
                                think_chunk, todo_meta_wrapper
                            )

                        if parsed.content:
                            aggregated += parsed.content
                        if parsed.tool_result:
                            tool_result = dict(parsed.tool_result)
                            content = tool_result.get("content")
                            if not isinstance(content, str):
                                try:
                                    tool_result["content"] = json.dumps(
                                        content,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                except Exception:
                                    tool_result["content"] = str(content)
                            tool_results.append(tool_result)

                        if is_last_todo:
                            delta = self._visible_text_delta(
                                parsed.content or "", visible_state, all_tool_names
                            )
                            if delta:
                                yield create_response_chunk(
                                    f"mcp-{str(uuid.uuid4())}", delta
                                )
                                visible_accum = visible_state.get("clean", delta)

                    # Once the stream completes, turn the aggregated content into
                    # a stored result. If no content was aggregated, mark an error.
                    result = None
                    if aggregated:
                        try:
                            # Attempt to interpret aggregated as a JSON
                            try:
                                result = json.loads(aggregated)
                            except Exception:
                                result = aggregated
                        except Exception as exc:
                            result = {"error": str(exc)}
                    if tool_results:
                        if isinstance(result, dict):
                            existing = result.get("tool_results")
                            if isinstance(existing, list):
                                existing.extend(tool_results)
                            else:
                                result["tool_results"] = tool_results
                        else:
                            result = {"response": result, "tool_results": tool_results}
                    if result is None:
                        result = "No content returned from step."
                except Exception as exc:
                    result = {"error": str(exc)}

                stringified = self._stringify_result(result)
                if isinstance(stringified, str):
                    cleaned_result = self._strip_tool_call_blocks(
                        self._strip_think_tags(stringified), all_tool_names
                    )
                else:
                    cleaned_result = stringified

                if not is_last_todo:
                    feedback = self._extract_feedback_from_result(
                        result, all_tool_names
                    )
                    if not feedback and isinstance(cleaned_result, str):
                        feedback = self._extract_feedback_from_text(
                            cleaned_result, all_tool_names
                        )
                    if feedback:
                        todo_manager.finish_todo(todo.id, feedback)
                        yield create_response_chunk(
                            f"mcp-{str(uuid.uuid4())}", feedback
                        )
                        yield create_response_chunk(
                            f"mcp-{str(uuid.uuid4())}", "[DONE]"
                        )
                        return

                todo_manager.finish_todo(todo.id, cleaned_result)
                visible_accum = visible_state.get("clean", visible_accum)

                if is_last_todo and isinstance(cleaned_result, str):
                    emitted_len = visible_state.get("emitted", len(visible_accum))
                    if len(cleaned_result) > emitted_len:
                        remaining = cleaned_result[emitted_len:]
                        if remaining:
                            yield create_response_chunk(
                                f"mcp-{str(uuid.uuid4())}", remaining
                            )
                            visible_accum = cleaned_result

                metadata_context = (
                    result if isinstance(result, (dict, list)) else cleaned_result
                )
                summary = f"<think>I have completed the todo '{todo.name}'. Result summary: {self._stringify_result(cleaned_result)[:200]}...</think>"
                summary_metadata = {"context": metadata_context, **todo_meta_wrapper}
                yield self._create_metadata_chunk(
                    summary,
                    summary_metadata,
                    model_name=request.model or self.model_name,
                    chunk_id=f"mcp-{str(uuid.uuid4())}",
                )

            # yield the result of the final todo as a last chunk
            final_result = todos[-1].result if todos else "No todos were processed."
            # If we already streamed visible content for the last todo, avoid duplicating it.
            if not (todos):
                yield create_response_chunk(f"mcp-{str(uuid.uuid4())}", final_result)
            yield create_response_chunk(f"mcp-{str(uuid.uuid4())}", "[DONE]")

        return _generator()

    def _extract_response_format_schema(
        self, response_format: Optional[dict]
    ) -> Optional[dict]:
        if not isinstance(response_format, dict):
            return None
        rtype = response_format.get("type")
        if rtype == "json_schema":
            json_schema = response_format.get("json_schema") or {}
            schema = json_schema.get("schema")
            return schema if isinstance(schema, dict) else None
        if rtype == "json_object":
            return {"type": "object"}
        return None

    def _build_intent_response_schema(self, response_format: Optional[dict]) -> dict:
        schema = copy.deepcopy(intent_response_schema)
        answer_schema = self._extract_response_format_schema(response_format)
        if answer_schema:
            props = dict(schema.get("properties") or {})
            props["answer"] = {"oneOf": [answer_schema, {"type": "null"}]}
            schema["properties"] = props
        return schema

    def _stringify_payload(self, payload: Any) -> str:
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if payload is None:
            return ""
        return str(payload)

    async def _well_planned_non_streaming(
        self,
        todo_manager: TodoManager,
        session,
        request: ChatCompletionRequest,
        available_tools: List[Any],
        access_token: Optional[str],
        span,
        original_messages: Optional[List[Message]] = None,
    ) -> ChatCompletionResponse:
        todos = todo_manager.list_todos()
        names = [t.name for t in todos]
        results_payload: List[dict] = []
        all_tool_names = self._extract_tool_names(available_tools)

        for idx, todo in enumerate(todos):
            is_last_todo = idx == len(todos) - 1
            todo_manager.start_todo(todo.id)

            needed_info = self._normalize_needed_info(todo.needed_info)
            if needed_info:
                question = self._format_needed_info_question(needed_info)
                todo_manager.finish_todo(todo.id, question)
                return self._single_message_response(question, request)

            focused_messages, focused_request = self._build_focused_request(
                todo_manager,
                todo,
                request,
                original_messages,
                is_final_multistep_todo=is_last_todo and len(todos) > 1,
                is_final_step=is_last_todo,
            )
            filtered_tools = self._select_tools_for_todo(todo, available_tools)

            if len(filtered_tools) == 0 and len(todo.tools) > 0:
                available_tool_names = [
                    t.function.name if hasattr(t, "function") else str(t)
                    for t in (available_tools or [])
                ]
                if logger:
                    logger.warning(
                        f"[WellPlanned] No matching tools found for todo {todo.id} with requested tools {todo.tools}. Available: {available_tool_names}. Proceeding without tools."
                    )

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

            # Ensure we store a stringified version of the result in the history
            # so that subsequent steps can read it properly (avoiding ChatCompletionResponse objects)
            stored_result = self._stringify_result(result)
            if isinstance(stored_result, str):
                stored_result = self._strip_tool_call_blocks(
                    self._strip_think_tags(stored_result), all_tool_names
                )
            feedback = self._extract_feedback_from_result(result, all_tool_names)
            if not feedback and isinstance(stored_result, str):
                feedback = self._extract_feedback_from_text(
                    stored_result, all_tool_names
                )
            if feedback:
                todo_manager.finish_todo(todo.id, feedback)
                return self._single_message_response(feedback, request)
            todo_manager.finish_todo(todo.id, stored_result)

            results_payload.append(
                {
                    "id": todo.id,
                    "name": todo.name,
                    "goal": todo.goal,
                    "result": stored_result,
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

        available_tool_names: List[str] = []
        for tool in available_tools or []:
            function = getattr(tool, "function", None)
            if function and getattr(function, "name", None):
                available_tool_names.append(function.name)
            elif isinstance(tool, dict):
                name = tool.get("function", {}).get("name")
                if name:
                    available_tool_names.append(name)

        unique_tool_names = sorted(set(available_tool_names))
        tool_list_text = ", ".join(unique_tool_names) if unique_tool_names else "none"
        conversation_text = "\n".join([f"{m.role}: {m.content}" for m in messages])

        intent_schema = self._build_intent_response_schema(request.response_format)

        intent_prompt = (
            "You are an assistant that must decide between three intents: "
            "'plan', 'answer', or 'reroute'. Always include the 'intent' property. "
            "Choose 'plan' when the user request needs multiple steps or tool calls. "
            "Choose 'answer' when the request is simple (e.g., greetings, short questions, "
            "or anything that can be answered immediately without tools). "
            "Choose 'reroute' when the request is out of scope for this agent or not "
            "supported by the available tools or system prompt.\n\n"
            "If you pick 'plan', create a short, ordered todo list that progressively reaches "
            "the user's goal. Each todo should be specific, build on previous results, avoid "
            "repeating the same tool unless necessary, list "
            "only the tools it needs (using exact names), and include an 'expected_result' "
            "field with examples of success/failure. Use the 'needed_info' field only for "
            "user-provided inputs that are missing. If user input is required, set needed_info "
            "to an object with required_from_user (array of strings). If info should come from "
            "previous steps/tools, put it in needed_info.context instead of required_from_user. "
            "If nothing is needed from the user and no context is required, set needed_info to null. "
            "If no tools are required, use an empty array. "
            "If more then one todo is needed, ensure the final todo delivers a clear user-facing answer. "
            f"Available tools: {tool_list_text}.\n\n"
            "Return only JSON that conforms to the following JSON Schema (response_format):\n\n"
            + json.dumps(intent_schema, ensure_ascii=False)
            + "\n\nConversation:\n"
            + conversation_text
            + "\n\nReturn only the JSON object that matches the schema above."
        )

        plan_request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=intent_prompt),
                Message(role=MessageRole.USER, content=messages[-1].content or "Do it"),
            ],
            model=request.model or self.model_name,
            stream=True,
            tools=available_tools,
        )

        intent_result, plan_error = await self._stream_intent_decision(
            plan_request, access_token, span, intent_schema=intent_schema
        )

        if plan_error:
            if request.stream:
                return self._intent_error_generator(plan_error)
            return self._intent_error_response(plan_error, request)

        intent = (intent_result or {}).get("intent", "plan")

        if span:
            span.set_attribute("chat.intent", intent)
            if intent != "plan":
                span.set_attribute("chat.todos_count", 0)

        if intent == "reroute":
            reason = intent_result.get("reason") if intent_result else None
            if isinstance(reason, str):
                reason = self._strip_think_tags(reason)
            message = (
                f"<reroute_requested>{reason}</reroute_requested>"
                if reason
                else "<reroute_requested>This request should be handled by another system.</reroute_requested>"
            )
            if request.stream:
                return self._stream_single_message(message)
            return self._single_message_response(message, request)

        if intent == "answer":
            answer_payload = (intent_result or {}).get("answer")
            if isinstance(answer_payload, str):
                answer_payload = self._strip_think_tags(answer_payload)
                if request.response_format:
                    try:
                        answer_payload = json.loads(answer_payload)
                    except Exception:
                        pass
            answer_text = self._stringify_payload(answer_payload)
            if request.stream:
                return self._stream_single_message(answer_text)
            return self._single_message_response(answer_text, request)

        todos_json = intent_result.get("todos") if intent_result else []

        if not isinstance(todos_json, list):
            if request.stream:
                return self._intent_error_generator("todo plan must be a JSON array")
            return self._intent_error_response(
                "todo plan must be a JSON array", request
            )

        todo_manager = self._create_todo_manager(todos_json)
        user_message = messages[-1].content if messages else ""
        self._enforce_final_answer_step(todo_manager, user_message)

        if span:
            span.set_attribute("chat.todos_count", len(todo_manager.list_todos()))

        if request.stream:
            return self._well_planned_streaming(
                todo_manager,
                session,
                request,
                available_tools,
                access_token,
                span,
                original_messages=messages,
            )

        return await self._well_planned_non_streaming(
            todo_manager,
            session,
            request,
            available_tools,
            access_token,
            span,
            original_messages=messages,
        )
