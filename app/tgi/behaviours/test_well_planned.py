import pytest
import json
import time
from unittest.mock import AsyncMock, Mock

from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.tgi.behaviours.well_planned_orchestrator import WellPlannedOrchestrator
from app.tgi.behaviours.todos.todo_manager import TodoManager, TodoItem
from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
    Usage,
)


class DummyResponse:
    """Emulate an OpenAI-like response.

    If stream=False, this mimics the non-streaming response shape:
    {
      "id": "chatcmpl-...",
      "object": "chat.completion",
      "created": 123456,
      "model": "gpt-4o-mini",
      "choices": [{"index":0, "message": {"role":"assistant","content": "..."}, "finish_reason": "stop"}],
      "usage": {...}
    }

    If stream=True, this produces a chunk-like shape similar to streaming events:
    {
      "id": "chatcmpl-...",
      "object": "chat.completion.chunk",
      "created": 123456,
      "model": "gpt-4o-mini",
      "choices": [{"delta": {"role":"assistant","content":"..."}, "index":0, "finish_reason": null}]
    }
    """

    def __init__(self, content, model: str = "gpt-4o-mini", stream: bool = False):
        self.id = "chatcmpl-xyz789" if not stream else "chatcmpl-abc123"
        self.object = "chat.completion" if not stream else "chat.completion.chunk"
        self.created = int(time.time())
        self.model = model
        if stream:
            # streaming chunk shape - use plain dicts here
            self.choices = [
                {
                    "delta": {"role": "assistant", "content": content},
                    "index": 0,
                    "finish_reason": None,
                }
            ]
        else:
            # non-streaming full completion shape
            choice = Mock()
            msg = Mock()
            msg.role = "assistant"
            msg.content = content
            choice.index = 0
            choice.message = msg
            choice.finish_reason = "stop"
            self.choices = [choice]
            # minimal usage info so callers that inspect it won't break
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def configure_plan_stream(service, todos, stream_calls=None, intent_payload=None):
    payload = intent_payload or json.dumps(
        {"intent": "plan", "todos": todos}, ensure_ascii=False
    )
    chunk_payload = json.dumps(
        {"choices": [{"delta": {"content": payload}, "index": 0}]}
    )
    plan_chunks = [
        f"data: {chunk_payload}" + "\n\n",
        "data: [DONE]" + "\n\n",
    ]

    async def plan_generator():
        for chunk in plan_chunks:
            yield chunk

    def fake_stream_completion(llm_request, access_token, span):
        if stream_calls is not None:
            stream_calls.append(llm_request)
        return plan_generator()

    service.llm_client.stream_completion = fake_stream_completion
    service.llm_client.non_stream_completion = AsyncMock(
        side_effect=AssertionError(
            "non_stream_completion should not be used when streaming the todo plan"
        )
    )


def _parse_stream_json(chunk: str):
    """Parse a streaming chunk string prefixed with 'data: ' into JSON.

    Returns the parsed object, or None if the chunk is a [DONE] marker or not JSON.
    """
    if not chunk:
        return None
    s = chunk.strip()
    # Typical SSE style prefix used in tests
    if s.startswith("data:"):
        payload = s[len("data:") :].strip()
    else:
        payload = s
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def test_enforce_final_answer_appends_when_last_todo_has_tools():
    class DummyLLM:
        def create_completion_id(self):
            return "dummy"

        def create_usage_stats(self):
            return {}

    orchestrator = WellPlannedOrchestrator(
        llm_client=DummyLLM(),
        prompt_service=None,
        tool_service=None,
        non_stream_chat_with_tools_callable=lambda *a, **k: None,
        stream_chat_with_tools_callable=lambda *a, **k: None,
        tool_resolution=None,
        logger_obj=None,
        model_name="test-model",
    )

    todo_manager = TodoManager()
    todo_manager.add_todos(
        [
            TodoItem(
                id="t1",
                name="get-account",
                goal="Fetch account metadata",
                tools=["get_account"],
            ),
            TodoItem(
                id="t2",
                name="query-jira",
                goal="Query Jira for open issues",
                tools=["search_jira"],
            ),
        ]
    )

    orchestrator._enforce_final_answer_step(todo_manager, "List my open issues")

    todos = todo_manager.list_todos()
    assert len(todos) == 3
    assert todos[-1].name == "final-answer"
    assert todos[-1].tools == []
    assert todos[-2].name == "query-jira"
    assert todos[-2].tools == ["search_jira"]


@pytest.mark.asyncio
async def test_well_planned_streaming_includes_tool_results_in_next_step():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "search-issues",
            "goal": "Search for issues",
            "needed_info": None,
            "tools": ["search_issues"],
        },
        {
            "id": "t2",
            "name": "final-answer",
            "goal": "Summarize issues",
            "needed_info": None,
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    calls = {"count": 0}

    async def fake_stream_chat(session, messages, tools, req, access_token, span):
        calls["count"] += 1
        if calls["count"] == 1:
            tool_result_event = {
                "choices": [
                    {
                        "delta": {
                            "tool_result": {
                                "name": "search_issues",
                                "content": '{"issues":[{"key":"ISSUE-1"}]}',
                            }
                        },
                        "index": 0,
                    }
                ]
            }
            yield "data: " + json.dumps(tool_result_event) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        history = "\n".join(
            [m.content or "" for m in messages if m.role == MessageRole.ASSISTANT]
        )
        assert "search_issues" in history
        assert "ISSUE-1" in history

        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "Final answer"}, "index": 0}]}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="List issues")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    async for _chunk in gen:
        pass


@pytest.mark.asyncio
async def test_well_planned_streaming_flow():
    service = ProxiedTGIService()

    # Craft a todo list JSON with two items
    todos = [
        {
            "id": "t1",
            "name": "step1",
            "goal": "Do step one",
            "needed_info": "none",
            "tools": [],
        },
        {
            "id": "t2",
            "name": "step2",
            "goal": "Do step two",
            "needed_info": "none",
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    # Patch _non_stream_chat_with_tools to return a predictable result per todo
    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        # simulate a typical helper returning a dict with messages
        return {"ok": True, "messages": [getattr(m, "content", m) for m in messages]}

    service._non_stream_chat_with_tools = fake_non_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Please do X")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    # gen should be an async generator
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    # The first chunk should be a streaming plan chunk containing a JSON with choices->0->delta->content
    first_obj = _parse_stream_json(chunks[0])
    assert first_obj is not None, f"Expected JSON in first chunk, got: {chunks[0]}"
    assert "choices" in first_obj
    assert isinstance(first_obj["choices"], list) and len(first_obj["choices"]) > 0
    choice0 = first_obj["choices"][0]
    assert "delta" in choice0
    assert "content" in choice0["delta"]


@pytest.mark.asyncio
async def test_well_planned_streaming_stops_for_needed_info():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "needs-input",
            "goal": "Search for issues",
            "needed_info": {
                "required_from_user": ["Confirm what counts as open"],
                "context": None,
            },
            "tools": ["search_issues"],
        },
        {
            "id": "t2",
            "name": "final-answer",
            "goal": "Summarize issues",
            "needed_info": None,
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    calls = {"count": 0}

    async def fake_stream_chat(*_args, **_kwargs):
        calls["count"] += 1
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "should not run"}, "index": 0}]}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="List open issues")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    assert calls["count"] == 0
    contents = []
    for chunk in chunks:
        parsed = _parse_stream_json(chunk)
        if parsed and parsed.get("choices"):
            delta = parsed["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                contents.append(content)
    assert any("Confirm what counts as open" in c for c in contents)


@pytest.mark.asyncio
async def test_well_planned_streaming_ignores_previous_step_needed_info():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "fetch-cloud",
            "goal": "Fetch cloudId",
            "needed_info": None,
            "tools": ["get_cloud"],
        },
        {
            "id": "t2",
            "name": "query-issues",
            "goal": "Query open issues",
            "needed_info": {
                "required_from_user": [],
                "context": "cloudId from previous step. Optional: confirm preferred JQL.",
            },
            "tools": ["search_issues"],
        },
    ]

    configure_plan_stream(service, todos)

    calls = {"count": 0}

    async def fake_stream_chat(*_args, **_kwargs):
        calls["count"] += 1
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "ok"}, "index": 0}]}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="List issues")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    assert calls["count"] >= 1
    contents = []
    for chunk in chunks:
        parsed = _parse_stream_json(chunk)
        if parsed and parsed.get("choices"):
            delta = parsed["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                contents.append(content)
    assert any(c == "ok" for c in contents)


@pytest.mark.asyncio
async def test_well_planned_streaming_ignores_todo_reference_needed_info():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "fetch-cloud",
            "goal": "Fetch cloudId",
            "needed_info": None,
            "tools": ["get_cloud"],
        },
        {
            "id": "t2",
            "name": "query-issues",
            "goal": "Query open issues",
            "needed_info": {"required_from_user": [], "context": "cloudId from todo_1"},
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    calls = {"count": 0}

    async def fake_stream_chat(*_args, **_kwargs):
        calls["count"] += 1
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "ok"}, "index": 0}]}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="List issues")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    assert calls["count"] == 2
    contents = []
    for chunk in chunks:
        parsed = _parse_stream_json(chunk)
        if parsed and parsed.get("choices"):
            delta = parsed["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                contents.append(content)
    assert not any("Before I can continue" in c for c in contents)


@pytest.mark.asyncio
async def test_well_planned_streaming_ignores_crawl_needed_info():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "crawl-site",
            "goal": "Crawl site",
            "needed_info": None,
            "tools": ["crawl"],
        },
        {
            "id": "t2",
            "name": "summarize",
            "goal": "Summarize",
            "needed_info": {
                "required_from_user": [],
                "context": (
                    "Content from the crawled blog including posts, "
                    "biographical information, and any personal or professional details"
                ),
            },
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    calls = {"count": 0}

    async def fake_stream_chat(*_args, **_kwargs):
        calls["count"] += 1
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "ok"}, "index": 0}]}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Summarize the blog")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    assert calls["count"] == 2
    contents = []
    for chunk in chunks:
        parsed = _parse_stream_json(chunk)
        if parsed and parsed.get("choices"):
            delta = parsed["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                contents.append(content)
    assert not any("Before I can continue" in c for c in contents)


@pytest.mark.asyncio
async def test_well_planned_streaming_stops_on_user_feedback_tag():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "step1",
            "goal": "Do step one",
            "needed_info": None,
            "tools": [],
        },
        {
            "id": "t2",
            "name": "step2",
            "goal": "Do step two",
            "needed_info": None,
            "tools": [],
        },
    ]

    configure_plan_stream(service, todos)

    calls = {"count": 0}

    async def fake_stream_chat(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] > 1:
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": "extra"}, "index": 0}]}
            ) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        yield "data: " + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "<user_feedback_needed>Please choose a project</user_feedback_needed>"
                        },
                        "index": 0,
                    }
                ]
            }
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Please do X")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    assert calls["count"] == 1
    contents = []
    for chunk in chunks:
        parsed = _parse_stream_json(chunk)
        if parsed and parsed.get("choices"):
            delta = parsed["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                contents.append(content)
    assert any("Please choose a project" in c for c in contents)


@pytest.mark.asyncio
async def test_well_planned_no_premature_done():
    """Test that verifies [DONE] is not sent immediately - the full sequence should be yielded."""
    service = ProxiedTGIService()

    # Create a todo list with at least one item
    todos = [
        {
            "id": "t1",
            "name": "analyze",
            "goal": "Analyze the data",
            "needed_info": None,
            "tools": ["tool_a"],
        },
    ]
    configure_plan_stream(service, todos)

    # Mock _non_stream_chat_with_tools to return a result
    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        return {"result": "Task completed successfully"}

    service._non_stream_chat_with_tools = fake_non_stream_chat

    # Stream path should not leak plan content; return a simple streamed answer
    async def fake_stream_chat(session, messages, tools, req, access_token, span):
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "done"}, "index": 0}]}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do analysis")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    # Collect all chunks
    chunks = []
    async for chunk in gen:
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)

    # Should have at least 3 chunks: 1) todos plan chunk, 2) start event, 3) done event, 4) [DONE]
    assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}: {chunks}"

    # First chunk should be the streaming plan chunk (JSON with choices->0->delta->content)
    first_obj = _parse_stream_json(chunks[0])
    assert first_obj is not None, f"Expected JSON plan chunk, got: {chunks[0]}"
    assert "choices" in first_obj
    choice0 = first_obj["choices"][0]
    assert "delta" in choice0 and "content" in choice0["delta"]

    # Second chunk should be the start event (parse and inspect)
    second_obj = _parse_stream_json(chunks[1]).get("choices", [{}])[0].get("delta", {})
    content = second_obj.get("content", "") if second_obj else ""
    assert (
        second_obj is not None
        and "<think>I marked 'analyze'" in content
        and "current todo" in content
    ), f"Second chunk should be start event, got: {chunks[1]}"

    # A later chunk should be the done event and include the result
    found_done = False
    for c in chunks[2:]:
        obj = _parse_stream_json(c)
        if not obj:
            continue
        delta = obj.get("choices", [{}])[0].get("delta", {})
        if "completed the todo" in delta.get("content", ""):
            found_done = True
            break
    assert found_done, f"Did not find completion chunk in sequence: {chunks}"

    # Last chunk should be [DONE]
    assert any(
        "[DONE]" in c for c in chunks
    ), f"Last chunk should be [DONE], got: {chunks[-1]}"


@pytest.mark.asyncio
async def test_well_planned_reroute_intent_short_circuit_streaming():
    service = ProxiedTGIService()

    reroute_payload = json.dumps(
        {"intent": "reroute", "reason": "Out of scope"}, ensure_ascii=False
    )
    configure_plan_stream(service, [], intent_payload=reroute_payload)

    service._stream_chat_with_tools = AsyncMock(
        side_effect=AssertionError("Tool flow should not run for reroute intent")
    )

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do something unrelated")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    chunks = []
    async for chunk in gen:
        if chunk.strip():
            chunks.append(chunk)

    assert len(chunks) == 2, f"Expected reroute + DONE, got {chunks}"
    first_obj = _parse_stream_json(chunks[0])
    content = first_obj["choices"][0]["delta"]["content"]
    assert content.startswith("<reroute_requested>")
    assert "Out of scope" in content
    assert chunks[-1].strip() == "data: [DONE]"


@pytest.mark.asyncio
async def test_well_planned_answer_intent_short_circuit_streaming():
    service = ProxiedTGIService()

    answer_payload = json.dumps(
        {"intent": "answer", "answer": "Hello there!"}, ensure_ascii=False
    )
    configure_plan_stream(service, [], intent_payload=answer_payload)

    service._stream_chat_with_tools = AsyncMock(
        side_effect=AssertionError("Tool flow should not run for answer intent")
    )

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Say hi")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    chunks = []
    async for chunk in gen:
        if chunk.strip():
            chunks.append(chunk)

    assert len(chunks) == 2, f"Expected answer + DONE, got {chunks}"
    first_obj = _parse_stream_json(chunks[0])
    content = first_obj["choices"][0]["delta"]["content"]
    assert content == "Hello there!"
    assert chunks[-1].strip() == "data: [DONE]"


@pytest.mark.asyncio
async def test_chat_completion_routes_to_well_planned_for_streaming():
    """Test that chat_completion routes to well_planned when tool_choice is set and stream=True.

    This reproduces the bug where streaming requests with tool_choice='required'
    were incorrectly routed to one_off_chat_completion instead of well_planned_chat_completion.
    """
    service = ProxiedTGIService()

    # Create a todo list
    todos = [
        {
            "id": "t1",
            "name": "task",
            "goal": "Do task",
            "needed_info": None,
            "tools": [],
        }
    ]

    stream_calls = []
    configure_plan_stream(service, todos, stream_calls=stream_calls)

    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        return {"result": "done"}

    service._non_stream_chat_with_tools = fake_non_stream_chat

    async def fake_stream_chat(session, messages, tools, req, access_token, span):
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "done"}, "index": 0}]}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()

    # Make a streaming request with tool_choice set (not auto)
    # Before the fix, this would route to one_off_chat_completion and immediately send [DONE]
    # After the fix, this should route to well_planned_chat_completion
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do task")],
        model="test-model",
        stream=True,
        tool_choice="auto",  # This should trigger well-planned path
    )

    gen = await service.chat_completion(
        session, chat_request, access_token=None, prompt=None
    )

    # Collect chunks, skipping over '\n' only chunks
    chunks = []
    async for chunk in gen:
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)

    # Should see the well-planned sequence (not just immediate [DONE])
    # If the bug exists, we'd only get 1 chunk with [DONE]
    # After fix, we should get: plan chunk (choices->delta->content), start event, done event, [DONE]
    assert (
        stream_calls
    ), "well-planned path should invoke stream_completion for todo plan"
    assert (
        len(chunks) >= 4
    ), f"Expected at least 4 chunks from well-planned, got {len(chunks)}: {chunks}"
    # Validate the plan chunk shape
    first_obj = _parse_stream_json(chunks[0])
    assert first_obj is not None, f"Expected JSON in first chunk, got: {chunks[0]}"
    assert "choices" in first_obj and isinstance(first_obj["choices"], list)
    c0 = first_obj["choices"][0]
    assert "delta" in c0 and "content" in c0["delta"]

    second_obj = _parse_stream_json(chunks[1]).get("choices", [{}])[0].get("delta", {})
    content = second_obj.get("content", "") if second_obj else ""
    assert (
        second_obj is not None
        and "<think>I marked 'task'" in content
        and "current todo" in content
    ), f"Should see start event, got: {chunks[1]}"

    found_done = False
    for c in chunks[2:]:
        obj = _parse_stream_json(c)
        if not obj:
            continue
        delta = obj.get("choices", [{}])[0].get("delta", {})
        if "completed" in delta.get("content", ""):
            found_done = True
            break
    assert found_done, f"Should see done event, got sequence: {chunks}"
    assert any(
        "[DONE]" in c for c in chunks
    ), f"Last chunk should be [DONE], got: {chunks[-1]}"


@pytest.mark.asyncio
async def test_well_planned_streaming_uses_stream_completion_for_plan():
    """Ensure the todo plan is retrieved via streaming when streaming is requested."""
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "first",
            "goal": "Do the first thing",
            "needed_info": None,
            "tools": [],
        }
    ]
    todos_payload = json.dumps({"intent": "plan", "todos": todos}, ensure_ascii=False)
    plan_chunks = [
        f"data: {json.dumps({'choices': [{'delta': {'content': todos_payload}, 'index': 0, 'finish_reason': 'stop'}]})}"
        + "\n\n",
        "data: [DONE]" + "\n\n",
    ]

    stream_calls = []

    async def plan_generator():
        for chunk in plan_chunks:
            yield chunk

    def fake_stream_completion(llm_request, access_token, span):
        stream_calls.append(llm_request)
        return plan_generator()

    service.llm_client.stream_completion = fake_stream_completion
    service.llm_client.non_stream_completion = AsyncMock(
        side_effect=AssertionError(
            "non_stream_completion should not be used when streaming the todo plan"
        )
    )

    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        return {"result": "ok"}

    service._non_stream_chat_with_tools = fake_non_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Plan this")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    assert stream_calls, "stream_completion should be invoked to fetch the todo plan"
    # first chunk should be a streaming chunk with choices->delta->content
    first_obj = _parse_stream_json(chunks[0])
    assert first_obj is not None
    assert "choices" in first_obj
    assert "delta" in first_obj["choices"][0]
    assert "content" in first_obj["choices"][0]["delta"]
    assert chunks[-1].strip() == "data: [DONE]"


@pytest.mark.asyncio
async def test_well_planned_returns_chat_completion_when_not_streaming():
    """When streaming is disabled, the response should be a ChatCompletionResponse."""
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "analyze",
            "goal": "Analyze the thing",
            "needed_info": None,
            "tools": [],
        }
    ]
    todos_payload = json.dumps({"intent": "plan", "todos": todos}, ensure_ascii=False)
    plan_chunks = [
        f"data: {json.dumps({'choices': [{'delta': {'content': todos_payload}, 'index': 0, 'finish_reason': 'stop'}]})}\n\n",
        "data: [DONE]\n\n",
    ]

    stream_calls = []

    async def plan_generator():
        for chunk in plan_chunks:
            yield chunk

    def fake_stream_completion(llm_request, access_token, span):
        stream_calls.append(llm_request)
        return plan_generator()

    service.llm_client.stream_completion = fake_stream_completion
    service.llm_client.non_stream_completion = AsyncMock(
        side_effect=AssertionError(
            "non_stream_completion should not be used when streaming the todo plan"
        )
    )

    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        return ChatCompletionResponse(
            id="chatcmpl-test",
            created=int(time.time()),
            model=req.model or "test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Completed: {messages[-1].content if messages else ''}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    service._non_stream_chat_with_tools = fake_non_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Please analyse")],
        model="test-model",
        stream=False,
        tool_choice="auto",
    )

    result = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    assert isinstance(result, ChatCompletionResponse)
    assert stream_calls, "stream_completion should be used to fetch the todo plan"
    assert result.choices
    assert result.choices[0].message
    assert "Completed" in result.choices[0].message.content


@pytest.mark.asyncio
async def test_well_planned_strips_think_tags_from_result():
    """Test that <think> tags are stripped from the result before being stored/returned."""
    service = ProxiedTGIService()

    # Create a todo list
    todos = [
        {
            "id": "t1",
            "name": "think_task",
            "goal": "Do thinking",
            "needed_info": None,
            "tools": [],
        }
    ]
    configure_plan_stream(service, todos)

    # Mock _stream_chat_with_tools to return content with <think> tags
    async def fake_stream_chat(session, messages, tools, req, access_token, span):
        # Yield chunks that form: "Here is the result.<think>Thinking process</think>"
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "Here is "}, "index": 0}]}
        ) + "\n\n"
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "the result."}, "index": 0}]}
        ) + "\n\n"
        yield "data: " + json.dumps(
            {
                "choices": [
                    {
                        "delta": {"content": "<think>Thinking process</think>"},
                        "index": 0,
                    }
                ]
            }
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do thinking")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    chunks = []
    async for chunk in gen:
        if chunk.strip():
            chunks.append(chunk)

    # Find the chunk that contains the result summary
    # It should be one of the chunks after the start event

    summary_chunk = None
    visible_parts = []

    for chunk in chunks:
        parsed = _parse_stream_json(chunk)
        if not parsed:
            continue
        content = parsed["choices"][0]["delta"].get("content", "")

        if "<think>I have completed" in content:
            summary_chunk = parsed
            break
        elif not content.startswith("<think>"):
            visible_parts.append(content)

    assert summary_chunk is not None, "Summary chunk not found"

    # Check summary chunk
    content = summary_chunk["choices"][0]["delta"]["content"]
    # The summary should NOT contain <think> tags inside the result part
    # The summary itself is wrapped in <think>...</think> by the orchestrator
    # format: <think>I have completed... Result summary: ...</think>
    assert "Thinking process" not in content
    assert "Here is the result." in content

    combined = "".join(visible_parts)
    assert combined.strip() == "Here is the result."


@pytest.mark.asyncio
async def test_well_planned_streams_final_message_chunks():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "streaming",
            "goal": "Send updates",
            "needed_info": None,
            "tools": [],
        }
    ]

    configure_plan_stream(service, todos)

    async def fake_stream_chat(*args, **kwargs):
        payloads = ["Hello ", "World"]
        for piece in payloads:
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": piece}, "index": 0}]}
            ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Do it")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    contents = []
    async for chunk in gen:
        parsed = _parse_stream_json(chunk)
        if parsed and parsed.get("choices"):
            contents.append(parsed["choices"][0]["delta"].get("content", ""))

    assert any(
        "Hello " in c for c in contents
    ), f"Streamed chunks missing 'Hello ': {contents}"
    assert any(
        "World" in c for c in contents
    ), f"Streamed chunks missing 'World': {contents}"

    summary_idx = next(i for i, c in enumerate(contents) if "completed the todo" in c)
    hello_idx = next(i for i, c in enumerate(contents) if "Hello " in c)
    assert (
        hello_idx < summary_idx
    ), "Visible content should arrive before the completion summary"

    # Ensure we did not emit a duplicated full result chunk after the summary
    full_text = "Hello World"
    assert not any(
        full_text in c for c in contents[summary_idx + 1 :]
    ), f"Duplicate final chunk detected after summary: {contents}"


@pytest.mark.asyncio
async def test_well_planned_streaming_preserves_all_visible_text():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "final-answer",
            "goal": "Describe capabilities",
            "needed_info": None,
            "tools": [],
        }
    ]

    configure_plan_stream(service, todos)

    async def fake_stream_chat(*args, **kwargs):
        pieces = [
            "Hello! I'm INXM, ",
            "<think>internal reasoning</think>",
            "an orchestrator that builds ",
            "repeatable ",
            '<get_tools server-id="x">ignored</get_tools>',
            "workflows with teams.",
        ]
        for piece in pieces:
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": piece}, "index": 0}]}
            ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Tell me what you do")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    visible_parts = []
    async for chunk in gen:
        parsed = _parse_stream_json(chunk)
        if not parsed or not parsed.get("choices"):
            continue
        content = parsed["choices"][0]["delta"].get("content", "") or ""
        if content.startswith("<think>"):
            continue
        if "completed the todo" in content:
            break
        visible_parts.append(content)

    combined = "".join(visible_parts)
    expected = (
        "Hello! I'm INXM, an orchestrator that builds repeatable workflows with teams."
    )
    assert (
        combined == expected
    ), f"Visible stream content lost pieces.\nExpected: {expected}\nGot: {combined}"


@pytest.mark.asyncio
async def test_well_planned_strips_tool_call_blocks_from_final_output():
    service = ProxiedTGIService()

    todos = [
        {
            "id": "t1",
            "name": "strip-tools",
            "goal": "Produce answer",
            "needed_info": None,
            "tools": [],
        }
    ]

    configure_plan_stream(service, todos)

    service.tool_service.get_all_mcp_tools = AsyncMock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "get_tools",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        ]
    )

    async def fake_stream_chat(*args, **kwargs):
        yield "data: " + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "content": 'Result <get_tools server-id="abc"></get_tools> value'
                        },
                        "index": 0,
                    }
                ]
            }
        ) + "\n\n"
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": " Final."}, "index": 0}]}
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    service._stream_chat_with_tools = fake_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Answer please")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    gen = await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    contents = []
    async for chunk in gen:
        parsed = _parse_stream_json(chunk)
        if parsed and parsed.get("choices"):
            contents.append(parsed["choices"][0]["delta"].get("content", ""))

    assert all("<get_tools" not in (c or "") for c in contents)
    combined = "".join(c for c in contents if c and not c.startswith("<think>"))
    assert "Result" in combined and "Final." in combined


@pytest.mark.asyncio
async def test_well_planned_streaming_does_not_repeat_tool_trace_in_final_chunk():
    class DummyLLM:
        def create_completion_id(self):
            return "dummy"

        def create_usage_stats(self):
            return {}

    orchestrator = WellPlannedOrchestrator(
        llm_client=DummyLLM(),
        prompt_service=None,
        tool_service=None,
        non_stream_chat_with_tools_callable=lambda *a, **k: None,
        stream_chat_with_tools_callable=lambda *a, **k: None,
        tool_resolution=None,
        logger_obj=None,
        model_name="test-model",
    )

    final_raw = """
<think>The user wants me to achieve the current goal, which is to retrieve docs.</think>
I'll retrieve all documents from the collection 'your-collection-id' for you.
<list_collection_documents>{"collection_id": "your-collection-id"}</list_collection_documents>
<think>Executing the following tools
- list_collection_documents(collection_id='your-collection-id')</think><think>I executed the tools successfully, resuming response generation...</think>
I encountered an error with the collection ID format.
"""

    async def fake_stream_chat(*args, **kwargs):
        payloads = [
            "data: "
            + json.dumps({"choices": [{"delta": {"content": final_raw}, "index": 0}]})
            + "\n\n",
            "data: [DONE]\n\n",
        ]
        for p in payloads:
            yield p

    orchestrator._stream_chat_with_tools = fake_stream_chat

    todo_manager = TodoManager()
    todo_manager.add_todos(
        [
            TodoItem(
                id="t1",
                name="list-docs",
                goal="List docs",
                needed_info=None,
                expected_result=None,
                tools=["list_collection_documents"],
            )
        ]
    )

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="List docs")],
        model="test-model",
        stream=True,
        tool_choice="auto",
    )

    available_tools = [{"function": {"name": "list_collection_documents"}}]

    gen = orchestrator._well_planned_streaming(
        todo_manager,
        session=None,
        request=request,
        available_tools=available_tools,
        access_token=None,
        span=None,
        original_messages=request.messages,
    )

    contents = []
    async for chunk in gen:
        parsed = _parse_stream_json(chunk)
        if parsed and parsed.get("choices"):
            contents.append(parsed["choices"][0]["delta"].get("content", ""))

    visible = [c for c in contents if c and not c.startswith("<think>")]
    combined_visible = "".join(visible)
    expected_visible = orchestrator._strip_tool_call_blocks(
        orchestrator._strip_think_tags(final_raw), ["list_collection_documents"]
    )

    assert combined_visible.strip() == expected_visible.strip()
    assert "<think>" not in combined_visible
    assert "<list_collection_documents" not in combined_visible
    assert "Executing the following tools" not in combined_visible
