import pytest
import json
import time
from unittest.mock import AsyncMock, patch, Mock

from app.tgi.proxied_tgi_service import ProxiedTGIService
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


def configure_plan_stream(service, todos, stream_calls=None):
    todos_payload = json.dumps(todos, ensure_ascii=False)
    chunk_payload = json.dumps({
        'choices': [{'delta': {'content': todos_payload}, 'index': 0}]
    })
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
        payload = s[len("data:"):].strip()
    else:
        payload = s
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


@pytest.mark.asyncio
async def test_well_planned_streaming_flow():
    service = ProxiedTGIService()

    # Craft a todo list JSON with two items
    todos = [
        {"id": "t1", "name": "step1", "goal": "Do step one", "needed_info": "none", "tools": []},
        {"id": "t2", "name": "step2", "goal": "Do step two", "needed_info": "none", "tools": []},
    ]

    configure_plan_stream(service, todos)

    # Patch _non_stream_chat_with_tools to return a predictable result per todo
    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        # simulate a typical helper returning a dict with messages
        return {"ok": True, "messages": [getattr(m, 'content', m) for m in messages]}

    service._non_stream_chat_with_tools = fake_non_stream_chat

    class DummySession:
        async def list_tools(self):
            return []

        async def list_prompts(self):
            return []

    session = DummySession()
    chat_request = ChatCompletionRequest(messages=[Message(role=MessageRole.USER, content="Please do X")], model="test-model", stream=True, tool_choice="auto")

    gen = await service.well_planned_chat_completion(session, chat_request, access_token=None, prompt=None, span=None)

    # gen should be an async generator
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    # Expect at least the todos list, two start and two done, and final [DONE]
    all_text = "".join(chunks)
    # The first chunk should be a streaming plan chunk containing a JSON with choices->0->delta->content
    first_obj = _parse_stream_json(chunks[0])
    assert first_obj is not None, f"Expected JSON in first chunk, got: {chunks[0]}"
    assert "choices" in first_obj
    assert isinstance(first_obj["choices"], list) and len(first_obj["choices"]) > 0
    choice0 = first_obj["choices"][0]
    assert "delta" in choice0
    assert "content" in choice0["delta"]    


@pytest.mark.asyncio
async def test_well_planned_no_premature_done():
    """Test that verifies [DONE] is not sent immediately - the full sequence should be yielded."""
    service = ProxiedTGIService()

    # Create a todo list with at least one item
    todos = [
        {"id": "t1", "name": "analyze", "goal": "Analyze the data", "needed_info": None, "tools": ["tool_a"]},
    ]
    configure_plan_stream(service, todos)

    # Mock _non_stream_chat_with_tools to return a result
    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        return {"result": "Task completed successfully"}

    service._non_stream_chat_with_tools = fake_non_stream_chat

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

    gen = await service.well_planned_chat_completion(session, chat_request, access_token=None, prompt=None, span=None)

    # Collect all chunks
    chunks = []
    async for chunk in gen:
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
    assert second_obj is not None and second_obj.get("content") == "<think>I marked 'analyze' as current todo, and will work on the following goal: Analyze the data</think>", f"Second chunk should be start event, got: {chunks[1]}"

    # Third chunk should be the done event and include the result
    third_obj = _parse_stream_json(chunks[2]).get("choices", [{}])[0].get("delta", {})
    assert third_obj is not None and "Task completed successfully" in third_obj.get("content", ""), f"Third chunk should be done event, got: {chunks[2]}"

    # Last chunk should be [DONE]
    assert any("[DONE]" in c for c in chunks), f"Last chunk should be [DONE], got: {chunks[-1]}"


@pytest.mark.asyncio
async def test_chat_completion_routes_to_well_planned_for_streaming():
    """Test that chat_completion routes to well_planned when tool_choice is set and stream=True.
    
    This reproduces the bug where streaming requests with tool_choice='required' 
    were incorrectly routed to one_off_chat_completion instead of well_planned_chat_completion.
    """
    service = ProxiedTGIService()

    # Create a todo list
    todos = [{"id": "t1", "name": "task", "goal": "Do task", "needed_info": None, "tools": []}]
    llm_raw = json.dumps(todos)

    stream_calls = []
    configure_plan_stream(service, todos, stream_calls=stream_calls)

    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        return {"result": "done"}

    service._non_stream_chat_with_tools = fake_non_stream_chat

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
        tool_choice="auto"  # This should trigger well-planned path
    )

    gen = await service.chat_completion(session, chat_request, access_token=None, prompt=None)

    # Collect chunks
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)

    # Should see the well-planned sequence (not just immediate [DONE])
    # If the bug exists, we'd only get 1 chunk with [DONE]
    # After fix, we should get: plan chunk (choices->delta->content), start event, done event, [DONE]
    assert stream_calls, "well-planned path should invoke stream_completion for todo plan"
    assert len(chunks) >= 4, f"Expected at least 4 chunks from well-planned, got {len(chunks)}: {chunks}"
    # Validate the plan chunk shape
    first_obj = _parse_stream_json(chunks[0])
    assert first_obj is not None, f"Expected JSON in first chunk, got: {chunks[0]}"
    assert "choices" in first_obj and isinstance(first_obj["choices"], list)
    c0 = first_obj["choices"][0]
    assert "delta" in c0 and "content" in c0["delta"]

    second_obj = _parse_stream_json(chunks[1]).get("choices", [{}])[0].get("delta", {})
    assert second_obj is not None and second_obj.get("content") == "<think>I marked 'task' as current todo, and will work on the following goal: Do task</think>", f"Should see start event, got: {chunks[1]}"
    third_obj = _parse_stream_json(chunks[2]).get("choices", [{}])[0].get("delta", {})
    assert third_obj is not None and "done" in third_obj.get("content", ""), f"Should see done event, got: {chunks[2]}"
    assert any("[DONE]" in c for c in chunks), f"Last chunk should be [DONE], got: {chunks[-1]}"


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
    todos_payload = json.dumps(todos, ensure_ascii=False)
    plan_chunks = [
        f"data: {json.dumps({'choices': [{'delta': {'content': todos_payload}, 'index': 0, 'finish_reason': 'stop'}]})}" + "\n\n",
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
    todos_payload = json.dumps(todos, ensure_ascii=False)
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
