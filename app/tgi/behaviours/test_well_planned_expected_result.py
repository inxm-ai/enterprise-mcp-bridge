import pytest
import json
from unittest.mock import AsyncMock

from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
)


def configure_plan_stream(service, todos, stream_calls=None):
    todos_payload = json.dumps(todos, ensure_ascii=False)
    chunk_payload = json.dumps(
        {"choices": [{"delta": {"content": todos_payload}, "index": 0}]}
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


@pytest.mark.asyncio
async def test_well_planned_expected_result_flow():
    service = ProxiedTGIService()

    # Craft a todo list JSON with expected_result
    todos = [
        {
            "id": "t1",
            "name": "step1",
            "goal": "Do step one",
            "needed_info": "none",
            "expected_result": "A JSON object with key 'status'",
            "tools": [],
        }
    ]

    configure_plan_stream(service, todos)

    # Patch _non_stream_chat_with_tools to verify expected_result is in the prompt
    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        # Check if expected_result is in the system prompt
        system_message = next(
            (m for m in messages if m.role == MessageRole.SYSTEM), None
        )
        assert system_message is not None
        assert (
            "Expected Result: A JSON object with key 'status'" in system_message.content
        )

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

    # Consume the generator to trigger the execution
    async for _ in gen:
        pass
