import pytest
import json
import time
from unittest.mock import AsyncMock

from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
    Usage,
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
async def test_well_planned_context_forwarding():
    """Test that the result of the first todo is correctly forwarded to the second todo as a string content."""
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

    # We will capture the messages passed to _non_stream_chat_with_tools
    captured_messages_per_call = []

    async def fake_non_stream_chat(session, messages, tools, req, access_token, span):
        captured_messages_per_call.append(messages)

        # Return a ChatCompletionResponse object, which was causing issues when str() was used on it
        return ChatCompletionResponse(
            id="chatcmpl-test",
            created=int(time.time()),
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Result for {req.messages[0].content}",  # Just some content
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
        messages=[Message(role=MessageRole.USER, content="Do the job")],
        model="test-model",
        stream=False,
        tool_choice="auto",
    )

    await service.well_planned_chat_completion(
        session, chat_request, access_token=None, prompt=None, span=None
    )

    assert len(captured_messages_per_call) == 2

    # Check the messages for the second todo (index 1)
    second_call_messages = captured_messages_per_call[1]

    # The messages should contain the result of the first todo
    # It should be an ASSISTANT message
    # And crucially, the content should be the string content, not the object representation

    found_result = False
    for msg in second_call_messages:
        if msg.role == MessageRole.ASSISTANT:
            # We expect "Result for ..." in the content
            if "Result for" in msg.content:
                found_result = True
                # Verify it is NOT the string representation of the object
                assert "ChatCompletionResponse" not in msg.content
                assert "Result for" in msg.content

    assert (
        found_result
    ), "Did not find the result of the first todo in the messages for the second todo"
