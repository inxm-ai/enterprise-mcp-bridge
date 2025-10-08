"""
Tests for the MessageSummarizationService.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from app.tgi.services.message_summarization_service import MessageSummarizationService
from app.tgi.models import (
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
    ChatCompletionResponse,
    Choice,
    Usage,
)
from app.tgi.clients.llm_client import LLMClient


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    return client


@pytest.fixture
def summarization_service(mock_llm_client):
    """Create a MessageSummarizationService with mocked dependencies."""
    return MessageSummarizationService(llm_client=mock_llm_client)


def test_should_summarize_below_threshold(summarization_service):
    """Test that messages below threshold are not recommended for summarization."""
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi there!"),
    ]

    assert not summarization_service.should_summarize(messages, threshold=15)


def test_should_summarize_above_threshold(summarization_service):
    """Test that messages above threshold are recommended for summarization."""
    messages = [
        Message(role=MessageRole.USER, content=f"Message {i}") for i in range(20)
    ]

    assert summarization_service.should_summarize(messages, threshold=15)


@pytest.mark.asyncio
async def test_summarize_messages_too_few(summarization_service, mock_llm_client):
    """Test that too few messages are not summarized."""
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello"),
    ]

    result = await summarization_service.summarize_messages(messages)

    # Should return original messages unchanged
    assert result == messages
    # LLM should not be called
    mock_llm_client.non_stream_completion.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_messages_preserves_system_and_tool(
    summarization_service, mock_llm_client
):
    """Test that summarization preserves system prompt and last tool result."""
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="What's the weather?"),
        Message(
            role=MessageRole.ASSISTANT,
            content="Let me check that for you.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments=json.dumps({"location": "New York"}),
                    ),
                )
            ],
        ),
        Message(
            role=MessageRole.TOOL,
            content="Sunny, 72F",
            tool_call_id="call_1",
            name="get_weather",
        ),
    ]

    # Mock the LLM response
    mock_response = ChatCompletionResponse(
        id="test_id",
        object="chat.completion",
        created=1234567890,
        model="test_model",
        choices=[
            Choice(
                index=0,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="User asked about weather, assistant used get_weather tool for New York.",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    mock_llm_client.non_stream_completion = AsyncMock(return_value=mock_response)

    result = await summarization_service.summarize_messages(messages)

    # Should have 3 messages: system, summary, tool result
    assert len(result) == 3
    assert result[0].role == MessageRole.SYSTEM
    assert result[0].content == "You are a helpful assistant."
    assert result[1].role == MessageRole.ASSISTANT
    assert "[Conversation Summary]" in result[1].content
    assert result[2].role == MessageRole.TOOL
    assert result[2].content == "Sunny, 72F"

    # LLM should be called once
    mock_llm_client.non_stream_completion.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_messages_no_system_prompt(
    summarization_service, mock_llm_client
):
    """Test summarization when there's no system prompt."""
    messages = [
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi!"),
        Message(role=MessageRole.USER, content="How are you?"),
        Message(role=MessageRole.ASSISTANT, content="I'm good!"),
    ]

    # Mock the LLM response
    mock_response = ChatCompletionResponse(
        id="test_id",
        object="chat.completion",
        created=1234567890,
        model="test_model",
        choices=[
            Choice(
                index=0,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="User greeted and asked how the assistant was doing.",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    mock_llm_client.non_stream_completion = AsyncMock(return_value=mock_response)

    result = await summarization_service.summarize_messages(messages)

    # Should have 1 message: just the summary
    assert len(result) == 1
    assert result[0].role == MessageRole.ASSISTANT
    assert "[Conversation Summary]" in result[0].content


@pytest.mark.asyncio
async def test_summarize_messages_no_tool_result(
    summarization_service, mock_llm_client
):
    """Test summarization when there's no tool result at the end."""
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi!"),
        Message(role=MessageRole.USER, content="How are you?"),
    ]

    # Mock the LLM response
    mock_response = ChatCompletionResponse(
        id="test_id",
        object="chat.completion",
        created=1234567890,
        model="test_model",
        choices=[
            Choice(
                index=0,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content="User greeted and asked how the assistant was doing.",
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    mock_llm_client.non_stream_completion = AsyncMock(return_value=mock_response)

    result = await summarization_service.summarize_messages(messages)

    # Should have 2 messages: system and summary
    assert len(result) == 2
    assert result[0].role == MessageRole.SYSTEM
    assert result[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_summarize_messages_error_handling(
    summarization_service, mock_llm_client
):
    """Test that errors during summarization return original messages."""
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi!"),
        Message(role=MessageRole.USER, content="How are you?"),
    ]

    # Mock the LLM to raise an exception
    mock_llm_client.non_stream_completion = AsyncMock(
        side_effect=Exception("LLM error")
    )

    result = await summarization_service.summarize_messages(messages)

    # Should return original messages on error
    assert result == messages


def test_build_summarization_prompt(summarization_service):
    """Test that summarization prompt is built correctly."""
    messages = [
        Message(role=MessageRole.USER, content="What's 2+2?"),
        Message(role=MessageRole.ASSISTANT, content="The answer is 4."),
    ]

    prompt = summarization_service._build_summarization_prompt(messages)

    assert "USER: What's 2+2?" in prompt
    assert "ASSISTANT: The answer is 4." in prompt
    assert "summarize" in prompt.lower()


def test_build_summarization_prompt_with_tool_calls(summarization_service):
    """Test that tool calls are included in the summarization prompt."""
    messages = [
        Message(
            role=MessageRole.ASSISTANT,
            content="Let me check that.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments=json.dumps({"location": "NYC"}),
                    ),
                )
            ],
        ),
        Message(
            role=MessageRole.TOOL,
            content="Sunny",
            tool_call_id="call_1",
            name="get_weather",
        ),
    ]

    prompt = summarization_service._build_summarization_prompt(messages)

    assert "get_weather" in prompt
    assert "Tool" in prompt
