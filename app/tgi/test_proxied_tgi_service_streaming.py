import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from app.tgi.proxied_tgi_service import ProxiedTGIService
from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
)


@pytest.mark.asyncio
async def test_streaming_tool_call_merging_and_status_messages():
    service = ProxiedTGIService()
    # Simulate a streaming LLM response with tool_calls split across chunks
    tool_call_id = "toolcall-123"
    chunks = []
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [{"id": tool_call_id, "index": 0, "function": {}}]
                },
                "index": 0,
            }
        ]
    }
    chunk2 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "index": 0,
                            "function": {"name": "get_weather"},
                        }
                    ]
                },
                "index": 0,
            }
        ]
    }
    chunk3 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "index": 0,
                            "function": {"arguments": '{"city": "Berlin"}'},
                        }
                    ]
                },
                "index": 0,
            }
        ]
    }
    chunk4 = {
        "choices": [{"delta": {"content": "Here is the weather info."}, "index": 0}]
    }
    chunks.append("data: " + json.dumps(chunk1) + "\n\n")
    chunks.append("data: " + json.dumps(chunk2) + "\n\n")
    chunks.append("data: " + json.dumps(chunk3) + "\n\n")
    chunks.append("data: " + json.dumps(chunk4) + "\n\n")
    chunks.append("data: [DONE]\n\n")

    # Patch llm_client.stream_completion to yield our chunks
    async def mock_stream_llm_completion(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    service.llm_client.stream_completion = mock_stream_llm_completion
    # Patch tool_service.execute_tool_calls to simulate tool execution
    service.tool_service.execute_tool_calls = AsyncMock(
        return_value=(
            [
                Message(
                    role=MessageRole.TOOL,
                    content='{"result": "sunny"}',
                    tool_call_id=tool_call_id,
                    name="get_weather",
                )
            ],
            True,
        )
    )
    # Patch MCPSessionBase
    session = MagicMock()
    messages = [Message(role=MessageRole.USER, content="What's the weather in Berlin?")]
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "",
                "parameters": {},
            },
        }
    ]
    chat_request = ChatCompletionRequest(
        messages=messages, model="test-model", stream=True
    )
    # Collect all yielded chunks
    result_chunks = []
    async for chunk in service._stream_chat_with_tools(
        session, messages, available_tools, chat_request, "token", None
    ):
        result_chunks.append(chunk)
    # Check status messages and merging
    assert any("Preparing tools to call" in c for c in result_chunks)
    assert any("I will run <code>get_weather</code>" in c for c in result_chunks)
    assert any("Here is the weather info." in c for c in result_chunks)
    assert any("[DONE]" in c for c in result_chunks)


@pytest.mark.asyncio
async def test_streaming_multiple_tool_calls_parallel():
    service = ProxiedTGIService()
    # Simulate two parallel tool calls
    chunks = []
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"id": "id1", "index": 0, "function": {"name": "func1"}},
                        {"id": "id2", "index": 1, "function": {"name": "func2"}},
                    ]
                },
                "index": 0,
            }
        ]
    }
    chunk2 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": "id1",
                            "index": 0,
                            "function": {"arguments": '{"foo": 1}'},
                        },
                        {
                            "id": "id2",
                            "index": 1,
                            "function": {"arguments": '{"bar": 2}'},
                        },
                    ]
                },
                "index": 0,
            }
        ]
    }
    chunks.append("data: " + json.dumps(chunk1) + "\n\n")
    chunks.append("data: " + json.dumps(chunk2) + "\n\n")
    chunks.append("data: [DONE]\n\n")

    async def mock_stream_llm_completion(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    service.llm_client.stream_completion = mock_stream_llm_completion
    service.tool_service.execute_tool_calls = AsyncMock(
        return_value=(
            [
                Message(
                    role=MessageRole.TOOL,
                    content='{"result": "ok1"}',
                    tool_call_id="id1",
                    name="func1",
                ),
                Message(
                    role=MessageRole.TOOL,
                    content='{"result": "ok2"}',
                    tool_call_id="id2",
                    name="func2",
                ),
            ],
            True,
        )
    )
    session = MagicMock()
    messages = [Message(role=MessageRole.USER, content="Do two things.")]
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "func1",
                "description": "",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "func2",
                "description": "",
                "parameters": {},
            },
        },
    ]
    chat_request = ChatCompletionRequest(
        messages=messages, model="test-model", stream=True
    )
    result_chunks = []
    async for chunk in service._stream_chat_with_tools(
        session, messages, available_tools, chat_request, "token", None
    ):
        result_chunks.append(chunk)
    assert any("I will run <code>func1</code>" in c for c in result_chunks)
    assert any("I will run <code>func2</code>" in c for c in result_chunks)
    assert any("[DONE]" in c for c in result_chunks)


@pytest.mark.asyncio
async def test_streaming_tool_call_missing_arguments():
    service = ProxiedTGIService()
    # Simulate a tool call with missing arguments
    chunks = []
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"id": "id1", "index": 0, "function": {"name": "func1"}}
                    ]
                },
                "index": 0,
            }
        ]
    }
    chunks.append("data: " + json.dumps(chunk1) + "\n\n")
    chunks.append("data: [DONE]\n\n")

    async def mock_stream_llm_completion(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    service.llm_client.stream_completion = mock_stream_llm_completion
    service.tool_service.execute_tool_calls = AsyncMock(
        return_value=(
            [
                Message(
                    role=MessageRole.TOOL,
                    content='{"error": "missing arguments"}',
                    tool_call_id="id1",
                    name="func1",
                ),
            ],
            True,
        )
    )
    session = MagicMock()
    messages = [Message(role=MessageRole.USER, content="Call func1.")]
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "func1",
                "description": "",
                "parameters": {},
            },
        }
    ]
    chat_request = ChatCompletionRequest(
        messages=messages, model="test-model", stream=True
    )
    result_chunks = []
    async for chunk in service._stream_chat_with_tools(
        session, messages, available_tools, chat_request, "token", None
    ):
        result_chunks.append(chunk)
    assert any("I will run <code>func1</code>" in c for c in result_chunks)
    assert any("[DONE]" in c for c in result_chunks)


@pytest.mark.asyncio
async def test_streaming_content_only():
    service = ProxiedTGIService()
    # Simulate a content-only stream (no tool calls)
    chunks = []
    chunk1 = {"choices": [{"delta": {"content": "Just a reply."}, "index": 0}]}
    chunks.append("data: " + json.dumps(chunk1) + "\n\n")
    chunks.append("data: [DONE]\n\n")

    async def mock_stream_llm_completion(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    service.llm_client.stream_completion = mock_stream_llm_completion
    service.tool_service.execute_tool_calls = AsyncMock()
    session = MagicMock()
    messages = [Message(role=MessageRole.USER, content="Say something.")]
    available_tools = []
    chat_request = ChatCompletionRequest(
        messages=messages, model="test-model", stream=True
    )
    result_chunks = []
    async for chunk in service._stream_chat_with_tools(
        session, messages, available_tools, chat_request, "token", None
    ):
        result_chunks.append(chunk)
    assert any("Just a reply." in c for c in result_chunks)
    assert any("[DONE]" in c for c in result_chunks)


@pytest.mark.asyncio
async def test_streaming_tool_call_chunked_arguments():
    service = ProxiedTGIService()
    # Arguments split across multiple chunks
    tool_call_id = "id1"
    chunks = []
    # First chunk: tool call with name
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"id": tool_call_id, "index": 0, "function": {"name": "func1"}}
                    ]
                },
                "index": 0,
            }
        ]
    }
    chunks.append(f"data: {json.dumps(chunk1)}\n\n")
    # Second chunk: arguments part 1
    chunk2 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "index": 0,
                            "function": {"arguments": '{"foo":'},
                        }
                    ]
                },
                "index": 0,
            }
        ]
    }
    chunks.append("data: " + json.dumps(chunk2) + "\n\n")
    # Third chunk: arguments part 2
    chunk3 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "index": 0,
                            "function": {"arguments": " 1}"},
                        }
                    ]
                },
                "index": 0,
            }
        ]
    }
    chunks.append(f"data: {json.dumps(chunk3)}\n\n")
    # End chunk
    chunks.append("data: [DONE]\n\n")

    async def mock_stream_llm_completion(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    service.llm_client.stream_completion = mock_stream_llm_completion
    service.tool_service.execute_tool_calls = AsyncMock(
        return_value=(
            [
                Message(
                    role=MessageRole.TOOL,
                    content='{"result": "ok"}',
                    tool_call_id=tool_call_id,
                    name="func1",
                ),
            ],
            True,
        )
    )
    session = MagicMock()
    messages = [Message(role=MessageRole.USER, content="Call func1.")]
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "func1",
                "description": "",
                "parameters": {},
            },
        }
    ]
    chat_request = ChatCompletionRequest(
        messages=messages, model="test-model", stream=True
    )
    result_chunks = []
    async for chunk in service._stream_chat_with_tools(
        session, messages, available_tools, chat_request, "token", None
    ):
        result_chunks.append(chunk)
    # Arguments should be merged
    assert any("I will run <code>func1</code>" in c for c in result_chunks)
    assert any("[DONE]" in c for c in result_chunks)
