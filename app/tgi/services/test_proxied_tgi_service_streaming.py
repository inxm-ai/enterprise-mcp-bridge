import pytest
import json
from unittest.mock import AsyncMock, MagicMock, Mock
from app.tgi.services.proxied_tgi_service import ProxiedTGIService
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
async def test_streaming_tool_calls_without_assistant_content_appends_tool_calls():
    """Ensure that when the LLM only emits tool_calls (no assistant content),
    an assistant message with the tool_calls is still appended to history so
    subsequent LLM requests (or the backend) see tool messages as a response
    to an assistant message.
    """
    service = ProxiedTGIService()

    # Simulate a stream that only emits tool_calls and then [DONE]
    chunks = []
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"id": "only1", "index": 0, "function": {"name": "solo"}}
                    ]
                },
                "index": 0,
            }
        ]
    }
    chunks.append("data: " + json.dumps(chunk1) + "\n\n")
    chunks.append("data: [DONE]\n\n")

    async def mock_stream_llm_completion(*args, **kwargs):
        # capture the request messages passed in the first arg for inspection
        for chunk in chunks:
            yield chunk

    service.llm_client.stream_completion = mock_stream_llm_completion

    # make execute_tool_calls return an empty result set (we don't need to run tools)
    service.tool_service.execute_tool_calls = AsyncMock(return_value=([], True))

    session = MagicMock()
    messages = [Message(role=MessageRole.USER, content="Trigger tool.")]
    available_tools = [
        {
            "type": "function",
            "function": {"name": "solo", "description": "", "parameters": {}},
        }
    ]

    chat_request = ChatCompletionRequest(
        messages=messages, model="test-model", stream=True
    )

    async for _chunk in service._stream_chat_with_tools(
        session, messages, available_tools, chat_request, "token", None
    ):
        # drain the stream
        pass

    # execute_tool_calls should have been called once with the resolved tool call
    assert service.tool_service.execute_tool_calls.called


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


def test_deduplicate_retry_hints_keeps_latest():
    service = ProxiedTGIService()
    messages = [
        Message(role=MessageRole.USER, content="original"),
        Message(
            role=MessageRole.USER,
            name="mcp_tool_retry_hint",
            content="Please retry",
        ),
        Message(
            role=MessageRole.USER,
            name="mcp_tool_retry_hint",
            content="Please retry",
        ),
    ]

    service._deduplicate_retry_hints(messages)

    retry_hints = [m for m in messages if m.name == "mcp_tool_retry_hint"]
    assert len(retry_hints) == 1


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


@pytest.mark.asyncio
async def test_stream_chat_with_tools_yields_correct_newlines():
    """
    Test that _stream_chat_with_tools yields SSE events with correct newlines (\n\n)
    instead of escaped newlines (\\n\\n).
    """
    service = ProxiedTGIService()

    # Mock LLM client stream_completion
    # We simulate a tool call response
    tool_call_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "test_tool", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }

    done_chunk = {"choices": [{"finish_reason": "tool_calls"}]}

    async def mock_stream_completion(*args, **kwargs):
        yield f"data: {json.dumps(tool_call_chunk)}\n\n"
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    service.llm_client.stream_completion = mock_stream_completion

    # Mock tool service execute_tool_calls
    service.tool_service.execute_tool_calls = AsyncMock(
        return_value=(
            [
                Message(
                    role=MessageRole.TOOL, content="Tool result", tool_call_id="call_1"
                )
            ],
            True,
        )
    )

    # Mock tool resolution
    mock_tool_call = Mock()
    mock_tool_call.name = "test_tool"
    mock_tool_call.arguments = {}
    mock_tool_call.id = "call_1"
    mock_tool_call.format = Mock()
    mock_tool_call.format.value = "openai"

    service.tool_resolution.resolve_tool_calls = Mock(
        return_value=([mock_tool_call], True)
    )

    # Prepare request
    session = MagicMock()
    messages = [Message(role=MessageRole.USER, content="Run tool")]
    available_tools = [{"function": {"name": "test_tool"}}]
    request = ChatCompletionRequest(messages=messages, tools=available_tools)

    # Run stream
    chunks = []
    async for chunk in service._stream_chat_with_tools(
        session, messages, available_tools, request, None, None
    ):
        chunks.append(chunk)

    # Verify chunks
    # We expect chunks containing <think> messages

    think_chunks = [c for c in chunks if "<think>" in c]

    # Check for double backslashes
    for chunk in think_chunks:
        # The chunk is an SSE string: data: {...}\n\n
        # We want to ensure it doesn't contain literal \\n\\n inside the JSON string
        # But wait, the chunk itself ends with \n\n (actual newlines)

        # Let's inspect the raw string
        print(f"Chunk: {chunk!r}")

        # If the fix worked, the chunk should end with \n\n (actual newlines)
        # and the JSON content inside should NOT have \\n\\n at the end of the content string
        # unless it was intended.

        # The problematic code was: yield f"data: ...\\n\\n"
        # This produced a string ending with literal \n\n (backslash n backslash n)

        # Correct code is: yield f"data: ...\n\n"
        # This produces a string ending with actual newlines

        assert chunk.endswith(
            "\n\n"
        ), f"Chunk should end with actual newlines: {chunk!r}"
        assert not chunk.endswith(
            "\\n\\n"
        ), f"Chunk should not end with literal escaped newlines: {chunk!r}"

    # Specifically check the execution message
    execution_chunk = next(
        (c for c in think_chunks if "Executing the following tools" in c), None
    )
    assert execution_chunk is not None

    # Parse the JSON to check content
    if execution_chunk.startswith("data: "):
        json_str = execution_chunk[6:].strip()
        data = json.loads(json_str)
        content = data["choices"][0]["delta"]["content"]
        # The content itself should end with \n\n if that was the intention
        assert content.endswith(
            "\n\n"
        ), f"Content should end with newlines: {content!r}"

    # Check success message
    success_chunk = next(
        (c for c in think_chunks if "executed the tools successfully" in c), None
    )
    assert success_chunk is not None
    if success_chunk.startswith("data: "):
        json_str = success_chunk[6:].strip()
        data = json.loads(json_str)
        content = data["choices"][0]["delta"]["content"]
        assert content.endswith(
            "\n\n"
        ), f"Content should end with newlines: {content!r}"
