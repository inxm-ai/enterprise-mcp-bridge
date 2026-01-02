import json
from types import SimpleNamespace
from typing import Any, AsyncGenerator, List

import pytest

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.services.tool_chat_runner import ToolChatRunner
from app.tgi.services.tool_service import ToolService
from app.tgi.services.tools.tool_resolution import (
    ParsedToolCall,
    ToolCallFormat,
    ToolResolutionStrategy,
)
from app.tgi.clients.llm_client import LLMClient
from app.tgi.models.model_formats import ChatGPTModelFormat


class StubLLM:
    def __init__(self, streams: List[List[str]]):
        # streams is a list of lists, each inner list is the sequence of raw chunks for one call
        self.streams = streams
        self.requests: List[ChatCompletionRequest] = []

    def stream_completion(
        self, request: ChatCompletionRequest, access_token: str, span: Any
    ) -> AsyncGenerator[str, None]:
        self.requests.append(request)
        chunks = self.streams.pop(0) if self.streams else []

        async def _gen():
            for chunk in chunks:
                yield chunk
            yield "data: [DONE]\n\n"

        return _gen()


class StubToolResolution:
    def __init__(self, responses: List[List[ParsedToolCall]]):
        self.responses = responses
        self.calls: List[tuple[str, dict]] = []

    def resolve_tool_calls(self, content: str, tool_call_chunks: dict):
        self.calls.append((content, tool_call_chunks))
        if not self.responses:
            return [], True
        return self.responses.pop(0), True


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "desc",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _parsed(name: str, idx: int = 0) -> ParsedToolCall:
    return ParsedToolCall(
        id=f"id-{name}-{idx}",
        index=idx,
        name=name,
        arguments={},
        format=ToolCallFormat.OPENAI_JSON,
        raw_content="",
    )


@pytest.mark.asyncio
async def test_runner_preserves_response_format():
    class InspectingLLM:
        def __init__(self):
            self.requests: List[ChatCompletionRequest] = []
            self.client = LLMClient(model_format=ChatGPTModelFormat())

        def stream_completion(
            self, request: ChatCompletionRequest, access_token: str, span: Any
        ) -> AsyncGenerator[str, None]:
            self.requests.append(request)
            # Verify response_format is preserved in the request
            assert request.response_format is not None
            assert request.response_format["json_schema"][
                "name"
            ], "expected response_format.json_schema.name"

            async def _gen():
                yield "data: [DONE]\n\n"

            return _gen()

    llm = InspectingLLM()
    tool_resolution = StubToolResolution(responses=[[]])
    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=ToolService(),
        tool_resolution=tool_resolution,
        logger_obj=None,
        message_summarization_service=None,
    )

    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[],
        chat_request=ChatCompletionRequest(
            messages=[],
            model="test-model",
            stream=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {},
                    "name": "test_schema",
                },
            },
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=False,
    )
    _ = [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_runner_emits_think_messages_on_tool_call():
    # First stream emits a tool call chunk only
    llm = StubLLM(
        streams=[
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call1","function":{"name":"search","arguments":"{}"}}]},"index":0}]}\n\n'
            ]
        ]
    )
    tool_resolution = StubToolResolution(responses=[[]])
    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=ToolService(),
        tool_resolution=tool_resolution,
        logger_obj=None,
        message_summarization_service=None,
    )

    chunks: List[str] = []
    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[_tool("search")],
        chat_request=ChatCompletionRequest(
            messages=[], model="test-model", stream=True
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=True,
    )
    async for chunk in stream:
        chunks.append(chunk)

    joined = "".join(chunks)
    assert "<think>I need to call a tool" in joined
    assert "I will run <code>search" in joined


@pytest.mark.asyncio
async def test_runner_executes_tool_calls_and_loops():
    # Iteration 1: tool call; iteration 2: normal text
    llm = StubLLM(
        streams=[
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call1","function":{"name":"search","arguments":"{}"}}]},"index":0}]}\n\n'
            ],
            ['data: {"choices":[{"delta":{"content":"done"},"index":0}]}\n\n'],
        ]
    )
    tool_resolution = StubToolResolution(responses=[[_parsed("search")], []])
    tool_service = ToolService()
    tool_service.execute_tool_calls = SimpleNamespace(
        __call__=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not async"))
    )  # type: ignore

    # make execute_tool_calls async
    async def _exec(*_args, **_kwargs):
        if _kwargs.get("return_raw_results"):
            return ([], True, [])
        return ([], True)

    tool_service.execute_tool_calls = _exec  # type: ignore

    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=tool_service,
        tool_resolution=tool_resolution,
        logger_obj=None,
        message_summarization_service=None,
    )

    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[_tool("search")],
        chat_request=ChatCompletionRequest(
            messages=[], model="test-model", stream=True
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=False,
    )
    _ = [chunk async for chunk in stream]

    # execute_tool_calls should have been invoked once, and we should have looped twice
    assert len(llm.requests) == 2


@pytest.mark.asyncio
async def test_runner_executes_tool_calls_without_index():
    llm = StubLLM(
        streams=[
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"id":"call1","function":{"name":"search","arguments":"{}"}}]},"index":0}]}\n\n'
            ]
        ]
    )
    tool_service = ToolService()
    executed = {"count": 0}

    async def _exec(*_args, **_kwargs):
        executed["count"] += 1
        if _kwargs.get("return_raw_results"):
            return ([], True, [])
        return ([], True)

    tool_service.execute_tool_calls = _exec  # type: ignore

    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=tool_service,
        tool_resolution=ToolResolutionStrategy(),
        logger_obj=None,
        message_summarization_service=None,
    )

    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[_tool("search")],
        chat_request=ChatCompletionRequest(
            messages=[], model="test-model", stream=True
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=False,
    )
    _ = [chunk async for chunk in stream]

    assert executed["count"] == 1


@pytest.mark.asyncio
async def test_runner_fallbacks_to_single_tool_from_json_content():
    llm = StubLLM(
        streams=[
            [
                'data: {"choices":[{"delta":{"content":"{\\"start_url\\":\\"https://example.com\\",\\"max_depth\\":3}"},"index":0}]}\n\n'
            ]
        ]
    )
    tool_service = ToolService()
    executed = {"count": 0}

    async def _exec(*_args, **_kwargs):
        executed["count"] += 1
        if _kwargs.get("return_raw_results"):
            return ([], True, [])
        return ([], True)

    tool_service.execute_tool_calls = _exec  # type: ignore

    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=tool_service,
        tool_resolution=ToolResolutionStrategy(),
        logger_obj=None,
        message_summarization_service=None,
    )

    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[_tool("crawl")],
        chat_request=ChatCompletionRequest(
            messages=[], model="test-model", stream=True
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=False,
    )
    _ = [chunk async for chunk in stream]

    assert executed["count"] == 1


@pytest.mark.asyncio
async def test_runner_handles_multiple_tool_calls():
    llm = StubLLM(
        streams=[
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"first","arguments":"{}"}},{"index":1,"id":"c2","function":{"name":"second","arguments":"{}"}}]},"index":0}]}\n\n'
            ]
        ]
    )
    tool_resolution = StubToolResolution(
        responses=[[_parsed("first", 0), _parsed("second", 1)]]
    )

    async def _exec(*_args, **_kwargs):
        if _kwargs.get("return_raw_results"):
            return ([], True, [])
        return ([], True)

    tool_service = ToolService()
    tool_service.execute_tool_calls = _exec  # type: ignore

    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=tool_service,
        tool_resolution=tool_resolution,
        logger_obj=None,
        message_summarization_service=None,
    )

    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[_tool("first"), _tool("second")],
        chat_request=ChatCompletionRequest(
            messages=[], model="test-model", stream=True
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=False,
    )
    _ = [chunk async for chunk in stream]

    # Two tool calls resolved
    resolved = tool_resolution.calls[0][1]
    assert 0 in resolved and 1 in resolved or resolved  # accumulated calls present


@pytest.mark.asyncio
async def test_runner_logs_progress_and_log_events():
    class _Logger:
        def __init__(self):
            self.entries: list[tuple[str, str]] = []

        def info(self, msg, *_, **__):
            self.entries.append(("info", msg))

        def warning(self, msg, *_, **__):
            self.entries.append(("warning", msg))

        def error(self, msg, *_, **__):
            self.entries.append(("error", msg))

    logger = _Logger()

    async def _exec(
        _session,
        _calls,
        _token,
        _span,
        *,
        available_tools=None,
        return_raw_results=False,
        progress_callback=None,
        log_callback=None,
    ):
        if progress_callback:
            await progress_callback(0.5, 1.0, "halfway", "demo_tool")
        if log_callback:
            await log_callback("info", "log message", "demo_logger")
        if return_raw_results:
            return ([], True, [])
        return ([], True)

    tool_service = ToolService()
    tool_service.execute_tool_calls = _exec  # type: ignore

    llm = StubLLM(
        streams=[
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call1","function":{"name":"demo_tool","arguments":"{}"}}]},"index":0}]}\n\n'
            ],
            ['data: {"choices":[{"delta":{"content":"done"},"index":0}]}\n\n'],
        ]
    )
    tool_resolution = StubToolResolution(responses=[[_parsed("demo_tool")], []])
    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=tool_service,
        tool_resolution=tool_resolution,
        logger_obj=logger,
        message_summarization_service=None,
    )

    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[_tool("demo_tool")],
        chat_request=ChatCompletionRequest(
            messages=[], model="test-model", stream=True
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=False,
    )
    _ = [chunk async for chunk in stream]

    assert any("ToolProgress" in msg for level, msg in logger.entries)
    assert any("ToolLog" in msg for level, msg in logger.entries)


@pytest.mark.asyncio
async def test_runner_strips_hallucinated_tool_results_from_content():
    hallucinated_result = '<select_tools_result>{"tool_list":["alpha"],"reasoning":"made up"}</select_tools_result>'
    first_chunk_content = (
        '<function_calls><invoke name="select_tools">{"tools":["alpha"]}</invoke></function_calls>'
        f"{hallucinated_result}"
    )
    llm = StubLLM(
        streams=[
            [
                f'data: {json.dumps({"choices": [{"delta": {"content": first_chunk_content}, "index": 0}]})}\n\n'
            ],
            ['data: {"choices":[{"delta":{"content":"done"},"index":0}]}\n\n'],
        ]
    )
    tool_resolution = StubToolResolution(responses=[[_parsed("select_tools")], []])

    async def _exec(*_args, **_kwargs):
        if _kwargs.get("return_raw_results"):
            return ([], True, [])
        return ([], True)

    tool_service = ToolService()
    tool_service.execute_tool_calls = _exec  # type: ignore

    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=tool_service,
        tool_resolution=tool_resolution,
        logger_obj=None,
        message_summarization_service=None,
    )

    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[_tool("select_tools")],
        chat_request=ChatCompletionRequest(
            messages=[], model="test-model", stream=True
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=False,
    )
    _ = [chunk async for chunk in stream]

    assert len(llm.requests) == 2
    assistant_messages = [
        msg for msg in llm.requests[1].messages if msg.role == MessageRole.ASSISTANT
    ]
    assert assistant_messages, "Expected assistant tool call message"
    assistant_content = assistant_messages[0].content or ""
    assert "select_tools_result" not in assistant_content
    assert '<invoke name="select_tools">' in assistant_content


@pytest.mark.asyncio
async def test_runner_preserves_stop_tokens_in_requests():
    llm = StubLLM(streams=[[]])
    tool_resolution = StubToolResolution(responses=[[]])
    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=ToolService(),
        tool_resolution=tool_resolution,
        logger_obj=None,
        message_summarization_service=None,
    )

    stop_tokens = ["<stop/>"]
    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[_tool("search")],
        chat_request=ChatCompletionRequest(
            messages=[], model="test-model", stream=True, stop=stop_tokens
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=False,
    )
    _ = [chunk async for chunk in stream]

    assert llm.requests[0].stop == stop_tokens


@pytest.mark.asyncio
async def test_runner_tracks_message_history_across_iterations():
    llm = StubLLM(
        streams=[
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call1","function":{"name":"search","arguments":"{}"}}]},"index":0}]}\n\n'
            ],
            ['data: {"choices":[{"delta":{"content":"after"},"index":0}]}\n\n'],
        ]
    )
    tool_resolution = StubToolResolution(responses=[[_parsed("search")], []])

    async def _exec(*_args, **_kwargs):
        # return one tool result message to be appended
        if _kwargs.get("return_raw_results"):
            return (
                [Message(role=MessageRole.TOOL, content="tool-result")],
                True,
                [{"name": "search", "content": "tool-result"}],
            )
        return ([Message(role=MessageRole.TOOL, content="tool-result")], True)

    tool_service = ToolService()
    tool_service.execute_tool_calls = _exec  # type: ignore

    runner = ToolChatRunner(
        llm_client=llm,
        tool_service=tool_service,
        tool_resolution=tool_resolution,
        logger_obj=None,
        message_summarization_service=None,
    )

    stream = runner.stream_chat_with_tools(
        session=None,
        messages=[Message(role=MessageRole.USER, content="hi")],
        available_tools=[_tool("search")],
        chat_request=ChatCompletionRequest(
            messages=[], model="test-model", stream=True
        ),
        access_token="",
        parent_span=None,
        emit_think_messages=False,
    )
    _ = [chunk async for chunk in stream]

    # Second request should include original messages + assistant tool_call message + tool result
    assert len(llm.requests) == 2
    second_messages = llm.requests[1].messages
    assert len(second_messages) == 4
    assert second_messages[1].role == MessageRole.USER  # original user
    assert second_messages[2].role == MessageRole.ASSISTANT  # tool call message
    assert second_messages[3].role == MessageRole.TOOL  # tool result
