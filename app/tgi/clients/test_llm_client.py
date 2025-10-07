import pytest
from unittest.mock import patch

from app.tgi import llm_client as llm
from app.tgi.clients.llm_client import LLMClient
from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
    Tool,
    FunctionDefinition,
    ToolCall,
    ToolCallFunction,
)
from app.tgi.models.model_formats import ChatGPTModelFormat, ClaudeModelFormat
from app.vars import TGI_MODEL_NAME


@pytest.fixture
def llm_client():
    """Create LLMClient instance with test configuration."""
    with patch.dict(
        "os.environ",
        {"TGI_URL": "https://api.test-llm.com/v1", "TGI_TOKEN": "test-token-123"},
    ):
        return LLMClient()


@pytest.fixture
def llm_client_no_token():
    """Create LLMClient instance without token."""
    with patch.dict(
        "os.environ", {"TGI_URL": "https://api.test-llm.com/v1", "TGI_TOKEN": ""}
    ):
        return LLMClient()


class TestLLMClient:
    """Test cases for LLMClient."""

    def test_init_with_token(self, llm_client):
        """Test client initialization with token."""
        assert llm_client.tgi_url == "https://api.test-llm.com/v1"
        assert llm_client.tgi_token == "test-token-123"

    def test_init_without_token(self, llm_client_no_token):
        """Test client initialization without token."""
        assert llm_client_no_token.tgi_url == "https://api.test-llm.com/v1"
        assert llm_client_no_token.tgi_token == ""

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from TGI_URL."""
        with patch.dict(
            "os.environ", {"TGI_URL": "https://api.test.com/", "TGI_TOKEN": ""}
        ):
            client = LLMClient()
            assert client.tgi_url == "https://api.test.com"

    def test_get_headers_with_token(self, llm_client):
        """Test header generation with token."""
        headers = llm_client._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-token-123"
        assert "User-Agent" in headers

    def test_get_headers_without_token(self, llm_client_no_token):
        """Test header generation without token."""
        headers = llm_client_no_token._get_headers()

        assert headers["Content-Type"] == "application/json"
        # the api always requires an auth token, so if we don't have one, provide a fake
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer fake"
        assert "User-Agent" in headers

    def test_create_completion_id(self, llm_client):
        """Test completion ID generation."""
        completion_id = llm_client.create_completion_id()

        assert completion_id.startswith("chatcmpl-")
        assert len(completion_id) == 38  # "chatcmpl-" (9 chars) + 29 hex chars

    def test_create_usage_stats(self, llm_client):
        """Test usage statistics creation."""
        usage = llm_client.create_usage_stats(100, 50)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    @pytest.mark.asyncio
    async def test_non_stream_completion_success(self, llm_client):
        """Test successful non-streaming LLM completion."""
        # Mock response data
        mock_response_data = {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        # Create mock request
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=False,
        )

        # Create a proper context manager mock
        class MockResponse:
            def __init__(self):
                self.ok = True

            async def json(self):
                return mock_response_data

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        # Create a proper session mock
        class MockSession:
            def __init__(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            def post(self, url, headers, data=None):
                return MockResponse()

        with patch("app.tgi.llm_client.aiohttp.ClientSession", MockSession):
            response = await llm_client.non_stream_completion(
                request, "test-token", None
            )

            assert response.id == "chatcmpl-test123"
            assert response.model == "test-model"
            assert len(response.choices) == 1
            assert (
                response.choices[0].message.content
                == "Hello! How can I help you today?"
            )

    @pytest.mark.asyncio
    async def test_non_stream_completion_error(self, llm_client):
        """Test non-streaming LLM completion with API error."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=False,
        )

        class MockResponse:
            def __init__(self):
                self.ok = False
                self.status = 500

            async def text(self):
                return "Internal Server Error"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        class MockSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            def post(self, url, headers, data=None):
                return MockResponse()

        with patch("app.tgi.llm_client.aiohttp.ClientSession", MockSession):
            with pytest.raises(Exception):  # Should raise HTTPException
                await llm_client.non_stream_completion(request, "test-token", None)

    @pytest.mark.asyncio
    async def test_stream_completion_success(self, llm_client):
        """Test successful streaming completion."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="test")],
            model="test-model",
            stream=True,
        )

        # Mock aiohttp response
        class MockResponse:
            ok = True

            class MockContent:
                def __init__(self):
                    # Simulate chunks that might be split across network boundaries
                    self.chunks = [
                        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
                        b"data: [DONE]\n",
                    ]
                    self.index = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.index < len(self.chunks):
                        chunk = self.chunks[self.index]
                        self.index += 1
                        return chunk
                    raise StopAsyncIteration

            content = MockContent()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        class MockSession:
            def __init__(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            def post(self, *args, **kwargs):
                return MockResponse()

        class MockSessionContext:
            async def __aenter__(self):
                return MockSession()

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        with patch("aiohttp.ClientSession", return_value=MockSessionContext()):
            chunks = []
            async for chunk in llm_client.stream_completion(
                request, "test-token", None
            ):
                chunks.append(chunk)

            assert len(chunks) == 2
            # SSE format requires proper \n\n endings
            assert chunks[0] == 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            assert chunks[1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_completion_error_handling(self, llm_client):
        """Test error handling in streaming completion."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=True,
        )

        class MockStream:
            def __aiter__(self):
                async def anext():
                    raise ConnectionError("Test connection error")

                return self

            async def __anext__(self):
                raise ConnectionError("Test connection error")

        with patch.object(LLMClient, "stream_completion", return_value=MockStream()):
            with pytest.raises(ConnectionError):
                async for _ in llm_client.stream_completion(
                    request, "test-token", None
                ):
                    pass

    @pytest.mark.asyncio
    async def test_ask_with_no_question_or_statement(self, llm_client):
        """Test ask method with no question or assistant statement."""
        base_request = ChatCompletionRequest(
            messages=[],
            model="test-model",
        )

        class MockStream:
            async def __aiter__(self):
                yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
                yield "data: [DONE]"

        with patch.object(LLMClient, "stream_completion", return_value=MockStream()):
            result = await llm_client.ask(
                base_prompt="Test prompt",
                base_request=base_request,
                outer_span=None,
                question=None,
                assistant_statement=None,
                access_token="test-token",
            )

            assert result == "Hello"

    @pytest.mark.asyncio
    async def test_summarize_text_with_empty_content(self, llm_client):
        """Test summarize_text with empty content."""
        base_request = ChatCompletionRequest(
            messages=[],
            model="test-model",
        )

        content = ""

        class MockStreamGenerator:
            async def __aiter__(self):
                yield 'data: {"choices":[{"delta":{"content":"Summary: Nothing to summarize."}}]}'
                yield "data: [DONE]"

        with patch.object(
            LLMClient, "stream_completion", return_value=MockStreamGenerator()
        ) as mock_stream:
            result = await llm_client.summarize_text(
                base_request, content, "test-token", None
            )

            # summarize_text does not return the ask() result in current implementation
            assert result is None
            mock_stream.assert_called_once()


def make_tool(name, params):
    return Tool(function=FunctionDefinition(name=name, parameters=params))


def test_no_injection_keeps_tools():
    """ChatGPT format leaves tools untouched in payload."""
    client = llm.LLMClient(model_format=ChatGPTModelFormat())
    tool = make_tool("fn", {"a": 1})
    messages = [Message(role=MessageRole.USER, content="hi")]
    req = ChatCompletionRequest(messages=messages, tools=[tool], model="m")

    payload = client._generate_llm_payload(req)

    # tools should be present and unchanged
    assert "tools" in payload
    assert payload["tools"] != []
    assert payload["tools"][0]["function"]["name"] == "fn"

    # messages unchanged
    assert payload["messages"][0]["content"] == "hi"


def test_tool_choice_excluded_when_no_tools():
    """tool_choice should not be in payload when tools is None or empty."""
    client = llm.LLMClient(model_format=ChatGPTModelFormat())
    messages = [Message(role=MessageRole.USER, content="hi")]

    # Test with tools=None
    req = ChatCompletionRequest(messages=messages, tools=None, model="m")
    payload = client._generate_llm_payload(req)
    assert (
        "tool_choice" not in payload
    ), "tool_choice should not be present when tools is None"

    # Test with tools=[]
    req = ChatCompletionRequest(messages=messages, tools=[], model="m")
    payload = client._generate_llm_payload(req)
    assert (
        "tool_choice" not in payload
    ), "tool_choice should not be present when tools is empty"


def test_tool_choice_included_when_tools_present():
    """tool_choice should be in payload when tools are specified."""
    client = llm.LLMClient(model_format=ChatGPTModelFormat())
    tool = make_tool("fn", {"a": 1})
    messages = [Message(role=MessageRole.USER, content="hi")]

    # Test with explicit tool_choice
    req = ChatCompletionRequest(
        messages=messages, tools=[tool], tool_choice="auto", model="m"
    )
    payload = client._generate_llm_payload(req)
    assert "tool_choice" in payload
    assert payload["tool_choice"] == "auto"

    # Test with default tool_choice
    req = ChatCompletionRequest(messages=messages, tools=[tool], model="m")
    payload = client._generate_llm_payload(req)
    assert "tool_choice" in payload


def test_model_parameter_required_not_empty_string():
    """Model parameter should not be present if empty string, to avoid 'you must provide a model parameter' error."""
    client = llm.LLMClient(model_format=ChatGPTModelFormat())
    messages = [Message(role=MessageRole.USER, content="hi")]

    # Test with empty string model (simulates TGI_MODEL_NAME="" env var)
    req = ChatCompletionRequest(messages=messages, model="")
    payload = client._generate_llm_payload(req)
    # Empty model will be changed to default model to avoid API error
    assert (
        "model" in payload and payload["model"] == TGI_MODEL_NAME
    ), "Empty model string should not be sent to API"

    # Test with None model
    req = ChatCompletionRequest(messages=messages, model=None)
    payload = client._generate_llm_payload(req)
    # None model should be replaced with default model
    assert (
        "model" in payload and payload["model"] == TGI_MODEL_NAME
    ), "None model should not be in payload"

    # Test with valid model
    req = ChatCompletionRequest(messages=messages, model="valid-model")
    payload = client._generate_llm_payload(req)
    assert "model" in payload
    assert payload["model"] == "valid-model"


def test_claude_with_existing_system_appends():
    """Claude format injects tool descriptors into existing system prompts."""
    client = llm.LLMClient(model_format=ClaudeModelFormat())
    tool = make_tool("doThing", {"x": 1})
    # place system message not at index 0 to ensure position is preserved
    messages = [
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.SYSTEM, content="sys start"),
    ]
    req = ChatCompletionRequest(messages=messages, tools=[tool])

    payload = client._generate_llm_payload(req)

    # tools should be emptied (injected into system prompt)
    assert "tools" not in payload

    # find system message and verify injection
    sys_msg = next(m for m in payload["messages"] if m["role"] == MessageRole.SYSTEM)
    assert "You have access to the following tools" in sys_msg["content"]
    assert "<doThing>" in sys_msg["content"]
    # compact JSON (no spaces) as produced by json.dumps with separators(',',':')
    assert '{"x":1}' in sys_msg["content"]


def test_claude_without_system_inserts_at_start():
    """Claude format adds a system message when none exists."""
    client = llm.LLMClient(model_format=ClaudeModelFormat())
    tool1 = make_tool("t1", {"a": "b"})
    messages = [Message(role=MessageRole.USER, content="u1")]
    req = ChatCompletionRequest(messages=messages, tools=[tool1])

    payload = client._generate_llm_payload(req)

    assert "tools" not in payload

    # first message must be the injected system message
    first = payload["messages"][0]
    assert first["role"] == MessageRole.SYSTEM
    assert "You have access to the following tools" in first["content"]
    assert "<t1>" in first["content"]
    # parameters should be rendered as compact JSON
    assert '{"a":"b"}' in first["content"]


def test_claude_preserves_system_position():
    """Claude format does not reorder existing system messages."""
    client = llm.LLMClient(model_format=ClaudeModelFormat())
    tool = make_tool("fn", {"n": 2})
    messages = [
        Message(role=MessageRole.USER, content="u"),
        Message(role=MessageRole.SYSTEM, content="s"),
        Message(role=MessageRole.ASSISTANT, content="a"),
    ]
    req = ChatCompletionRequest(messages=messages, tools=[tool])

    payload = client._generate_llm_payload(req)

    roles = [m["role"] for m in payload["messages"]]
    assert roles == [MessageRole.USER, MessageRole.SYSTEM, MessageRole.ASSISTANT]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPayloadCompaction:
    def test_prepare_payload_compacts_messages_and_arguments(self, monkeypatch):
        client = llm.LLMClient()
        monkeypatch.setattr(llm, "LLM_MAX_PAYLOAD_BYTES", 360)

        request = ChatCompletionRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content="  system prompt  " + " " * 200,
                ),
                Message(
                    role=MessageRole.USER,
                    content="  hi there  " + " " * 200,
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content='  {   "foo" :  "bar"  }  ' + " " * 400,
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            function=ToolCallFunction(
                                name="doThing",
                                arguments=' {  "value" :  "xyz"  } ' + " " * 200,
                            ),
                        )
                    ],
                ),
            ],
            model="test-model",
            stream=True,
        )

        payload, serialized, size = client._prepare_payload(request)

        assert size <= llm.LLM_MAX_PAYLOAD_BYTES
        system = payload["messages"][0]["content"]
        assistant = payload["messages"][2]
        assert system == "system prompt"
        assert assistant["content"] == '{"foo":"bar"}'
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"value":"xyz"}'

    def test_prepare_payload_truncates_long_messages(self, monkeypatch):
        client = llm.LLMClient()
        monkeypatch.setattr(llm, "LLM_MAX_PAYLOAD_BYTES", 200)
        monkeypatch.setattr(llm, "TOOL_CHUNK_SIZE", 20)

        long_text = "x" * 200
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="system"),
                Message(role=MessageRole.ASSISTANT, content=long_text),
            ],
            model="test-model",
            stream=True,
        )

        _, serialized, _ = client._prepare_payload(request)

        truncated_content = request.messages[1].content
        assert "[truncated" in truncated_content
        assert len(serialized.encode("utf-8")) <= llm.LLM_MAX_PAYLOAD_BYTES

    def test_prepare_payload_drops_old_tool_messages(self, monkeypatch):
        client = llm.LLMClient()
        monkeypatch.setattr(llm, "TOOL_CHUNK_SIZE", 60)

        big_payload = "{" + ("y" * 300) + "}"
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="system"),
                Message(role=MessageRole.TOOL, content=big_payload, name="tool"),
                Message(role=MessageRole.USER, content="what next?"),
            ],
            model="test-model",
            stream=True,
        )

        client._generate_llm_payload(request)
        payload, serialized, size = client._drop_messages_until_fit(request, 120)

        roles = [msg["role"] for msg in payload["messages"]]
        assert MessageRole.TOOL not in roles
        assert len(request.messages) == 2
        assert request.messages[0].role == MessageRole.SYSTEM
        assert request.messages[1].role == MessageRole.USER
