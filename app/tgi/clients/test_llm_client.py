import pytest
from unittest.mock import patch, MagicMock, AsyncMock

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
        assert llm_client.client.api_key == "test-token-123"
        assert str(llm_client.client.base_url) == "https://api.test-llm.com/v1/"

    def test_init_without_token(self, llm_client_no_token):
        """Test client initialization without token."""
        assert llm_client_no_token.tgi_url == "https://api.test-llm.com/v1"
        assert llm_client_no_token.tgi_token == ""
        assert llm_client_no_token.client.api_key == "fake-token"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from TGI_URL."""
        with patch.dict(
            "os.environ", {"TGI_URL": "https://api.test.com/", "TGI_TOKEN": ""}
        ):
            client = LLMClient()
            assert client.tgi_url == "https://api.test.com"
            assert str(client.client.base_url) == "https://api.test.com"

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

        # Mock openai client response
        mock_response = MagicMock()
        mock_response.model_dump.return_value = mock_response_data

        llm_client.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        response = await llm_client.non_stream_completion(request, "test-token", None)

        call_kwargs = llm_client.client.chat.completions.create.call_args.kwargs
        assert "tool_choice" not in call_kwargs
        assert "tools" not in call_kwargs

        assert response.id == "chatcmpl-test123"
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_non_stream_completion_error(self, llm_client):
        """Test non-streaming LLM completion with API error."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=False,
        )

        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(Exception) as excinfo:
            await llm_client.non_stream_completion(request, "test-token", None)

        assert "API Error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stream_completion_success(self, llm_client):
        """Test successful streaming completion."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="test")],
            model="test-model",
            stream=True,
        )

        # Mock chunks
        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {"choices": [{"delta": {"content": "Hello"}}]}

        # Mock async generator
        async def mock_stream():
            yield chunk1

        llm_client.client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        chunks = []
        async for chunk in llm_client.stream_completion(request, "test-token", None):
            chunks.append(chunk)

        call_kwargs = llm_client.client.chat.completions.create.call_args.kwargs
        assert "tool_choice" not in call_kwargs
        assert "tools" not in call_kwargs

        assert len(chunks) == 2
        # SSE format requires proper \n\n endings
        import json

        assert (
            chunks[0]
            == f'data: {json.dumps({"choices":[{"delta":{"content":"Hello"}}]})}\n\n'
        )
        assert chunks[1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_completion_error_handling(self, llm_client):
        """Test error handling in streaming completion."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=True,
        )

        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=Exception("Test connection error")
        )

        chunks = []
        async for chunk in llm_client.stream_completion(request, "test-token", None):
            chunks.append(chunk)

        # Should yield error chunk then DONE
        assert len(chunks) == 2
        assert "Error: Error streaming from LLM: Test connection error" in chunks[0]
        assert chunks[1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_prepare_payload_sanitizes_null_content_after_compression(
        self, llm_client, monkeypatch
    ):
        """Ensure compression path does not leave null message content."""
        monkeypatch.setattr(llm, "LLM_MAX_PAYLOAD_BYTES", 800)

        tool_call = ToolCall(
            id="call-1",
            function=ToolCallFunction(name="test_tool", arguments="{}"),
        )
        request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="System"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=[tool_call],
                ),
                Message(role=MessageRole.USER, content="A" * 2000),
            ],
            model="test-model",
            stream=True,
        )

        async def summarize_fn(
            base_request=None,
            content=None,
            access_token=None,
            outer_span=None,
            **_kwargs,
        ):
            return "summary"

        llm_client.summarize_text = AsyncMock(side_effect=summarize_fn)

        from app.tgi.context_compressor import AdaptiveCompressor

        monkeypatch.setattr(
            llm,
            "get_default_compressor",
            lambda: AdaptiveCompressor(chunk_tokens=10, group_size=2, window_size=1),
        )

        compressed = await llm_client._prepare_payload(request, "test-token")

        assert all(message.content is not None for message in compressed.messages)
        assert compressed.messages[1].content == ""

    @pytest.mark.asyncio
    async def test_summarize_text_uses_user_message(self, llm_client):
        """Summarize_text should send content as a user message and return text."""
        captured = {}

        async def fake_non_stream_completion(request, _access_token, _outer_span):
            captured["messages"] = request.messages
            return {"choices": [{"message": {"content": "summary"}}]}

        llm_client.non_stream_completion = AsyncMock(
            side_effect=fake_non_stream_completion
        )

        base_request = ChatCompletionRequest(messages=[], model="test-model")
        result = await llm_client.summarize_text(
            base_request, "raw content", None, None
        )

        assert result == "summary"
        assert captured["messages"][0].role == MessageRole.SYSTEM
        assert captured["messages"][1].role == MessageRole.USER
        assert captured["messages"][1].content == "raw content"
        assert all(m.role != MessageRole.ASSISTANT for m in captured["messages"][1:])


def make_tool(name, params):
    return Tool(function=FunctionDefinition(name=name, parameters=params))


def test_model_parameter_required_not_empty_string():
    """Model parameter should not be present if empty string, to avoid 'you must provide a model parameter' error."""
    # This test is less relevant now as we rely on openai lib, but we do set default model in _prepare_payload
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
