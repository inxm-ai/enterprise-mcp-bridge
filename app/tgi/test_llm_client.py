import pytest
from unittest.mock import patch, Mock, AsyncMock
import json

from app.tgi.llm_client import LLMClient
from app.tgi.models import ChatCompletionRequest, Message, MessageRole


@pytest.fixture
def llm_client():
    """Create LLMClient instance with test configuration."""
    with patch.dict(
        "os.environ", 
        {"TGI_URL": "https://api.test-llm.com/v1", "TGI_TOKEN": "test-token-123"}
    ):
        return LLMClient()


@pytest.fixture
def llm_client_no_token():
    """Create LLMClient instance without token."""
    with patch.dict(
        "os.environ", 
        {"TGI_URL": "https://api.test-llm.com/v1", "TGI_TOKEN": ""}
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
            "os.environ", 
            {"TGI_URL": "https://api.test.com/", "TGI_TOKEN": ""}
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
            
            def post(self, url, headers, json):
                return MockResponse()

        with patch("app.tgi.llm_client.aiohttp.ClientSession", MockSession):
            response = await llm_client.non_stream_completion(request, "test-token", None)
            
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
            
            def post(self, url, headers, json):
                return MockResponse()

        with patch("app.tgi.llm_client.aiohttp.ClientSession", MockSession):
            with pytest.raises(Exception):  # Should raise HTTPException
                await llm_client.non_stream_completion(request, "test-token", None)

    @pytest.mark.asyncio
    async def test_stream_completion_success(self, llm_client):
        """Test successful streaming LLM completion."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=True,
        )

        mock_chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" there!"}}]}\n\n',
            b'data: [DONE]\n\n'
        ]

        class MockResponse:
            def __init__(self):
                self.ok = True
                self.content = MockAsyncIterator(mock_chunks)

            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        class MockAsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        class MockSession:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            
            def post(self, url, headers, json):
                return MockResponse()

        with patch("app.tgi.llm_client.aiohttp.ClientSession", MockSession):
            chunks = []
            async for chunk in llm_client.stream_completion(request, "test-token", None):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert 'data: {"choices":[{"delta":{"content":"Hello"}}]}' in chunks[0]
            assert 'data: {"choices":[{"delta":{"content":" there!"}}]}' in chunks[1]
            assert 'data: [DONE]' in chunks[2]

    @pytest.mark.asyncio
    async def test_summarize_text(self, llm_client):
        """Test text summarization."""
        base_request = ChatCompletionRequest(
            messages=[],
            model="test-model",
        )
        
        content = "This is a long text that needs to be summarized."
        
        # Mock the stream_completion method
        async def mock_stream_completion(request, access_token, outer_span):
            yield 'data: {"choices":[{"delta":{"content":"Summary: "}}]}\n\n'
            yield 'data: {"choices":[{"delta":{"content":"Short text."}}]}\n\n'
            yield 'data: [DONE]\n\n'
        
        llm_client.stream_completion = mock_stream_completion
        
        result = await llm_client.summarize_text(base_request, content, "test-token", None)
        
        assert result == "Summary: Short text."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
