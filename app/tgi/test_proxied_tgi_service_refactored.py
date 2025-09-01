import pytest
from unittest.mock import Mock, AsyncMock, patch

from app.tgi.proxied_tgi_service import ProxiedTGIService
from app.tgi.models import ChatCompletionRequest, Message, MessageRole


class MockMCPSession:
    """Mock MCP session for testing."""

    def __init__(self):
        pass

    async def list_prompts(self):
        """Mock list_prompts method."""
        mock_result = Mock()
        mock_result.prompts = []
        return mock_result

    async def list_tools(self):
        """Mock list_tools method."""
        return []


@pytest.fixture
def proxied_tgi_service():
    """Create ProxiedTGIService instance."""
    return ProxiedTGIService()


class TestProxiedTGIServiceRefactored:
    """Test cases for the refactored ProxiedTGIService."""

    def test_init(self, proxied_tgi_service):
        """Test service initialization."""
        assert proxied_tgi_service.prompt_service is not None
        assert proxied_tgi_service.tool_service is not None
        assert proxied_tgi_service.llm_client is not None

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming(self, proxied_tgi_service):
        """Test non-streaming chat completion flow."""
        session = MockMCPSession()
        
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=False,
        )

        # Mock the service methods
        mock_response = Mock()
        mock_response.choices = []  # No tool calls, should return immediately
        
        with patch.object(proxied_tgi_service.prompt_service, 'prepare_messages') as mock_prepare:
            with patch.object(proxied_tgi_service.tool_service, 'get_all_mcp_tools') as mock_get_tools:
                with patch.object(proxied_tgi_service.llm_client, 'non_stream_completion') as mock_llm:
                    mock_prepare.return_value = request.messages
                    mock_get_tools.return_value = []
                    mock_llm.return_value = mock_response
                    
                    result = await proxied_tgi_service.chat_completion(session, request)
                    
                    assert result == mock_response
                    mock_prepare.assert_called_once()
                    mock_get_tools.assert_called_once()
                    mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(self, proxied_tgi_service):
        """Test streaming chat completion flow."""
        session = MockMCPSession()
        
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=True,
        )

        # Mock the service methods
        async def mock_stream_generator():
            yield "data: test\n\n"
            yield "data: [DONE]\n\n"
        
        with patch.object(proxied_tgi_service.prompt_service, 'prepare_messages') as mock_prepare:
            with patch.object(proxied_tgi_service.tool_service, 'get_all_mcp_tools') as mock_get_tools:
                with patch.object(proxied_tgi_service, '_stream_chat_with_tools') as mock_stream:
                    mock_prepare.return_value = request.messages
                    mock_get_tools.return_value = []
                    mock_stream.return_value = mock_stream_generator()
                    
                    result = await proxied_tgi_service.chat_completion(session, request)
                    
                    # Result should be an async generator
                    chunks = []
                    async for chunk in result:
                        chunks.append(chunk)
                    
                    assert len(chunks) == 2
                    assert "data: test" in chunks[0]
                    assert "data: [DONE]" in chunks[1]

    @pytest.mark.asyncio
    async def test_stream_chat_with_tools_no_tool_calls(self, proxied_tgi_service):
        """Test streaming chat with no tool calls."""
        session = MockMCPSession()
        messages = [Message(role=MessageRole.USER, content="Hello")]
        available_tools = []
        
        request = ChatCompletionRequest(
            messages=messages,
            model="test-model",
            stream=True,
        )

        # Mock LLM stream that returns content only
        async def mock_llm_stream():
            yield 'data: {"choices":[{"delta":{"content":"Hello!"}}]}\n\n'
            yield 'data: [DONE]\n\n'

        with patch.object(proxied_tgi_service.llm_client, 'stream_completion') as mock_stream:
            mock_stream.return_value = mock_llm_stream()
            
            chunks = []
            async for chunk in proxied_tgi_service._stream_chat_with_tools(
                session, messages, available_tools, request, "token", None
            ):
                chunks.append(chunk)
            
            # Should yield the content and final [DONE]
            assert len(chunks) >= 2
            assert any("Hello!" in chunk for chunk in chunks)
            assert any("[DONE]" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_non_stream_chat_with_tools_no_tool_calls(self, proxied_tgi_service):
        """Test non-streaming chat with no tool calls."""
        session = MockMCPSession()
        messages = [Message(role=MessageRole.USER, content="Hello")]
        available_tools = []
        
        request = ChatCompletionRequest(
            messages=messages,
            model="test-model",
            stream=False,
        )

        # Mock response with no tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = None

        with patch.object(proxied_tgi_service.llm_client, 'non_stream_completion') as mock_llm:
            mock_llm.return_value = mock_response
            
            result = await proxied_tgi_service._non_stream_chat_with_tools(
                session, messages, available_tools, request, "token", None
            )
            
            assert result == mock_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
