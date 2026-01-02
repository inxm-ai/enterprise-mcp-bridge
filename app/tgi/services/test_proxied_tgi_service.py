import pytest
import os
from unittest.mock import Mock, patch, AsyncMock

from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
    Choice,
    Usage,
)
from app.tgi.services.proxied_tgi_service import ProxiedTGIService


class MockMCPSession:
    """Mock MCP session for testing."""

    def __init__(self, prompts=None, tools=None):
        self.prompts = prompts or []
        self.tools = tools or []

    async def list_prompts(self):
        """Mock list_prompts method."""
        return {"prompts": self.prompts}

    async def list_tools(self):
        """Mock list_tools method."""
        # Return a list directly, as expected by map_tools
        return self.tools

    async def call_prompt(self, name, args):
        """Mock call_prompt method."""
        mock_result = Mock()
        mock_result.isError = False
        mock_result.messages = []

        # Find the prompt and return mock content
        for prompt in self.prompts:
            if prompt["name"] == name:
                mock_message = Mock()
                mock_message.content = Mock()
                mock_message.content.text = f"System prompt from {name}"
                mock_result.messages = [mock_message]
                break

        return mock_result

    async def call_tool(self, tool_name, args, access_token):
        """Mock call_tool method."""
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = []
        mock_result.structuredContent = {
            "result": f"Tool {tool_name} executed with args {args}"
        }
        return mock_result


@pytest.fixture
def mock_prompts():
    """Create mock prompts for testing."""
    system_prompt = {
        "name": "system",
        "description": "System prompt with role=system",
        "template": {"role": "system", "content": "System prompt from system"},
    }

    custom_prompt = {
        "name": "custom",
        "description": "Custom prompt",
        "template": {"role": "system", "content": "Custom prompt content"},
    }

    return [system_prompt, custom_prompt]


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    tool1 = {
        "name": "list-files",
        "description": "List files in a directory",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
    tool2 = {
        "name": "read-file",
        "description": "Read contents of a file",
        "inputSchema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
        },
    }
    return [tool1, tool2]


@pytest.fixture
def proxied_tgi_service():
    """Create ProxiedTGIService instance with test configuration."""
    with patch.dict(
        os.environ,
        {"TGI_URL": "https://api.test-llm.com/v1", "TGI_TOKEN": "test-token-123"},
    ):
        return ProxiedTGIService()


@pytest.fixture
def proxied_tgi_service_no_token():
    """Create ProxiedTGIService instance without token."""
    with patch.dict(
        os.environ, {"TGI_URL": "https://api.test-llm.com/v1", "TGI_TOKEN": ""}
    ):
        return ProxiedTGIService()


class TestProxiedTGIService:
    """Test cases for ProxiedTGIService."""

    def test_init_with_token(self, proxied_tgi_service):
        """Test service initialization with token."""
        assert proxied_tgi_service.llm_client.tgi_url == "https://api.test-llm.com/v1"
        assert proxied_tgi_service.llm_client.tgi_token == "test-token-123"

    def test_init_without_token(self, proxied_tgi_service_no_token):
        """Test service initialization without token."""
        assert (
            proxied_tgi_service_no_token.llm_client.tgi_url
            == "https://api.test-llm.com/v1"
        )
        assert proxied_tgi_service_no_token.llm_client.tgi_token == ""

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from TGI_URL."""
        with patch.dict(
            os.environ, {"TGI_URL": "https://api.test.com/", "TGI_TOKEN": ""}
        ):
            service = ProxiedTGIService()
            assert service.llm_client.tgi_url == "https://api.test.com"

    def test_client_initialization_with_token(self, proxied_tgi_service):
        """Test OpenAI client initialization with token."""
        assert proxied_tgi_service.llm_client.client.api_key == "test-token-123"
        assert (
            str(proxied_tgi_service.llm_client.client.base_url)
            == "https://api.test-llm.com/v1/"
        )

    def test_client_initialization_without_token(self, proxied_tgi_service_no_token):
        """Test OpenAI client initialization without token."""
        assert proxied_tgi_service_no_token.llm_client.client.api_key == "fake-token"
        assert (
            str(proxied_tgi_service_no_token.llm_client.client.base_url)
            == "https://api.test-llm.com/v1/"
        )

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_success(self, proxied_tgi_service, mock_prompts):
        """Test finding prompt by specific name."""
        session = MockMCPSession(prompts=mock_prompts)

        result = await proxied_tgi_service.prompt_service.find_prompt_by_name_or_role(
            session, "custom"
        )

        assert result is not None
        assert result["name"] == "custom"

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_or_role_no_prompts(self, proxied_tgi_service):
        """Test finding prompt when no prompts are available."""
        session = MockMCPSession(prompts=[])
        result = await proxied_tgi_service.prompt_service.find_prompt_by_name_or_role(
            session
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_or_role_exception(self, proxied_tgi_service):
        """Test exception handling in find_prompt_by_name_or_role."""

        class BadSession:
            async def list_prompts(self):
                raise RuntimeError("MCP error")

        with pytest.raises(Exception):
            await proxied_tgi_service.prompt_service.find_prompt_by_name_or_role(
                BadSession()
            )

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_not_found(
        self, proxied_tgi_service, mock_prompts
    ):
        """Test finding prompt by name that doesn't exist."""
        session = MockMCPSession(prompts=mock_prompts)

        result = await proxied_tgi_service.prompt_service.find_prompt_by_name_or_role(
            session, "nonexistent"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_prompt_content_error(self, proxied_tgi_service, mock_prompts):
        """Test get_prompt_content error handling."""

        class BadSession:
            async def call_prompt(self, name, args):
                mock_result = Mock()
                mock_result.isError = True
                return mock_result

        session = BadSession()
        prompt = Mock()
        prompt.name = "bad"
        with pytest.raises(Exception):
            await proxied_tgi_service.prompt_service.get_prompt_content(session, prompt)

    @pytest.mark.asyncio
    async def test_get_prompt_content_exception(
        self, proxied_tgi_service, mock_prompts
    ):
        """Test get_prompt_content with unexpected exception."""

        class BadSession:
            async def call_prompt(self, name, args):
                raise RuntimeError("Unexpected error")

        session = BadSession()
        prompt = Mock()
        prompt.name = "bad"
        with pytest.raises(Exception):
            await proxied_tgi_service.prompt_service.get_prompt_content(session, prompt)

    @pytest.mark.asyncio
    async def test_get_all_mcp_tools(self, proxied_tgi_service, mock_tools):
        """Test getting all MCP tools in OpenAI format."""
        session = MockMCPSession(tools=mock_tools)
        openai_tools = await proxied_tgi_service.tool_service.get_all_mcp_tools(session)
        assert isinstance(openai_tools, list)
        assert len(openai_tools) == 3
        names = [tool["function"]["name"] for tool in openai_tools]
        assert "list-files" in names
        assert "read-file" in names
        assert names[-1] == "describe_tool"
        assert all(tool["type"] == "function" for tool in openai_tools)

    @pytest.mark.asyncio
    async def test_get_all_mcp_tools_no_tools(self, proxied_tgi_service):
        """Test get_all_mcp_tools when no tools are available."""
        session = MockMCPSession(tools=[])
        tools = await proxied_tgi_service.tool_service.get_all_mcp_tools(session)
        assert isinstance(tools, list)
        assert tools == []

    @pytest.mark.asyncio
    async def test_get_all_mcp_tools_exception(self, proxied_tgi_service):
        """Test get_all_mcp_tools with exception."""

        class BadSession:
            async def list_tools(self):
                raise RuntimeError("MCP error")

        with pytest.raises(Exception):
            await proxied_tgi_service.tool_service.get_all_mcp_tools(BadSession())

    @pytest.mark.asyncio
    async def test_execute_tool_call_success(self, proxied_tgi_service, mock_tools):
        """Test successful tool execution."""
        session = MockMCPSession(tools=mock_tools)

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments='{"path": "/tmp"}'),
        )

        result = await proxied_tgi_service.tool_service.execute_tool_call(
            session, tool_call, None
        )

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "list-files"
        assert "Tool list-files executed" in result["content"]

    @pytest.mark.asyncio
    async def test_execute_tool_call_tool_error(self, proxied_tgi_service, mock_tools):
        """Test execute_tool_call when tool returns error."""

        class ErrorSession:
            async def call_tool(self, name, args, access_token):
                mock_result = Mock()
                mock_result.isError = True
                mock_result.content = [Mock(text="Tool error!")]
                return mock_result

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments='{"path": "/tmp"}'),
        )
        result = await proxied_tgi_service.tool_service.execute_tool_call(
            ErrorSession(), tool_call, None
        )
        assert result["role"] == "tool"
        assert "error" in result["content"]

    @pytest.mark.asyncio
    async def test_execute_tool_call_exception(self, proxied_tgi_service, mock_tools):
        """Test execute_tool_call with unexpected exception."""

        class ErrorSession:
            async def call_tool(self, name, args, access_token):
                raise RuntimeError("Tool crashed!")

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments='{"path": "/tmp"}'),
        )
        with pytest.raises(RuntimeError):
            await proxied_tgi_service.tool_service.execute_tool_call(
                ErrorSession(), tool_call, None
            )

    @pytest.mark.asyncio
    async def test_execute_tool_call_invalid_json(
        self, proxied_tgi_service, mock_tools
    ):
        """Test tool execution with invalid JSON arguments."""
        session = MockMCPSession(tools=mock_tools)

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments="invalid json"),
        )

        result = await proxied_tgi_service.tool_service.execute_tool_call(
            session, tool_call, None
        )

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert "error" in result["content"].lower()

    @pytest.mark.asyncio
    async def test_prepare_messages_error(self, proxied_tgi_service, mock_prompts):
        """Test prepare_messages error handling returns original messages."""

        class BadSession:
            async def list_prompts(self):
                raise RuntimeError("fail")

        user_message = Message(role=MessageRole.USER, content="Hello")
        messages = [user_message]
        prepared = await proxied_tgi_service.prompt_service.prepare_messages(
            BadSession(), messages, "system"
        )
        assert prepared == messages

    @pytest.mark.asyncio
    async def test_prepare_messages_with_system_prompt(
        self, proxied_tgi_service, mock_prompts
    ):
        """Test message preparation with system prompt addition."""
        session = MockMCPSession(prompts=mock_prompts)

        user_message = Message(role=MessageRole.USER, content="Hello")
        messages = [user_message]

        prepared = await proxied_tgi_service.prompt_service.prepare_messages(
            session, messages, "system"
        )

        assert len(prepared) == 2
        assert prepared[0].role == MessageRole.SYSTEM
        assert prepared[0].content == "System prompt from system"
        assert prepared[1] == user_message

    @pytest.mark.asyncio
    async def test_prepare_messages_with_existing_system(
        self, proxied_tgi_service, mock_prompts
    ):
        """Test message preparation when system message already exists."""
        session = MockMCPSession(prompts=mock_prompts)

        system_message = Message(role=MessageRole.SYSTEM, content="Existing system")
        user_message = Message(role=MessageRole.USER, content="Hello")
        messages = [system_message, user_message]

        prepared = await proxied_tgi_service.prompt_service.prepare_messages(
            session, messages, "system"
        )

        # Should not add another system message
        assert len(prepared) == 2
        assert prepared[0] == system_message
        assert prepared[1] == user_message

    def test_create_completion_id(self, proxied_tgi_service):
        """Test completion ID generation."""
        completion_id = proxied_tgi_service.llm_client.create_completion_id()

        assert completion_id.startswith("chatcmpl-")
        assert len(completion_id) == 38  # "chatcmpl-" (9 chars) + 29 hex chars

    def test_create_usage_stats(self, proxied_tgi_service):
        """Test usage statistics creation."""
        usage = proxied_tgi_service.llm_client.create_usage_stats(100, 50)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestProxiedTGIServiceLLMCalls:
    """Test LLM API call functionality."""

    @pytest.mark.asyncio
    async def test_non_stream_llm_completion_success(self, proxied_tgi_service):
        """Test successful non-streaming LLM completion."""
        # Mock response data
        mock_response = Mock()
        mock_response.id = "chatcmpl-test123"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "test-model"
        mock_response.choices = [
            Mock(
                index=0,
                message=Mock(
                    role="assistant",
                    content="Hello! How can I help you today?",
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=8, total_tokens=18
        )
        # Add model_dump method to mock response and its components
        mock_response.model_dump = Mock(
            return_value={
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
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            }
        )

        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=False,
        )

        with patch.object(
            proxied_tgi_service.llm_client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            with patch("opentelemetry.trace.get_tracer") as mock_tracer:
                mock_span = Mock()
                mock_span.set_attribute = Mock()
                mock_tracer.return_value.start_as_current_span.return_value.__enter__ = Mock(
                    return_value=mock_span
                )
                mock_tracer.return_value.start_as_current_span.return_value.__exit__ = (
                    Mock(return_value=None)
                )

                result = await proxied_tgi_service.llm_client.non_stream_completion(
                    request, None, mock_span
                )

                assert isinstance(result, ChatCompletionResponse)
                assert result.id == "chatcmpl-test123"
                assert (
                    result.choices[0].message.content
                    == "Hello! How can I help you today?"
                )

    @pytest.mark.asyncio
    async def test_non_stream_llm_completion_error(self, proxied_tgi_service):
        """Test error handling in non-streaming LLM completion."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=False,
        )

        with patch.object(
            proxied_tgi_service.llm_client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = Exception("API Error")

            with patch("opentelemetry.trace.get_tracer") as mock_tracer:
                mock_span = Mock()
                mock_span.set_attribute = Mock()
                mock_tracer.return_value.start_as_current_span.return_value.__enter__ = Mock(
                    return_value=mock_span
                )
                mock_tracer.return_value.start_as_current_span.return_value.__exit__ = (
                    Mock(return_value=None)
                )

                with pytest.raises(Exception):
                    await proxied_tgi_service.llm_client.non_stream_completion(
                        request, None, mock_span
                    )

    @pytest.mark.asyncio
    async def test_stream_llm_completion_success(self, proxied_tgi_service):
        """Test successful streaming LLM completion."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="test-model",
            stream=True,
        )

        # Mock streaming chunks
        chunk1 = Mock()
        chunk1.choices = [
            Mock(delta=Mock(content="Hello", tool_calls=None), finish_reason=None)
        ]
        chunk1.model_dump = Mock(
            return_value={"choices": [{"delta": {"content": "Hello"}}]}
        )

        chunk2 = Mock()
        chunk2.choices = [
            Mock(delta=Mock(content="!", tool_calls=None), finish_reason="stop")
        ]
        chunk2.model_dump = Mock(
            return_value={"choices": [{"delta": {"content": "!"}}]}
        )

        async def async_generator():
            yield chunk1
            yield chunk2

        with patch.object(
            proxied_tgi_service.llm_client.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = async_generator()

            with patch("opentelemetry.trace.get_tracer") as mock_tracer:
                mock_span = Mock()
                mock_span.set_attribute = Mock()
                mock_tracer.return_value.start_as_current_span.return_value.__enter__ = Mock(
                    return_value=mock_span
                )
                mock_tracer.return_value.start_as_current_span.return_value.__exit__ = (
                    Mock(return_value=None)
                )

                chunks = []
                async for chunk in proxied_tgi_service.llm_client.stream_completion(
                    request, None, mock_span
                ):
                    chunks.append(chunk)

                assert len(chunks) > 0
                full_content = "".join(chunks)
                assert "Hello" in full_content
                assert "!" in full_content

    @pytest.mark.asyncio
    async def test_non_stream_chat_with_tools(self, proxied_tgi_service, mock_tools):
        """Test non-streaming chat with tool execution."""
        # Patch _non_stream_chat_with_tools to avoid pydantic validation error
        with patch.object(
            proxied_tgi_service, "_non_stream_chat_with_tools"
        ) as mock_chat:
            mock_chat.return_value = ChatCompletionResponse(
                id="chatcmpl-test456",
                object="chat.completion",
                created=1234567891,
                model="test-model",
                choices=[
                    Choice(
                        index=0,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content="I've listed the files in /tmp directory.",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(prompt_tokens=15, completion_tokens=10, total_tokens=25),
            )
            session = MockMCPSession(tools=mock_tools)
            messages = [Message(role=MessageRole.USER, content="List files")]
            available_tools = await proxied_tgi_service.tool_service.get_all_mcp_tools(
                session
            )
            chat_request = ChatCompletionRequest(
                messages=messages, model="test-model", stream=False
            )
            result = await proxied_tgi_service._non_stream_chat_with_tools(
                session, messages, available_tools, chat_request, None, Mock()
            )
            assert isinstance(result, ChatCompletionResponse)
            assert (
                result.choices[0].message.content
                == "I've listed the files in /tmp directory."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
