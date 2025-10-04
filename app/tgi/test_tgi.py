import pytest
from unittest.mock import Mock

from app.tgi.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
    Tool,
    ToolCall,
    ToolCallFunction,
    FunctionDefinition,
)
from app.tgi.services.service import TGIService


class MockMCPSession:
    """Mock MCP session for testing."""

    def __init__(self, prompts=None, tools=None):
        self.prompts = prompts or []
        self.tools = tools or []

    async def list_prompts(self):
        """Mock list_prompts method."""
        mock_result = Mock()
        mock_result.prompts = self.prompts
        return mock_result

    async def list_tools(self):
        """Mock list_tools method."""
        mock_result = Mock()
        mock_result.tools = self.tools
        return mock_result

    async def call_prompt(self, name, args):
        """Mock call_prompt method."""
        mock_result = Mock()
        mock_result.isError = False
        mock_result.messages = []

        # Find the prompt and return mock content
        for prompt in self.prompts:
            if prompt.name == name:
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
def tgi_service():
    """Create TGI service instance."""
    return TGIService()


@pytest.fixture
def mock_prompts():
    """Create mock prompts for testing."""
    system_prompt = Mock()
    system_prompt.name = "system"
    system_prompt.description = "System prompt with role=system"

    custom_prompt = Mock()
    custom_prompt.name = "custom"
    custom_prompt.description = "Custom prompt"

    return [system_prompt, custom_prompt]


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    tool1 = Mock()
    tool1.name = "list-files"
    tool1.description = "List files in a directory"
    tool1.inputSchema = {"type": "object", "properties": {"path": {"type": "string"}}}

    tool2 = Mock()
    tool2.name = "read-file"
    tool2.description = "Read contents of a file"
    tool2.inputSchema = {
        "type": "object",
        "properties": {"filename": {"type": "string"}},
    }

    return [tool1, tool2]


class TestTGIService:
    """Test cases for TGIService."""

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_success(self, tgi_service, mock_prompts):
        """Test finding prompt by specific name."""
        session = MockMCPSession(prompts=mock_prompts)

        result = await tgi_service.find_prompt_by_name_or_role(session, "custom")

        assert result is not None
        assert result.name == "custom"

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_not_found(self, tgi_service, mock_prompts):
        """Test finding prompt by name that doesn't exist."""
        session = MockMCPSession(prompts=mock_prompts)

        result = await tgi_service.find_prompt_by_name_or_role(session, "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_find_system_prompt(self, tgi_service, mock_prompts):
        """Test finding system prompt when no specific name provided."""
        session = MockMCPSession(prompts=mock_prompts)

        result = await tgi_service.find_prompt_by_name_or_role(session)

        assert result is not None
        assert result.name == "system"

    @pytest.mark.asyncio
    async def test_find_first_prompt_fallback(self, tgi_service):
        """Test fallback to first prompt when no system prompt exists."""
        other_prompt = Mock()
        other_prompt.name = "other"
        other_prompt.description = "Some other prompt"

        session = MockMCPSession(prompts=[other_prompt])

        result = await tgi_service.find_prompt_by_name_or_role(session)

        assert result is not None
        assert result.name == "other"

    @pytest.mark.asyncio
    async def test_find_no_prompts_available(self, tgi_service):
        """Test when no prompts are available."""
        session = MockMCPSession(prompts=[])

        result = await tgi_service.find_prompt_by_name_or_role(session)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_prompt_content_success(self, tgi_service, mock_prompts):
        """Test getting prompt content successfully."""
        session = MockMCPSession(prompts=mock_prompts)

        content = await tgi_service.get_prompt_content(session, mock_prompts[0])

        assert content == "System prompt from system"

    @pytest.mark.asyncio
    async def test_filter_available_tools(self, tgi_service, mock_tools):
        """Test filtering tools to only available ones."""
        session = MockMCPSession(tools=mock_tools)

        # Create requested tools (some available, some not)
        requested_tools = [
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="list-files", description="List files"
                ),
            ),
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="unavailable-tool", description="Not available"
                ),
            ),
        ]

        filtered_tools = await tgi_service.filter_available_tools(
            session, requested_tools
        )

        assert len(filtered_tools) == 1
        assert filtered_tools[0].function.name == "list-files"

    @pytest.mark.asyncio
    async def test_get_all_mcp_tools(self, tgi_service, mock_tools):
        """Test getting all MCP tools in OpenAI format."""
        session = MockMCPSession(tools=mock_tools)

        openai_tools = await tgi_service.get_all_mcp_tools(session)

        assert len(openai_tools) == 2
        assert openai_tools[0].function.name == "list-files"
        assert openai_tools[1].function.name == "read-file"
        assert all(tool.type == "function" for tool in openai_tools)

    @pytest.mark.asyncio
    async def test_execute_tool_call_success(self, tgi_service, mock_tools):
        """Test successful tool execution."""
        session = MockMCPSession(tools=mock_tools)

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments='{"path": "/tmp"}'),
        )

        result = await tgi_service.execute_tool_call(session, tool_call, None)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "list-files"
        assert "Tool list-files executed" in result["content"]

    @pytest.mark.asyncio
    async def test_execute_tool_call_invalid_json(self, tgi_service, mock_tools):
        """Test tool execution with invalid JSON arguments."""
        session = MockMCPSession(tools=mock_tools)

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments="invalid json"),
        )

        result = await tgi_service.execute_tool_call(session, tool_call, None)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert "error" in result["content"].lower()

    @pytest.mark.asyncio
    async def test_prepare_messages_with_system_prompt(self, tgi_service, mock_prompts):
        """Test message preparation with system prompt addition."""
        session = MockMCPSession(prompts=mock_prompts)

        user_message = Message(role=MessageRole.USER, content="Hello")
        messages = [user_message]

        prepared = await tgi_service.prepare_messages(session, messages, "system")

        assert len(prepared) == 2
        assert prepared[0].role == MessageRole.SYSTEM
        assert prepared[0].content == "System prompt from system"
        assert prepared[1] == user_message

    @pytest.mark.asyncio
    async def test_prepare_messages_with_existing_system(
        self, tgi_service, mock_prompts
    ):
        """Test message preparation when system message already exists."""
        session = MockMCPSession(prompts=mock_prompts)

        system_message = Message(role=MessageRole.SYSTEM, content="Existing system")
        user_message = Message(role=MessageRole.USER, content="Hello")
        messages = [system_message, user_message]

        prepared = await tgi_service.prepare_messages(session, messages, "system")

        # Should not add another system message
        assert len(prepared) == 2
        assert prepared[0] == system_message
        assert prepared[1] == user_message

    def test_create_completion_id(self, tgi_service):
        """Test completion ID generation."""
        completion_id = tgi_service.create_completion_id()

        assert completion_id.startswith("chatcmpl-")
        assert len(completion_id) == 38  # "chatcmpl-" (9 chars) + 29 hex chars

    def test_create_usage_stats(self, tgi_service):
        """Test usage statistics creation."""
        usage = tgi_service.create_usage_stats(100, 50)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestTGIModels:
    """Test cases for TGI models."""

    def test_message_model(self):
        """Test Message model validation."""
        message = Message(role=MessageRole.USER, content="Hello, world!")

        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.tool_calls is None

    def test_chat_completion_request_model(self):
        """Test ChatCompletionRequest model validation."""
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="gpt-4",
            stream=True,
        )

        assert len(request.messages) == 1
        assert request.model == "gpt-4"
        assert request.stream is True
        assert request.tool_choice == "auto"

    def test_tool_model(self):
        """Test Tool model validation."""
        tool = Tool(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            ),
        )

        assert tool.type == "function"
        assert tool.function.name == "get_weather"
        assert "location" in tool.function.parameters["properties"]


@pytest.mark.asyncio
async def test_tgi_route_integration():
    """Integration test for TGI route (mocked)."""
    # TODO: This would require setting up the full FastAPI test client
    # with mocked dependencies. For now, we'll skip this test.
    # Let's think of something later, maybe like the dummy-llm from the example?
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
