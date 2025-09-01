import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from app.tgi.tool_service import ToolService
from app.tgi.models import ToolCall, ToolCallFunction, Message, MessageRole


class DummySession:
    async def call_tool(self, name, args, access_token):
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = []
        mock_result.structuredContent = {"result": f"Tool {name} executed with args {args}"}
        return mock_result

    async def list_tools(self):
        return [
            {
                "name": "list-files",
                "description": "List files in a directory",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
            {
                "name": "read-file",
                "description": "Read contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                },
            },
        ]


class ErrorSession:
    async def call_tool(self, name, args, access_token):
        mock_result = Mock()
        mock_result.isError = True
        mock_result.content = [Mock(text="Tool error!")]
        mock_result.structuredContent = {}
        return mock_result

    async def list_tools(self):
        return []


class ExceptionSession:
    async def call_tool(self, name, args, access_token):
        raise RuntimeError("Tool crashed!")

    async def list_tools(self):
        raise RuntimeError("Session crashed!")


def make_tool_call(name="tool1", args='{"foo": "bar"}', id="call_1"):
    return ToolCall(
        id=id,
        type="function",
        function=ToolCallFunction(name=name, arguments=args),
    )


@pytest.fixture
def tool_service():
    """Create ToolService instance."""
    return ToolService()


class TestToolService:
    """Test cases for ToolService."""

    @pytest.mark.asyncio
    async def test_get_all_mcp_tools(self, tool_service):
        """Test getting all MCP tools in OpenAI format."""
        session = DummySession()
        with patch('app.tgi.tool_service.map_tools') as mock_map_tools:
            mock_map_tools.return_value = [
                {"type": "function", "function": {"name": "list-files"}},
                {"type": "function", "function": {"name": "read-file"}},
            ]
            
            openai_tools = await tool_service.get_all_mcp_tools(session)
            assert isinstance(openai_tools, list)
            assert len(openai_tools) == 2
            assert openai_tools[0]["function"]["name"] == "list-files"
            assert openai_tools[1]["function"]["name"] == "read-file"
            assert all(tool["type"] == "function" for tool in openai_tools)

    @pytest.mark.asyncio
    async def test_get_all_mcp_tools_no_tools(self, tool_service):
        """Test get_all_mcp_tools when no tools are available."""
        session = ErrorSession()
        with patch('app.tgi.tool_service.map_tools') as mock_map_tools:
            mock_map_tools.return_value = []
            
            tools = await tool_service.get_all_mcp_tools(session)
            assert isinstance(tools, list)
            assert tools == []

    @pytest.mark.asyncio
    async def test_get_all_mcp_tools_exception(self, tool_service):
        """Test get_all_mcp_tools with exception."""
        session = ExceptionSession()
        
        with pytest.raises(RuntimeError):
            await tool_service.get_all_mcp_tools(session)

    @pytest.mark.asyncio
    async def test_execute_tool_call_success(self, tool_service):
        """Test successful tool execution."""
        session = DummySession()

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments='{"path": "/tmp"}'),
        )

        result = await tool_service.execute_tool_call(session, tool_call, None)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "list-files"
        assert "Tool list-files executed" in result["content"]

    @pytest.mark.asyncio
    async def test_execute_tool_call_tool_error(self, tool_service):
        """Test execute_tool_call when tool returns error."""
        session = ErrorSession()

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments='{"path": "/tmp"}'),
        )
        result = await tool_service.execute_tool_call(session, tool_call, None)
        assert result["role"] == "tool"
        assert "error" in result["content"]

    @pytest.mark.asyncio
    async def test_execute_tool_call_exception(self, tool_service):
        """Test execute_tool_call with unexpected exception."""
        session = ExceptionSession()

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments='{"path": "/tmp"}'),
        )
        with pytest.raises(RuntimeError):
            await tool_service.execute_tool_call(session, tool_call, None)

    @pytest.mark.asyncio
    async def test_execute_tool_call_invalid_json(self, tool_service):
        """Test tool execution with invalid JSON arguments."""
        session = DummySession()

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=ToolCallFunction(name="list-files", arguments="invalid json"),
        )

        result = await tool_service.execute_tool_call(session, tool_call, None)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert "error" in result["content"].lower()

    def test_parse_json_array_from_message_valid(self, tool_service):
        msg = "Some error occurred: [\n {\"code\": \"invalid_type\", \"expected\": \"boolean\", \"received\": \"string\", \"path\": [\"foo\"], \"message\": \"Type error\"}\n]"
        result = tool_service._parse_json_array_from_message(msg)
        assert isinstance(result, list)
        assert result[0]["code"] == "invalid_type"

    def test_parse_json_array_from_message_no_array(self, tool_service):
        msg = "No brackets here"
        result = tool_service._parse_json_array_from_message(msg)
        assert result is None

    def test_parse_json_array_from_message_invalid_json(self, tool_service):
        msg = "Some error: [not a valid json]"
        result = tool_service._parse_json_array_from_message(msg)
        assert result is None

    def test_format_errors_invalid_type(self, tool_service):
        errors = [
            {
                "code": "invalid_type",
                "expected": "boolean",
                "received": "string",
                "path": ["foo"],
                "message": "Type error"
            }
        ]
        formatted = tool_service._format_errors(errors)
        assert "data type mismatch" in formatted
        assert "foo" in formatted
        assert "boolean" in formatted
        assert "string" in formatted

    def test_format_errors_other_error(self, tool_service):
        errors = [
            {
                "code": "other_error",
                "message": "Something went wrong",
                "path": ["bar"]
            }
        ]
        formatted = tool_service._format_errors(errors)
        assert "An error occurred" in formatted
        assert "Something went wrong" in formatted

    def test_format_errors_empty_list(self, tool_service):
        formatted = tool_service._format_errors([])
        assert formatted == "No errors to format."

    def test_format_errors_not_list(self, tool_service):
        formatted = tool_service._format_errors(None)
        assert formatted == "No errors to format."

    @pytest.mark.asyncio
    async def test_execute_tool_calls_success(self, tool_service):
        session = DummySession()
        tool_call = make_tool_call()
        result, success = await tool_service.execute_tool_calls(session, [tool_call], None, None)
        assert success
        assert len(result) == 1
        assert result[0].role == MessageRole.TOOL
        assert result[0].tool_call_id == tool_call.id
        assert result[0].name == tool_call.function.name
        assert "Tool tool1 executed" in result[0].content

    @pytest.mark.asyncio
    async def test_execute_tool_calls_error(self, tool_service):
        session = ErrorSession()
        tool_call = make_tool_call()
        result, success = await tool_service.execute_tool_calls(session, [tool_call], None, None)
        assert not success
        assert len(result) == 1
        assert result[0].role == MessageRole.TOOL
        assert '"error"' in result[0].content

    @pytest.mark.asyncio
    async def test_execute_tool_calls_exception(self, tool_service):
        session = ExceptionSession()
        tool_call = make_tool_call()
        result, success = await tool_service.execute_tool_calls(session, [tool_call], None, None)
        assert not success
        assert len(result) == 2
        assert result[0].role == MessageRole.TOOL
        assert "Failed to execute tool" in result[0].content
        assert result[1].role == MessageRole.USER
        assert "Please fix the error" in result[1].content or "No errors to format." in result[1].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
