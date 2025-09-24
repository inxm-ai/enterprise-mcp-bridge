import json
import pytest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from app.tgi.tool_service import ToolService
from app.tgi.models import Message, MessageRole
from app.tgi.tool_resolution import ToolCallFormat
from app.tgi.models import ToolCall, ToolCallFunction
from app.tgi.tool_service import (
    process_tool_arguments,
    parse_and_clean_tool_call,
    extract_tool_call_from_streamed_content,
)


class DummySession:
    async def call_tool(self, name, args, access_token):
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = []
        mock_result.structuredContent = {
            "result": f"Tool {name} executed with args {args}"
        }
        return mock_result

    async def list_tools(self):
        return [
            {
                "name": "list-files",
                "description": "List files in a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
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
    def __init__(self, list_tools_fails=True, call_tool_fails=True):
        self.list_tools_fails = list_tools_fails
        self.call_tool_fails = call_tool_fails

    async def call_tool(self, name, args, access_token):
        if self.call_tool_fails:
            raise RuntimeError("Tool crashed!")
        else:
            return await DummySession().call_tool(name, args, access_token)

    async def list_tools(self):
        if self.list_tools_fails:
            raise RuntimeError("Session crashed!")
        else:
            return await DummySession().list_tools()


def make_tool_call(name="list-files", args='{"path": "bar"}', id="call_1"):
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
        with patch("app.tgi.tool_service.map_tools") as mock_map_tools:
            mock_map_tools.return_value = [
                {"type": "function", "function": {"name": "list-files"}},
                {"type": "function", "function": {"name": "read-file"}},
                {"type": "function", "function": {"name": "describe_tool"}},
            ]

            openai_tools = await tool_service.get_all_mcp_tools(session)
            assert isinstance(openai_tools, list)
            assert len(openai_tools) == 4
            names = [tool["function"]["name"] for tool in openai_tools]
            assert "list-files" in names
            assert "read-file" in names
            assert names[-1] == "describe_tool"
            assert all(tool["type"] == "function" for tool in openai_tools)

    @pytest.mark.asyncio
    async def test_get_all_mcp_tools_no_tools(self, tool_service):
        """Test get_all_mcp_tools when no tools are available."""
        session = ErrorSession()
        with patch("app.tgi.tool_service.map_tools") as mock_map_tools:
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
        msg = 'Some error occurred: [\n {"code": "invalid_type", "expected": "boolean", "received": "string", "path": ["foo"], "message": "Type error"}\n]'
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
                "message": "Type error",
            }
        ]
        formatted = tool_service._format_errors(errors)
        assert "data type mismatch" in formatted
        assert "foo" in formatted
        assert "boolean" in formatted
        assert "string" in formatted

    def test_format_errors_other_error(self, tool_service):
        errors = [
            {"code": "other_error", "message": "Something went wrong", "path": ["bar"]}
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
        result, success = await tool_service.execute_tool_calls(
            session, [(tool_call, ToolCallFormat.OPENAI_JSON)], None, None
        )
        print(result)
        assert success
        assert len(result) == 1
        assert result[0].role == MessageRole.TOOL
        assert result[0].tool_call_id == tool_call.id
        assert result[0].name == tool_call.function.name
        assert "Tool list-files executed" in result[0].content

    @pytest.mark.asyncio
    async def test_execute_tool_calls_error(self, tool_service):
        session = ErrorSession()
        tool_call = make_tool_call()
        result, success = await tool_service.execute_tool_calls(
            session, [(tool_call, ToolCallFormat.OPENAI_JSON)], None, None
        )
        assert not success
        assert len(result) == 1
        assert result[0].role == MessageRole.TOOL
        assert '"error"' in result[0].content

    @pytest.mark.asyncio
    async def test_execute_tool_calls_exception(self, tool_service):
        session = ExceptionSession(list_tools_fails=False)
        tool_call = make_tool_call()
        result, success = await tool_service.execute_tool_calls(
            session, [(tool_call, ToolCallFormat.OPENAI_JSON)], None, None
        )
        assert not success
        assert len(result) == 2
        assert result[0].role == MessageRole.TOOL
        assert "Failed to execute tool" in result[0].content
        assert result[1].role == MessageRole.USER
        assert (
            "Please fix the error" in result[1].content
            or "No errors to format." in result[1].content
        )


def test_process_tool_arguments_valid_simple_json():
    """Test processing valid simple JSON."""
    input_str = '{"key": "value"}'
    result = process_tool_arguments(input_str)
    expected = '{"key":"value"}'
    assert result == expected


def test_process_tool_arguments_nested_json_string():
    """Test processing JSON with nested JSON string that gets parsed."""
    input_str = '{"key": "{\\"nested\\": \\"value\\"}"}'
    result = process_tool_arguments(input_str)
    expected = '{"key":{"nested":"value"}}'
    assert result == expected


def test_process_tool_arguments_multiple_nested():
    """Test processing JSON with multiple nested JSON strings."""
    input_str = '{"arg1": "{\\"a\\": 1}", "arg2": "{\\"b\\": 2}"}'
    result = process_tool_arguments(input_str)
    expected = '{"arg1":{"a":1},"arg2":{"b":2}}'
    assert result == expected

    def test_process_tool_arguments_invalid_json():
        """Test processing invalid JSON returns original string."""
        input_str = "invalid json"
        result = process_tool_arguments(input_str)
        assert result == input_str


def test_process_tool_arguments_empty_string():
    """Test processing empty string."""
    input_str = ""
    result = process_tool_arguments(input_str)
    assert result == input_str


def test_process_tool_arguments_non_string_nested():
    """Test processing JSON with non-string nested values."""
    input_str = '{"key": 123}'
    result = process_tool_arguments(input_str)
    expected = '{"key": 123}'
    assert json.loads(result) == json.loads(expected)


def test_process_tool_arguments_mixed_types():
    """Test processing JSON with mixed types including valid nested JSON."""
    input_str = '{"str": "value", "num": 42, "nested": "{\\"inner\\": \\"data\\"}"}'
    result = process_tool_arguments(input_str)
    expected = '{"str": "value", "num": 42, "nested": {"inner": "data"}}'
    assert json.loads(result) == json.loads(expected)


def test_parse_and_clean_tool_call_valid_dict():
    """Test parsing and cleaning valid tool call dict."""
    tool_call = {"function": {"arguments": '{"param": "{\\"nested\\": \\"value\\"}"}'}}
    result = parse_and_clean_tool_call(tool_call)
    expected_arguments = '{"param": {"nested": "value"}}'
    assert json.loads(result["function"]["arguments"]) == json.loads(expected_arguments)


def test_parse_and_clean_tool_call_missing_function():
    """Test parsing dict without function key."""
    tool_call = {"other": "data"}
    result = parse_and_clean_tool_call(tool_call)
    assert result == tool_call  # Should return unchanged


def test_parse_and_clean_tool_call_with_llama_failure_example():
    tool_call = {
        "id": "call_w5gydic5",
        "index": 0,
        "type": "function",
        "function": {
            "name": "create_entities",
            "arguments": '{"entities":"[{\\"entityType\\": \\"Person\\", \\"name\\": \\"Matthias\\", \\"observations\\": []}, {\\"entityType\\": \\"Department\\", \\"name\\": \\"HR\\", \\"observations\\": []}]"}',
        },
    }
    result = parse_and_clean_tool_call(tool_call)
    expected_arguments = '{"entities": [{"entityType": "Person", "name": "Matthias", "observations": []}, {"entityType": "Department", "name": "HR", "observations": []}]}'
    assert json.loads(result["function"]["arguments"]) == json.loads(expected_arguments)


def test_parse_and_clean_tool_call_missing_arguments():
    """Test parsing dict with function but no arguments."""
    tool_call = {"function": {"name": "test"}}
    result = parse_and_clean_tool_call(tool_call)
    assert result == tool_call  # Should return unchanged


def test_parse_and_clean_tool_call_arguments_not_string():
    """Test parsing dict with non-string arguments."""
    tool_call = {"function": {"arguments": 123}}
    result = parse_and_clean_tool_call(tool_call)
    assert result == tool_call  # Should return unchanged


def test_parse_and_clean_tool_call_invalid_arguments_json():
    """Test parsing dict with invalid JSON in arguments."""
    tool_call = {"function": {"arguments": "invalid json"}}
    result = parse_and_clean_tool_call(tool_call)
    assert result["function"]["arguments"] == "invalid json"  # Should remain unchanged


def test_extract_tool_call_from_streamed_content_valid_tool_call():
    """Test extracting valid tool call from content."""
    content = 'Some text {"id": "call_1", "index": 0, "type": "function", "function": {"name": "test_tool", "arguments": "{}"}} more text'
    result = extract_tool_call_from_streamed_content(content)
    expected = {
        "id": "call_1",
        "index": 0,
        "type": "function",
        "function": {"name": "test_tool", "arguments": "{}"},
    }
    assert result == expected


def test_extract_tool_call_from_streamed_content_with_nested_arguments():
    """Test extracting tool call with nested JSON in arguments."""
    nested_arguments = json.dumps({"param": json.dumps({"inner": "value"})})
    tool_call = {
        "id": "call_2",
        "index": 1,
        "type": "function",
        "function": {
            "name": "nested_tool",
            "arguments": nested_arguments,
        },
    }
    content = json.dumps(tool_call)
    result = extract_tool_call_from_streamed_content(content)
    expected = {
        "id": "call_2",
        "index": 1,
        "type": "function",
        "function": {
            "name": "nested_tool",
            "arguments": json.dumps({"param": {"inner": "value"}}),
        },
    }
    assert result is not None
    assert result["id"] == expected["id"]
    assert result["index"] == expected["index"]
    assert result["type"] == expected["type"]
    assert result["function"]["name"] == expected["function"]["name"]
    assert json.loads(result["function"]["arguments"]) == json.loads(
        expected["function"]["arguments"]
    )


def test_extract_tool_call_from_streamed_content_multiple_json_first_valid():
    """Test extracting from content with multiple JSON objects, first is valid."""
    content = '{"id": "call_3", "index": 0, "type": "function", "function": {"name": "first_tool", "arguments": "{}"}} {"invalid": json}'
    result = extract_tool_call_from_streamed_content(content)
    expected = {
        "id": "call_3",
        "index": 0,
        "type": "function",
        "function": {"name": "first_tool", "arguments": "{}"},
    }
    assert result == expected


def test_extract_tool_call_from_streamed_content_multiple_json_second_valid():
    """Test extracting from content with multiple JSON objects, second is valid."""
    tool_call = {
        "id": "call_4",
        "index": 1,
        "type": "function",
        "function": {
            "name": "second_tool",
            "arguments": json.dumps({"key": "value"}),
        },
    }
    content = json.dumps({"invalid": "json"}) + " " + json.dumps(tool_call)
    result = extract_tool_call_from_streamed_content(content)
    expected = tool_call
    assert result is not None
    assert result["id"] == expected["id"]
    assert result["index"] == expected["index"]
    assert result["type"] == expected["type"]
    assert result["function"]["name"] == expected["function"]["name"]
    assert json.loads(result["function"]["arguments"]) == json.loads(
        expected["function"]["arguments"]
    )


def test_extract_tool_call_from_streamed_content_no_valid_json():
    """Test extracting from content with no valid JSON."""
    content = "Just some plain text without any JSON objects."
    result = extract_tool_call_from_streamed_content(content)
    assert result is None


def test_extract_tool_call_from_streamed_content_invalid_json_only():
    """Test extracting from content with only invalid JSON."""
    content = '{"invalid": json syntax}'
    result = extract_tool_call_from_streamed_content(content)
    assert result is None


def test_extract_tool_call_from_streamed_content_missing_required_keys():
    """Test extracting from content with JSON missing required keys."""
    content = '{"id": "call_5", "index": 0}'  # Missing type and function
    result = extract_tool_call_from_streamed_content(content)
    assert result is None


def test_extract_tool_call_from_streamed_content_missing_function_keys():
    """Test extracting from content with JSON missing function sub-keys."""
    content = '{"id": "call_6", "index": 0, "type": "function", "function": {"name": "test"}}'  # Missing arguments
    result = extract_tool_call_from_streamed_content(content)
    assert result is None


def test_extract_tool_call_from_streamed_content_empty_content():
    """Test extracting from empty content."""
    content = ""
    result = extract_tool_call_from_streamed_content(content)
    assert result is None


def test_extract_tool_call_from_streamed_content_json_with_extra_keys():
    """Test extracting from content with valid JSON plus extra keys."""
    content = '{"id": "call_7", "index": 0, "type": "function", "function": {"name": "extra_tool", "arguments": "{}"}, "extra": "data"}'
    result = extract_tool_call_from_streamed_content(content)
    expected = {
        "id": "call_7",
        "index": 0,
        "type": "function",
        "function": {"name": "extra_tool", "arguments": "{}"},
        "extra": "data",
    }
    assert result == expected


@pytest.mark.asyncio
async def test_describe_tool_returns_full_schema(tool_service):
    await tool_service.get_all_mcp_tools(DummySession())
    tool_call = make_tool_call(
        name="describe_tool", args=json.dumps({"name": "list-files"}), id="call_desc"
    )
    result, success = await tool_service.execute_tool_calls(
        DummySession(), [(tool_call, ToolCallFormat.OPENAI_JSON)], None, None
    )
    assert success
    assert len(result) == 1
    payload = json.loads(result[0].content)
    assert payload["name"] == "list-files"
    assert "inputSchema" in payload
    assert payload["inputSchema"]["type"] == "object"


@pytest.mark.asyncio
async def test_describe_tool_unknown_name(tool_service):
    await tool_service.get_all_mcp_tools(DummySession())
    tool_call = make_tool_call(
        name="describe_tool", args=json.dumps({"name": "missing"}), id="call_desc2"
    )
    result, success = await tool_service.execute_tool_calls(
        DummySession(), [(tool_call, ToolCallFormat.OPENAI_JSON)], None, None
    )
    assert success
    payload = json.loads(result[0].content)
    assert "error" in payload


@pytest.mark.asyncio
async def test_create_result_message_basic():
    service = ToolService()
    tool_result = {
        "content": "result content",
        "tool_call_id": "abc123",
        "name": "my_tool",
    }
    msg = await service.create_result_message(ToolCallFormat.OPENAI_JSON, tool_result)
    assert isinstance(msg, Message)
    assert msg.role == MessageRole.TOOL
    assert msg.content == "result content"
    assert msg.tool_call_id == "abc123"
    assert msg.name == "my_tool"


@pytest.mark.asyncio
async def test_create_result_message_claude_xml():
    service = ToolService()
    tool_result = {
        "content": "42",
        "tool_call_id": "id789",
        "name": "sum_numbers",
    }
    msg = await service.create_result_message(ToolCallFormat.CLAUDE_XML, tool_result)
    assert isinstance(msg, Message)
    assert msg.role == MessageRole.ASSISTANT
    assert msg.tool_call_id == "id789"
    assert msg.name == "sum_numbers"
    assert msg.content == "<sum_numbers_result>42</sum_numbers_result><stop/>"


@pytest.mark.asyncio
async def test_create_result_message_claude_xml_missing_fields():
    service = ToolService()
    tool_result = {}
    msg = await service.create_result_message(ToolCallFormat.CLAUDE_XML, tool_result)
    assert msg.role == MessageRole.ASSISTANT
    assert msg.content == "<None_result></None_result><stop/>"
    assert msg.tool_call_id is None
    assert msg.name is None


@pytest.mark.asyncio
async def test_create_result_message_missing_fields():
    service = ToolService()
    tool_result = {}
    msg = await service.create_result_message(ToolCallFormat.OPENAI_JSON, tool_result)
    assert msg.content == ""
    assert msg.tool_call_id is None
    assert msg.name is None


@pytest.mark.asyncio
async def test_create_result_message_with_extra_fields():
    service = ToolService()
    tool_result = {
        "content": "extra content",
        "tool_call_id": "id456",
        "name": "tool_x",
        "extra": "should be ignored",
    }
    msg = await service.create_result_message(ToolCallFormat.OPENAI_JSON, tool_result)
    assert msg.content == "extra content"
    assert msg.tool_call_id == "id456"
    assert msg.name == "tool_x"


@pytest.mark.asyncio
async def test_create_result_message_summarizes_long_content(monkeypatch):
    service = ToolService()
    long_text = "a" * 12000
    tool_result = {
        "content": long_text,
        "tool_call_id": "long1",
        "name": "big_tool",
    }

    async def fake_summarize(base_request, content, access_token, outer_span):
        return "SHORT_SUMMARY"

    # attach fake llm client
    service.llm_client = SimpleNamespace(summarize_text=fake_summarize)

    msg = await service.create_result_message(ToolCallFormat.OPENAI_JSON, tool_result)
    assert msg.content == "SHORT_SUMMARY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
