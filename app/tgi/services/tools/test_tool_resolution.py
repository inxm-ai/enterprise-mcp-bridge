"""
Comprehensive tests for the tool resolution module.

Tests cover various formats, edge cases, and error conditions for tool call resolution.
"""

from unittest.mock import patch

from app.tgi.services.tools.tool_resolution import (
    ToolResolutionStrategy,
    ToolCallFormat,
)


class TestToolResolutionStrategy:
    """Test suite for ToolResolutionStrategy class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ToolResolutionStrategy()

    # ===== FORMAT DETECTION TESTS =====

    def test_detect_openai_json_format(self):
        """Test detection of OpenAI JSON format."""
        content = 'Some text {"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}} more text'
        format_type = self.strategy.detect_format(content)
        assert format_type == ToolCallFormat.OPENAI_JSON

    def test_detect_unknown_format(self):
        """Test detection when no known format is present."""
        content = "Just some plain text without any tool calls."
        format_type = self.strategy.detect_format(content)
        assert format_type == ToolCallFormat.UNKNOWN

    def test_detect_format_with_chunks(self):
        """Test format detection with existing tool call chunks."""
        content = "Plain text"
        chunks = {0: {"id": "call_1", "name": "test", "arguments": "{}"}}
        format_type = self.strategy.detect_format(content, chunks)
        assert format_type == ToolCallFormat.OPENAI_JSON

    # ===== OPENAI FORMAT RESOLUTION TESTS =====

    def test_resolve_openai_format_from_content(self):
        """Test resolving OpenAI format tool calls from content."""
        content = '{"id": "call_1", "type": "function", "index": 0, "function": {"name": "test_tool", "arguments": "{\\"param\\": \\"value\\"}"}}'
        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.id == "call_1"
        assert call.name == "test_tool"
        assert call.arguments == {"param": "value"}
        assert call.format == ToolCallFormat.OPENAI_JSON
        assert call.index == 0

    def test_resolve_openai_format_from_chunks(self):
        """Test resolving OpenAI format tool calls from chunks."""
        chunks = {
            0: {"id": "call_1", "name": "test_tool", "arguments": '{"param": "value"}'},
            1: {"id": "call_2", "name": "another_tool", "arguments": '{"key": "data"}'},
        }
        content = "Some regular text"

        tool_calls, success = self.strategy.resolve_tool_calls(content, chunks)

        assert success
        assert len(tool_calls) == 2

        call1 = tool_calls[0]
        assert call1.id == "call_1"
        assert call1.name == "test_tool"
        assert call1.arguments == {"param": "value"}

        call2 = tool_calls[1]
        assert call2.id == "call_2"
        assert call2.name == "another_tool"
        assert call2.arguments == {"key": "data"}

    def test_resolve_openai_format_mixed_chunks_and_content(self):
        """Test resolving OpenAI format from both chunks and content."""
        chunks = {
            0: {"id": "call_1", "name": "chunk_tool", "arguments": '{"from": "chunk"}'}
        }
        content = '{"id": "call_2", "type": "function", "index": 1, "function": {"name": "content_tool", "arguments": "{\\"from\\": \\"content\\"}"}}'

        tool_calls, success = self.strategy.resolve_tool_calls(content, chunks)

        assert success
        assert len(tool_calls) == 2

        # Check that both chunk and content tool calls are present
        names = [call.name for call in tool_calls]
        assert "chunk_tool" in names
        assert "content_tool" in names

    def test_resolve_openai_format_invalid_json_arguments(self):
        """Test handling of invalid JSON in arguments."""
        chunks = {0: {"id": "call_1", "name": "test_tool", "arguments": "invalid json"}}

        tool_calls, success = self.strategy.resolve_tool_calls("", chunks)

        assert success
        assert len(tool_calls) == 1
        assert tool_calls[0].arguments == {"raw": "invalid json"}

    # ===== ERROR HANDLING AND EDGE CASES =====

    def test_resolve_empty_content(self):
        """Test handling of empty content."""
        tool_calls, success = self.strategy.resolve_tool_calls("")

        assert not success
        assert len(tool_calls) == 0

    def test_resolve_invalid_json_content(self):
        """Test handling of invalid JSON in content."""
        content = '{"invalid": json syntax}'

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert not success
        assert len(tool_calls) == 0

    def test_resolve_with_logging(self):
        """Test that appropriate logging occurs."""
        with patch.object(self.strategy, "logger") as mock_logger:
            content = '{"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}'

            tool_calls, success = self.strategy.resolve_tool_calls(content)

            # Verify debug logs were called
            mock_logger.debug.assert_called()
            assert success

    def test_resolve_error_handling(self):
        """Test error handling during resolution."""
        # Patch to simulate an exception
        with patch.object(
            self.strategy, "_extract_json_objects", side_effect=Exception("Test error")
        ):
            tool_calls, success = self.strategy.resolve_tool_calls('{"test": "data"}')

            assert not success
            assert len(tool_calls) == 0

    # ===== UTILITY METHOD TESTS =====

    def test_extract_json_objects(self):
        """Test JSON object extraction."""
        content = 'Text {"obj1": "value1"} more text {"obj2": "value2"} end'

        json_objects = self.strategy._extract_json_objects(content)

        assert len(json_objects) == 2
        assert json_objects[0] == {"obj1": "value1"}
        assert json_objects[1] == {"obj2": "value2"}

    def test_extract_json_objects_nested(self):
        """Test extraction of nested JSON objects."""
        content = (
            '{"outer": {"inner": {"deep": "value"}}, "array": [1, 2, {"nested": true}]}'
        )

        json_objects = self.strategy._extract_json_objects(content)

        assert len(json_objects) == 1
        assert json_objects[0]["outer"]["inner"]["deep"] == "value"
        assert json_objects[0]["array"][2]["nested"] is True

    def test_is_valid_openai_tool_call(self):
        """Test validation of OpenAI tool call structure."""
        valid_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "test", "arguments": "{}"},
        }
        invalid_call = {"id": "call_1", "type": "function"}  # Missing function

        assert self.strategy._is_valid_openai_tool_call(valid_call)
        assert not self.strategy._is_valid_openai_tool_call(invalid_call)

    def test_parse_arguments_various_formats(self):
        """Test argument parsing with various input formats."""
        # JSON string
        result1 = self.strategy._parse_arguments('{"key": "value"}')
        assert result1 == {"key": "value"}

        # Dictionary
        result2 = self.strategy._parse_arguments({"key": "value"})
        assert result2 == {"key": "value"}

        # Invalid JSON
        result3 = self.strategy._parse_arguments("invalid json")
        assert result3 == {"raw": "invalid json"}

        # Empty string
        result4 = self.strategy._parse_arguments("")
        assert result4 == {}

    def test_parse_arguments_null_returns_empty(self):
        """Ensure null/None arguments are coerced to an empty dict."""
        result = self.strategy._parse_arguments("null")
        assert result == {}


# ===== INTEGRATION TESTS =====


class TestToolResolutionIntegration:
    """Integration tests for tool resolution with real-world scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ToolResolutionStrategy()

    def test_streaming_openai_tool_calls(self):
        """Test streaming OpenAI tool calls scenario."""
        # Simulate streaming chunks
        chunks = {
            0: {
                "id": "call_abc123",
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "unit": "celsius"}',
            },
            1: {
                "id": "call_def456",
                "name": "search_events",
                "arguments": '{"city": "San Francisco", "date": "2024-01-15"}',
            },
        }

        content = "I'll help you with the weather and events information."

        tool_calls, success = self.strategy.resolve_tool_calls(content, chunks)

        assert success
        assert len(tool_calls) == 2

        weather_call = next(
            (call for call in tool_calls if call.name == "get_weather"), None
        )
        assert weather_call is not None
        assert weather_call.arguments["location"] == "San Francisco"

        events_call = next(
            (call for call in tool_calls if call.name == "search_events"), None
        )
        assert events_call is not None
        assert events_call.arguments["city"] == "San Francisco"
