"""
Comprehensive tests for the tool resolution module.

Tests cover various formats, edge cases, and error conditions for tool call resolution.
"""

import pytest
from unittest.mock import patch

from app.tgi.tool_resolution import (
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

    def test_detect_claude_xml_format_mcp(self):
        """Test detection of Claude XML format with MCP tags."""
        content = 'Some text <create_entities>{"entity_id": "test"}</create_entities> more text'
        format_type = self.strategy.detect_format(content)
        assert format_type == ToolCallFormat.CLAUDE_XML

    def test_detect_claude_xml_format_generic(self):
        """Test detection of Claude XML format with generic function tags."""
        content = (
            'Some text <search_database>{"query": "test"}</search_database> more text'
        )
        format_type = self.strategy.detect_format(content)
        assert format_type == ToolCallFormat.CLAUDE_XML

    def test_detect_claude_xml_format_invoke(self):
        """Test detection of Claude XML format with invoke tags."""
        content = 'Text <function_calls><invoke name="test_function">{"param": "value"}</invoke></function_calls> more'
        format_type = self.strategy.detect_format(content)
        assert format_type == ToolCallFormat.CLAUDE_XML

    def test_detect_mixed_format(self):
        """Test detection of mixed formats."""
        content = """
        Some text {"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}
        and also <create_entities>{"entity_id": "test"}</create_entities> more text
        """
        format_type = self.strategy.detect_format(content)
        assert format_type == ToolCallFormat.MIXED

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

    # ===== CLAUDE XML FORMAT RESOLUTION TESTS =====

    def test_resolve_claude_mcp_create_entities(self):
        """Test resolving Claude MCP create_entities call."""
        content = """
        <create_entities>
        {
            "entity_id": "person_123",
            "entity_type": "person",
            "name": "John Doe",
            "description": "Software engineer"
        }
        </create_entities>
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.name == "create_entities"
        assert call.format == ToolCallFormat.CLAUDE_XML
        assert "entity_id" in call.arguments
        assert call.arguments["entity_id"] == "person_123"

    def test_resolve_claude_mcp_add_observations(self):
        """Test resolving Claude MCP add_observations call."""
        content = """
        <add_observations>
        {
            "entity_id": "person_123",
            "observations": ["Works at tech company", "Likes Python"]
        }
        </add_observations>
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.name == "add_observations"
        assert call.format == ToolCallFormat.CLAUDE_XML
        assert "entity_id" in call.arguments
        assert call.arguments["entity_id"] == "person_123"

    def test_resolve_claude_generic_function_call(self):
        """Test resolving generic Claude function call."""
        content = '<search_database>{"query": "SELECT * FROM users", "limit": 10}</search_database>'

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.name == "search_database"
        assert call.format == ToolCallFormat.CLAUDE_XML
        assert call.arguments["query"] == "SELECT * FROM users"
        assert call.arguments["limit"] == 10

    def test_resolve_claude_invoke_format(self):
        """Test resolving Claude invoke format."""
        content = """
        <function_calls>
        <invoke name="calculate_sum">
        <parameter name="numbers">[1, 2, 3, 4, 5]</parameter>
        <parameter name="operation">sum</parameter>
        </invoke>
        </function_calls>
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.name == "calculate_sum"
        assert call.format == ToolCallFormat.CLAUDE_XML
        assert "numbers" in call.arguments
        assert "operation" in call.arguments

    def test_resolve_claude_multiple_calls(self):
        """Test resolving multiple Claude XML calls."""
        content = """
        <create_entities>{"entity_id": "entity1", "name": "Entity One"}</create_entities>
        Some text in between
        <search_memory>{"query": "test query"}</search_memory>
        More text
        <add_observations>{"entity_id": "entity1", "observations": ["observation1"]}</add_observations>
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 3

        names = [call.name for call in tool_calls]
        assert "create_entities" in names
        assert "search_memory" in names
        assert "add_observations" in names

    def test_resolve_claude_xml_non_json_content(self):
        """Test resolving Claude XML with non-JSON content."""
        content = "<custom_function>param1=value1,param2=value2</custom_function>"

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.name == "custom_function"
        assert "param1" in call.arguments
        assert call.arguments["param1"] == "value1"

    def test_resolve_claude_xml_empty_content(self):
        """Test resolving Claude XML with empty content."""
        content = "<empty_function></empty_function>"

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.name == "empty_function"
        assert call.arguments == {}

    def test_resolve_claude_xml_ignore_html_tags(self):
        """Test that HTML-like tags are ignored."""
        content = """
        <p>This is a paragraph</p>
        <create_entities>{"entity_id": "test"}</create_entities>
        <div>This is a div</div>
        <think>This is thinking</think>
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "create_entities"

    # ===== MIXED FORMAT TESTS =====

    def test_resolve_mixed_format(self):
        """Test resolving mixed OpenAI and Claude formats."""
        content = """
        {"id": "call_1", "type": "function", "index": 0, "function": {"name": "openai_tool", "arguments": "{}"}}
        Some text
        <create_entities>{"entity_id": "claude_entity"}</create_entities>
        More text
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 2

        names = [call.name for call in tool_calls]
        assert "openai_tool" in names
        assert "create_entities" in names

        # Check that different formats are preserved
        formats = [call.format for call in tool_calls]
        assert ToolCallFormat.OPENAI_JSON in formats
        assert ToolCallFormat.CLAUDE_XML in formats

    def test_resolve_mixed_format_deduplication(self):
        """Test deduplication in mixed formats."""
        content = """
        {"id": "call_1", "type": "function", "function": {"name": "duplicate_tool", "arguments": "{\\"param\\": \\"value\\"}"}}
        <duplicate_tool>{"param": "value"}</duplicate_tool>
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        # Should deduplicate based on name and arguments
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "duplicate_tool"

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

    def test_resolve_malformed_xml(self):
        """Test handling of malformed XML."""
        content = '<malformed_tag>{"param": "value"}</malformed_tag_wrong>'

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        # Should not crash, but also not find valid tool calls
        assert not success
        assert len(tool_calls) == 0

    def test_resolve_nested_json_in_xml(self):
        """Test handling of nested JSON within XML."""
        content = """
        <complex_tool>
        {
            "nested": {
                "array": [1, 2, 3],
                "object": {"key": "value"}
            },
            "escaped_json": "{\\"inner\\": \\"value\\"}"
        }
        </complex_tool>
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 1

        call = tool_calls[0]
        assert call.name == "complex_tool"
        assert "nested" in call.arguments
        assert call.arguments["nested"]["array"] == [1, 2, 3]

    def test_resolve_with_logging(self):
        """Test that appropriate logging occurs."""
        with patch.object(self.strategy, "logger") as mock_logger:
            content = '<test_tool>{"param": "value"}</test_tool>'

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

    def test_parse_xml_content_json(self):
        """Test XML content parsing with JSON content."""
        content = '{"param1": "value1", "param2": 42}'

        result = self.strategy._parse_xml_content(content)

        assert result == {"param1": "value1", "param2": 42}

    def test_parse_xml_content_parameters(self):
        """Test XML content parsing with parameter tags."""
        content = """
        <parameter name="param1">value1</parameter>
        <parameter name="param2">42</parameter>
        <parameter name="param3">true</parameter>
        """

        result = self.strategy._parse_xml_content(content)

        assert result["param1"] == "value1"
        assert result["param2"] == "42"  # String values from XML
        assert result["param3"] == "true"

    def test_parse_xml_content_key_value_pairs(self):
        """Test XML content parsing with key=value format."""
        content = "key1=value1,key2=value2,key3=123"

        result = self.strategy._parse_xml_content(content)

        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert result["key3"] == "123"

    def test_parse_xml_content_fallback(self):
        """Test XML content parsing fallback to raw text."""
        content = "Some unstructured text content"

        result = self.strategy._parse_xml_content(content)

        assert result == {"raw": "Some unstructured text content"}


# ===== INTEGRATION TESTS =====


class TestToolResolutionIntegration:
    """Integration tests for tool resolution with real-world scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ToolResolutionStrategy()

    def test_claude_mcp_conversation_flow(self):
        """Test a realistic Claude MCP conversation flow."""
        content = """
        I'll help you track this information. Let me create a new entity for you.
        
        <create_entities>
        {
            "entity_id": "matthias_001",
            "entity_type": "person",
            "name": "Matthias",
            "description": "Software developer/programmer"
        }
        </create_entities>
        
        <add_observations>
        {
            "entity_id": "matthias_001",
            "observations": [
                "Is currently writing software",
                "Working on programming projects"
            ]
        }
        </add_observations>
        
        Great! I've created your profile and added some initial observations.
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content)

        assert success
        assert len(tool_calls) == 2

        # Verify the create_entities call
        create_call = next(
            (call for call in tool_calls if call.name == "create_entities"), None
        )
        assert create_call is not None
        assert create_call.arguments["entity_id"] == "matthias_001"
        assert create_call.arguments["name"] == "Matthias"

        # Verify the add_observations call
        obs_call = next(
            (call for call in tool_calls if call.name == "add_observations"), None
        )
        assert obs_call is not None
        assert obs_call.arguments["entity_id"] == "matthias_001"
        assert len(obs_call.arguments["observations"]) == 2

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

    def test_mixed_format_real_world(self):
        """Test mixed format in a real-world scenario."""
        # OpenAI tool call from streaming
        chunks = {
            0: {
                "id": "call_weather",
                "name": "get_current_weather",
                "arguments": '{"location": "New York"}',
            }
        }

        # Claude memory management in response
        content = """
        I'll get the weather for you and remember your location preference.
        
        <create_entities>
        {
            "entity_id": "user_location_pref",
            "entity_type": "preference",
            "name": "Location Preference",
            "description": "User's preferred location for weather queries"
        }
        </create_entities>
        
        <add_observations>
        {
            "entity_id": "user_location_pref",
            "observations": ["User asked for New York weather"]
        }
        </add_observations>
        """

        tool_calls, success = self.strategy.resolve_tool_calls(content, chunks)

        assert success
        assert len(tool_calls) == 3

        # Should have one OpenAI format call and two Claude format calls
        formats = [call.format for call in tool_calls]
        assert ToolCallFormat.OPENAI_JSON in formats
        assert ToolCallFormat.CLAUDE_XML in formats

        # Verify all expected tools are present
        names = [call.name for call in tool_calls]
        assert "get_current_weather" in names
        assert "create_entities" in names
        assert "add_observations" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
