"""
Tests for SSE streaming functionality.

These tests verify the behavior of the SSE streaming module for MCP tool calls
with progress reporting. Tests focus on behavior rather than implementation details.
"""

import asyncio
import json
import pytest
from typing import Any, Dict, List, Optional

from app.sse.streaming import (
    SSEEvent,
    SSEEventType,
    ProgressNotificationHandler,
    stream_tool_call,
    _serialize_result,
)


class TestSSEEvent:
    """Tests for SSEEvent data class."""

    def test_progress_event_creation(self):
        """Test creating a progress event with all fields."""
        event = SSEEvent.progress_event(50.0, 100.0, "Processing...")

        assert event.type == SSEEventType.PROGRESS
        assert event.progress == 50.0
        assert event.total == 100.0
        assert event.message == "Processing..."

    def test_progress_event_minimal(self):
        """Test creating a progress event with minimal fields."""
        event = SSEEvent.progress_event(25.0)

        assert event.type == SSEEventType.PROGRESS
        assert event.progress == 25.0
        assert event.total is None
        assert event.message is None

    def test_log_event_creation(self):
        """Test creating a log event."""
        event = SSEEvent.log_event("info", "Log message", "mylogger")

        assert event.type == SSEEventType.LOG
        assert event.level == "info"
        assert event.data == "Log message"
        assert event.logger_name == "mylogger"

    def test_result_event_creation(self):
        """Test creating a result event."""
        result_data = {"content": [{"text": "Done"}], "isError": False}
        event = SSEEvent.result_event(result_data)

        assert event.type == SSEEventType.RESULT
        assert event.data == result_data

    def test_error_event_creation(self):
        """Test creating an error event."""
        event = SSEEvent.error_event("Something went wrong", {"code": 500})

        assert event.type == SSEEventType.ERROR
        assert event.data["message"] == "Something went wrong"
        assert event.data["details"] == {"code": 500}

    def test_error_event_without_details(self):
        """Test creating an error event without details."""
        event = SSEEvent.error_event("Simple error")

        assert event.type == SSEEventType.ERROR
        assert event.data["message"] == "Simple error"
        assert "details" not in event.data

    def test_to_sse_string_progress(self):
        """Test SSE string format for progress events."""
        event = SSEEvent.progress_event(50.0, 100.0, "Half done")
        sse_string = event.to_sse_string()

        assert sse_string.startswith("data: ")
        assert sse_string.endswith("\n\n")

        # Parse the JSON payload
        json_str = sse_string[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        payload = json.loads(json_str)

        assert payload["type"] == "progress"
        assert payload["progress"] == 50.0
        assert payload["total"] == 100.0
        assert payload["message"] == "Half done"

    def test_to_sse_string_omits_none_values(self):
        """Test that None values are not included in SSE output."""
        event = SSEEvent.progress_event(25.0)
        sse_string = event.to_sse_string()

        json_str = sse_string[6:-2]
        payload = json.loads(json_str)

        assert "total" not in payload
        assert "message" not in payload

    def test_to_sse_string_result(self):
        """Test SSE string format for result events."""
        result = {"status": "success", "value": 42}
        event = SSEEvent.result_event(result)
        sse_string = event.to_sse_string()

        json_str = sse_string[6:-2]
        payload = json.loads(json_str)

        assert payload["type"] == "result"
        assert payload["data"]["status"] == "success"
        assert payload["data"]["value"] == 42


class TestProgressNotificationHandler:
    """Tests for ProgressNotificationHandler."""

    @pytest.mark.asyncio
    async def test_on_progress_adds_to_queue(self):
        """Test that progress notifications are added to the queue."""
        queue: asyncio.Queue = asyncio.Queue()
        handler = ProgressNotificationHandler(queue)

        await handler.on_progress(50.0, 100.0, "Processing")

        event = await queue.get()
        assert event.type == SSEEventType.PROGRESS
        assert event.progress == 50.0
        assert event.total == 100.0
        assert event.message == "Processing"

    @pytest.mark.asyncio
    async def test_on_log_adds_to_queue(self):
        """Test that log notifications are added to the queue."""
        queue: asyncio.Queue = asyncio.Queue()
        handler = ProgressNotificationHandler(queue)

        await handler.on_log("warning", "Watch out!", "test_logger")

        event = await queue.get()
        assert event.type == SSEEventType.LOG
        assert event.level == "warning"
        assert event.data == "Watch out!"
        assert event.logger_name == "test_logger"

    @pytest.mark.asyncio
    async def test_multiple_events_preserve_order(self):
        """Test that multiple events are queued in order."""
        queue: asyncio.Queue = asyncio.Queue()
        handler = ProgressNotificationHandler(queue)

        await handler.on_progress(20.0)
        await handler.on_progress(40.0)
        await handler.on_progress(60.0)

        events = []
        while not queue.empty():
            events.append(await queue.get())

        assert len(events) == 3
        assert events[0].progress == 20.0
        assert events[1].progress == 40.0
        assert events[2].progress == 60.0


class TestStreamToolCall:
    """Tests for stream_tool_call function."""

    @pytest.mark.asyncio
    async def test_successful_tool_call_with_progress(self):
        """Test streaming a successful tool call with progress updates."""
        events_received: List[Dict[str, Any]] = []

        async def mock_call_tool(
            tool_name: str,
            args: Dict,
            access_token: Optional[str],
            progress_callback,
            log_callback,
        ):
            # Simulate progress updates
            await progress_callback(25.0, 100.0, "Starting...")
            await progress_callback(50.0, 100.0, "Halfway...")
            await progress_callback(100.0, 100.0, "Complete!")
            return {"content": [{"text": "Done"}], "isError": False}

        async for event_str in stream_tool_call(
            mock_call_tool, "test_tool", {"arg": "value"}, None
        ):
            # Parse the SSE string
            json_str = event_str[6:-2]
            events_received.append(json.loads(json_str))

        # Should have 3 progress events + 1 result event
        assert len(events_received) == 4

        # Check progress events
        assert events_received[0]["type"] == "progress"
        assert events_received[0]["progress"] == 25.0
        assert events_received[1]["type"] == "progress"
        assert events_received[1]["progress"] == 50.0
        assert events_received[2]["type"] == "progress"
        assert events_received[2]["progress"] == 100.0

        # Check result event
        assert events_received[3]["type"] == "result"
        assert events_received[3]["data"]["isError"] is False

    @pytest.mark.asyncio
    async def test_tool_call_with_error(self):
        """Test streaming a tool call that raises an error."""
        events_received: List[Dict[str, Any]] = []

        async def mock_call_tool_error(
            tool_name: str,
            args: Dict,
            access_token: Optional[str],
            progress_callback,
            log_callback,
        ):
            await progress_callback(10.0, 100.0, "Starting...")
            raise ValueError("Tool execution failed")

        async for event_str in stream_tool_call(
            mock_call_tool_error, "failing_tool", {}, None
        ):
            json_str = event_str[6:-2]
            events_received.append(json.loads(json_str))

        # Should have 1 progress event + 1 error event
        assert len(events_received) == 2

        assert events_received[0]["type"] == "progress"
        assert events_received[1]["type"] == "error"
        assert "Tool execution failed" in events_received[1]["data"]["message"]

    @pytest.mark.asyncio
    async def test_tool_call_without_progress(self):
        """Test streaming a tool call that doesn't report progress."""
        events_received: List[Dict[str, Any]] = []

        async def mock_call_tool_no_progress(
            tool_name: str,
            args: Dict,
            access_token: Optional[str],
            progress_callback,
            log_callback,
        ):
            # Don't call progress_callback
            return {"content": [{"text": "Quick result"}], "isError": False}

        async for event_str in stream_tool_call(
            mock_call_tool_no_progress, "quick_tool", {}, None
        ):
            json_str = event_str[6:-2]
            events_received.append(json.loads(json_str))

        # Should only have the result event
        assert len(events_received) == 1
        assert events_received[0]["type"] == "result"

    @pytest.mark.asyncio
    async def test_arguments_passed_correctly(self):
        """Test that tool name and arguments are passed correctly."""
        received_args = {}

        async def mock_call_tool_capture(
            tool_name: str,
            args: Dict,
            access_token: Optional[str],
            progress_callback,
            log_callback,
        ):
            received_args["tool_name"] = tool_name
            received_args["args"] = args
            received_args["access_token"] = access_token
            return {"content": [], "isError": False}

        async for _ in stream_tool_call(
            mock_call_tool_capture,
            "my_tool",
            {"key": "value", "number": 42},
            "test_token",
        ):
            pass

        assert received_args["tool_name"] == "my_tool"
        assert received_args["args"] == {"key": "value", "number": 42}
        assert received_args["access_token"] == "test_token"


class TestSerializeResult:
    """Tests for _serialize_result helper function."""

    def test_serialize_dict(self):
        """Test serializing a dictionary."""
        result = {"key": "value", "nested": {"a": 1}}
        serialized = _serialize_result(result)
        assert serialized == result

    def test_serialize_none(self):
        """Test serializing None."""
        assert _serialize_result(None) is None

    def test_serialize_pydantic_model(self):
        """Test serializing an object with model_dump method."""

        class MockModel:
            def model_dump(self):
                return {"field": "value"}

        result = MockModel()
        serialized = _serialize_result(result)
        assert serialized == {"field": "value"}

    def test_serialize_mcp_result_like(self):
        """Test serializing an MCP-like result object."""

        class MockMCPResult:
            isError = False
            content = [{"text": "Result text"}]
            structuredContent = {"key": "value"}

        result = MockMCPResult()
        serialized = _serialize_result(result)

        assert serialized["isError"] is False
        assert serialized["content"] == [{"text": "Result text"}]
        assert serialized["structuredContent"] == {"key": "value"}

    def test_serialize_unknown_type(self):
        """Test serializing an unknown type falls back to string."""

        class UnknownType:
            pass

        result = UnknownType()
        serialized = _serialize_result(result)
        assert isinstance(serialized, str)
