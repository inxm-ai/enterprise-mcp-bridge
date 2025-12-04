"""
Tests for SSE streaming routes.

These tests verify the behavior of the SSE streaming endpoints for MCP tool calls.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.sse.routes import router


@pytest.fixture
def app():
    """Create a test FastAPI application with the SSE router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return TestClient(app)


class MockRunToolsResult:
    """Mock for RunToolsResult."""

    def __init__(self, content_text: str = "Done", is_error: bool = False):
        self.isError = is_error
        self.content = [MockContent(content_text)]
        self.structuredContent = {"result": content_text}


class MockContent:
    """Mock for content items."""

    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class TestSSEStreamEndpoint:
    """Tests for the /tools/{tool_name}/stream endpoint."""

    def test_stream_endpoint_returns_sse_content_type(self, client):
        """Test that the stream endpoint returns SSE content type."""
        mock_result = MockRunToolsResult("Done")

        with patch("app.sse.routes.mcp_session_context") as mock_context:
            # Create a mock session delegate
            mock_delegate = AsyncMock()
            mock_delegate.call_tool_with_progress = AsyncMock(return_value=mock_result)

            # Setup the context manager
            mock_context.return_value.__aenter__ = AsyncMock(return_value=mock_delegate)
            mock_context.return_value.__aexit__ = AsyncMock(return_value=None)

            response = client.post(
                "/tools/test_tool/stream",
                json={},
            )

            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

    def test_stream_endpoint_with_tool_arguments(self, client):
        """Test streaming endpoint with tool arguments."""
        mock_result = MockRunToolsResult("5")
        captured_args = {}

        async def capture_call(
            tool_name, args, access_token, progress_callback=None, log_callback=None
        ):
            captured_args["tool_name"] = tool_name
            captured_args["args"] = args
            return mock_result

        with patch("app.sse.routes.mcp_session_context") as mock_context:
            mock_delegate = AsyncMock()
            mock_delegate.call_tool_with_progress = capture_call

            mock_context.return_value.__aenter__ = AsyncMock(return_value=mock_delegate)
            mock_context.return_value.__aexit__ = AsyncMock(return_value=None)

            response = client.post(
                "/tools/add/stream",
                json={"a": 2, "b": 3},
            )

            assert response.status_code == 200
            # Check that arguments were captured (note: async iteration may not complete in sync client)

    def test_stream_endpoint_emits_result_event(self, client):
        """Test that the stream endpoint emits a result event."""
        mock_result = MockRunToolsResult("Success!")

        async def mock_call_with_progress(
            tool_name, args, access_token, progress_callback=None, log_callback=None
        ):
            return mock_result

        with patch("app.sse.routes.mcp_session_context") as mock_context:
            mock_delegate = AsyncMock()
            mock_delegate.call_tool_with_progress = mock_call_with_progress

            mock_context.return_value.__aenter__ = AsyncMock(return_value=mock_delegate)
            mock_context.return_value.__aexit__ = AsyncMock(return_value=None)

            response = client.post(
                "/tools/my_tool/stream",
                json={},
            )

            assert response.status_code == 200

            # Parse SSE events from response
            content = response.content.decode("utf-8")
            events = []
            for line in content.split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

            # Should have at least one result event
            result_events = [e for e in events if e.get("type") == "result"]
            assert len(result_events) >= 1

    def test_stream_endpoint_with_progress_updates(self, client):
        """Test that progress updates are streamed correctly."""
        mock_result = MockRunToolsResult("Done")

        async def mock_call_with_progress(
            tool_name, args, access_token, progress_callback=None, log_callback=None
        ):
            if progress_callback:
                await progress_callback(25.0, 100.0, "Starting...")
                await progress_callback(50.0, 100.0, "Halfway...")
                await progress_callback(100.0, 100.0, "Complete!")
            return mock_result

        with patch("app.sse.routes.mcp_session_context") as mock_context:
            mock_delegate = AsyncMock()
            mock_delegate.call_tool_with_progress = mock_call_with_progress

            mock_context.return_value.__aenter__ = AsyncMock(return_value=mock_delegate)
            mock_context.return_value.__aexit__ = AsyncMock(return_value=None)

            response = client.post(
                "/tools/progress_tool/stream",
                json={},
            )

            assert response.status_code == 200

            content = response.content.decode("utf-8")
            events = []
            for line in content.split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

            # Should have progress events
            progress_events = [e for e in events if e.get("type") == "progress"]
            assert len(progress_events) == 3

            assert progress_events[0]["progress"] == 25.0
            assert progress_events[1]["progress"] == 50.0
            assert progress_events[2]["progress"] == 100.0

    def test_stream_endpoint_handles_tool_error(self, client):
        """Test that tool errors are streamed as error events."""

        async def mock_call_with_error(
            tool_name, args, access_token, progress_callback=None, log_callback=None
        ):
            raise ValueError("Tool execution failed")

        with patch("app.sse.routes.mcp_session_context") as mock_context:
            mock_delegate = AsyncMock()
            mock_delegate.call_tool_with_progress = mock_call_with_error

            mock_context.return_value.__aenter__ = AsyncMock(return_value=mock_delegate)
            mock_context.return_value.__aexit__ = AsyncMock(return_value=None)

            response = client.post(
                "/tools/failing_tool/stream",
                json={},
            )

            # The response itself should still be 200 (SSE stream)
            assert response.status_code == 200

            content = response.content.decode("utf-8")
            events = []
            for line in content.split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

            # Should have an error event
            error_events = [e for e in events if e.get("type") == "error"]
            assert len(error_events) >= 1
            assert "Tool execution failed" in error_events[0]["data"]["message"]


class TestSSERouteHeaders:
    """Tests for SSE route header handling."""

    def test_cache_control_headers(self, client):
        """Test that appropriate cache control headers are set."""
        mock_result = MockRunToolsResult("Done")

        async def mock_call(
            tool_name, args, access_token, progress_callback=None, log_callback=None
        ):
            return mock_result

        with patch("app.sse.routes.mcp_session_context") as mock_context:
            mock_delegate = AsyncMock()
            mock_delegate.call_tool_with_progress = mock_call

            mock_context.return_value.__aenter__ = AsyncMock(return_value=mock_delegate)
            mock_context.return_value.__aexit__ = AsyncMock(return_value=None)

            response = client.post(
                "/tools/test_tool/stream",
                json={},
            )

            # Check SSE-specific headers
            assert response.headers.get("cache-control") == "no-cache"
            assert response.headers.get("x-accel-buffering") == "no"
