import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.server import app as fastapi_app
from app.routes import HTTP_STATUS_TOOL_EXECUTION_ERROR
from pydantic import BaseModel


class MockContent(BaseModel):
    text: str
    type: str = "text"
    structuredContent: dict | None = None


class MockResult(BaseModel):
    content: list[MockContent]
    isError: bool = False
    structuredContent: dict | None = None


@pytest.fixture
def client():
    return TestClient(fastapi_app)


@pytest.fixture
def mock_session_context():
    with patch("app.routes.mcp_session_context") as mock_ctx:
        mock_session = AsyncMock()
        mock_ctx.return_value.__aenter__.return_value = mock_session
        yield mock_session


def _call_failing_tool(client, mock_session_context, text):
    mock_session_context.call_tool.return_value = MockResult(
        content=[MockContent(text=text)], isError=True
    )
    return client.post(
        "/tools/test_tool", headers={"x-inxm-mcp-session": "test-session"}, json={}
    )


def test_tool_execution_error_is_client_error_not_server_error(
    client, mock_session_context
):
    """A tool that ran and rejected the request must not look like a bridge fault.

    Callers key their retry decision off the status class, so a deterministic
    failure reported as 5xx gets retried until the budget is gone.
    """
    response = _call_failing_tool(
        client,
        mock_session_context,
        "Error executing tool search_social_media: Max 20 URLs are allowed.",
    )

    assert response.status_code == HTTP_STATUS_TOOL_EXECUTION_ERROR
    assert 400 <= response.status_code < 500
    assert "Max 20 URLs are allowed." in response.json()["detail"]


def test_unknown_tool_still_maps_to_404(client, mock_session_context):
    response = _call_failing_tool(
        client, mock_session_context, "Unknown tool: nope_tool"
    )

    assert response.status_code == 404


def test_validation_error_still_maps_to_400(client, mock_session_context):
    response = _call_failing_tool(
        client, mock_session_context, "1 validation error for test_tool"
    )

    assert response.status_code == 400


def test_error_result_with_empty_content_does_not_crash(client, mock_session_context):
    """An isError result carrying no content used to raise IndexError, which the
    outer handler turned into a generic 500 that discarded the real error."""
    mock_session_context.call_tool.return_value = MockResult(content=[], isError=True)

    response = client.post(
        "/tools/test_tool", headers={"x-inxm-mcp-session": "test-session"}, json={}
    )

    assert response.status_code == HTTP_STATUS_TOOL_EXECUTION_ERROR
