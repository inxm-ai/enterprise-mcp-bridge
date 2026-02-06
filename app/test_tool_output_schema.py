import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.server import app as fastapi_app
import app.vars
import json
from pydantic import BaseModel
import importlib
import tempfile
import os


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


def test_tool_output_schema_parsing(client, mock_session_context):
    # Setup the tool schema in app.vars
    test_schema = {"type": "object", "properties": {"result": {"type": "string"}}}

    with patch.dict(app.vars.TOOL_OUTPUT_SCHEMAS, {"test_tool": test_schema}):
        # Setup mock tool execution response
        # The tool returns a text that is a JSON string
        tool_response_content = MockContent(text=json.dumps({"result": "success"}))
        tool_result = MockResult(content=[tool_response_content])

        mock_session_context.call_tool.return_value = tool_result

        # Call the tool
        response = client.post(
            "/tools/test_tool", headers={"x-inxm-mcp-session": "test-session"}, json={}
        )

        assert response.status_code == 200
        data = response.json()

        # Check structuredContent in the result object
        assert "structuredContent" in data
        assert data["structuredContent"] == {"result": "success"}


def test_tool_output_schema_in_details(client, mock_session_context):
    # Setup the tool schema in app.vars
    test_schema = {"type": "object", "properties": {"foo": {"type": "string"}}}

    with patch.dict(app.vars.TOOL_OUTPUT_SCHEMAS, {"test_tool_2": test_schema}):
        # Mock list_tools output
        # list_tools returns a list of tools (dicts or objects). Assuming dicts based on current code.
        mock_session_context.list_tools.return_value = [
            {"name": "test_tool_2", "description": "Test Tool"}
        ]

        response = client.get(
            "/tools/test_tool_2", headers={"x-inxm-mcp-session": "test-session"}
        )

        assert response.status_code == 200
        data = response.json()

        # This assertion is expected to fail initially
        assert "outputSchema" in data
        assert data["outputSchema"] == test_schema


def test_tool_output_schema_from_file():
    # Create a dummy schema file
    file_schema = {"type": "object", "properties": {"bar": {"type": "number"}}}
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(file_schema, tmp)
        tmp_path = tmp.name

    try:
        # Set env var
        env_val = json.dumps({"test_tool_file": tmp_path})
        with patch.dict(os.environ, {"TOOL_OUTPUT_SCHEMAS": env_val}):
            # Reload module to trigger _load_tool_output_schemas
            importlib.reload(app.vars)

            assert "test_tool_file" in app.vars.TOOL_OUTPUT_SCHEMAS
            assert app.vars.TOOL_OUTPUT_SCHEMAS["test_tool_file"] == file_schema

    finally:
        os.remove(tmp_path)
        # cleanup: reload vars with empty env to reset state
        with patch.dict(os.environ, {"TOOL_OUTPUT_SCHEMAS": "{}"}):
            importlib.reload(app.vars)
