import json
import types
import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
import app.well_known.agent as agent_module
from app.well_known.agent import get_agent_card
from app.vars import SERVICE_NAME


@pytest.fixture
def mock_llm_client():
    # Patch where LLMClient is used (inside the agent module)
    with patch("app.well_known.agent.LLMClient", autospec=True) as mock_cls:
        instance = mock_cls.return_value
        # Return a simple constant for all ask() calls to avoid side-effect exhaustion
        instance.ask = AsyncMock(return_value="Service description summary")
        yield mock_cls


@pytest.fixture
def mock_mcp_session():
    # Patch where mcp_session is used (inside the agent module)
    with patch("app.well_known.agent.mcp_session") as mock_session:

        # Helper to convert dicts to objects with attribute access, filling missing keys
        def dict_to_obj(d):
            # List of all expected keys in tool objects
            expected_keys = [
                "id",
                "name",
                "title",
                "description",
                "inputSchema",
                "outputSchema",
                "annotations",
                "meta",
            ]
            # Fill missing keys with None
            filled = {k: d.get(k, None) for k in expected_keys}
            # Also include any extra keys from d
            filled.update(d)
            # Ensure 'id' is present and matches d['id'] if available, else fallback to d['name']
            if not filled.get("id"):
                filled["id"] = d.get("id") or d.get("name")
            return types.SimpleNamespace(**filled)

        class ToolsWrapper:
            def __init__(self, tools):
                self.tools = tools

        # Async context manager class
        class _AsyncMCPContext:
            def __init__(self, tools):
                self._tools = ToolsWrapper(tools)

            async def __aenter__(self):
                obj = types.SimpleNamespace()
                obj.list_tools = AsyncMock(return_value=self._tools)
                return obj

            async def __aexit__(self, exc_type, exc, tb):
                return False

        # Store tool objects (not dicts)
        tools_state = []

        def factory(*args, **kwargs):
            return _AsyncMCPContext(tools_state)

        # Accepts a list of tool dicts, converts to objects
        def set_tools(new_tools):
            tools_state.clear()
            tools_state.extend([dict_to_obj(t) for t in new_tools])

        mock_session.side_effect = factory
        mock_session.set_tools = set_tools  # helper for tests
        yield mock_session


@pytest.fixture
def mock_get_server_params():
    # Avoid executing real get_server_params
    with patch("app.well_known.agent.get_server_params", autospec=True) as mock_gsp:
        mock_gsp.return_value = object()
        yield mock_gsp


@pytest.fixture
def ensure_default_model():
    # Ensure DEFAULT_MODEL is truthy for tests that need it
    with patch("app.well_known.agent.DEFAULT_MODEL", "test-model"):
        yield


# Helper to reset agent module globals before each test
@pytest.fixture(autouse=True)
def reset_agent_globals():
    agent_module._running = False
    agent_module._agent_card_cache = None
    yield


@pytest.mark.asyncio
async def test_get_agent_card_happy_case(
    mock_llm_client, mock_mcp_session, mock_get_server_params, ensure_default_model
):
    # Configure the fake session to return two tools
    mock_mcp_session.set_tools(
        [
            {"id": "tool1", "name": "Tool 1", "description": "Tool 1 description"},
            {"id": "tool2", "name": "Tool 2", "description": "Tool 2 description"},
        ]
    )

    response = await get_agent_card()

    assert response.status_code == 200
    agent_card = json.loads(response.body.decode("utf-8"))
    assert agent_card["name"] == SERVICE_NAME
    assert agent_card["description"] == "Service description summary"
    assert len(agent_card["skills"]) == 3  # Default skill + 2 tools


@pytest.mark.asyncio
async def test_get_agent_card_no_default_model():
    agent_module._agent_card_cache = None

    # Patch the symbol as used inside the agent module
    with patch("app.well_known.agent.DEFAULT_MODEL", None):
        with pytest.raises(HTTPException) as exc_info:
            await get_agent_card()

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not Found"


@pytest.mark.asyncio
async def test_get_agent_card_cache_hit(
    mock_llm_client, mock_mcp_session, ensure_default_model
):
    agent_module._agent_card_cache = {
        "name": "Cached Agent",
        "description": "Cached description",
    }

    response = await get_agent_card()

    assert response.status_code == 200
    agent_card = json.loads(response.body.decode("utf-8"))
    assert agent_card["name"] == "Cached Agent"
    assert agent_card["description"] == "Cached description"


@pytest.mark.asyncio
async def test_get_agent_card_empty_tools(
    mock_llm_client, mock_mcp_session, mock_get_server_params, ensure_default_model
):
    agent_module._agent_card_cache = None

    # Leave tools empty via default fixture configuration
    mock_mcp_session.set_tools([])

    response = await get_agent_card()

    assert response.status_code == 200
    agent_card = json.loads(response.body.decode("utf-8"))
    assert agent_card["name"] == SERVICE_NAME
    assert agent_card["description"] == "Service description summary"
    assert len(agent_card["skills"]) == 1  # Only default skill


@pytest.mark.asyncio
async def test_get_agent_card_tool_with_schemas(
    mock_llm_client, mock_mcp_session, mock_get_server_params, ensure_default_model
):
    # Verify inputModes/outputModes and parameter_schema when tool has schemas
    tools = [
        {
            "name": "Schema Tool",
            "description": "Uses schemas",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object"},
        }
    ]
    # Inject tools via helper
    mock_mcp_session.set_tools(tools)

    response = await get_agent_card()
    assert response.status_code == 200
    agent_card = json.loads(response.body.decode("utf-8"))
    # Skills: default + 1 tool
    assert len(agent_card["skills"]) == 2
    schema_skill = next(s for s in agent_card["skills"] if s["name"] == "Schema Tool")
    assert schema_skill["inputModes"] == ["data"]
    assert schema_skill["outputModes"] == ["data"]
    assert "parameter_schema" in schema_skill
