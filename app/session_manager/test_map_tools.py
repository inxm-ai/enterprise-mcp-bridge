import pytest
from app.session_manager.session_context import map_tools
from unittest.mock import patch


class MockTool:
    def __init__(
        self, name, title, description, inputSchema, outputSchema, annotations, meta
    ):
        self.name = name
        self.title = title
        self.description = description
        self.inputSchema = inputSchema
        self.outputSchema = outputSchema
        self.annotations = annotations
        self.meta = meta


class MockTools:
    def __init__(self, tools):
        self.tools = tools


@pytest.fixture
def mock_tools():
    return MockTools(
        [
            MockTool("tool1", "Tool 1", "Description 1", {}, {}, {}, {}),
            MockTool("tool2", "Tool 2", "Description 2", {}, {}, {}, {}),
            MockTool(
                "special_tool", "Special Tool", "Special Description", {}, {}, {}, {}
            ),
        ]
    )


def test_works_without_filters(mock_tools):
    with patch("app.session_manager.session_context.INCLUDE_TOOLS", []), patch(
        "app.session_manager.session_context.EXCLUDE_TOOLS", []
    ):
        result = map_tools(mock_tools)
        assert len(result) == 3
        assert all(
            tool["name"] in ["tool1", "tool2", "special_tool"] for tool in result
        )


def test_include_tools(mock_tools):
    with patch("app.session_manager.session_context.INCLUDE_TOOLS", ["tool*"]), patch(
        "app.session_manager.session_context.EXCLUDE_TOOLS", []
    ):
        result = map_tools(mock_tools)
        assert len(result) == 2
        assert all(tool["name"] in ["tool1", "tool2"] for tool in result)


def test_exclude_tools(mock_tools):
    with patch("app.session_manager.session_context.INCLUDE_TOOLS", []), patch(
        "app.session_manager.session_context.EXCLUDE_TOOLS", ["tool*"]
    ):
        result = map_tools(mock_tools)
        assert len(result) == 1
        assert result[0]["name"] == "special_tool"


def test_include_and_exclude_tools(mock_tools):
    with patch("app.session_manager.session_context.INCLUDE_TOOLS", ["*"]), patch(
        "app.session_manager.session_context.EXCLUDE_TOOLS", ["special*"]
    ):
        result = map_tools(mock_tools)
        assert len(result) == 2
        assert all(tool["name"] in ["tool1", "tool2"] for tool in result)
