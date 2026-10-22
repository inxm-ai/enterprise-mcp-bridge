import pytest
from app.session_manager.session_context import map_tools
from unittest.mock import patch
import app.session_manager.session_context as session_context
from mcp import types as mcp_types


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
    with (
        patch("app.session_manager.session_context.INCLUDE_TOOLS", []),
        patch("app.session_manager.session_context.EXCLUDE_TOOLS", []),
    ):
        result = map_tools(mock_tools)
        assert len(result) == 3
        assert all(
            tool["name"] in ["tool1", "tool2", "special_tool"] for tool in result
        )


def test_include_tools(mock_tools):
    with (
        patch("app.session_manager.session_context.INCLUDE_TOOLS", ["tool*"]),
        patch("app.session_manager.session_context.EXCLUDE_TOOLS", []),
    ):
        result = map_tools(mock_tools)
        assert len(result) == 2
        assert all(tool["name"] in ["tool1", "tool2"] for tool in result)


def test_exclude_tools(mock_tools):
    with (
        patch("app.session_manager.session_context.INCLUDE_TOOLS", []),
        patch("app.session_manager.session_context.EXCLUDE_TOOLS", ["tool*"]),
    ):
        result = map_tools(mock_tools)
        assert len(result) == 1
        assert result[0]["name"] == "special_tool"


def test_include_and_exclude_tools(mock_tools):
    with (
        patch("app.session_manager.session_context.INCLUDE_TOOLS", ["*"]),
        patch("app.session_manager.session_context.EXCLUDE_TOOLS", ["special*"]),
    ):
        result = map_tools(mock_tools)
        assert len(result) == 2
        assert all(tool["name"] in ["tool1", "tool2"] for tool in result)


def test_output_schema_pluralized_name_alias_is_applied():
    expected_schema = {
        "type": "object",
        "properties": {"value": {"type": "array"}},
        "required": ["value"],
    }
    tools = MockTools(
        [
            MockTool(
                "list-team-channels",
                None,
                "Retrieve the list of channels in this team.",
                {"type": "object"},
                None,
                {},
                None,
            )
        ]
    )

    with patch.dict(
        session_context.TOOL_OUTPUT_SCHEMAS,
        {"list-teams-channels": expected_schema},
        clear=True,
    ):
        mapped = map_tools(tools)

    assert mapped[0]["outputSchema"] == expected_schema


def test_read_only_tools_are_annotated_read_only_over_upstream_hints():
    tools = MockTools(
        [
            MockTool("search_traces", None, "", {}, None, None, None),
            MockTool(
                "get_services",
                None,
                "",
                {},
                None,
                {"title": "Services", "readOnlyHint": False},
                None,
            ),
            MockTool("delete_trace", None, "", {}, None, {"title": "Delete"}, None),
        ]
    )

    with patch.object(session_context, "READ_ONLY_TOOLS", ["get_*", "search_traces"]):
        mapped = {tool["name"]: tool["annotations"] for tool in map_tools(tools)}

    assert mapped["search_traces"] == {"readOnlyHint": True, "destructiveHint": False}
    assert mapped["get_services"] == {
        "title": "Services",
        "readOnlyHint": True,
        "destructiveHint": False,
    }
    assert mapped["delete_trace"] == {"title": "Delete"}


def test_read_only_hint_keeps_the_fields_of_sdk_annotations():
    tools = MockTools(
        [
            MockTool(
                "get_services",
                None,
                "",
                {},
                None,
                mcp_types.ToolAnnotations(title="Services", openWorldHint=False),
                None,
            )
        ]
    )

    with patch.object(session_context, "READ_ONLY_TOOLS", ["get_services"]):
        mapped = map_tools(tools)

    assert mapped[0]["annotations"] == {
        "title": "Services",
        "openWorldHint": False,
        "readOnlyHint": True,
        "destructiveHint": False,
    }
