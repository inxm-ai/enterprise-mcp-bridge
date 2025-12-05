"""Unit tests for ArgInjector and ToolResultCapture."""

from app.tgi.workflows.arg_injector import (
    ArgInjector,
    ToolArgMapping,
    ToolResultCapture,
)


class TestArgInjector:
    """Tests for ArgInjector class."""

    def test_inject_returns_original_args_when_no_mapping(self):
        """When tool has no mapping, return args unchanged."""
        injector = ArgInjector([])
        args = {"title": "My Plan"}
        context = {}

        result = injector.inject("plan", args, context)

        assert result == {"title": "My Plan"}

    def test_inject_merges_mapped_arg_from_context(self):
        """When mapping exists, inject value from context."""
        injector = ArgInjector(
            [ToolArgMapping("plan", {"selected_tools": "select_tools.selected_tools"})]
        )
        args = {"title": "My Plan"}
        context = {
            "agents": {
                "select_tools": {
                    "selected_tools": ["tool1", "tool2"],
                }
            }
        }

        result = injector.inject("plan", args, context)

        assert result == {
            "title": "My Plan",
            "selected_tools": ["tool1", "tool2"],
        }

    def test_inject_preserves_original_args_object(self):
        """Injection should not mutate the original args dict."""
        injector = ArgInjector(
            [ToolArgMapping("plan", {"selected_tools": "select_tools.selected_tools"})]
        )
        args = {"title": "My Plan"}
        context = {"agents": {"select_tools": {"selected_tools": ["t1"]}}}

        result = injector.inject("plan", args, context)

        assert "selected_tools" not in args
        assert "selected_tools" in result

    def test_inject_handles_missing_agent_in_context(self):
        """When agent not in context, don't inject (value is None)."""
        injector = ArgInjector(
            [ToolArgMapping("plan", {"selected_tools": "select_tools.selected_tools"})]
        )
        args = {"title": "My Plan"}
        context = {"agents": {}}  # select_tools not present

        result = injector.inject("plan", args, context)

        # Original args returned, no injection happened
        assert result == {"title": "My Plan"}

    def test_inject_handles_missing_field_in_agent(self):
        """When field not in agent data, don't inject."""
        injector = ArgInjector(
            [ToolArgMapping("plan", {"selected_tools": "select_tools.selected_tools"})]
        )
        args = {"title": "My Plan"}
        context = {"agents": {"select_tools": {"content": "something else"}}}

        result = injector.inject("plan", args, context)

        assert result == {"title": "My Plan"}

    def test_inject_handles_nested_paths(self):
        """Support nested paths like 'agent.result.items'."""
        injector = ArgInjector(
            [ToolArgMapping("process", {"items": "fetch.result.items"})]
        )
        args = {}
        context = {"agents": {"fetch": {"result": {"items": [1, 2, 3]}}}}

        result = injector.inject("process", args, context)

        assert result == {"items": [1, 2, 3]}

    def test_inject_multiple_args(self):
        """Inject multiple args at once."""
        injector = ArgInjector(
            [
                ToolArgMapping(
                    "plan",
                    {
                        "selected_tools": "select_tools.selected_tools",
                        "context_info": "analyzer.summary",
                    },
                )
            ]
        )
        args = {"title": "My Plan"}
        context = {
            "agents": {
                "select_tools": {"selected_tools": ["t1"]},
                "analyzer": {"summary": "Analysis complete"},
            }
        }

        result = injector.inject("plan", args, context)

        assert result == {
            "title": "My Plan",
            "selected_tools": ["t1"],
            "context_info": "Analysis complete",
        }

    def test_has_mapping(self):
        """Check if injector has mapping for a tool."""
        injector = ArgInjector([ToolArgMapping("plan", {"x": "a.b"})])

        assert injector.has_mapping("plan") is True
        assert injector.has_mapping("other") is False

    def test_from_agent_def_extracts_mappings(self):
        """Create injector from an agent def with tool configs."""

        class MockAgentDef:
            tools = [
                {"plan": {"args": {"selected_tools": "select_tools.selected_tools"}}},
                "simple_tool",  # No args mapping
            ]

        injector = ArgInjector.from_agent_def(MockAgentDef())

        assert injector is not None
        assert injector.has_mapping("plan")
        assert not injector.has_mapping("simple_tool")

    def test_from_agent_def_returns_none_for_no_mappings(self):
        """Return None when no tools have args mappings."""

        class MockAgentDef:
            tools = ["tool1", "tool2"]

        injector = ArgInjector.from_agent_def(MockAgentDef())

        assert injector is None


class TestToolResultCapture:
    """Tests for ToolResultCapture class."""

    def test_capture_extracts_specified_fields(self):
        """Capture specified fields from tool result JSON."""
        capture = ToolResultCapture("select_tools", ["selected_tools"])
        context = {"agents": {}}

        result = capture.capture(
            '{"selected_tools": ["t1", "t2"], "other": "ignored"}', context
        )

        assert result["agents"]["select_tools"]["selected_tools"] == ["t1", "t2"]
        assert "other" not in result["agents"]["select_tools"]

    def test_capture_creates_agent_entry_if_missing(self):
        """Create agent entry in context if it doesn't exist."""
        capture = ToolResultCapture("new_agent", ["result"])
        context = {}

        result = capture.capture('{"result": 42}', context)

        assert result["agents"]["new_agent"]["result"] == 42

    def test_capture_handles_invalid_json(self):
        """Return context unchanged for invalid JSON."""
        capture = ToolResultCapture("agent", ["field"])
        context = {"agents": {"agent": {"existing": "value"}}}

        result = capture.capture("not valid json", context)

        assert result == {"agents": {"agent": {"existing": "value"}}}

    def test_capture_handles_non_dict_json(self):
        """Return context unchanged when JSON is not an object."""
        capture = ToolResultCapture("agent", ["field"])
        context = {"agents": {}}

        result = capture.capture("[1, 2, 3]", context)

        assert result == {"agents": {}}

    def test_capture_ignores_missing_fields(self):
        """Only capture fields that exist in the result."""
        capture = ToolResultCapture("agent", ["present", "missing"])
        context = {"agents": {}}

        result = capture.capture('{"present": "value"}', context)

        assert result["agents"]["agent"]["present"] == "value"
        assert "missing" not in result["agents"]["agent"]

    def test_capture_multiple_fields(self):
        """Capture multiple fields from result."""
        capture = ToolResultCapture("agent", ["field1", "field2"])
        context = {"agents": {}}

        result = capture.capture(
            '{"field1": "a", "field2": "b", "field3": "c"}', context
        )

        assert result["agents"]["agent"] == {"field1": "a", "field2": "b"}

    def test_capture_preserves_existing_agent_data(self):
        """Capture adds to existing agent data, doesn't replace it."""
        capture = ToolResultCapture("agent", ["new_field"])
        context = {"agents": {"agent": {"existing": "value", "content": "text"}}}

        result = capture.capture('{"new_field": "new_value"}', context)

        assert result["agents"]["agent"]["existing"] == "value"
        assert result["agents"]["agent"]["content"] == "text"
        assert result["agents"]["agent"]["new_field"] == "new_value"
