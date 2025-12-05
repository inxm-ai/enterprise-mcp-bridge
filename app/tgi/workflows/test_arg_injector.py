"""Unit tests for ArgInjector and ToolResultCapture."""

import json

from app.tgi.workflows.arg_injector import (
    ArgInjector,
    ReturnSpec,
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

    def test_capture_nested_path_with_dotted_notation(self):
        """Capture nested fields using dotted notation like 'payload.result'.

        This test reproduces a real-world bug where tool results have nested
        structure like:
        {
            "isError": false,
            "payload": {
                "result": { ... the actual plan object ... }
            }
        }

        And the workflow wants to access it via 'create_plan.payload.result'.
        """
        capture = ToolResultCapture("create_plan", ["payload.result"])
        context = {"agents": {}}

        # This is the structure returned by MCP tools with structuredContent
        tool_result = json.dumps(
            {
                "isError": False,
                "payload": {
                    "result": {
                        "name": "My Plan",
                        "plan_id": "123",
                        "phases": ["phase1", "phase2"],
                    }
                },
            }
        )

        result = capture.capture(tool_result, context)

        # The nested value should be stored at the dotted path
        assert result["agents"]["create_plan"]["payload"]["result"] == {
            "name": "My Plan",
            "plan_id": "123",
            "phases": ["phase1", "phase2"],
        }

    def test_capture_nested_path_integration_with_arg_injector(self):
        """End-to-end test: capture nested result, then inject it into another tool.

        This tests the full workflow:
        1. Tool result with nested 'payload.result' is captured
        2. Another agent's tool args reference 'create_plan.payload.result'
        3. ArgInjector correctly resolves the nested path
        """
        # Step 1: Capture the nested result
        capture = ToolResultCapture("create_plan", ["payload.result"])
        context = {"agents": {}}

        tool_result = json.dumps(
            {
                "isError": False,
                "payload": {
                    "result": {"name": "Email Monitoring Plan", "plan_id": "uuid-123"}
                },
            }
        )

        capture.capture(tool_result, context)

        # Step 2: Use ArgInjector to inject the captured value into save_plan tool
        injector = ArgInjector(
            [ToolArgMapping("save_plan", {"plan": "create_plan.payload.result"})]
        )

        result = injector.inject("save_plan", {}, context)

        # The plan should be injected correctly
        assert result == {
            "plan": {"name": "Email Monitoring Plan", "plan_id": "uuid-123"}
        }

    def test_capture_deeply_nested_path(self):
        """Capture deeply nested fields like 'data.response.items[0].value'."""
        capture = ToolResultCapture("agent", ["data.nested.deep"])
        context = {"agents": {}}

        tool_result = json.dumps({"data": {"nested": {"deep": {"key": "value"}}}})

        result = capture.capture(tool_result, context)

        assert result["agents"]["agent"]["data"]["nested"]["deep"] == {"key": "value"}

    def test_capture_mixed_top_level_and_nested_returns(self):
        """Capture both top-level and nested fields in the same returns list."""
        capture = ToolResultCapture("agent", ["simple", "nested.field"])
        context = {"agents": {}}

        tool_result = json.dumps(
            {"simple": "top-level value", "nested": {"field": "nested value"}}
        )

        result = capture.capture(tool_result, context)

        assert result["agents"]["agent"]["simple"] == "top-level value"
        assert result["agents"]["agent"]["nested"]["field"] == "nested value"


class TestReturnSpec:
    """Tests for ReturnSpec parsing."""

    def test_parse_simple_string(self):
        """Parse a simple string field name."""
        spec = ReturnSpec.parse("field")
        assert spec.field == "field"
        assert spec.from_tool is None
        assert spec.as_name is None

    def test_parse_dotted_string(self):
        """Parse a dotted path string."""
        spec = ReturnSpec.parse("payload.result")
        assert spec.field == "payload.result"
        assert spec.from_tool is None
        assert spec.as_name is None

    def test_parse_dict_with_from(self):
        """Parse a dict with 'from' to specify tool."""
        spec = ReturnSpec.parse({"field": "payload.result", "from": "plan"})
        assert spec.field == "payload.result"
        assert spec.from_tool == "plan"
        assert spec.as_name is None

    def test_parse_dict_with_as_alias(self):
        """Parse a dict with 'as' to specify storage alias."""
        spec = ReturnSpec.parse({"field": "payload.result", "as": "plan_data"})
        assert spec.field == "payload.result"
        assert spec.from_tool is None
        assert spec.as_name == "plan_data"

    def test_parse_dict_with_all_options(self):
        """Parse a dict with all options."""
        spec = ReturnSpec.parse(
            {"field": "payload.result", "from": "plan", "as": "my_plan"}
        )
        assert spec.field == "payload.result"
        assert spec.from_tool == "plan"
        assert spec.as_name == "my_plan"


class TestToolResultCaptureWithToolFilter:
    """Tests for tool-specific result capture."""

    def test_capture_from_specific_tool(self):
        """Capture only from a specific tool when 'from' is specified."""
        capture = ToolResultCapture(
            "agent", [{"field": "result", "from": "target_tool"}]
        )
        context = {"agents": {}}

        # Result from wrong tool - should NOT be captured
        wrong_result = json.dumps({"result": "wrong"})
        capture.capture(wrong_result, context, tool_name="other_tool")

        assert "result" not in context["agents"].get("agent", {})

        # Result from correct tool - should be captured
        correct_result = json.dumps({"result": "correct"})
        capture.capture(correct_result, context, tool_name="target_tool")

        assert context["agents"]["agent"]["result"] == "correct"

    def test_capture_from_any_tool_when_no_filter(self):
        """Capture from any tool when 'from' is not specified."""
        capture = ToolResultCapture("agent", ["result"])
        context = {"agents": {}}

        tool_result = json.dumps({"result": "captured"})
        capture.capture(tool_result, context, tool_name="any_tool")

        assert context["agents"]["agent"]["result"] == "captured"

    def test_capture_with_alias(self):
        """Store captured value under an alias using 'as'."""
        capture = ToolResultCapture(
            "agent", [{"field": "payload.result", "as": "plan"}]
        )
        context = {"agents": {}}

        tool_result = json.dumps(
            {"payload": {"result": {"id": "123", "name": "My Plan"}}}
        )
        capture.capture(tool_result, context)

        # Should be stored under the alias, not the original path
        assert context["agents"]["agent"]["plan"] == {"id": "123", "name": "My Plan"}
        assert "payload" not in context["agents"]["agent"]

    def test_capture_multiple_tools_different_fields(self):
        """Capture different fields from different tools."""
        capture = ToolResultCapture(
            "agent",
            [
                {"field": "tools", "from": "select_tools"},
                {"field": "payload.result", "from": "plan", "as": "plan_data"},
            ],
        )
        context = {"agents": {}}

        # First tool result
        select_result = json.dumps({"tools": ["tool1", "tool2"]})
        capture.capture(select_result, context, tool_name="select_tools")

        # Second tool result
        plan_result = json.dumps({"payload": {"result": {"phases": ["p1", "p2"]}}})
        capture.capture(plan_result, context, tool_name="plan")

        assert context["agents"]["agent"]["tools"] == ["tool1", "tool2"]
        assert context["agents"]["agent"]["plan_data"] == {"phases": ["p1", "p2"]}

    def test_capture_ignores_non_matching_tool(self):
        """When from is specified, results from other tools are ignored."""
        capture = ToolResultCapture(
            "agent", [{"field": "data", "from": "specific_tool"}]
        )
        context = {"agents": {}}

        # Results from multiple tools, only one matches
        for tool in ["tool_a", "tool_b", "tool_c"]:
            result = json.dumps({"data": f"from_{tool}"})
            capture.capture(result, context, tool_name=tool)

        # None should be captured
        assert "data" not in context["agents"].get("agent", {})

        # Now the matching tool
        result = json.dumps({"data": "from_specific"})
        capture.capture(result, context, tool_name="specific_tool")

        assert context["agents"]["agent"]["data"] == "from_specific"

    def test_mixed_filtered_and_unfiltered_returns(self):
        """Mix of filtered and unfiltered return specs in same agent."""
        capture = ToolResultCapture(
            "agent",
            [
                "any_field",  # Capture from any tool
                {"field": "specific_field", "from": "my_tool"},  # Only from my_tool
            ],
        )
        context = {"agents": {}}

        # Result from random tool
        result1 = json.dumps({"any_field": "value1", "specific_field": "wrong"})
        capture.capture(result1, context, tool_name="random_tool")

        assert context["agents"]["agent"]["any_field"] == "value1"
        assert "specific_field" not in context["agents"]["agent"]

        # Result from the specific tool
        result2 = json.dumps({"specific_field": "correct"})
        capture.capture(result2, context, tool_name="my_tool")

        assert context["agents"]["agent"]["specific_field"] == "correct"

    def test_backward_compatible_with_no_tool_name(self):
        """When tool_name is not provided, all specs without 'from' match."""
        capture = ToolResultCapture("agent", ["field"])
        context = {"agents": {}}

        # No tool_name provided (backward compatibility)
        result = json.dumps({"field": "value"})
        capture.capture(result, context)

        assert context["agents"]["agent"]["field"] == "value"

    def test_integration_with_arg_injector(self):
        """Full integration: capture from specific tool, then inject into another."""
        # Agent calls select_tools and plan, captures from plan only
        capture = ToolResultCapture(
            "create_plan", [{"field": "payload.result", "from": "plan", "as": "plan"}]
        )
        context = {"agents": {}}

        # select_tools result (should be ignored for this capture)
        select_result = json.dumps({"selected_tools": ["t1", "t2"]})
        capture.capture(select_result, context, tool_name="select_tools")

        # plan result (should be captured)
        plan_result = json.dumps(
            {"payload": {"result": {"name": "My Plan", "phases": ["phase1"]}}}
        )
        capture.capture(plan_result, context, tool_name="plan")

        # Now inject into save_plan
        injector = ArgInjector(
            [ToolArgMapping("save_plan", {"plan": "create_plan.plan"})]
        )

        result = injector.inject("save_plan", {}, context)

        assert result == {"plan": {"name": "My Plan", "phases": ["phase1"]}}
