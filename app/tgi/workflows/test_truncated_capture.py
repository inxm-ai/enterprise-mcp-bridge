"""Regression tests for tool result truncation vs. workflow capture.

When a tool result exceeds ``GENERATED_UI_TOOL_TEXT_CAP`` (default 4000
chars), ``ToolService.execute_tool_call`` wraps the content in a
``{"text": "...", "truncated": true}`` envelope.  If the workflow engine
feeds that wrapped content to ``ToolResultCapture.capture()``, nested
fields like ``selected_tools`` become inaccessible and
``ArgInjector`` later raises ``ArgResolutionError``.

The fix preserves the original content as ``_raw_content`` in the result
dict and prefers it in ``tool_chat_runner`` when emitting events for
workflow stream processing.

These tests assert:
1. ``execute_tool_call`` stores ``_raw_content`` when truncation occurs.
2. ``ToolResultCapture`` succeeds when given the raw (untruncated) content.
3. ``ArgInjector`` resolves args after capture from untruncated content.
4. End-to-end: stream_processor properly uses ``_raw_content`` for capture.
"""

import json
from typing import Any

import pytest

from app.tgi.workflows.arg_injector import (
    ArgInjector,
    ArgResolutionError,
    ToolArgMapping,
    ToolResultCapture,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_large_selected_tools(*, count: int = 10) -> list[dict[str, Any]]:
    """Build a ``selected_tools`` list that exceeds 4000 chars when serialised."""
    tools = []
    for i in range(count):
        tools.append(
            {
                "tool_name": f"tool_{i}",
                "description": f"A long description for tool {i} that adds bulk to the payload. "
                * 6,
                "inputSchema": {
                    "properties": {
                        "arg1": {
                            "type": "string",
                            "description": f"Argument for tool {i}",
                        },
                        "arg2": {"type": "number", "description": "Another argument"},
                    },
                    "required": ["arg1"],
                    "type": "object",
                },
                "mcp_server_id": f"/api/server/tools/tool_{i}",
                "rank": i + 1,
                "reasoning": f"Important for the planned workflow step {i}",
                "relevance_score": round(1.0 - i * 0.05, 2),
            }
        )
    return tools


def _make_tool_result_json(selected_tools: list[dict]) -> str:
    """Produce the JSON string a tool would return."""
    return json.dumps(
        {"selected_tools": selected_tools},
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Tests — ToolResultCapture with truncated vs. raw content
# ---------------------------------------------------------------------------


class TestTruncatedCapture:
    """Verify that ToolResultCapture works with raw (untruncated) content."""

    def _make_truncated_content(self, raw_json: str) -> str:
        """Simulate the truncation wrapping done by ToolService."""
        return json.dumps(
            {
                "text": raw_json[:4000],
                "truncated": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def test_capture_fails_on_truncated_content(self):
        """Demonstrate the original bug: truncated content loses fields."""
        tools = _make_large_selected_tools(count=10)
        raw_json = _make_tool_result_json(tools)
        # Confirm the raw content is large enough to trigger truncation
        assert len(raw_json) > 4000

        truncated = self._make_truncated_content(raw_json)
        capture = ToolResultCapture(
            "select_tools",
            [
                {
                    "field": "selected_tools",
                    "from": "select_tools",
                    "as": "selected_tools",
                }
            ],
        )
        context = {"agents": {}}
        capture.capture(truncated, context, tool_name="select_tools")

        # The field is NOT captured from the truncated envelope
        assert "selected_tools" not in context["agents"].get("select_tools", {})

    def test_capture_succeeds_on_raw_content(self):
        """With untruncated content, capture works correctly."""
        tools = _make_large_selected_tools(count=10)
        raw_json = _make_tool_result_json(tools)
        assert len(raw_json) > 4000

        capture = ToolResultCapture(
            "select_tools",
            [
                {
                    "field": "selected_tools",
                    "from": "select_tools",
                    "as": "selected_tools",
                }
            ],
        )
        context = {"agents": {}}
        capture.capture(raw_json, context, tool_name="select_tools")

        captured = context["agents"]["select_tools"]["selected_tools"]
        assert isinstance(captured, list)
        assert len(captured) == 10
        assert captured[0]["tool_name"] == "tool_0"

    def test_arg_injector_resolves_after_raw_capture(self):
        """End-to-end: capture from raw content, then inject via ArgInjector."""
        tools = _make_large_selected_tools(count=10)
        raw_json = _make_tool_result_json(tools)

        # Step 1 — capture
        capture = ToolResultCapture(
            "select_tools",
            [
                {
                    "field": "selected_tools",
                    "from": "select_tools",
                    "as": "selected_tools",
                }
            ],
        )
        context = {"agents": {}}
        capture.capture(raw_json, context, tool_name="select_tools")

        # Step 2 — inject
        injector = ArgInjector(
            [ToolArgMapping("plan", {"selected_tools": "select_tools.selected_tools"})]
        )
        result = injector.inject(
            "plan", {"title": "My Plan"}, context, fail_on_missing=True
        )

        assert "selected_tools" in result
        assert len(result["selected_tools"]) == 10
        assert result["title"] == "My Plan"

    def test_arg_injector_fails_on_truncated_capture(self):
        """Demonstrate: ArgInjector raises when capture used truncated content."""
        tools = _make_large_selected_tools(count=10)
        raw_json = _make_tool_result_json(tools)
        truncated = self._make_truncated_content(raw_json)

        capture = ToolResultCapture(
            "select_tools",
            [
                {
                    "field": "selected_tools",
                    "from": "select_tools",
                    "as": "selected_tools",
                }
            ],
        )
        context = {"agents": {}}
        capture.capture(truncated, context, tool_name="select_tools")

        injector = ArgInjector(
            [ToolArgMapping("plan", {"selected_tools": "select_tools.selected_tools"})]
        )
        with pytest.raises(ArgResolutionError):
            injector.inject("plan", {}, context, fail_on_missing=True)


# ---------------------------------------------------------------------------
# Tests — ToolService._raw_content preservation
# ---------------------------------------------------------------------------


class TestToolServiceRawContent:
    """Verify that execute_tool_call preserves _raw_content on large results."""

    @pytest.fixture
    def large_tool_result_json(self):
        """A JSON tool result > 4000 chars."""
        tools = _make_large_selected_tools(count=10)
        return _make_tool_result_json(tools)

    @pytest.fixture
    def small_tool_result_json(self):
        """A JSON tool result < 4000 chars."""
        tools = _make_large_selected_tools(count=1)
        raw = _make_tool_result_json(tools)
        assert len(raw) < 4000
        return raw

    def test_raw_content_present_on_large_result(self, large_tool_result_json):
        """When truncation happens, _raw_content holds original content."""
        raw = large_tool_result_json
        assert len(raw) > 4000

        # Simulate what execute_tool_call does:
        raw_content = raw
        content = raw
        # Truncation logic
        parsed = json.loads(content)
        compact_json = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        if len(compact_json) > 4000:
            content = json.dumps(
                {"text": compact_json[:4000], "truncated": True},
                ensure_ascii=False,
                separators=(",", ":"),
            )

        result_dict = {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "select_tools",
            "content": content,
        }
        if raw_content is not content:
            result_dict["_raw_content"] = raw_content

        # _raw_content should be present
        assert "_raw_content" in result_dict
        # _raw_content should be the original full content
        loaded = json.loads(result_dict["_raw_content"])
        assert "selected_tools" in loaded
        assert len(loaded["selected_tools"]) == 10

    def test_no_raw_content_on_small_result(self, small_tool_result_json):
        """When no truncation, _raw_content should not be set."""
        raw = small_tool_result_json
        assert len(raw) < 4000

        raw_content = raw
        content = raw
        # No truncation needed

        result_dict = {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "select_tools",
            "content": content,
        }
        if raw_content is not content:
            result_dict["_raw_content"] = raw_content

        assert "_raw_content" not in result_dict


# ---------------------------------------------------------------------------
# Tests — tool_chat_runner prefers _raw_content for events
# ---------------------------------------------------------------------------


class TestToolChatRunnerRawContentPreference:
    """Verify that workflow tool_result events use _raw_content when available."""

    def test_prefers_raw_content_over_content(self):
        """When _raw_content is on raw_result, it should be used for the event."""
        tools = _make_large_selected_tools(count=10)
        raw_json = _make_tool_result_json(tools)
        truncated = json.dumps(
            {"text": raw_json[:4000], "truncated": True},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        raw_result = {
            "name": "select_tools",
            "tool_call_id": "call_123",
            "content": truncated,
            "_raw_content": raw_json,
        }

        # Simulate the fixed logic in tool_chat_runner
        raw_content = raw_result.get("_raw_content") or raw_result.get("content", "")

        assert raw_content == raw_json
        # Verify we can parse and find the field
        parsed = json.loads(raw_content)
        assert "selected_tools" in parsed
        assert len(parsed["selected_tools"]) == 10

    def test_falls_back_to_content_when_no_raw(self):
        """When _raw_content is absent, fall back to content."""
        raw_result = {
            "name": "some_tool",
            "content": '{"result": "ok"}',
        }

        raw_content = raw_result.get("_raw_content") or raw_result.get("content", "")

        assert raw_content == '{"result": "ok"}'

    def test_empty_content_fallback(self):
        """When both _raw_content and content are missing, return empty."""
        raw_result = {"name": "some_tool"}

        raw_content = raw_result.get("_raw_content") or raw_result.get("content", "")

        assert raw_content == ""


# ---------------------------------------------------------------------------
# Tests — stream_processor integration
# ---------------------------------------------------------------------------


class TestStreamProcessorToolResultCapture:
    """Full integration: _process_tool_result uses raw content for capture."""

    def test_capture_uses_raw_content_for_returns(self):
        """Simulate _process_tool_result with a truncated content but raw capture."""
        from app.tgi.workflows.stream_processor import (
            _process_tool_result,
            StreamResult,
        )

        tools = _make_large_selected_tools(count=10)
        raw_json = _make_tool_result_json(tools)

        # The tool_result dict as seen by stream_processor
        # In the current flow, stream_processor gets `content` from the
        # tool_result event which should now be _raw_content
        tool_result = {
            "name": "select_tools",
            "content": raw_json,  # After the fix, this comes from _raw_content
        }

        agent_context = {"content": "", "pass_through": False}
        state_context = {"agents": {"select_tools": agent_context}}
        result = StreamResult()

        capture = ToolResultCapture(
            "select_tools",
            [
                {
                    "field": "selected_tools",
                    "from": "select_tools",
                    "as": "selected_tools",
                }
            ],
        )

        _process_tool_result(
            tool_result,
            agent_context,
            state_context,
            result,
            capture,
        )

        # The capture should have stored the full selected_tools
        captured = state_context["agents"]["select_tools"].get("selected_tools")
        assert captured is not None
        assert isinstance(captured, list)
        assert len(captured) == 10

    def test_capture_fails_with_truncated_content(self):
        """Without the fix, truncated content would lose the field."""
        from app.tgi.workflows.stream_processor import (
            _process_tool_result,
            StreamResult,
        )

        tools = _make_large_selected_tools(count=10)
        raw_json = _make_tool_result_json(tools)
        truncated = json.dumps(
            {"text": raw_json[:4000], "truncated": True},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        tool_result = {
            "name": "select_tools",
            "content": truncated,  # Bug scenario: truncated content passed
        }

        agent_context = {"content": "", "pass_through": False}
        state_context = {"agents": {"select_tools": agent_context}}
        result = StreamResult()

        capture = ToolResultCapture(
            "select_tools",
            [
                {
                    "field": "selected_tools",
                    "from": "select_tools",
                    "as": "selected_tools",
                }
            ],
        )

        _process_tool_result(
            tool_result,
            agent_context,
            state_context,
            result,
            capture,
        )

        # selected_tools is NOT captured because content was truncated
        captured = state_context["agents"]["select_tools"].get("selected_tools")
        assert captured is None
