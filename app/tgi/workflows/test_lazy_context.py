"""
Tests for lazy context loading in workflows.

This tests the LazyContextProvider and its integration with the workflow engine,
ensuring agents can retrieve context on-demand without full payload embedding.
"""

import json
import pytest
from typing import Optional

from app.tgi.workflows.lazy_context import LazyContextProvider, ContextQueryResult
from app.tgi.workflows.state import WorkflowExecutionState


class StubStateStore:
    """Stub for testing without actual persistence."""

    def __init__(self, state: Optional[WorkflowExecutionState] = None):
        self.state = state

    def get_or_create(
        self, execution_id: str, flow_id: str
    ) -> Optional[WorkflowExecutionState]:
        return self.state


def test_lazy_context_get_summary():
    """Test context summary retrieval."""
    state = WorkflowExecutionState(
        execution_id="test-1",
        flow_id="test-flow",
        context={
            "agents": {
                "agent1": {"content": "result1", "completed": True},
                "agent2": {"content": "result2", "completed": False},
            },
            "user_messages": ["hello", "world"],
        },
    )

    store = StubStateStore(state)
    provider = LazyContextProvider(store, "test-1")

    summary = provider.get_context_summary()

    assert "agents" in summary
    assert "user_messages" in summary
    assert summary["agents"]["type"] == "dict"
    assert summary["agents"]["available"] is True
    assert summary["user_messages"]["type"] == "list"


def test_lazy_context_get_value_simple():
    """Test retrieving simple values by path."""
    state = WorkflowExecutionState(
        execution_id="test-2",
        flow_id="test-flow",
        context={
            "user_messages": ["hello", "world"],
        },
    )

    store = StubStateStore(state)
    provider = LazyContextProvider(store, "test-2")

    result = provider.get_context_value("user_messages")

    assert result.success is True
    assert result.data == ["hello", "world"]
    assert "Retrieved" in result.summary


def test_lazy_context_get_value_nested():
    """Test retrieving nested values by dot-separated path."""
    state = WorkflowExecutionState(
        execution_id="test-3",
        flow_id="test-flow",
        context={
            "agents": {
                "prior_agent": {
                    "content": "previous output",
                    "reason": "completed",
                }
            }
        },
    )

    store = StubStateStore(state)
    provider = LazyContextProvider(store, "test-3")

    result = provider.get_context_value("agents.prior_agent.content")

    assert result.success is True
    assert result.data == "previous output"


def test_lazy_context_get_value_not_found():
    """Test error handling for non-existent paths."""
    state = WorkflowExecutionState(
        execution_id="test-4", flow_id="test-flow", context={"agents": {}}
    )

    store = StubStateStore(state)
    provider = LazyContextProvider(store, "test-4")

    result = provider.get_context_value("agents.nonexistent.content")

    assert result.success is False
    assert "not found" in result.error.lower()


def test_lazy_context_get_agent_context():
    """Test retrieving context for a specific agent."""
    state = WorkflowExecutionState(
        execution_id="test-5",
        flow_id="test-flow",
        context={
            "agents": {
                "agent1": {
                    "content": "output",
                    "reason": "completed",
                    "completed": True,
                }
            }
        },
    )

    store = StubStateStore(state)
    provider = LazyContextProvider(store, "test-5")

    result = provider.get_agent_context("agent1")

    assert result.success is True
    assert "content" in result.data
    assert result.data["reason"] == "completed"


def test_lazy_context_get_agent_with_field_filter():
    """Test retrieving specific fields from agent context."""
    state = WorkflowExecutionState(
        execution_id="test-6",
        flow_id="test-flow",
        context={
            "agents": {
                "agent1": {
                    "content": "long output",
                    "reason": "completed",
                    "completed": True,
                    "internal_state": "not_needed",
                }
            }
        },
    )

    store = StubStateStore(state)
    provider = LazyContextProvider(store, "test-6")

    result = provider.get_agent_context("agent1", fields=["content", "reason"])

    assert result.success is True
    assert "content" in result.data
    assert "reason" in result.data
    assert "internal_state" not in result.data


def test_lazy_context_get_user_messages():
    """Test retrieving user message history."""
    state = WorkflowExecutionState(
        execution_id="test-7",
        flow_id="test-flow",
        context={"user_messages": ["msg1", "msg2", "msg3", "msg4", "msg5"]},
    )

    store = StubStateStore(state)
    provider = LazyContextProvider(store, "test-7")

    # Get all messages
    result = provider.get_user_messages()
    assert result.success is True
    assert len(result.data) == 5

    # Get last 2 messages
    result = provider.get_user_messages(limit=2)
    assert result.success is True
    assert len(result.data) == 2
    assert result.data == ["msg4", "msg5"]


def test_lazy_context_size_limit():
    """Test that oversized results are rejected."""
    large_content = "x" * 60000  # Larger than default 50KB limit
    state = WorkflowExecutionState(
        execution_id="test-8",
        flow_id="test-flow",
        context={"large_data": large_content},
    )

    store = StubStateStore(state)
    provider = LazyContextProvider(store, "test-8")

    result = provider.get_context_value("large_data")

    assert result.success is False
    assert "too large" in result.error.lower()


def test_lazy_context_tool_definition():
    """Test that tool definition is properly formatted."""
    tool_def = LazyContextProvider.create_tool_definition()

    assert tool_def["type"] == "function"
    assert tool_def["function"]["name"] == "get_workflow_context"
    assert "description" in tool_def["function"]

    # Verify operation enum
    params = tool_def["function"]["parameters"]
    assert params["type"] == "object"

    operation_prop = params["properties"]["operation"]
    assert set(operation_prop["enum"]) == {
        "summary",
        "get_value",
        "get_agent",
        "get_messages",
    }


def test_context_query_result_to_dict():
    """Test ContextQueryResult serialization."""
    result = ContextQueryResult(
        success=True, data={"key": "value"}, summary="Test summary"
    )

    as_dict = result.to_dict()

    assert as_dict["success"] is True
    assert as_dict["data"] == {"key": "value"}
    assert as_dict["summary"] == "Test summary"
    assert as_dict["error"] is None

    # Should be JSON serializable
    json_str = json.dumps(as_dict)
    assert "key" in json_str


def test_lazy_context_error_handling():
    """Test error handling when state is not found."""
    store = StubStateStore(None)
    provider = LazyContextProvider(store, "nonexistent")

    result = provider.get_context_summary()
    assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
