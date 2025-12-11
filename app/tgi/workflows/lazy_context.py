"""
Lazy context loading for workflows.

Instead of passing full workflow context to routing and orchestration agents,
this module provides tools for agents to retrieve context on-demand, reducing
initial payload size and avoiding expensive JSON serialization of unused data.
"""

import json
import logging
from typing import Any, Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger("uvicorn.error")


@dataclass
class ContextQueryResult:
    """Result of a context query operation."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "summary": self.summary,
        }


class LazyContextProvider:
    """
    Provides on-demand context retrieval for workflow agents.

    Instead of embedding full context in agent payloads, agents can call
    a tool to retrieve specific context when needed. This reduces:
    - Initial token count
    - JSON serialization overhead
    - Compression time for large workflows
    """

    def __init__(self, state_store: Any, state_id: str, logger_instance=None):
        """
        Initialize lazy context provider.

        Args:
            state_store: WorkflowStateStore instance for retrieving state
            state_id: Execution ID of the current workflow
            logger_instance: Optional logger instance
        """
        self.state_store = state_store
        self.state_id = state_id
        self.logger = logger_instance or logger

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available context without full data.

        Returns:
            Dictionary with keys and sizes of available context.
        """
        try:
            state = self.state_store.get_or_create(self.state_id, "")
            if not state:
                return {"error": "State not found"}

            context = state.context or {}
            summary = {}

            for key, value in context.items():
                if key.startswith("_"):
                    continue
                size = len(json.dumps(value, default=str))
                summary[key] = {
                    "type": type(value).__name__,
                    "size_bytes": size,
                    "available": True,
                }

            return summary
        except Exception as exc:
            self.logger.debug(f"[LazyContextProvider] Error in summary: {exc}")
            return {"error": str(exc)}

    def get_context_value(
        self,
        path: str,
        max_depth: int = 10,
        max_size_bytes: Optional[int] = None,
    ) -> ContextQueryResult:
        """
        Retrieve a specific context value by path.

        Args:
            path: Dot-separated path (e.g., "agents.prior_agent.content", "user_messages")
            max_depth: Maximum depth to traverse nested structures (prevents infinite loops)
            max_size_bytes: Maximum bytes to return (default 50KB)

        Returns:
            ContextQueryResult with the requested data or error details
        """
        max_size_bytes = max_size_bytes or 50000

        try:
            state = self.state_store.get_or_create(self.state_id, "")
            if not state:
                return ContextQueryResult(success=False, error="State not found")

            context = state.context or {}
            value = self._traverse_path(context, path.split("."), max_depth)

            if value is None:
                return ContextQueryResult(
                    success=False, error=f"Path not found: {path}"
                )

            # Serialize to check size
            serialized = json.dumps(value, default=str, ensure_ascii=False)
            size = len(serialized.encode("utf-8"))

            if size > max_size_bytes:
                return ContextQueryResult(
                    success=False,
                    error=f"Context too large ({size} bytes, limit {max_size_bytes}). "
                    f"Consider requesting a subset or summary instead.",
                )

            return ContextQueryResult(
                success=True,
                data=value,
                summary=f"Retrieved {size} bytes from path: {path}",
            )

        except Exception as exc:
            self.logger.debug(f"[LazyContextProvider] Error retrieving {path}: {exc}")
            return ContextQueryResult(success=False, error=str(exc))

    def get_agent_context(
        self,
        agent_name: str,
        fields: Optional[List[str]] = None,
    ) -> ContextQueryResult:
        """
        Retrieve context for a specific agent.

        Args:
            agent_name: Name of the agent
            fields: Optional list of fields to retrieve (e.g., ["content", "reason"])
                   If None, returns all non-empty fields

        Returns:
            ContextQueryResult with agent context
        """
        try:
            state = self.state_store.get_or_create(self.state_id, "")
            if not state:
                return ContextQueryResult(success=False, error="State not found")

            agents = state.context.get("agents", {})
            agent_ctx = agents.get(agent_name, {})

            if not agent_ctx:
                return ContextQueryResult(
                    success=False, error=f"No context found for agent: {agent_name}"
                )

            # Filter to requested fields
            if fields:
                result = {k: v for k, v in agent_ctx.items() if k in fields}
            else:
                # Return non-empty fields by default
                result = {k: v for k, v in agent_ctx.items() if v}

            if not result:
                return ContextQueryResult(
                    success=False,
                    error=f"No data found for agent {agent_name} with fields {fields}",
                )

            return ContextQueryResult(
                success=True,
                data=result,
                summary=f"Retrieved context for agent: {agent_name}",
            )

        except Exception as exc:
            self.logger.debug(
                f"[LazyContextProvider] Error retrieving agent context: {exc}"
            )
            return ContextQueryResult(success=False, error=str(exc))

    def get_user_messages(
        self,
        limit: Optional[int] = None,
    ) -> ContextQueryResult:
        """
        Retrieve user message history.

        Args:
            limit: Optional limit on number of messages to return

        Returns:
            ContextQueryResult with user messages
        """
        try:
            state = self.state_store.get_or_create(self.state_id, "")
            if not state:
                return ContextQueryResult(success=False, error="State not found")

            messages = state.context.get("user_messages", [])

            if not messages:
                return ContextQueryResult(success=False, error="No user messages found")

            if limit:
                messages = messages[-limit:]

            return ContextQueryResult(
                success=True,
                data=messages,
                summary=f"Retrieved {len(messages)} user messages",
            )

        except Exception as exc:
            self.logger.debug(
                f"[LazyContextProvider] Error retrieving user messages: {exc}"
            )
            return ContextQueryResult(success=False, error=str(exc))

    def _traverse_path(
        self,
        obj: Any,
        path_parts: List[str],
        max_depth: int = 2,
        current_depth: int = 0,
    ) -> Any:
        """
        Safely traverse a nested structure by path parts.

        Args:
            obj: Object to traverse
            path_parts: List of path components
            max_depth: Maximum allowed depth
            current_depth: Current depth (for recursion tracking)

        Returns:
            The value at the path, or None if not found
        """
        if not path_parts or current_depth >= max_depth:
            return obj

        part = path_parts[0]
        remaining = path_parts[1:]

        if isinstance(obj, dict):
            next_obj = obj.get(part)
        elif isinstance(obj, list):
            try:
                idx = int(part)
                next_obj = obj[idx] if 0 <= idx < len(obj) else None
            except (ValueError, IndexError):
                return None
        else:
            return None

        if next_obj is None:
            return None

        return self._traverse_path(next_obj, remaining, max_depth, current_depth + 1)

    @staticmethod
    def create_tool_definition() -> Dict[str, Any]:
        """
        Create tool definition for the lazy context loader.

        Returns:
            Tool definition in OpenAI format
        """
        return {
            "type": "function",
            "function": {
                "name": "get_workflow_context",
                "description": (
                    "Retrieve specific workflow context on-demand. Use this to avoid "
                    "processing large context blocks upfront. Supports retrieving context "
                    "summaries, specific agent outputs, user messages, or custom paths."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "summary",
                                "get_value",
                                "get_agent",
                                "get_messages",
                            ],
                            "description": (
                                "Operation to perform: "
                                "'summary' (list available context keys), "
                                "'get_value' (retrieve by dot-separated path), "
                                "'get_agent' (retrieve specific agent context), "
                                "'get_messages' (retrieve user message history)"
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "Dot-separated path for 'get_value' operation. "
                                "Examples: 'agents.prior_agent.content', 'user_messages'"
                            ),
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Agent name for 'get_agent' operation",
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional field filter for 'get_agent' operation. "
                                "Examples: ['content', 'reason', 'completed']"
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional limit on messages for 'get_messages' operation",
                        },
                        "max_size_bytes": {
                            "type": "integer",
                            "description": (
                                "Maximum bytes to return (default 50000). "
                                "Helps prevent oversized responses."
                            ),
                        },
                    },
                    "required": ["operation"],
                },
            },
        }
