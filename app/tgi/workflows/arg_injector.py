"""
ArgInjector: Resolves and injects arguments into tool calls based on workflow context.

This module handles the mapping of arguments from previous agent results into
tool calls for dependent agents. For example:

    tools:
      - plan:
          args:
            selected_tools: "select_tools.selected_tools"

When the "plan" tool is called, this injector will look up the value at
state.context["agents"]["select_tools"]["selected_tools"] and inject it
into the tool call arguments.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uvicorn.error")


@dataclass
class ToolArgMapping:
    """Mapping configuration for a single tool's arguments."""

    tool_name: str
    args_mapping: Dict[str, str]  # arg_name -> "agent_name.field_path"


class ArgInjector:
    """
    Resolves and injects arguments into tool calls based on workflow context.

    Usage:
        injector = ArgInjector([
            ToolArgMapping("plan", {"selected_tools": "select_tools.selected_tools"})
        ])

        # When a tool call is made, inject the mapped args
        merged_args = injector.inject("plan", {"title": "My Plan"}, context)
        # Returns: {"title": "My Plan", "selected_tools": ["tool1", "tool2"]}
    """

    def __init__(self, mappings: List[ToolArgMapping]):
        self._mappings: Dict[str, ToolArgMapping] = {m.tool_name: m for m in mappings}

    def has_mapping(self, tool_name: str) -> bool:
        """Check if there's a mapping for the given tool."""
        return tool_name in self._mappings

    def inject(
        self, tool_name: str, args: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Inject mapped arguments into the tool call args.

        Args:
            tool_name: Name of the tool being called
            args: Original arguments from the LLM
            context: Workflow context containing agent results

        Returns:
            Merged arguments with injected values
        """
        if tool_name not in self._mappings:
            return args

        mapping = self._mappings[tool_name]
        result = dict(args)

        for arg_name, source_path in mapping.args_mapping.items():
            value = self._resolve_path(source_path, context)
            if value is not None:
                result[arg_name] = value
                logger.debug(
                    f"ArgInjector: Injected {arg_name}={value!r} from {source_path}"
                )
            else:
                logger.warning(
                    f"ArgInjector: Could not resolve {source_path} for {arg_name}"
                )

        return result

    def _resolve_path(self, path: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a dotted path like "agent_name.field" from context.

        The path format is: "<agent_name>.<field_path>"
        This looks up context["agents"][<agent_name>][<field_path>]
        """
        parts = path.split(".", 1)
        if len(parts) != 2:
            return None

        agent_name, field_path = parts
        agents = context.get("agents", {})
        agent_data = agents.get(agent_name, {})

        # Navigate nested path (e.g., "result.items")
        value = agent_data
        for key in field_path.split("."):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
            if value is None:
                return None

        return value

    @classmethod
    def from_agent_def(cls, agent_def: Any) -> Optional["ArgInjector"]:
        """
        Create an ArgInjector from a WorkflowAgentDef.

        Extracts tool configurations with args mappings from the agent definition.
        """
        mappings = []

        for tool in agent_def.tools or []:
            if isinstance(tool, dict):
                for tool_name, config in tool.items():
                    if isinstance(config, dict) and "args" in config:
                        mappings.append(
                            ToolArgMapping(
                                tool_name=tool_name,
                                args_mapping=config["args"],
                            )
                        )

        return cls(mappings) if mappings else None


class ToolResultCapture:
    """
    Captures tool results and stores specified fields in the workflow context.

    When an agent has a `returns` field, this captures those fields from
    tool results and stores them in the agent's context data.
    """

    def __init__(self, agent_name: str, returns: List[str]):
        self.agent_name = agent_name
        self.returns = returns

    def capture(
        self, tool_result_content: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse tool result content and capture specified return values into context.

        Args:
            tool_result_content: JSON string from tool execution
            context: Workflow context to update

        Returns:
            Updated context with captured values
        """
        try:
            data = json.loads(tool_result_content)
        except (json.JSONDecodeError, TypeError):
            return context

        if not isinstance(data, dict):
            return context

        # Ensure agent entry exists
        if "agents" not in context:
            context["agents"] = {}
        if self.agent_name not in context["agents"]:
            context["agents"][self.agent_name] = {}

        agent_data = context["agents"][self.agent_name]

        for field in self.returns:
            if field in data:
                agent_data[field] = data[field]
                logger.debug(
                    f"ToolResultCapture: Stored {self.agent_name}.{field}={data[field]!r}"
                )

        return context
