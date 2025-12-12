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
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("uvicorn.error")


class ArgResolutionError(Exception):
    """Raised when an argument mapping cannot be resolved from workflow context."""

    def __init__(
        self,
        tool_name: str,
        arg_name: str,
        source_path: str,
        available_context: dict,
    ):
        self.tool_name = tool_name
        self.arg_name = arg_name
        self.source_path = source_path
        self.available_context = available_context
        super().__init__(
            f"Could not resolve '{arg_name}' from '{source_path}' for tool '{tool_name}'. "
            f"Available context for '{source_path.split('.')[0]}': {available_context}"
        )


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
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Dict[str, Any],
        *,
        fail_on_missing: bool = False,
    ) -> Dict[str, Any]:
        """
        Inject mapped arguments into the tool call args.

        Args:
            tool_name: Name of the tool being called
            args: Original arguments from the LLM
            context: Workflow context containing agent results
            fail_on_missing: If True, raise ArgResolutionError when a mapping
                cannot be resolved. If False, log a warning and skip the arg.

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
            else:
                agent_name = source_path.split(".", 1)[0]
                available = context.get("agents", {}).get(agent_name, {}) or {}
                if fail_on_missing:
                    # Surface a hard failure so misconfigured workflows stop immediately
                    raise ArgResolutionError(
                        tool_name=tool_name,
                        arg_name=arg_name,
                        source_path=source_path,
                        available_context=available,
                    )
                logger.warning(
                    "ArgInjector: Could not resolve %s for %s (available context: %s)",
                    source_path,
                    arg_name,
                    available,
                )

        return result

    def _resolve_path(self, path: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a path from context.

        Supports:
        - "agent.field" (or nested) -> context["agents"]["agent"]["field"]
        - "shared_key"              -> context["shared_key"]
        - "nested.shared.key"       -> context["nested"]["shared"]["key"]
        """
        if not path:
            return None

        # Support global context references (e.g., "plan_id") for shared values
        if "." not in path:
            return context.get(path)

        # Prefer agent-scoped lookup when the first segment matches an agent
        agent_name, field_path = path.split(".", 1)
        agents = context.get("agents", {})
        if agent_name in agents:
            agent_data = agents.get(agent_name, {}) or {}
            value = agent_data
            for key in field_path.split("."):
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return None
                if value is None:
                    return None
            return value

        # Fallback: allow dotted paths into the root context
        value: Any = context
        for key in path.split("."):
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


@dataclass
class ReturnSpec:
    """Specification for a single return value to capture.

    Attributes:
        field: The field path to capture (e.g., "payload.result")
        from_tool: Optional tool name to capture from (None means any tool)
        as_name: Optional alias for storing the value (defaults to field path)
    """

    field: str
    from_tool: Optional[str] = None
    as_name: Optional[str] = None

    @classmethod
    def parse(cls, spec: Union[str, Dict[str, Any]]) -> "ReturnSpec":
        """
        Parse a return specification from workflow config.

        Supports multiple formats:
        - "field" -> captures 'field' from any tool
        - "payload.result" -> captures nested path from any tool
        - {"field": "payload.result", "from": "plan"} -> captures from specific tool
        - {"field": "payload.result", "from": "plan", "as": "plan_data"} -> with alias
        """
        if isinstance(spec, str):
            return cls(field=spec)

        if isinstance(spec, dict):
            field = spec.get("field", "")
            from_tool = spec.get("from")
            as_name = spec.get("as")
            return cls(field=field, from_tool=from_tool, as_name=as_name)

        raise ValueError(f"Invalid return spec: {spec}")


class ToolResultCapture:
    """
    Captures tool results and stores specified fields in the workflow context.

    When an agent has a `returns` field, this captures those fields from
    tool results and stores them in the agent's context data.

    Supports multiple formats for returns:
    - ["field"] - captures top-level field from any tool
    - ["payload.result"] - captures nested field from any tool
    - [{"field": "payload.result", "from": "plan"}] - captures from specific tool
    - [{"field": "payload.result", "from": "plan", "as": "my_plan"}] - with alias
    """

    def __init__(self, agent_name: str, returns: List[Union[str, Dict[str, Any]]]):
        self.agent_name = agent_name
        self.return_specs = [ReturnSpec.parse(r) for r in returns]

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        Navigate a dotted path to get a nested value.

        Args:
            data: The source dictionary
            path: A dotted path like "payload.result"

        Returns:
            The value at the path, or None if not found
        """
        value = data
        for key in path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _set_nested_value(self, target: Dict[str, Any], path: str, value: Any) -> None:
        """
        Set a value at a dotted path, creating intermediate dicts as needed.

        Args:
            target: The target dictionary to modify
            path: A dotted path like "payload.result"
            value: The value to set
        """
        keys = path.split(".")
        current = target
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def capture(
        self,
        tool_result_content: str,
        context: Dict[str, Any],
        tool_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse tool result content and capture specified return values into context.

        Args:
            tool_result_content: JSON string from tool execution
            context: Workflow context to update
            tool_name: Name of the tool that produced this result (for filtering)

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

        for spec in self.return_specs:
            # Skip if this spec is for a different tool
            if spec.from_tool and tool_name and spec.from_tool != tool_name:
                continue

            # Get the value from the tool result
            if "." in spec.field:
                value = self._get_nested_value(data, spec.field)
            else:
                value = data.get(spec.field)

            if value is None:
                continue

            # Determine where to store it
            store_path = spec.as_name or spec.field

            # Store the value
            if "." in store_path:
                self._set_nested_value(agent_data, store_path, value)
            else:
                agent_data[store_path] = value

            logger.debug(
                f"ToolResultCapture: Stored {self.agent_name}.{store_path}={value!r} "
                f"(from tool={tool_name or 'any'})"
            )

        return context
