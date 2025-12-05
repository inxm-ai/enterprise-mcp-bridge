"""
Agent module for workflow execution.

This module provides:
- ToolConfig: Parse and manage tool configurations (string or object format)
- AgentExecutor: Execute agents with support for:
  - Tool argument mapping from previous agent results
  - Streaming tool execution
  - Returns extraction to context
  - Pass-through guidelines
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from app.tgi.workflows.models import WorkflowAgentDef, WorkflowExecutionState

logger = logging.getLogger("uvicorn.error")


@dataclass
class ToolConfig:
    """
    Parsed configuration for a tool.

    Supports both simple string tool names and complex object configurations
    with settings and argument mappings.
    """

    name: str
    streaming: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)
    args_mapping: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_definition(cls, definition: Union[str, Dict[str, Any]]) -> "ToolConfig":
        """
        Parse a tool definition into a ToolConfig.

        Args:
            definition: Can be:
                - str: Simple tool name
                - dict: {"tool_name": {"settings": {...}, "args": {...}}}

        Returns:
            Parsed ToolConfig instance.
        """
        if isinstance(definition, str):
            return cls(name=definition)

        if not isinstance(definition, dict) or len(definition) != 1:
            raise ValueError(f"Invalid tool definition format: {definition}")

        tool_name = next(iter(definition.keys()))
        config = definition[tool_name]

        if not isinstance(config, dict):
            return cls(name=tool_name)

        settings = config.get("settings", {})
        args = config.get("args", {})
        streaming = settings.get("streaming", False) if settings else False

        return cls(
            name=tool_name,
            streaming=streaming,
            settings=settings,
            args_mapping=args,
        )

    @classmethod
    def from_definitions(
        cls, definitions: Optional[List[Union[str, Dict[str, Any]]]]
    ) -> List["ToolConfig"]:
        """Parse a list of tool definitions into ToolConfigs."""
        if definitions is None:
            return []
        return [cls.from_definition(d) for d in definitions]


class AgentExecutor:
    """
    Executor for workflow agents.

    Handles:
    - Building agent prompts with pass-through guidelines
    - Resolving argument references from context
    - Modifying tool schemas to remove pre-mapped args
    - Extracting returns from tool results
    - Executing streaming tools
    """

    def __init__(self):
        pass

    def build_agent_prompt(self, agent_def: WorkflowAgentDef, base_prompt: str) -> str:
        """
        Build the system prompt for an agent.

        Includes pass-through guideline if specified.

        Args:
            agent_def: The agent definition.
            base_prompt: The base prompt text.

        Returns:
            The complete system prompt.
        """
        prompt = base_prompt

        if agent_def.pass_through_guideline:
            prompt += f"\n\nResponse guideline: {agent_def.pass_through_guideline}"

        return prompt

    def resolve_arg_reference(
        self, reference: str, context: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Resolve an argument reference from context.

        References are in the format "agent_name.field.nested_field".

        Args:
            reference: The reference string, e.g., "select_tools.selected-tools"
            context: The workflow execution context.

        Returns:
            The resolved value, or None if not found.
        """
        parts = reference.split(".")
        if len(parts) < 2:
            return None

        agent_name = parts[0]
        field_path = parts[1:]

        agents = context.get("agents", {})
        agent_ctx = agents.get(agent_name)
        if not agent_ctx:
            return None

        value = agent_ctx
        for part in field_path:
            # Handle both dict and object attribute access
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
            if value is None:
                return None

        return value

    def resolve_args_for_tool(
        self,
        tool_config: ToolConfig,
        context: Dict[str, Any],
        provided_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve all argument mappings for a tool call.

        Args:
            tool_config: The tool configuration with arg mappings.
            context: The workflow execution context.
            provided_args: Any directly provided arguments.

        Returns:
            Dict of resolved arguments.
        """
        resolved = dict(provided_args or {})

        for arg_name, reference in tool_config.args_mapping.items():
            value = self.resolve_arg_reference(reference, context)
            if value is not None:
                resolved[arg_name] = value

        return resolved

    def modify_tool_for_agent(
        self, tool: Dict[str, Any], tool_config: ToolConfig
    ) -> Dict[str, Any]:
        """
        Modify a tool schema to hide pre-mapped arguments from the LLM.

        When arguments are pre-mapped from context, they should be completely
        removed from both properties and required so the LLM doesn't try to
        fill them with huge JSON values. The original unmodified tools should
        be passed to execute_tool_calls for proper validation.

        Args:
            tool: The original tool definition.
            tool_config: The tool configuration with arg mappings.

        Returns:
            Modified tool definition with mapped args removed entirely.
        """
        if not tool_config.args_mapping:
            return tool

        modified = copy.deepcopy(tool)
        func = modified.get("function", {})
        params = func.get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})

        # Remove mapped args from both required AND properties
        # so the LLM doesn't see them at all
        for arg_name in tool_config.args_mapping.keys():
            if arg_name in required:
                required.remove(arg_name)
            if arg_name in properties:
                del properties[arg_name]

        return modified

    def extract_returns(
        self,
        agent_def: WorkflowAgentDef,
        tool_result: Dict[str, Any],
        state: WorkflowExecutionState,
    ) -> None:
        """
        Extract specified return fields from tool result into context.

        When multiple tool calls provide the same return field, values are
        aggregated into a list of lists.

        Args:
            agent_def: The agent definition with returns specification.
            tool_result: The tool execution result.
            state: The workflow execution state to update.
        """
        if not agent_def.returns:
            return

        agent_ctx = state.context.setdefault("agents", {}).setdefault(
            agent_def.agent, {"content": ""}
        )

        for field_name in agent_def.returns:
            if field_name not in tool_result:
                continue

            value = tool_result[field_name]

            # If the field already exists, aggregate into list of results
            if field_name in agent_ctx:
                existing = agent_ctx[field_name]
                if isinstance(existing, list) and len(existing) > 0:
                    # Check if already aggregating (list of lists/values)
                    first = existing[0]
                    if isinstance(first, list) or not isinstance(value, list):
                        # Already aggregating or value is not a list - append
                        existing.append(value)
                    else:
                        # First value was a list, convert to list of lists
                        agent_ctx[field_name] = [existing, value]
                else:
                    agent_ctx[field_name] = [existing, value]
            else:
                agent_ctx[field_name] = value

    def parse_tool_result_for_returns(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool result content to extract structured data.

        Args:
            content: The tool result content (may be JSON string).

        Returns:
            Parsed dict if content is valid JSON, None otherwise.
        """
        if not content:
            return None

        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None

    async def execute_streaming_tool(
        self,
        session: Any,
        tool_config: ToolConfig,
        args: Dict[str, Any],
        access_token: str,
        state: WorkflowExecutionState,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a tool using the streaming endpoint.

        Args:
            session: The MCP session.
            tool_config: The tool configuration.
            args: The resolved arguments.
            access_token: OAuth access token.
            state: The workflow execution state.

        Yields:
            Progress/result events from the streaming execution.
        """
        if not hasattr(session, "call_tool_streaming"):
            # Fallback to regular tool call if streaming not available
            result = await session.call_tool(tool_config.name, args, access_token)
            yield {"type": "result", "data": result}
            return

        stream = await session.call_tool_streaming(tool_config.name, args, access_token)

        async for event in stream:
            yield event

    def get_tool_configs_for_agent(
        self, agent_def: WorkflowAgentDef
    ) -> List[ToolConfig]:
        """
        Parse tool configurations from agent definition.

        Args:
            agent_def: The agent definition.

        Returns:
            List of ToolConfig objects.
        """
        if agent_def.tools is None:
            return []
        return ToolConfig.from_definitions(agent_def.tools)

    def get_tool_names_for_agent(
        self, agent_def: WorkflowAgentDef
    ) -> Optional[List[str]]:
        """
        Get simple tool names from agent definition for filtering.

        This handles both string tool names and object configurations.

        Args:
            agent_def: The agent definition.

        Returns:
            List of tool names, or None if all tools should be used.
        """
        if agent_def.tools is None:
            return None

        if isinstance(agent_def.tools, list) and len(agent_def.tools) == 0:
            return []

        names = []
        for tool_def in agent_def.tools:
            if isinstance(tool_def, str):
                names.append(tool_def)
            elif isinstance(tool_def, dict) and len(tool_def) == 1:
                names.append(next(iter(tool_def.keys())))

        return names if names else None

    def should_use_streaming(self, tool_config: ToolConfig) -> bool:
        """Check if a tool should use streaming execution."""
        return tool_config.streaming

    def build_tool_argument_instruction(self, tool_config: ToolConfig) -> Optional[str]:
        """
        Build instruction for LLM about pre-filled arguments.

        Args:
            tool_config: The tool configuration.

        Returns:
            Instruction string if there are mapped args, None otherwise.
        """
        if not tool_config.args_mapping:
            return None

        mapped_args = list(tool_config.args_mapping.keys())
        return (
            f"Note: The following arguments for '{tool_config.name}' will be "
            f"automatically filled and should not be specified: {', '.join(mapped_args)}"
        )
