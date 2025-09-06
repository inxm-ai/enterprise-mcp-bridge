import logging
import json

from opentelemetry import trace
from app.tgi.models import ToolCall, ToolCallFunction

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


def fix_tool_arguments(tool_call: ToolCall, available_tools: list[dict]) -> ToolCall:
    """
    Fix the arguments of a tool call by mapping them to the available tools.
    Args:
        tool_call: The original tool call
        available_tools: The list of available tools

    Returns:
        The fixed tool call
    """
    # Find the tool definition
    tool_definition = next(
        (tool for tool in available_tools if tool["name"] == tool_call.function.name),
        None,
    )

    if not tool_definition:
        logger.warning(
            f"[ToolArgumentFixerService] Tool '{tool_call.function.name}' not found"
        )
        return tool_call

    # Map the arguments to the tool definition
    fixed_arguments = {}

    # try to parse the arguments as JSON, if it's not already a dict
    args = {}
    if not isinstance(tool_call.function.arguments, dict):
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            logger.error(
                f"[ToolArgumentFixerService] Failed to parse tool call arguments: {e}"
            )
            return tool_call
    else:
        args = tool_call.function.arguments
        logger.debug("[ToolArgumentFixerService] Tool call arguments already a dict")

    # TODO: Implement argument fixing logic here
    # The tool_definition contains a "parameters" key with the expected argument structure as JSON Schema
    #  The LLM might have confused the argument names, and for instance changed argumentName to argument_name
    #  We need to map the arguments back to the expected names and types

    def normalize(name):
        import re

        if "_" in name:
            # snake_case to camelCase
            camel = "".join(
                word.capitalize() if i > 0 else word
                for i, word in enumerate(name.split("_"))
            )
            return [name, camel]
        else:
            # camelCase to snake_case
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
            return [name, snake]

    properties = tool_definition.get("parameters", {}).get("properties", {})
    normalized_map = {}
    for prop in properties:
        for norm in normalize(prop):
            normalized_map[norm] = prop

    for arg_name, arg_value in args.items():
        mapped_name = normalized_map.get(arg_name)
        if mapped_name:
            fixed_arguments[mapped_name] = arg_value
        else:
            logger.warning(
                f"[ToolArgumentFixerService] Argument '{arg_name}' not found in tool '{tool_call.function.name}' definition"
            )

    return ToolCall(
        id=tool_call.id,
        function=ToolCallFunction(
            name=tool_call.function.name, arguments=json.dumps(fixed_arguments)
        ),
    )
