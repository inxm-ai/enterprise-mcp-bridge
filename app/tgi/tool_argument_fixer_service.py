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
        (
            tool
            for tool in available_tools
            if tool["function"]["name"] == tool_call.function.name
        ),
        None,
    )

    if not tool_definition:
        logger.warning(
            f"[ToolArgumentFixerService] Tool '{tool_call.function.name}' not found"
        )
        return tool_call

    # Map the arguments to the tool definition recursively
    def normalize_for_compare(name):
        return "".join(c for c in name.upper() if c.isalnum())

    def coerce_type(value, schema):
        t = schema.get("type")
        if t == "integer":
            try:
                return int(value)
            except Exception:
                return value
        elif t == "number":
            try:
                return float(value)
            except Exception:
                return value
        elif t == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "1", "yes", "on"):
                    return True
                if v in ("false", "0", "no", "off"):
                    return False
            if isinstance(value, int):
                return bool(value)
            return value
        # Add more types as needed
        return value

    def fix_args(args, properties):
        if not isinstance(args, dict):
            return args
        fixed = {}
        direct_map = {prop: prop for prop in properties}
        normalized_map = {normalize_for_compare(prop): prop for prop in properties}
        for arg_name, arg_value in args.items():
            if arg_name in direct_map:
                prop_name = arg_name
            else:
                norm_arg = normalize_for_compare(arg_name)
                prop_name = normalized_map.get(norm_arg)
            if prop_name:
                prop_schema = properties[prop_name]
                if (
                    prop_schema.get("type") == "object"
                    and "properties" in prop_schema
                    and isinstance(arg_value, dict)
                ):
                    fixed[prop_name] = fix_args(arg_value, prop_schema["properties"])
                elif (
                    prop_schema.get("type") == "array"
                    and "items" in prop_schema
                    and isinstance(arg_value, list)
                ):
                    # Coerce each item in the array
                    fixed[prop_name] = [
                        coerce_type(item, prop_schema["items"]) for item in arg_value
                    ]
                else:
                    fixed[prop_name] = coerce_type(arg_value, prop_schema)
            else:
                logger.warning(
                    f"[ToolArgumentFixerService] Argument '{arg_name}' not found in tool '{tool_call.function.name}' definition"
                )
        return fixed

    # try to parse the arguments as JSON, if it's not already a dict
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

    properties = (
        tool_definition.get("function", {}).get("parameters", {}).get("properties", {})
    )

    fixed_arguments = fix_args(args, properties)

    return ToolCall(
        id=tool_call.id,
        function=ToolCallFunction(
            name=tool_call.function.name, arguments=json.dumps(fixed_arguments)
        ),
    )
