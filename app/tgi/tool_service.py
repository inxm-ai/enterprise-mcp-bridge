"""
Tool service module for handling MCP tool operations.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from opentelemetry import trace

from app.models import RunToolResultContent
from app.tgi.models import Message, MessageRole, Tool, ToolCall
from app.session import MCPSessionBase
from app.tgi.tools_map import map_tools

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class ToolService:
    """Service for handling MCP tool operations."""

    def __init__(self):
        self.logger = logger

    async def get_all_mcp_tools(
        self, session: MCPSessionBase, parent_span=None
    ) -> List[Tool]:
        """
        Get all available tools from the MCP server as OpenAI-compatible tools.

        Args:
            session: The MCP session to use
            parent_span: Optional parent span for tracing

        Returns:
            List of all available tools in OpenAI format
        """
        with tracer.start_as_current_span("get_all_mcp_tools") as span:
            try:
                # Get available tools from MCP server
                tools_result = map_tools(await session.list_tools())
                if not tools_result:
                    self.logger.debug(
                        "[ToolService] No tools available from MCP server"
                    )
                    span.set_attribute("tools.count", 0)
                    return []

                span.set_attribute("tools.count", len(tools_result))
                self.logger.debug(
                    f"[ToolService] Retrieved {len(tools_result)} MCP tools"
                )
                return tools_result

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                self.logger.error(f"[ToolService] Error getting MCP tools: {str(e)}")
                raise

    async def execute_tool_call(
        self, session: MCPSessionBase, tool_call: ToolCall, access_token: Optional[str]
    ) -> Dict[str, Any]:
        """
        Execute a single tool call via MCP.

        Args:
            session: The MCP session to use
            tool_call: The tool call to execute
            access_token: Optional access token for authentication

        Returns:
            The result of the tool execution
        """
        # Create span manually to avoid context issues
        span = None
        try:
            span = tracer.start_span("execute_tool_call")
            span.set_attribute("tool.name", tool_call.function.name)
            span.set_attribute("tool.call_id", tool_call.id)

            try:
                # Parse tool arguments
                try:
                    args = (
                        json.loads(tool_call.function.arguments)
                        if tool_call.function.arguments
                        else {}
                    )
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid tool arguments JSON: {str(e)}"
                    self.logger.error(f"[ToolService] {error_msg}")
                    if span:
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", error_msg)
                    return {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps({"error": error_msg}),
                    }

                # Check if session is still valid before calling
                if not hasattr(session, "call_tool"):
                    error_msg = "MCP session is invalid or closed"
                    self.logger.error(f"[ToolService] {error_msg}")
                    if span:
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", error_msg)
                    return {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps({"error": error_msg}),
                    }

                span.set_attribute("args", str(args))
                result = await session.call_tool(
                    tool_call.function.name, args, access_token
                )

                if result.isError:
                    error_content = ""
                    if hasattr(result, "content") and result.content:
                        error_content = (
                            str(result.content[0].text)
                            if result.content
                            else str(result)
                        )

                    self.logger.error(
                        f"[ToolService] Tool execution error: {error_content}"
                    )
                    if span:
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", error_content)

                    return {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps({"error": error_content}),
                    }

                # Format successful result
                content = ""
                if hasattr(result, "structuredContent") and result.structuredContent:
                    logger.debug(
                        f"[ToolService] Tool '{tool_call.function.name}' returned structured content: {result.structuredContent}"
                    )
                    content = json.dumps(result.structuredContent)
                elif hasattr(result, "content") and result.content:
                    logger.debug(
                        f"[ToolService] Tool '{tool_call.function.name}' returned content: {result.content}"
                    )

                    if isinstance(result.content, RunToolResultContent):
                        value = result.content.value
                        content = value
                        # this _might_ be valid json. Try to parse it if it's a string
                        if isinstance(value, str):
                            try:
                                content = json.loads(value)
                            except json.JSONDecodeError:
                                pass
                    else:
                        content = json.dumps(
                            [
                                {
                                    "text": (
                                        item.text
                                        if hasattr(item, "text")
                                        else str(item)
                                    )
                                }
                                for item in result.content
                            ]
                        )
                else:
                    logger.debug(
                        f"[ToolService] Tool '{tool_call.function.name}' returned content: {result.content}"
                    )
                    content = json.dumps({"result": str(result)})

                self.logger.debug(
                    f"[ToolService] Tool '{tool_call.function.name}' executed successfully"
                )
                if span:
                    span.set_attribute("tool.success", True)

                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": content,
                }

            except Exception as e:
                error_msg = f"Error executing tool: {type(e).__name__}"
                if str(e):
                    error_msg += f": {str(e)}"
                self.logger.error(f"[ToolService] {error_msg}")
                self.logger.exception("Full tool execution error traceback:")
                if span:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", error_msg)

                raise e

        finally:
            # Manually end span to avoid context issues
            if span:
                try:
                    span.end()
                except Exception:
                    pass  # Ignore any span cleanup errors

    def _parse_json_array_from_message(self, message):
        """
        Attempts to parse a valid JSON array from a string message.

        This function is useful for extracting structured data that might be
        embedded within a larger error message string.

        Args:
            message (str): The string to parse.

        Returns:
            list: The parsed list if a valid JSON array is found, otherwise None.
        """
        try:
            # Find the first and last brackets to isolate the potential JSON array
            start_index = message.find("[")
            end_index = message.rfind("]")

            if start_index == -1 or end_index == -1:
                return None

            json_string = message[start_index : end_index + 1]

            # Attempt to load the JSON string
            parsed_data = json.loads(json_string)

            # Check if the parsed data is a list
            if isinstance(parsed_data, list):
                return parsed_data
            else:
                return None
        except (json.JSONDecodeError, IndexError):
            # Return None if parsing fails
            return None

    def _format_errors(self, errors):
        """
        Takes a list of tool-use error dictionaries and formats them into a
        llm-followable string.

        Args:
            errors (list): A list of error dictionaries, typically from a failed
                        tool execution response.

        Returns:
            str: A single string containing a clear summary of all errors.
        """
        if not isinstance(errors, list) or not errors:
            return "No errors to format."

        formatted_messages = []

        for error in errors:
            # Safely get the values, providing defaults if keys are missing
            code = error.get("code", "N/A")
            expected_type = error.get("expected", "N/A")
            received_type = error.get("received", "N/A")
            path_list = error.get("path", [])
            message = error.get("message", "No message provided.")

            # Join the path parts to form a clear parameter name
            parameter_path = ".".join(path_list)

            # Check for the specific error type mentioned in the prompt
            if (
                code == "invalid_type"
                and parameter_path
                and expected_type
                and received_type
            ):
                formatted_message = (
                    f"A data type mismatch occurred. The tool expected the value for "
                    f"the parameter '{parameter_path}' to be a '{expected_type}', but it "
                    f"received a '{received_type}'.\n"
                    f"Example: If the tool expected 'boolean', you should use `true` or `false` instead of a string like 'true' or 'false'."
                )
                formatted_messages.append(formatted_message)
            else:
                # Fallback for other error types
                formatted_messages.append(
                    f"An error occurred: '{message}'\n"
                    f"Error details: {json.dumps(error, indent=2)}"
                )

        return "\n\n".join(formatted_messages)

    async def execute_tool_calls(
        self,
        session: MCPSessionBase,
        tool_calls: List[ToolCall],
        access_token: Optional[str],
        parent_span,
    ) -> Tuple[List[Message], bool]:
        """Execute multiple tool calls and return tool result messages."""
        tool_results = []
        success = True

        for tool_call in tool_calls:
            try:
                result = await self.execute_tool_call(session, tool_call, access_token)

                if "error" in result["content"]:
                    success = False
                tool_message = Message(
                    role=MessageRole.TOOL,
                    content=result["content"],
                    tool_call_id=result["tool_call_id"],
                    name=result["name"],
                )
                tool_results.append(tool_message)
            except Exception as e:
                # If tool execution fails completely, create an error message
                error_msg = (
                    f"Failed to execute tool {tool_call.function.name}: {str(e)}"
                )
                self.logger.error(f"[ToolService] {error_msg}")

                error_message = Message(
                    role=MessageRole.TOOL,
                    content=json.dumps({"error": error_msg}),
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                )
                tool_results.append(error_message)
                parsed_errors = self._parse_json_array_from_message(error_msg)
                user_asks_for_correction = Message(
                    role=MessageRole.USER,
                    content=(
                        self._format_errors(parsed_errors)
                        if parsed_errors
                        else f"Please fix the error, or use any other available tools you have to get the required information, and call the tool {tool_call.function.name} with the corrected arguments again."
                    ),
                )
                tool_results.append(user_asks_for_correction)
                success = False
                break

        return tool_results, success
