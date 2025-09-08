"""
Tool service module for handling MCP tool operations.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from app.vars import TOOL_CHUNK_SIZE
from opentelemetry import trace

from app.models import RunToolResultContent
from app.tgi.models import Message, MessageRole, Tool, ToolCall
from app.tgi.llm_client import LLMClient
from app.tgi.models import ChatCompletionRequest
from app.session import MCPSessionBase
from app.tgi.tool_argument_fixer_service import fix_tool_arguments
from app.tgi.tool_resolution import ToolCallFormat
from app.tgi.tools_map import map_tools

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class ToolService:
    """Service for handling MCP tool operations."""

    def __init__(self):
        self.logger = logger
        # llm client used for optional summarization of large tool outputs
        try:
            self.llm_client = LLMClient()
        except Exception:
            # Keep ToolService usable in tests/environments where LLMClient
            # may not initialize cleanly
            self.llm_client = None

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
                logger.debug(
                    f"[ToolService] Executing tool '{tool_call.function.name}' with args: {args}"
                )
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
                    f"Make sure you have correctly formatted the json, didn't put any data in strings, or failed to escape special characters. The following message should help you to identify the problem:\n"
                    f"Error details: {json.dumps(error, indent=2)}"
                )

        return "\n\n".join(formatted_messages)

    async def create_result_message(
        self, format: ToolCallFormat, tool_result: Dict[str, Any]
    ) -> Message:
        """Create a Message object from a tool result dictionary.

        If the tool result content is very large (>10000 characters) and an
        LLM client is available, try to summarize it via
        llm_client.summarize_text.
        """
        content = tool_result.get("content", "")

        # Normalize content into a string for size checks and message creation.
        # If content is not a string, try to JSON-dump it; if that fails, fallback to str().
        if isinstance(content, str):
            text_content = content
        else:
            try:
                text_content = json.dumps(content)
            except Exception:
                text_content = str(content)

        # If content is very large and we have an llm client, try to summarize it.
        if len(text_content) > TOOL_CHUNK_SIZE and getattr(self, "llm_client", None):
            try:
                # Create a minimal ChatCompletionRequest to allow summarize_text
                base_request = ChatCompletionRequest(messages=[], model=None)
                # ask the LLM client to summarize the large content
                summary = await self.llm_client.summarize_text(
                    base_request, text_content, None, None
                )
                # If summarize_text returns a value, use it; otherwise fall back
                if summary:
                    content = summary
                else:
                    # keep original long content if no summary returned
                    content = text_content
            except Exception:
                # On any summarization failure, keep the original content
                content = text_content
        else:
            # Ensure the content used in the Message is a string
            if not isinstance(content, str):
                content = text_content

        if format == ToolCallFormat.OPENAI_JSON:
            logger.debug(
                f"[ToolService] Creating OPENAI_JSON tool result message: {content}"
            )
            return Message(
                role=MessageRole.TOOL,
                content=content,
                tool_call_id=tool_result.get("tool_call_id"),
                name=tool_result.get("name"),
            )
        elif format == ToolCallFormat.CLAUDE_XML:
            logger.debug(
                f"[ToolService] Creating CLAUDE_XML tool result message: {content}"
            )
            name = tool_result.get("name")
            xml_tag = f"<{name}_result>{content}</{name}_result>"
            return Message(
                role=MessageRole.ASSISTANT,
                content=xml_tag,
                tool_call_id=tool_result.get("tool_call_id"),
                name=name,
            )
        logger.debug(f"[ToolService] Creating default tool result message: {content}")
        return Message(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_result.get("tool_call_id"),
            name=tool_result.get("name"),
        )

    async def execute_tool_calls(
        self,
        session: MCPSessionBase,
        tool_calls: List[Tuple[ToolCall, ToolCallFormat]],
        access_token: Optional[str],
        parent_span,
    ) -> Tuple[List[Message], bool]:
        """Execute multiple tool calls and return tool result messages."""
        tool_results = []
        success = True
        # Part of the todo to verify arguments before calling the tool
        available_tools = map_tools(await session.list_tools())

        for tool_call, tool_call_format in tool_calls:
            try:
                tool_call = fix_tool_arguments(tool_call, available_tools)
                result = await self.execute_tool_call(session, tool_call, access_token)

                content_check = result.get("content")
                if isinstance(content_check, str) and "error" in content_check:
                    success = False

                tool_message = await self.create_result_message(
                    tool_call_format, result
                )
                tool_results.append(tool_message)
            except Exception as e:
                # If tool execution fails completely, create an error message
                error_msg = f"Failed to execute tool {getattr(tool_call.function, 'name', 'unknown')}: {str(e)}"
                self.logger.error(f"[ToolService] {error_msg}")

                error_message = await self.create_result_message(
                    tool_call_format,
                    {
                        "name": getattr(tool_call.function, "name", None),
                        "tool_call_id": tool_call.id,
                        "content": {"error": error_msg},
                    },
                )
                tool_results.append(error_message)
                parsed_errors = self._parse_json_array_from_message(error_msg)
                if tool_call_format != ToolCallFormat.CLAUDE_XML:
                    user_asks_for_correction = Message(
                        role=MessageRole.USER,
                        content=(
                            self._format_errors(parsed_errors)
                            if parsed_errors
                            else f"Please fix the error, or use any other available tools you have to get the required information, and call the tool {getattr(tool_call.function, 'name', 'unknown')} with the corrected arguments again."
                        ),
                    )
                    tool_results.append(user_asks_for_correction)
                success = False
                break

        return tool_results, success


def process_tool_arguments(arguments_str):
    """
    Robustly processes a JSON string, handling nested JSON that may be
    incorrectly double-escaped.
    """

    def is_json_string(s):
        try:
            json.loads(s)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    try:
        data = json.loads(arguments_str)

        for key, value in data.items():
            if isinstance(value, str) and is_json_string(value):
                data[key] = json.loads(value)
        logger.debug(
            f"[ToolService] Processed tool arguments: {arguments_str} -> {json.dumps(data)}"
        )
        return json.dumps(data)

    except json.JSONDecodeError as e:
        logger.warning(f"[ToolService] Error decoding top-level JSON: {e}")
        return arguments_str
    except Exception as e:
        logger.error(f"[ToolService] An unexpected error occurred: {e}")
        return arguments_str


def parse_and_clean_tool_call(tool_call_dict):
    """
    Parses a complete tool call dictionary and cleans its 'arguments' field
    using the process_tool_arguments function.
    """
    if "function" in tool_call_dict and "arguments" in tool_call_dict["function"]:
        arguments_str = tool_call_dict["function"]["arguments"]
        cleaned_arguments_str = process_tool_arguments(arguments_str)
        tool_call_dict["function"]["arguments"] = cleaned_arguments_str
    elif "arguments" in tool_call_dict:
        arguments_str = tool_call_dict["arguments"]
        cleaned_arguments_str = process_tool_arguments(arguments_str)
        tool_call_dict["arguments"] = cleaned_arguments_str
    else:
        logger.warning(
            f"[ToolService] Tool call is missing 'function' or 'arguments': {tool_call_dict}"
        )
    return tool_call_dict


def extract_tool_call_from_streamed_content(content_str):
    """
    Finds and extracts a potential tool call dictionary from a streamed content string.

    It looks for a JSON object and validates if it has the required
    tool call structure.
    """
    # Find all complete JSON objects by balancing braces
    json_strings = []
    start = None
    brace_count = 0
    i = 0
    while i < len(content_str):
        if content_str[i] == "{":
            if brace_count == 0:
                start = i
            brace_count += 1
        elif content_str[i] == "}":
            brace_count -= 1
            if brace_count == 0 and start is not None:
                json_strings.append(content_str[start : i + 1])
                start = None
        i += 1

    for json_str in json_strings:
        try:
            potential_tool_call = json.loads(json_str)
            if all(
                key in potential_tool_call
                for key in ["id", "index", "type", "function"]
            ):
                function_part = potential_tool_call["function"]
                if all(key in function_part for key in ["name", "arguments"]):
                    potential_tool_call["function"]["arguments"] = (
                        process_tool_arguments(function_part["arguments"])
                    )
                    return potential_tool_call
        except json.JSONDecodeError:
            continue  # Move to the next potential JSON object

    return None
