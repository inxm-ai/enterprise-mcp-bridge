"""
Tool service module for handling MCP tool operations.
"""

import json
import copy
import logging
import inspect
from typing import Callable, List, Optional, Dict, Any, Tuple, Union
from app.vars import TGI_MODEL_NAME, TOOL_CHUNK_SIZE
from opentelemetry import trace

from app.models import RunToolResultContent
from app.tgi.models import Message, MessageRole, Tool, ToolCall
from app.tgi.clients.llm_client import LLMClient
from app.tgi.models import ChatCompletionRequest
from app.session import MCPSessionBase
from app.tgi.services.tools.tool_argument_fixer_service import fix_tool_arguments
from app.tgi.services.tools.tool_resolution import ToolCallFormat
from app.tgi.services.tools.tools_map import map_tools, inline_schema
from app.tgi.models.model_formats import BaseModelFormat, get_model_format_for
from app.session_manager.session_context import filter_tools, get_tool_name

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)
_WORKFLOW_DESCRIPTION_TOOL_NAMES = {"plan"}


def _compact_text(text: str) -> str:
    """Trim leading whitespace and minify JSON-like payloads."""
    if not isinstance(text, str):
        return text
    compact = text.strip()
    if not compact:
        return compact
    if compact[0] in ("{", "["):
        try:
            compact = json.dumps(
                json.loads(compact),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (json.JSONDecodeError, TypeError):
            pass
    return compact


class ToolService:
    """Service for handling MCP tool operations."""

    def __init__(
        self,
        *,
        model_format: Optional[BaseModelFormat] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.logger = logger
        self.model_format = model_format or get_model_format_for()
        # llm client used for optional summarization of large tool outputs
        try:
            self.llm_client = llm_client or LLMClient(self.model_format)
        except Exception:
            # Keep ToolService usable in tests/environments where LLMClient
            # may not initialize cleanly
            self.llm_client = None
        self._tool_registry: dict[str, dict] = {}

    async def get_all_mcp_tools(
        self, session: MCPSessionBase, parent_span=None, include_output_schema=False
    ) -> List[Tool]:
        """
        Get all available tools from the MCP server as OpenAI-compatible tools.

        Args:
            session: The MCP session to use
            parent_span: Optional parent span for tracing
            include_output_schema: If True, include outputSchema in tool definitions.
                                   Useful for UI generation. Defaults to False.

        Returns:
            List of all available tools in OpenAI format
        """
        with tracer.start_as_current_span("get_all_mcp_tools") as span:
            try:
                # Get available tools from MCP server
                raw_tools = await session.list_tools()
                filtered_tools = filter_tools(raw_tools)
                # Build registry while handling both raw and already-mapped tools
                self._tool_registry = {}
                for tool in filtered_tools:
                    name = get_tool_name(tool)
                    if name:
                        self._tool_registry[name] = tool

                # Detect if the tools are already in OpenAI function-call format (dict with function, or object with function.name)
                def _is_mapped(tool):
                    if isinstance(tool, dict):
                        func = tool.get("function") or {}
                        return isinstance(func, dict) and bool(func.get("name"))
                    func = getattr(tool, "function", None)
                    return bool(getattr(func, "name", None))

                is_already_mapped = bool(filtered_tools) and all(
                    _is_mapped(t) for t in filtered_tools
                )

                if is_already_mapped:
                    # Normalize to plain dicts for downstream consumers
                    normalized = []
                    for tool in filtered_tools:
                        if hasattr(tool, "model_dump"):
                            normalized.append(tool.model_dump(exclude_none=True))
                        elif isinstance(tool, dict):
                            normalized.append(tool)
                        else:
                            # best-effort object -> dict conversion
                            func = getattr(tool, "function", None)
                            normalized.append(
                                {
                                    "type": getattr(tool, "type", "function"),
                                    "function": {
                                        "name": getattr(func, "name", None),
                                        "description": getattr(
                                            func, "description", None
                                        ),
                                        "parameters": getattr(func, "parameters", None),
                                    },
                                }
                            )
                    tools_result = normalized
                else:
                    tools_result = map_tools(
                        filtered_tools, include_output_schema=include_output_schema
                    )
                if not tools_result:
                    self.logger.debug(
                        "[ToolService] No tools available from MCP server"
                    )
                    span.set_attribute("tools.count", 0)
                    return []

                self._augment_workflow_description_arg(tools_result)

                tools_result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "describe_tool",
                            "description": "Return the full schema for one of your available tools. Use this when you need parameter details.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Tool name exactly as provided in the tool list",
                                    }
                                },
                                "required": ["name"],
                            },
                        },
                    }
                )

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

    def _augment_workflow_description_arg(self, tools_result: List[Tool]) -> None:
        """
        Ensure workflow-creation tools request a short description argument.

        This adds or updates the "description" input schema for workflow creation
        so the LLM provides a 3-5 word identifier derived from the user's prompt.
        """
        if not tools_result:
            return
        for tool in tools_result:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function") or {}
            name = func.get("name")
            if name not in _WORKFLOW_DESCRIPTION_TOOL_NAMES:
                continue

            params = func.get("parameters")
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}}
                func["parameters"] = params
            if params.get("type") is None:
                params["type"] = "object"

            properties = params.get("properties")
            if not isinstance(properties, dict):
                properties = {}
                params["properties"] = properties

            desc_schema = properties.get("description")
            if not isinstance(desc_schema, dict):
                desc_schema = {"type": "string"}
                properties["description"] = desc_schema
            if desc_schema.get("type") is None:
                desc_schema["type"] = "string"

            desc_text = desc_schema.get("description") or ""
            desc_lower = desc_text.lower()
            if not desc_text:
                desc_schema["description"] = (
                    "Short identifier derived from the user's request (3-5 words max)."
                )
            elif "3-5" not in desc_lower and "three to five" not in desc_lower:
                desc_schema["description"] = f"{desc_text} (3-5 words max)"

            required = params.get("required")
            if not isinstance(required, list):
                required = []
                params["required"] = required
            if "description" not in required:
                required.append("description")

    async def execute_tool_call(
        self,
        session: MCPSessionBase,
        tool_call: ToolCall,
        access_token: Optional[str],
        progress_callback: Optional[
            Callable[[float, Optional[float], Optional[str], Optional[str]], Any]
        ] = None,
        log_callback: Optional[Callable[[str, Any, Optional[str]], Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single tool call via MCP.

        Args:
            session: The MCP session to use
            tool_call: The tool call to execute
            access_token: Optional access token for authentication
            progress_callback: Optional async callback for progress updates. Signature:
                (progress: float, total: float | None, message: str | None, tool_name: str | None)
            log_callback: Optional async callback for log updates. Signature:
                (level: str, data: Any, logger_name: str | None)

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
                raw_arguments = getattr(tool_call.function, "arguments", None)
                try:
                    if isinstance(raw_arguments, str):
                        raw_arguments = raw_arguments.strip()
                    args = json.loads(raw_arguments) if raw_arguments else {}
                    if args is None:
                        args = {}
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
                        "content": json.dumps(
                            {"error": error_msg},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
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
                        "content": json.dumps(
                            {"error": error_msg},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }

                span.set_attribute("args", str(args))
                logger.debug(
                    f"[ToolService] Executing tool '{tool_call.function.name}' with args: {args}"
                )
                call_args = args
                try:
                    tool_info = self._tool_registry.get(tool_call.function.name)
                    schema = (
                        tool_info.get("inputSchema")
                        if isinstance(tool_info, dict)
                        else None
                    )
                    if schema is not None:
                        props = (
                            schema.get("properties", {})
                            if isinstance(schema, dict)
                            else {}
                        )
                        # Some tools explicitly expect "no args"; allow None only when schema is known and empty
                        if not props and args == {}:
                            call_args = None
                except Exception:
                    call_args = args

                async def _progress_wrapper(
                    progress: float,
                    total: Optional[float] = None,
                    message: Optional[str] = None,
                ):
                    if not progress_callback:
                        return
                    try:
                        await progress_callback(
                            progress, total, message, tool_call.function.name
                        )
                    except Exception:
                        logger.debug(
                            "[ToolService] Progress callback failed for %s",
                            tool_call.function.name,
                        )

                call_kwargs: dict[str, Any] = {}
                call_fn = getattr(session, "call_tool", None)

                # Prefer explicit progress-aware call if available
                if progress_callback and hasattr(session, "call_tool_with_progress"):
                    call_fn = getattr(session, "call_tool_with_progress")
                    sig = inspect.signature(call_fn)
                    if "progress_callback" in sig.parameters:
                        call_kwargs["progress_callback"] = _progress_wrapper
                    if log_callback and "log_callback" in sig.parameters:
                        call_kwargs["log_callback"] = log_callback
                elif progress_callback and call_fn:
                    try:
                        sig = inspect.signature(call_fn)
                        if "progress_callback" in sig.parameters:
                            call_kwargs["progress_callback"] = _progress_wrapper
                    except Exception:
                        pass

                if not call_fn:
                    raise RuntimeError("MCP session missing call_tool implementation")

                result = await call_fn(
                    tool_call.function.name, call_args, access_token, **call_kwargs
                )
                if inspect.isawaitable(result):
                    # Some MCP clients may return a coroutine from the handler; ensure it is awaited
                    result = await result

                # Normalize common result shapes (object or dict)
                result_is_error = getattr(result, "isError", None)
                if result_is_error is None and isinstance(result, dict):
                    result_is_error = result.get("isError")

                structured = getattr(result, "structuredContent", None)
                if structured is None and isinstance(result, dict):
                    structured = result.get("structuredContent")

                content_entries = getattr(result, "content", None)
                if content_entries is None and isinstance(result, dict):
                    content_entries = result.get("content")

                if result_is_error:
                    error_content = ""
                    if content_entries:
                        error_content = (
                            str(content_entries[0].text)
                            if content_entries
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
                        "content": json.dumps(
                            {"error": error_content},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }

                # Format successful result
                content = ""
                if structured:
                    logger.debug(
                        f"[ToolService] Tool '{tool_call.function.name}' returned structured content: {str(structured)[1000:]}"
                    )
                    content = json.dumps(
                        structured,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                elif content_entries:
                    logger.debug(
                        f"[ToolService] Tool '{tool_call.function.name}' returned content: {str(content_entries)[1000:]}"
                    )

                    if isinstance(content_entries, RunToolResultContent):
                        value = content_entries.value
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
                                for item in content_entries
                            ],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                else:
                    logger.debug(
                        f"[ToolService] Tool '{tool_call.function.name}' returned content: {str(content_entries)[1000:] if content_entries else str(result)[1000:]}"
                    )
                    content = json.dumps(
                        {"result": str(result)},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )

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
                    f"Error details: {json.dumps(error, ensure_ascii=False, separators=(',', ':'))}"
                )

        return "\n\n".join(formatted_messages)

    async def create_result_message(
        self,
        format: ToolCallFormat,
        tool_result: Dict[str, Any],
        *,
        summarize: bool = True,
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
                text_content = _compact_text(
                    json.dumps(content, ensure_ascii=False, separators=(",", ":"))
                )
            except Exception:
                text_content = str(content)

        # If content is very large and we have an llm client, try to summarize it.
        if (
            summarize
            and len(text_content) > TOOL_CHUNK_SIZE
            and getattr(self, "llm_client", None)
        ):
            try:
                # Create a minimal ChatCompletionRequest to allow summarize_text
                base_request = ChatCompletionRequest(messages=[], model=TGI_MODEL_NAME)
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

        if isinstance(content, str):
            content = _compact_text(content)

        message = self.model_format.build_tool_message(format, tool_result, content)
        logger.debug(
            "[ToolService] Created tool result message role=%s content=%s",
            message.role,
            message.content,
        )
        return message

    def _result_has_error(self, result: dict) -> bool:
        """Detect common error signals in tool execution results."""
        if not isinstance(result, dict):
            return False

        if result.get("isError") is True:
            return True
        if result.get("success") is False:
            return True

        content = result.get("content")
        parsed_content = None

        if isinstance(content, str):
            try:
                parsed_content = json.loads(content)
            except Exception:
                parsed_content = None

            # Only check for "error" string if we COULD NOT parse it as JSON
            if parsed_content is None and "error" in content.lower():
                return True
        elif isinstance(content, dict):
            parsed_content = content
        elif isinstance(content, list):
            parsed_content = content

        if isinstance(parsed_content, dict) and parsed_content.get("error"):
            return True

        if isinstance(parsed_content, list):
            for item in parsed_content:
                if isinstance(item, dict) and item.get("error"):
                    return True

        return False

    async def execute_tool_calls(
        self,
        session: MCPSessionBase,
        tool_calls: Union[List[Tuple[ToolCall, ToolCallFormat]], List[ToolCall]],
        access_token: Optional[str],
        parent_span,
        *,
        available_tools: Optional[List[dict]] = None,
        return_raw_results: bool = False,
        progress_callback: Optional[
            Callable[[float, Optional[float], Optional[str], Optional[str]], Any]
        ] = None,
        log_callback: Optional[Callable[[str, Any, Optional[str]], Any]] = None,
        summarize_tool_results: bool = True,
        build_messages: bool = True,
    ) -> Tuple[List[Message], bool] | Tuple[List[Message], bool, List[Dict[str, Any]]]:
        """Execute multiple tool calls and return tool result messages.

        Args:
            return_raw_results: If True, also return the raw result dicts before
                summarization. This is useful for workflow engines that need to
                capture structured data from tool results.
            progress_callback: Optional async callback for progress updates from tools.
            log_callback: Optional async callback for log events from tools.
            summarize_tool_results: If False, skip LLM-backed summarization of
                large tool outputs when building tool messages.
            build_messages: If False, skip creating tool result messages entirely.

        Returns:
            If return_raw_results is False: (messages, success)
            If return_raw_results is True: (messages, success, raw_results)
        """
        tool_results = []
        raw_results = []  # Track raw results before summarization
        success = True
        mapped_tools = available_tools or map_tools(
            filter_tools(await session.list_tools())
        )

        def _coerce_tool_calls():
            # Allow legacy callers to pass just ToolCall objects
            if not tool_calls:
                return []
            first = tool_calls[0]
            if isinstance(first, tuple):
                return tool_calls
            return [(call, ToolCallFormat.OPENAI_JSON) for call in tool_calls]

        for tool_call, tool_call_format in _coerce_tool_calls():
            try:
                if tool_call.function.name == "describe_tool":
                    result = await self._handle_describe_tool(tool_call)
                    if return_raw_results:
                        raw_results.append(result)
                    if build_messages:
                        tool_message = await self.create_result_message(
                            tool_call_format,
                            result,
                            summarize=summarize_tool_results,
                        )
                        tool_results.append(tool_message)
                    continue
                tool_call = fix_tool_arguments(tool_call, mapped_tools)
                result = await self.execute_tool_call(
                    session,
                    tool_call,
                    access_token,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                )

                if return_raw_results:
                    raw_results.append(result)

                if self._result_has_error(result):
                    success = False

                if build_messages:
                    tool_message = await self.create_result_message(
                        tool_call_format,
                        result,
                        summarize=summarize_tool_results,
                    )
                    tool_results.append(tool_message)
            except Exception as e:
                # If tool execution fails completely, create an error message
                error_msg = f"Failed to execute tool {getattr(tool_call.function, 'name', 'unknown')}: {str(e)}"
                err_lower = str(e).lower()
                if "coroutine" in err_lower:
                    error_msg += (
                        " Hint: the MCP tool appears to have returned a coroutine "
                        "without awaiting it. Define the tool as async or await the "
                        "returned coroutine before returning."
                    )
                self.logger.error(f"[ToolService] {error_msg}")

                error_result = {
                    "name": getattr(tool_call.function, "name", None),
                    "tool_call_id": tool_call.id,
                    "content": {"error": error_msg},
                }
                if return_raw_results:
                    raw_results.append(error_result)
                if build_messages:
                    error_message = await self.create_result_message(
                        tool_call_format,
                        error_result,
                        summarize=summarize_tool_results,
                    )
                    tool_results.append(error_message)
                parsed_errors = self._parse_json_array_from_message(error_msg)
                user_asks_for_correction = Message(
                    role=MessageRole.USER,
                    name="mcp_tool_retry_hint",
                    content=(
                        self._format_errors(parsed_errors)
                        if parsed_errors
                        else f"Please fix the error, or use any other available tools you have to get the required information, and call the tool {getattr(tool_call.function, 'name', 'unknown')} with the corrected arguments again."
                    ),
                )
                if user_asks_for_correction.content:
                    user_asks_for_correction.content = _compact_text(
                        user_asks_for_correction.content
                    )
                tool_results.append(user_asks_for_correction)
                success = False
                break

        if return_raw_results:
            return tool_results, success, raw_results
        return tool_results, success

    async def _handle_describe_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        try:
            args = (
                json.loads(tool_call.function.arguments)
                if tool_call.function.arguments
                else {}
            )
        except json.JSONDecodeError as exc:
            error_content = json.dumps(
                {"error": f"Invalid JSON: {exc}"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": error_content,
            }

        target = args.get("name")
        tool_info = self._tool_registry.get(target)
        if not tool_info:
            content = json.dumps(
                {"error": f"Unknown tool '{target}'"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": content,
            }

        schema = tool_info.get("inputSchema") or {}
        schema = inline_schema(copy.deepcopy(schema), schema)

        content = json.dumps(
            {
                "name": tool_info.get("name"),
                "description": tool_info.get("description"),
                "inputSchema": schema,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": content,
        }


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
        compact = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        logger.debug(
            f"[ToolService] Processed tool arguments: {arguments_str} -> {compact}"
        )
        return compact

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
