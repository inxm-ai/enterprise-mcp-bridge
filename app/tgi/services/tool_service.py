"""
Tool service module for handling MCP tool operations.
"""

import json
import copy
import logging
import inspect
import time
from typing import Callable, List, Optional, Dict, Any, Tuple, Union
from app.vars import TGI_MODEL_NAME, TOOL_CHUNK_SIZE, GENERATED_UI_TOOL_TEXT_CAP
from opentelemetry import trace

from app.elicitation import (
    ElicitationRequiredError,
    InvalidUserFeedbackError,
    UnsupportedElicitationSchemaError,
)
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
_TOOL_RESULT_CACHE_LIMIT = 50
_SELECT_TOOL_NAME = "select-from-tool-response"


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


def _extract_json_block(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    start = -1
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if start < 0:
            if char == "{":
                start = idx
                depth = 1
            continue
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0:
            return text[start : idx + 1]
    return None


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
                self._ensure_select_from_tool_response(tools_result)

                existing_names = {
                    tool.get("function", {}).get("name")
                    for tool in tools_result
                    if isinstance(tool, dict)
                }
                if "describe_tool" not in existing_names:
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

    def _ensure_select_from_tool_response(self, tools_result: List[Tool]) -> None:
        if not tools_result:
            return
        existing = {
            tool.get("function", {}).get("name")
            for tool in tools_result
            if isinstance(tool, dict)
        }
        if _SELECT_TOOL_NAME in existing:
            return
        tools_result.append(
            {
                "type": "function",
                "function": {
                    "name": _SELECT_TOOL_NAME,
                    "description": (
                        "Select a subset of a previous tool response using a JSON Pointer "
                        "(RFC 6901). Use tool_call_id to target a specific tool call, or "
                        "tool_name to select the most recent result for that tool."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_call_id": {
                                "type": "string",
                                "description": "Tool call ID to select from (preferred).",
                            },
                            "tool_name": {
                                "type": "string",
                                "description": "Tool name to select the most recent result from.",
                            },
                            "pointer": {
                                "type": "string",
                                "description": "JSON Pointer path. Use '' to return the full result.",
                            },
                            "default": {
                                "description": "Optional fallback value if the pointer cannot be resolved.",
                            },
                        },
                        "required": ["pointer"],
                        "additionalProperties": False,
                    },
                },
            }
        )

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
                                parsed = self._try_parse_json_value(value)
                                if parsed is not None:
                                    content = parsed
                                else:
                                    content = self._compact_large_text(value)
                        if isinstance(content, (dict, list)):
                            content = json.dumps(
                                content,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                    else:
                        json_from_text = self._parse_json_from_text_entries(
                            content_entries
                        )
                        if json_from_text is not None:
                            if self._payload_has_error_signal(json_from_text):
                                resolved_error = self._extract_error_message(
                                    json_from_text
                                )
                                if not resolved_error:
                                    resolved_error = await self._explain_error_with_llm(
                                        json_from_text, access_token, span
                                    )
                                if not resolved_error:
                                    resolved_error = "Tool returned an error response."
                                content = json.dumps(
                                    {"error": resolved_error},
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                            else:
                                content = json.dumps(
                                    json_from_text,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                        else:
                            compact_text_payload = self._compact_text_entries_payload(
                                content_entries
                            )
                            if compact_text_payload is not None:
                                content = compact_text_payload
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

                # Preserve the original content before any truncation so
                # that workflow engines can capture structured data from the
                # full result (e.g. ToolResultCapture / ArgInjector).
                raw_content = content

                if (
                    isinstance(content, str)
                    and len(content) > GENERATED_UI_TOOL_TEXT_CAP
                ):
                    parsed = self._try_parse_json_value(content)
                    if parsed is not None:
                        compact_json = _compact_text(
                            json.dumps(
                                parsed,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                        if len(compact_json) <= GENERATED_UI_TOOL_TEXT_CAP:
                            content = compact_json
                        else:
                            content = json.dumps(
                                {
                                    "text": compact_json[:GENERATED_UI_TOOL_TEXT_CAP],
                                    "truncated": True,
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                    else:
                        content = json.dumps(
                            {
                                "text": self._compact_large_text(content),
                                "truncated": True,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )

                self.logger.debug(
                    f"[ToolService] Tool '{tool_call.function.name}' executed successfully"
                )
                if span:
                    span.set_attribute("tool.success", True)

                result_dict = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": content,
                }
                # Attach the untruncated content only when truncation happened
                if raw_content is not content:
                    result_dict["_raw_content"] = raw_content
                return result_dict

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

    @staticmethod
    def _try_parse_json_value(value: Any) -> Optional[Any]:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            block = _extract_json_block(text)
            if not block:
                return None
            try:
                return json.loads(block)
            except Exception:
                return None

    def _extract_text_entries(self, content_entries: Any) -> List[str]:
        texts: List[str] = []
        if not isinstance(content_entries, list):
            return texts
        for item in content_entries:
            if isinstance(item, str):
                texts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                texts.append(text)
        return texts

    def _compact_large_text(
        self, text: str, *, limit: int = GENERATED_UI_TOOL_TEXT_CAP
    ) -> str:
        compact = _compact_text(text or "")
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]}...[truncated {len(compact) - limit} chars]"

    def _compact_text_entries_payload(self, content_entries: Any) -> Optional[str]:
        texts = self._extract_text_entries(content_entries)
        if not texts:
            return None
        joined = "\n".join(t for t in texts if isinstance(t, str)).strip()
        if not joined:
            return None

        compact_joined = self._compact_large_text(joined)
        is_truncated = len(compact_joined) < len(_compact_text(joined))
        payload: Dict[str, Any] = {"text": compact_joined}
        if is_truncated:
            payload["truncated"] = True
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _parse_json_from_text_entries(self, content_entries: Any) -> Optional[Any]:
        texts = self._extract_text_entries(content_entries)
        if not texts:
            return None
        if len(texts) == 1:
            return self._try_parse_json_value(texts[0])
        joined = "".join(texts).strip()
        if not joined:
            return None
        return self._try_parse_json_value(joined)

    def _payload_has_error_signal(self, payload: Any) -> bool:
        if isinstance(payload, dict):
            if payload.get("isError") is True:
                return True
            if payload.get("success") is False:
                return True
            if payload.get("error"):
                return True
            if payload.get("errors"):
                return True
            text_json = self._try_parse_json_value(payload.get("text"))
            if text_json is not None and self._payload_has_error_signal(text_json):
                return True
            nested_content = payload.get("content")
            nested_json = self._parse_json_from_text_entries(nested_content)
            if nested_json is not None and self._payload_has_error_signal(nested_json):
                return True
            return False

        if isinstance(payload, list):
            return any(self._payload_has_error_signal(item) for item in payload)

        if isinstance(payload, str):
            parsed = self._try_parse_json_value(payload)
            if parsed is not None:
                return self._payload_has_error_signal(parsed)
            lower = payload.lower()
            return "error" in lower or "exception" in lower

        return False

    def _extract_error_message(self, payload: Any) -> Optional[str]:
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return None
            parsed = self._try_parse_json_value(text)
            if parsed is not None:
                return self._extract_error_message(parsed)
            return text

        if isinstance(payload, dict):
            for key in (
                "error",
                "error_message",
                "errorMessage",
                "message",
                "detail",
                "reason",
                "description",
                "title",
            ):
                if key not in payload:
                    continue
                resolved = self._extract_error_message(payload.get(key))
                if resolved:
                    return resolved
            text_json = self._try_parse_json_value(payload.get("text"))
            if text_json is not None:
                resolved = self._extract_error_message(text_json)
                if resolved:
                    return resolved
            nested_content = payload.get("content")
            nested_json = self._parse_json_from_text_entries(nested_content)
            if nested_json is not None:
                resolved = self._extract_error_message(nested_json)
                if resolved:
                    return resolved
            return None

        if isinstance(payload, list):
            for item in payload:
                resolved = self._extract_error_message(item)
                if resolved:
                    return resolved
            return None

        return None

    async def _explain_error_with_llm(
        self,
        payload: Any,
        access_token: Optional[str],
        parent_span,
    ) -> Optional[str]:
        if not getattr(self, "llm_client", None):
            return None

        payload_text = str(payload)
        try:
            payload_text = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            )
        except Exception:
            pass

        base_request = ChatCompletionRequest(messages=[], model=TGI_MODEL_NAME)
        try:
            explanation = await self.llm_client.ask(
                base_prompt=(
                    "You receive raw tool error payloads. Return one concise "
                    "user-facing error message. Do not return JSON or markdown."
                ),
                base_request=base_request,
                outer_span=parent_span,
                question=f"Explain this tool error payload:\n{payload_text}",
                access_token=access_token or "",
            )
        except Exception as exc:
            self.logger.debug(
                "[ToolService] Failed to explain tool error payload via LLM: %s",
                exc,
            )
            return None

        if not explanation:
            return None
        cleaned = explanation.strip()
        return cleaned or None

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
            parsed_content = self._try_parse_json_value(content)
            if parsed_content is not None:
                try:
                    content = _compact_text(
                        json.dumps(
                            parsed_content,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                except Exception:
                    content = _compact_text(content)
            else:
                content = _compact_text(content)

            if len(content) > GENERATED_UI_TOOL_TEXT_CAP:
                content = json.dumps(
                    {
                        "text": content[:GENERATED_UI_TOOL_TEXT_CAP],
                        "truncated": True,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

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

        if result.get("isError") is True or result.get("success") is False:
            return True

        content = result.get("content")
        parsed_content: Any = content
        if isinstance(content, str):
            parsed_json = self._try_parse_json_value(content)
            if parsed_json is not None:
                parsed_content = parsed_json
            else:
                return "error" in content.lower()

        return self._payload_has_error_signal(parsed_content)

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
                    self._cache_tool_result(session, result)
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
                if tool_call.function.name == _SELECT_TOOL_NAME:
                    result = await self._handle_select_from_tool_response(
                        session, tool_call
                    )
                    self._cache_tool_result(session, result)
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
                missing_required = self._missing_required_tool_args(
                    tool_call, available_tools or []
                )
                if missing_required:
                    self.logger.info(
                        "[ToolService] Skipping tool '%s' due to missing required args: %s",
                        tool_call.function.name,
                        ", ".join(missing_required),
                    )
                    error_result = {
                        "name": tool_call.function.name,
                        "tool_call_id": tool_call.id,
                        "content": {
                            "error": "Missing required tool arguments",
                            "missing": missing_required,
                        },
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
                        user_asks_for_correction = Message(
                            role=MessageRole.USER,
                            name="mcp_tool_retry_hint",
                            content=(
                                "Missing required tool inputs: "
                                f"{', '.join(missing_required)}. "
                                "Ask the user for these values before calling the tool again."
                            ),
                        )
                        tool_results.append(user_asks_for_correction)
                    success = False
                    break
                self.logger.info(
                    "[ToolService] Required args present for tool '%s'; executing.",
                    tool_call.function.name,
                )
                result = await self.execute_tool_call(
                    session,
                    tool_call,
                    access_token,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                )

                self._cache_tool_result(session, result)

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
                if isinstance(
                    e,
                    (
                        ElicitationRequiredError,
                        InvalidUserFeedbackError,
                        UnsupportedElicitationSchemaError,
                    ),
                ):
                    raise
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

    def _missing_required_tool_args(
        self, tool_call: ToolCall, available_tools: List[dict]
    ) -> List[str]:
        tool_name = getattr(tool_call.function, "name", None)
        if not tool_name:
            return []
        tool_def = next(
            (
                tool
                for tool in available_tools
                if isinstance(tool, dict)
                and (tool.get("function") or {}).get("name") == tool_name
            ),
            None,
        )
        if not tool_def:
            return []
        params = (tool_def.get("function") or {}).get("parameters") or {}
        if not isinstance(params, dict):
            return []
        required = params.get("required")
        if not isinstance(required, list) or not required:
            return []
        try:
            raw_args = tool_call.function.arguments
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {}
        if not isinstance(args, dict):
            return list(required)
        missing = []
        for key in required:
            if key not in args or args.get(key) is None:
                missing.append(key)
        return missing

    def _get_tool_result_cache(self, session: MCPSessionBase) -> dict:
        cache = getattr(session, "_tool_result_cache", None)
        if not isinstance(cache, dict):
            cache = {"order": [], "items": {}}
            setattr(session, "_tool_result_cache", cache)
        cache.setdefault("order", [])
        cache.setdefault("items", {})
        return cache

    def _cache_tool_result(
        self, session: MCPSessionBase, result: Dict[str, Any]
    ) -> None:
        if not session or not isinstance(result, dict):
            return
        tool_call_id = result.get("tool_call_id")
        if not tool_call_id:
            return
        content = result.get("content")
        parsed = None
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None
        else:
            parsed = content
        cache = self._get_tool_result_cache(session)
        items = cache["items"]
        order = cache["order"]
        if tool_call_id in order:
            order.remove(tool_call_id)
        items[tool_call_id] = {
            "tool_call_id": tool_call_id,
            "name": result.get("name"),
            "content": content,
            "parsed": parsed,
            "ts": time.time(),
        }
        order.append(tool_call_id)
        while len(order) > _TOOL_RESULT_CACHE_LIMIT:
            oldest = order.pop(0)
            items.pop(oldest, None)

    def _resolve_cached_tool_result(
        self,
        session: MCPSessionBase,
        tool_call_id: Optional[str],
        tool_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not session:
            return None
        cache = self._get_tool_result_cache(session)
        items = cache.get("items", {})
        order = cache.get("order", [])
        if tool_call_id:
            return items.get(tool_call_id)
        if tool_name:
            for call_id in reversed(order):
                entry = items.get(call_id)
                if entry and entry.get("name") == tool_name:
                    return entry
        return None

    @staticmethod
    def _apply_json_pointer(data: Any, pointer: str) -> Any:
        if pointer == "":
            return data
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise ValueError("pointer must be a JSON Pointer string starting with '/'")
        current = data
        for raw_part in pointer.split("/")[1:]:
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, list):
                try:
                    index = int(part)
                except Exception as exc:
                    raise KeyError(f"Invalid array index '{part}'") from exc
                if index < 0 or index >= len(current):
                    raise KeyError(f"Index '{part}' out of range")
                current = current[index]
            elif isinstance(current, dict):
                if part not in current:
                    raise KeyError(f"Key '{part}' not found")
                current = current[part]
            else:
                raise KeyError("Pointer cannot be applied to non-container value")
        return current

    async def _handle_select_from_tool_response(
        self, session: MCPSessionBase, tool_call: ToolCall
    ) -> Dict[str, Any]:
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

        tool_call_id = args.get("tool_call_id") or args.get("toolCallId")
        tool_name = args.get("tool_name") or args.get("toolName")
        pointer = args.get("pointer")
        default_value = args.get("default", None)

        if pointer is None:
            content = json.dumps(
                {"error": "Missing required 'pointer' argument"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": content,
            }

        entry = self._resolve_cached_tool_result(session, tool_call_id, tool_name)
        if not entry:
            content = json.dumps(
                {
                    "error": "No cached tool result found for selection",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
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

        data = entry.get("parsed")
        if data is None:
            data = entry.get("content")

        try:
            if isinstance(data, str):
                if pointer != "":
                    raise ValueError("Pointer selection requires JSON content")
                selected = data
            else:
                selected = self._apply_json_pointer(data, pointer)
        except Exception as exc:
            if "default" in args:
                selected = default_value
            else:
                content = json.dumps(
                    {
                        "error": f"Pointer selection failed: {exc}",
                        "pointer": pointer,
                        "tool_call_id": entry.get("tool_call_id"),
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

        content = json.dumps(
            {
                "tool_call_id": entry.get("tool_call_id"),
                "pointer": pointer,
                "value": selected,
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
