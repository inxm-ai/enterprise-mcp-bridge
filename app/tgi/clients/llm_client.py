"""
LLM client module for handling direct communication with the LLM API.
"""

import logging
import os
import time
import uuid
import json
import copy
import ast
import re
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    PermissionDeniedError,
)
from app.vars import (
    LLM_MAX_PAYLOAD_BYTES,
    TGI_MODEL_NAME,
    normalize_tgi_conversation_mode,
)
from opentelemetry import trace

from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    Message,
    MessageRole,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from app.utils import mask_token
from fastapi import HTTPException

from app.tgi.models.model_formats import BaseModelFormat, get_model_format_for
from app.tgi.context_compressor import get_default_compressor

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)
_WARNED_INVALID_CONVERSATION_MODES: set[str] = set()
_RESPONSES_JSON_OBJECT_ONLY_SCHEMAS: set[str] = set()


class LLMClient:
    """Client for communicating with the LLM API."""

    def __init__(self, model_format: Optional[BaseModelFormat] = None):
        self.logger = logger
        self.tgi_url = os.environ.get("TGI_URL", "")
        self.tgi_token = os.environ.get("TGI_TOKEN", "")
        raw_mode = os.environ.get("TGI_CONVERSATION_MODE", "chat/completions")
        self.conversation_mode = normalize_tgi_conversation_mode(raw_mode)
        normalized_input = (raw_mode or "").strip().lower()
        known_values = {
            "chat/completions",
            "/chat/completions",
            "chat",
            "chat_completions",
            "",
            "responses",
            "/responses",
        }
        if normalized_input not in known_values:
            if normalized_input not in _WARNED_INVALID_CONVERSATION_MODES:
                self.logger.warning(
                    "[LLMClient] Invalid TGI_CONVERSATION_MODE='%s'. Falling back to 'chat/completions'.",
                    raw_mode,
                )
                _WARNED_INVALID_CONVERSATION_MODES.add(normalized_input)
        self.model_format = model_format or get_model_format_for()

        # Ensure TGI_URL doesn't end with slash for consistent URL building
        if self.tgi_url.endswith("/"):
            self.tgi_url = self.tgi_url[:-1]

        self.logger.info(f"[LLMClient] Initialized with URL: {self.tgi_url}")
        self.logger.info("[LLMClient] Conversation mode: %s", self.conversation_mode)
        if self.tgi_token:
            self.logger.info(
                mask_token(
                    f"[LLMClient] Using authentication token: {self.tgi_token[:10]}...",
                    self.tgi_token[:10],
                )
            )
        else:
            self.logger.info("[LLMClient] No authentication token configured")

        self.client = AsyncOpenAI(
            api_key=self.tgi_token or "fake-token", base_url=self.tgi_url
        )

    def _payload_size(self, request: ChatCompletionRequest) -> int:
        """Calculate payload size in bytes."""
        return len(json.dumps(request.model_dump(exclude_none=True)).encode("utf-8"))

    def _message_size_summary(self, request: ChatCompletionRequest) -> List[dict]:
        summaries: List[dict] = []
        for index, message in enumerate(request.messages or []):
            content = message.content if message.content is not None else ""
            size = len(str(content).encode("utf-8"))
            summaries.append(
                {
                    "index": index,
                    "role": getattr(message, "role", "unknown"),
                    "bytes": size,
                }
            )
        summaries.sort(key=lambda item: item.get("bytes", 0), reverse=True)
        return summaries[:5]

    def _message_json_section_summary(
        self, request: ChatCompletionRequest
    ) -> List[dict]:
        sections: List[dict] = []
        for index, message in enumerate(request.messages or []):
            content = message.content if message.content is not None else ""
            if not isinstance(content, str):
                continue
            stripped = content.strip()
            if not stripped.startswith("{"):
                continue
            try:
                parsed = json.loads(stripped)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            for key, value in parsed.items():
                size = len(
                    json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
                )
                sections.append(
                    {
                        "index": index,
                        "role": getattr(message, "role", "unknown"),
                        "section": key,
                        "bytes": size,
                    }
                )
        sections.sort(key=lambda item: item.get("bytes", 0), reverse=True)
        return sections[:8]

    def _sanitize_message_contents(self, request: ChatCompletionRequest) -> None:
        """Ensure all message contents are strings to satisfy API validation."""
        for message in request.messages or []:
            if message.content is None:
                message.content = ""

    def _build_request_params(self, request: ChatCompletionRequest) -> dict:
        """Build chat-completions request params, removing invalid tool fields."""
        self.model_format.prepare_request(request)
        self._sanitize_message_contents(request)
        params = request.model_dump(exclude_none=True)
        tools = params.get("tools")
        if tools and not params.get("tool_choice"):
            if len(tools) == 1:
                params["tool_choice"] = tools[0]
            else:
                params["tool_choice"] = "auto"
        if not tools:
            params.pop("tool_choice", None)
            params.pop("tools", None)
        params.pop("persist_inner_thinking", None)
        return params

    @staticmethod
    def _to_dict(value: Any) -> dict:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(mode="json", exclude_none=True)
            except Exception:
                pass
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                pass
        return {}

    @staticmethod
    def _message_content_text(message: Message) -> str:
        content = message.content
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _normalized_tool_choice(tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            choice = tool_choice.strip().lower()
            if choice in ("auto", "none", "required"):
                return choice
            if choice:
                return {"type": "function", "name": choice}
            return None
        if isinstance(tool_choice, dict):
            choice_type = str(tool_choice.get("type") or "").strip().lower()
            if choice_type in ("auto", "none", "required"):
                return choice_type
            if choice_type == "function":
                function = tool_choice.get("function") or {}
                name = function.get("name") or tool_choice.get("name")
                if name:
                    return {"type": "function", "name": str(name)}
        return None

    def _messages_to_responses_input(self, messages: List[Message]) -> List[dict]:
        input_items: List[dict] = []
        seen_call_ids: set[str] = set()
        for index, message in enumerate(messages or []):
            if message.role == MessageRole.ASSISTANT and message.tool_calls:
                for tc in message.tool_calls:
                    call_id = getattr(tc, "id", None) or f"tool_call_{index}"
                    seen_call_ids.add(call_id)

        for index, message in enumerate(messages or []):
            role = (
                message.role.value
                if isinstance(message.role, MessageRole)
                else str(message.role)
            )
            text = self._message_content_text(message)

            if message.role == MessageRole.TOOL:
                call_id = message.tool_call_id or f"tool_call_{index}"
                if call_id in seen_call_ids:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": text,
                        }
                    )
                else:
                    tool_name = message.name or "tool"
                    input_items.append(
                        {
                            "role": "user",
                            "content": f"[{tool_name} result]\n{text}",
                        }
                    )
                continue

            if text:
                input_items.append({"role": role, "content": text})

            if message.role == MessageRole.ASSISTANT and message.tool_calls:
                for tc in message.tool_calls:
                    function = tc.function if tc else None
                    name = getattr(function, "name", None)
                    arguments = getattr(function, "arguments", None)
                    call_id = getattr(tc, "id", None) or f"tool_call_{index}"
                    if not name:
                        continue
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": str(name),
                            "arguments": (
                                str(arguments) if arguments is not None else "{}"
                            ),
                        }
                    )
            elif not text:
                input_items.append({"role": role, "content": ""})
        return input_items

    @staticmethod
    def _tool_to_responses_format(tool: Any) -> Optional[dict]:
        tool_dict = LLMClient._to_dict(tool)
        function = tool_dict.get("function") if isinstance(tool_dict, dict) else {}
        if not isinstance(function, dict):
            function = {}
        name = function.get("name") or tool_dict.get("name")
        if not name:
            return None
        mapped = {
            "type": "function",
            "name": str(name),
            "description": function.get("description"),
            "parameters": function.get("parameters") or {},
        }
        return {k: v for k, v in mapped.items() if v is not None}

    @staticmethod
    def _json_type_for_value(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    @staticmethod
    def _preferred_type_from_union(values: List[Any]) -> Optional[str]:
        normalized: List[str] = []
        for value in values:
            if isinstance(value, str) and value.strip():
                normalized.append(value.strip())
        if not normalized:
            return None

        preferred_order = (
            "object",
            "array",
            "string",
            "number",
            "integer",
            "boolean",
            "null",
        )
        for candidate in preferred_order:
            if candidate in normalized:
                return candidate
        return normalized[0]

    @staticmethod
    def _value_matches_declared_type(value: Any, declared_type: str) -> bool:
        actual_type = LLMClient._json_type_for_value(value)
        if declared_type == "number" and actual_type in {"number", "integer"}:
            return True
        if declared_type == "integer" and actual_type == "integer":
            return True
        return actual_type == declared_type

    @staticmethod
    def _preferred_type_from_union_for_node(
        values: List[Any], schema_node: Dict[str, Any]
    ) -> Optional[str]:
        normalized: List[str] = []
        for value in values:
            if isinstance(value, str) and value.strip():
                normalized.append(value.strip())
        if not normalized:
            return None

        enum_values = schema_node.get("enum")
        if isinstance(enum_values, list) and enum_values:
            enum_types = {
                LLMClient._json_type_for_value(enum_value) for enum_value in enum_values
            }
            if len(enum_types) == 1:
                only_type = next(iter(enum_types))
                if only_type in normalized:
                    return only_type
                if only_type == "integer" and "number" in normalized:
                    return "number"

        const_value = schema_node.get("const")
        if const_value is not None:
            const_type = LLMClient._json_type_for_value(const_value)
            if const_type in normalized:
                return const_type
            if const_type == "integer" and "number" in normalized:
                return "number"

        if "items" in schema_node and "array" in normalized:
            return "array"
        if (
            "properties" in schema_node
            or "required" in schema_node
            or "additionalProperties" in schema_node
        ) and "object" in normalized:
            return "object"

        return LLMClient._preferred_type_from_union(values)

    @staticmethod
    def _path_to_text(path: Tuple[str, ...]) -> str:
        if not path:
            return "$"
        return "$." + ".".join(path)

    @staticmethod
    def _schema_node_at_path(schema: Any, path: Tuple[str, ...]) -> Any:
        node = schema
        for token in path:
            if isinstance(node, list):
                if not token.isdigit():
                    return None
                idx = int(token)
                if idx < 0 or idx >= len(node):
                    return None
                node = node[idx]
                continue
            if isinstance(node, dict):
                if token not in node:
                    return None
                node = node[token]
                continue
            return None
        return node

    @staticmethod
    def _parse_invalid_schema_error_details(
        message: str,
    ) -> Tuple[Optional[str], Optional[Tuple[str, ...]], Optional[str]]:
        if (
            not isinstance(message, str)
            or "Invalid schema for response_format" not in message
        ):
            return None, None, None

        pattern = re.compile(
            r"Invalid schema for response_format '([^']+)': (?:In )?context=\((.*?)\), (.*)"
        )
        match = pattern.search(message)
        if not match:
            return None, None, message

        schema_name = match.group(1)
        raw_context = match.group(2).strip()
        detail = match.group(3).strip()
        if not raw_context:
            return schema_name, tuple(), detail

        try:
            parsed_context = ast.literal_eval(f"({raw_context},)")
            if not isinstance(parsed_context, tuple):
                return schema_name, None, detail
            return schema_name, tuple(str(item) for item in parsed_context), detail
        except Exception:
            return schema_name, None, detail

    def _log_responses_schema_rejection(
        self,
        *,
        request: ChatCompletionRequest,
        params: Dict[str, Any],
        exc: Exception,
    ) -> None:
        message = str(exc)
        schema_name, context_path, detail = self._parse_invalid_schema_error_details(
            message
        )
        if not schema_name:
            schema_name = (
                self._response_schema_name(request.response_format) or "response"
            )

        fmt = (params.get("text") or {}).get("format") or {}
        schema = fmt.get("schema")
        path_text = None
        node_preview = None
        if schema is not None and context_path is not None:
            path_text = self._path_to_text(context_path)
            node = self._schema_node_at_path(schema, context_path)
            if node is None:
                node_preview = "<path not found in request schema>"
            else:
                try:
                    node_preview = json.dumps(node, ensure_ascii=False)
                except Exception:
                    node_preview = str(node)
            if isinstance(node_preview, str) and len(node_preview) > 1000:
                node_preview = f"{node_preview[:1000]}...(trimmed)"

        self.logger.error(
            "[LLMClient] Responses strict json_schema rejected. schema=%s path=%s detail=%s node=%s",
            schema_name,
            path_text or "<unknown>",
            detail or message,
            node_preview or "<unavailable>",
        )

    def _log_responses_schema_normalization(
        self,
        *,
        schema_name: str,
        issues: List[str],
    ) -> None:
        if not issues:
            return
        preview = issues[:12]
        self.logger.warning(
            "[LLMClient] Responses strict json_schema normalization adjusted %s with %d issue(s): %s",
            schema_name,
            len(issues),
            " | ".join(preview),
        )
        if len(issues) > len(preview):
            self.logger.warning(
                "[LLMClient] Responses strict json_schema normalization for %s omitted %d additional issue(s).",
                schema_name,
                len(issues) - len(preview),
            )

    @staticmethod
    def _normalize_json_schema_for_responses(
        schema: Any,
    ) -> tuple[Any, List[str]]:
        """
        Normalize JSON schema for responses API strict validator subset.

        Returns:
            (normalized_schema, issues)
        """
        issues: List[str] = []

        def add_issue(path: Tuple[str, ...], reason: str) -> None:
            issues.append(f"{LLMClient._path_to_text(path)} {reason}")

        def walk_schema(node: Any, path: Tuple[str, ...]) -> Any:
            if isinstance(node, bool):
                return node

            if node is None:
                add_issue(path, "schema was null; replaced with {'type':'string'}")
                return {"type": "string"}

            if not isinstance(node, dict):
                add_issue(
                    path,
                    f"schema was {type(node).__name__}; replaced with {'type':'string'}",
                )
                return {"type": "string"}

            current: Dict[str, Any] = dict(node)

            schema_map_keywords = (
                "properties",
                "patternProperties",
                "$defs",
                "definitions",
                "dependentSchemas",
            )
            for keyword in schema_map_keywords:
                if keyword not in current:
                    continue
                value = current.get(keyword)
                if not isinstance(value, dict):
                    add_issue(
                        path + (keyword,),
                        "expected map of schemas; replaced with empty object",
                    )
                    current[keyword] = {}
                    continue
                converted: Dict[str, Any] = {}
                for map_key, map_value in value.items():
                    converted[str(map_key)] = walk_schema(
                        map_value, path + (keyword, str(map_key))
                    )
                current[keyword] = converted

            for keyword in ("anyOf", "oneOf", "allOf"):
                if keyword not in current:
                    continue
                value = current.get(keyword)
                if not isinstance(value, list):
                    add_issue(
                        path + (keyword,),
                        "was not a list; wrapped as single schema item",
                    )
                    value = [value]
                current[keyword] = [
                    walk_schema(item, path + (keyword, str(index)))
                    for index, item in enumerate(value)
                ]

            schema_value_keywords = (
                "items",
                "additionalProperties",
                "contains",
                "propertyNames",
                "if",
                "then",
                "else",
                "not",
                "unevaluatedItems",
                "unevaluatedProperties",
            )
            for keyword in schema_value_keywords:
                if keyword not in current:
                    continue
                value = current.get(keyword)
                if keyword == "items" and isinstance(value, list):
                    add_issue(
                        path + (keyword,),
                        "tuple-style items list converted to first schema entry",
                    )
                    value = value[0] if value else {"type": "string"}
                if isinstance(value, bool):
                    current[keyword] = value
                    continue
                current[keyword] = walk_schema(value, path + (keyword,))

            node_type = current.get("type")
            if isinstance(node_type, list):
                chosen = LLMClient._preferred_type_from_union_for_node(
                    node_type, current
                )
                if chosen:
                    add_issue(
                        path,
                        f"type union {node_type} collapsed to '{chosen}' for strict compatibility",
                    )
                    current["type"] = chosen
                else:
                    add_issue(path, "invalid type union replaced with 'string'")
                    current["type"] = "string"
            elif node_type is not None and not isinstance(node_type, str):
                add_issue(path, f"non-string type {node_type!r} replaced with 'string'")
                current["type"] = "string"

            if "type" not in current:
                inferred: Optional[str] = None
                if (
                    "properties" in current
                    or "required" in current
                    or "additionalProperties" in current
                ):
                    inferred = "object"
                elif "items" in current:
                    inferred = "array"
                elif any(keyword in current for keyword in ("anyOf", "oneOf", "allOf")):
                    branch_types: List[str] = []
                    for keyword in ("anyOf", "oneOf", "allOf"):
                        value = current.get(keyword)
                        if not isinstance(value, list):
                            continue
                        for branch in value:
                            if isinstance(branch, dict):
                                branch_type = branch.get("type")
                                if isinstance(branch_type, str):
                                    branch_types.append(branch_type)
                    if branch_types:
                        non_null = [t for t in branch_types if t != "null"]
                        inferred = non_null[0] if non_null else branch_types[0]
                elif "enum" in current and isinstance(current.get("enum"), list):
                    enum_types = {
                        LLMClient._json_type_for_value(enum_value)
                        for enum_value in current.get("enum") or []
                    }
                    if len(enum_types) == 1:
                        inferred = next(iter(enum_types))
                elif "const" in current:
                    inferred = LLMClient._json_type_for_value(current.get("const"))
                if inferred:
                    add_issue(path, f"inferred missing type='{inferred}'")
                    current["type"] = inferred
                else:
                    add_issue(path, "missing type; defaulted to 'string'")
                    current["type"] = "string"

            declared_type = current.get("type")
            enum_values = current.get("enum")
            if isinstance(enum_values, list) and isinstance(declared_type, str):
                filtered_enum = [
                    enum_value
                    for enum_value in enum_values
                    if LLMClient._value_matches_declared_type(enum_value, declared_type)
                ]
                if not filtered_enum:
                    add_issue(
                        path,
                        f"enum removed because no values matched declared type '{declared_type}'",
                    )
                    current.pop("enum", None)
                elif len(filtered_enum) != len(enum_values):
                    add_issue(
                        path,
                        f"enum filtered to values compatible with declared type '{declared_type}'",
                    )
                    current["enum"] = filtered_enum

            is_object_schema = (
                current.get("type") == "object" or "properties" in current
            )
            if is_object_schema:
                if current.get("type") != "object":
                    add_issue(path, "object-like schema forced to type='object'")
                    current["type"] = "object"

                properties = current.get("properties")
                if not isinstance(properties, dict):
                    if properties is not None:
                        add_issue(
                            path + ("properties",),
                            "invalid properties replaced with {}",
                        )
                    properties = {}
                    current["properties"] = properties

                required_properties = list(properties.keys())
                if current.get("required") != required_properties:
                    add_issue(
                        path,
                        "required adjusted to include exactly all object properties",
                    )
                    current["required"] = required_properties

                if current.get("additionalProperties") is not False:
                    add_issue(path, "additionalProperties forced to false")
                    current["additionalProperties"] = False

                if "items" in current:
                    add_issue(path, "items removed from object schema")
                    current.pop("items", None)

            if current.get("type") != "object":
                if "required" in current:
                    add_issue(path, "required removed from non-object schema")
                    current.pop("required", None)
                if "properties" in current:
                    add_issue(path, "properties removed from non-object schema")
                    current.pop("properties", None)
                if "additionalProperties" in current:
                    add_issue(
                        path, "additionalProperties removed from non-object schema"
                    )
                    current.pop("additionalProperties", None)

            if current.get("type") == "array" and "items" not in current:
                add_issue(path, "array missing items; defaulted to {'type':'string'}")
                current["items"] = {"type": "string"}

            if current.get("type") != "array" and "items" in current:
                add_issue(path, "items removed from non-array schema")
                current.pop("items", None)

            return current

        normalized = walk_schema(copy.deepcopy(schema), tuple())
        return normalized, issues

    def _response_format_to_text_config(
        self, response_format: Any, *, force_json_object: bool = False
    ) -> Optional[dict]:
        if not isinstance(response_format, dict):
            return None

        response_type = str(response_format.get("type") or "").strip().lower()
        if response_type == "json_schema":
            if force_json_object:
                return {"format": {"type": "json_object"}}
            schema_block = response_format.get("json_schema") or {}
            if not isinstance(schema_block, dict):
                schema_block = {}
            normalized_schema, normalization_issues = (
                self._normalize_json_schema_for_responses(
                    schema_block.get("schema") or {}
                )
            )
            schema_name = str(schema_block.get("name") or "response")
            self._log_responses_schema_normalization(
                schema_name=schema_name,
                issues=normalization_issues,
            )
            fmt: Dict[str, Any] = {
                "type": "json_schema",
                "name": schema_name,
                "schema": normalized_schema,
            }
            if "strict" in schema_block:
                fmt["strict"] = bool(schema_block.get("strict"))
            return {"format": fmt}

        if response_type == "json_object":
            return {"format": {"type": "json_object"}}

        return None

    def _build_responses_request_params(self, request: ChatCompletionRequest) -> dict:
        self._sanitize_message_contents(request)
        params: Dict[str, Any] = {
            "model": request.model,
            "input": self._messages_to_responses_input(request.messages or []),
        }

        tools = [
            mapped
            for mapped in (
                self._tool_to_responses_format(tool) for tool in (request.tools or [])
            )
            if mapped
        ]
        if tools:
            params["tools"] = tools
            tool_choice = self._normalized_tool_choice(request.tool_choice)
            if tool_choice is not None:
                params["tool_choice"] = tool_choice

        schema_name = self._response_schema_name(request.response_format)
        force_json_object = bool(
            schema_name and schema_name in _RESPONSES_JSON_OBJECT_ONLY_SCHEMAS
        )
        text = self._response_format_to_text_config(
            request.response_format,
            force_json_object=force_json_object,
        )
        if text:
            params["text"] = text
            fmt_type = (text.get("format") or {}).get("type")
            if fmt_type == "json_object" and not self._responses_input_mentions_json(
                params["input"]
            ):
                params["input"].append(
                    {"role": "system", "content": "Return valid JSON only."}
                )
            if fmt_type == "json_object":
                schema_guidance = self._responses_json_object_schema_guidance(
                    request.response_format
                )
                if schema_guidance:
                    params["input"].append(
                        {
                            "role": "system",
                            "content": schema_guidance,
                        }
                    )

        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.extra_headers:
            params["extra_headers"] = request.extra_headers
        if request.stop:
            self.logger.debug(
                "[LLMClient] 'stop' is not mapped in responses mode. Ignoring stop tokens."
            )
        return params

    @staticmethod
    def _response_schema_name(response_format: Any) -> Optional[str]:
        if not isinstance(response_format, dict):
            return None
        response_type = str(response_format.get("type") or "").strip().lower()
        if response_type != "json_schema":
            return None
        schema_block = response_format.get("json_schema") or {}
        if not isinstance(schema_block, dict):
            return None
        name = schema_block.get("name")
        if not name:
            return None
        return str(name)

    @staticmethod
    def _responses_json_object_schema_guidance(response_format: Any) -> Optional[str]:
        if not isinstance(response_format, dict):
            return None
        response_type = str(response_format.get("type") or "").strip().lower()
        if response_type != "json_schema":
            return None
        schema_block = response_format.get("json_schema") or {}
        if not isinstance(schema_block, dict):
            return None
        schema = schema_block.get("schema")
        if not isinstance(schema, dict):
            return None
        schema_name = str(schema_block.get("name") or "response")

        properties = schema.get("properties")
        prop_keys = list(properties.keys()) if isinstance(properties, dict) else []
        required = schema.get("required")
        req_keys = [str(k) for k in required] if isinstance(required, list) else []

        guidance_parts = [
            "Return a valid JSON object only. Do not return markdown.",
            f"The JSON must match the '{schema_name}' schema.",
        ]
        if prop_keys:
            guidance_parts.append(
                f"Top-level keys: {', '.join(str(k) for k in prop_keys)}."
            )
        if req_keys:
            guidance_parts.append(f"Required top-level keys: {', '.join(req_keys)}.")
        guidance_parts.append(
            "Schema:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        return " ".join(guidance_parts)

    @staticmethod
    def _is_json_schema_response_format(response_format: Any) -> bool:
        if not isinstance(response_format, dict):
            return False
        return str(response_format.get("type") or "").strip().lower() == "json_schema"

    def _should_retry_responses_with_json_object(
        self, request: ChatCompletionRequest, exc: Exception
    ) -> bool:
        if not self._is_json_schema_response_format(request.response_format):
            return False
        message = str(exc).lower()
        if "invalid_json_schema" in message:
            return True
        if (
            "text.format.schema" in message
            and "invalid schema for response_format" in message
        ):
            return True
        return False

    def _mark_schema_for_json_object_mode(self, request: ChatCompletionRequest) -> None:
        schema_name = self._response_schema_name(request.response_format)
        if schema_name:
            _RESPONSES_JSON_OBJECT_ONLY_SCHEMAS.add(schema_name)

    @staticmethod
    def _responses_input_mentions_json(input_items: List[dict]) -> bool:
        for item in input_items or []:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str) and "json" in content.lower():
                return True
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, str) and "json" in part.lower():
                        return True
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str) and "json" in text.lower():
                            return True
            output = item.get("output")
            if isinstance(output, str) and "json" in output.lower():
                return True
        return False

    @staticmethod
    def _extract_text_from_response_item(item: dict) -> str:
        item_type = item.get("type")
        if item_type == "message":
            content = item.get("content") or []
            if isinstance(content, str):
                return content
            pieces: List[str] = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        pieces.append(part)
                        continue
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in ("output_text", "text"):
                        text = part.get("text")
                        if text:
                            pieces.append(str(text))
            return "".join(pieces)
        if item_type in ("output_text", "text"):
            text = item.get("text")
            return str(text) if text is not None else ""
        return ""

    @staticmethod
    def _normalize_created(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            with_value = value
            if with_value.endswith("Z"):
                with_value = f"{with_value[:-1]}+00:00"
            try:
                return int(datetime.fromisoformat(with_value).timestamp())
            except Exception:
                pass
        return int(time.time())

    @staticmethod
    def _normalize_usage(raw_usage: Any) -> Optional[Usage]:
        usage = LLMClient._to_dict(raw_usage)
        if not usage:
            return None

        prompt_tokens = usage.get("input_tokens")
        if prompt_tokens is None:
            prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("output_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
        return Usage(
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            total_tokens=int(total_tokens or 0),
        )

    def _normalize_responses_api_result(
        self, response: Any, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        data = self._to_dict(response)
        output_items = data.get("output") or []

        content_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for item in output_items:
            if not isinstance(item, dict):
                item = self._to_dict(item)
            if not item:
                continue
            item_type = item.get("type")
            if item_type == "function_call":
                name = item.get("name")
                if not name:
                    continue
                tool_calls.append(
                    ToolCall(
                        id=str(item.get("call_id") or item.get("id") or uuid.uuid4()),
                        type="function",
                        function=ToolCallFunction(
                            name=str(name),
                            arguments=str(item.get("arguments") or "{}"),
                        ),
                    )
                )
                continue
            text = self._extract_text_from_response_item(item)
            if text:
                content_parts.append(text)

        output_text = data.get("output_text")
        if output_text and not content_parts:
            content_parts.append(str(output_text))

        message = Message(
            role=MessageRole.ASSISTANT,
            content="".join(content_parts),
        )
        if tool_calls:
            message.tool_calls = tool_calls

        usage = self._normalize_usage(data.get("usage"))

        return ChatCompletionResponse(
            id=str(data.get("id") or self.create_completion_id()),
            object="chat.completion",
            created=self._normalize_created(data.get("created_at")),
            model=str(data.get("model") or request.model or TGI_MODEL_NAME),
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
            usage=usage,
        )

    def _chat_chunk_json(
        self,
        *,
        request: ChatCompletionRequest,
        delta: Optional[dict] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        payload = {
            "id": self.create_completion_id(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model or TGI_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "delta": delta or {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _prepare_payload(
        self, request: ChatCompletionRequest, access_token: str = ""
    ) -> ChatCompletionRequest:
        """
        Prepare and compress payload if needed using adaptive compression strategy.

        Args:
            request: ChatCompletionRequest to prepare
            access_token: Access token for summarization operations

        Returns:
            ChatCompletionRequest
        """
        # Ensure model is set
        if not request.model:
            request.model = TGI_MODEL_NAME
        self._sanitize_message_contents(request)

        size = self._payload_size(request)

        if size <= LLM_MAX_PAYLOAD_BYTES:
            return request

        # Payload exceeds limit, apply compression
        self.logger.warning(
            "[LLMClient] Payload size %s exceeds limit %s, applying adaptive compression",
            size,
            LLM_MAX_PAYLOAD_BYTES,
        )
        self.logger.warning(
            "[LLMClient] Largest message contents before compression: %s",
            self._message_size_summary(request),
        )
        section_summary = self._message_json_section_summary(request)
        if section_summary:
            self.logger.warning(
                "[LLMClient] Largest JSON sections before compression: %s",
                section_summary,
            )

        compressor = get_default_compressor()
        compressed_request, stats = await compressor.compress(
            request,
            max_size=LLM_MAX_PAYLOAD_BYTES,
            summarize_fn=self.summarize_text,
        )
        request = compressed_request

        size = self._payload_size(request)

        self.logger.info(f"[LLMClient] Compression result: {stats.summary()}")
        oversized_sources = stats.metadata.get("oversized_sources") or []
        if oversized_sources:
            self.logger.warning(
                "[LLMClient] Oversized payload sources after analysis: %s",
                oversized_sources,
            )

        if size > LLM_MAX_PAYLOAD_BYTES:
            if oversized_sources:
                top = oversized_sources[0]
                self.logger.error(
                    "[LLMClient] Payload still too large; primary source role=%s index=%s source=%s total_bytes=%s",
                    top.get("role"),
                    top.get("index"),
                    top.get("source"),
                    top.get("total_bytes"),
                )
            raise HTTPException(
                status_code=413,
                detail=f"LLM payload size {size} remains above limit {LLM_MAX_PAYLOAD_BYTES} after compression",
            )

        return request

    def create_completion_id(self) -> str:
        """Generate a unique completion ID."""
        return f"chatcmpl-{uuid.uuid4().hex[:29]}"

    def create_usage_stats(
        self, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> Usage:
        """Create usage statistics."""
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    async def stream_completion(
        self,
        request: ChatCompletionRequest,
        access_token: str,
        parent_span,
    ) -> AsyncGenerator[str, None]:
        """Stream completion from the actual LLM."""
        # Remove all tracing to avoid context cleanup issues with async generators
        try:
            # Prepare payload with size-aware compression
            self.logger.info("[LLMClient] Preparing payload for streaming request.")
            request = await self._prepare_payload(request, access_token)

            self.logger.debug(f"[LLMClient] Opening stream to {self.tgi_url}")

            if self.conversation_mode == "responses":
                params = self._build_responses_request_params(request)
                params["stream"] = True
                self.logger.debug(
                    "[LLMClient] Responses streaming request params: %s", params
                )

                try:
                    stream = await self.client.responses.create(**params)
                except Exception as exc:
                    if not self._should_retry_responses_with_json_object(request, exc):
                        raise
                    self._log_responses_schema_rejection(
                        request=request,
                        params=params,
                        exc=exc,
                    )
                    self.logger.warning(
                        "[LLMClient] Responses streaming rejected json_schema response_format. Retrying once with text.format=json_object."
                    )
                    self._mark_schema_for_json_object_mode(request)
                    params = self._build_responses_request_params(request)
                    params["stream"] = True
                    stream = await self.client.responses.create(**params)
                tool_state: Dict[int, dict] = {}
                tool_index_by_id: Dict[str, int] = {}
                emitted_text = False
                emitted_tool_indexes: set[int] = set()

                def _tool_state_for(index: int) -> dict:
                    return tool_state.setdefault(
                        index,
                        {
                            "id": None,
                            "name": None,
                            "name_emitted": False,
                            "arguments_emitted": False,
                        },
                    )

                async for event in stream:
                    event_dict = self._to_dict(event)
                    event_type = event_dict.get("type") or getattr(event, "type", None)
                    if not event_type:
                        continue

                    if event_type == "response.output_text.delta":
                        delta = event_dict.get("delta")
                        if delta:
                            emitted_text = True
                            yield self._chat_chunk_json(
                                request=request,
                                delta={"content": str(delta)},
                            )
                        continue

                    if event_type in (
                        "response.output_item.added",
                        "response.output_item.done",
                    ):
                        item = event_dict.get("item") or {}
                        if not isinstance(item, dict):
                            item = self._to_dict(item)
                        if item.get("type") != "function_call":
                            continue
                        output_index = int(event_dict.get("output_index") or 0)
                        state = _tool_state_for(output_index)
                        call_id = item.get("call_id") or item.get("id")
                        if call_id:
                            state["id"] = str(call_id)
                            tool_index_by_id[str(call_id)] = output_index
                        if item.get("name"):
                            state["name"] = str(item.get("name"))

                        function_payload: Dict[str, Any] = {}
                        if state.get("name") and not state.get("name_emitted"):
                            function_payload["name"] = state["name"]
                            state["name_emitted"] = True
                        if (
                            event_type == "response.output_item.done"
                            and item.get("arguments") is not None
                            and not state.get("arguments_emitted")
                        ):
                            function_payload["arguments"] = str(item.get("arguments"))
                            state["arguments_emitted"] = True
                            emitted_tool_indexes.add(output_index)

                        if function_payload:
                            yield self._chat_chunk_json(
                                request=request,
                                delta={
                                    "tool_calls": [
                                        {
                                            "index": output_index,
                                            "id": state.get("id")
                                            or f"call_{output_index}",
                                            "type": "function",
                                            "function": function_payload,
                                        }
                                    ]
                                },
                            )
                        continue

                    if event_type == "response.function_call_arguments.delta":
                        output_index = int(event_dict.get("output_index") or 0)
                        state = _tool_state_for(output_index)
                        call_id = event_dict.get("call_id")
                        if call_id:
                            state["id"] = str(call_id)
                            tool_index_by_id[str(call_id)] = output_index
                        function_payload: Dict[str, Any] = {}
                        if state.get("name") and not state.get("name_emitted"):
                            function_payload["name"] = state["name"]
                            state["name_emitted"] = True
                        delta = event_dict.get("delta")
                        if delta is not None:
                            function_payload["arguments"] = str(delta)
                            state["arguments_emitted"] = True
                            emitted_tool_indexes.add(output_index)
                        if function_payload:
                            yield self._chat_chunk_json(
                                request=request,
                                delta={
                                    "tool_calls": [
                                        {
                                            "index": output_index,
                                            "id": state.get("id")
                                            or f"call_{output_index}",
                                            "type": "function",
                                            "function": function_payload,
                                        }
                                    ]
                                },
                            )
                        continue

                    if event_type == "response.function_call_arguments.done":
                        output_index = int(event_dict.get("output_index") or 0)
                        state = _tool_state_for(output_index)
                        call_id = event_dict.get("call_id")
                        if call_id:
                            state["id"] = str(call_id)
                            tool_index_by_id[str(call_id)] = output_index
                        done_arguments = event_dict.get("arguments")
                        if done_arguments is not None and not state.get(
                            "arguments_emitted"
                        ):
                            function_payload: Dict[str, Any] = {
                                "arguments": str(done_arguments)
                            }
                            if state.get("name") and not state.get("name_emitted"):
                                function_payload["name"] = state["name"]
                                state["name_emitted"] = True
                            state["arguments_emitted"] = True
                            emitted_tool_indexes.add(output_index)
                            yield self._chat_chunk_json(
                                request=request,
                                delta={
                                    "tool_calls": [
                                        {
                                            "index": output_index,
                                            "id": state.get("id")
                                            or f"call_{output_index}",
                                            "type": "function",
                                            "function": function_payload,
                                        }
                                    ]
                                },
                            )
                        continue

                    if event_type in ("response.completed", "response.done"):
                        raw_response = event_dict.get("response") or event_dict
                        final = self._normalize_responses_api_result(
                            raw_response, request
                        )
                        first_choice = final.choices[0] if final.choices else None
                        message = first_choice.message if first_choice else None
                        if message and message.content and not emitted_text:
                            emitted_text = True
                            yield self._chat_chunk_json(
                                request=request,
                                delta={"content": message.content},
                            )
                        if message and message.tool_calls:
                            for default_index, tc in enumerate(message.tool_calls):
                                call_id = tc.id
                                output_index = (
                                    tool_index_by_id.get(call_id, default_index)
                                    if call_id
                                    else default_index
                                )
                                if output_index in emitted_tool_indexes:
                                    continue
                                emitted_tool_indexes.add(output_index)
                                yield self._chat_chunk_json(
                                    request=request,
                                    delta={
                                        "tool_calls": [
                                            {
                                                "index": output_index,
                                                "id": call_id or f"call_{output_index}",
                                                "type": "function",
                                                "function": {
                                                    "name": tc.function.name,
                                                    "arguments": tc.function.arguments,
                                                },
                                            }
                                        ]
                                    },
                                )
                        continue

                yield self._chat_chunk_json(request=request, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            # Default chat/completions mode
            params = self._build_request_params(request)
            params["stream"] = True
            self.logger.debug(f"[LLMClient] Streaming request params: {params}")

            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                self.logger.debug(f"[LLMClient] Received openai chunk: {chunk}")
                chunk_dict = chunk.model_dump(mode="json", exclude_none=True)
                yield f"data: {json.dumps(chunk_dict)}\n\n"

            yield "data: [DONE]\n\n"

        except GeneratorExit:
            self.logger.debug(
                "[LLMClient] Generator closed - normal completion or iteration change"
            )
            return

        except APIConnectionError as e:
            error_msg = "Connection error occurred while streaming from LLM."
            self.logger.error(f"[LLMClient] {error_msg}: {str(e)}")
            raise

        except (RateLimitError, AuthenticationError, PermissionDeniedError) as e:
            # Fatal, non-retryable API errors must propagate so callers can fail
            # fast and surface the real error to the user instead of masking it
            # as a JSON-parse failure and spinning through useless retries.
            self.logger.error(
                "[LLMClient] Fatal API error during streaming (non-retryable): %s",
                str(e),
            )
            raise

        except Exception as e:
            error_msg = f"Error streaming from LLM: {str(e)}"
            self.logger.error(f"[LLMClient] {error_msg} - Stack trace:", exc_info=True)

            # Return error as streaming response
            error_chunk = ChatCompletionChunk(
                id=self.create_completion_id(),
                created=int(time.time()),
                model=request.model or TGI_MODEL_NAME,
                choices=[
                    Choice(
                        index=0,
                        delta=DeltaMessage(content=f"Error: {error_msg}"),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

    async def non_stream_completion(
        self,
        request: ChatCompletionRequest,
        access_token: str,
        parent_span,
    ) -> ChatCompletionResponse:
        """Get non-streaming completion from the actual LLM."""
        with tracer.start_as_current_span("non_stream_llm_completion") as span:
            span.set_attribute("llm.url", self.tgi_url)
            span.set_attribute("llm.model", request.model or TGI_MODEL_NAME)

            try:
                request.stream = False
                request = await self._prepare_payload(request, access_token)

                if self.conversation_mode == "responses":
                    params = self._build_responses_request_params(request)
                    try:
                        response = await self.client.responses.create(**params)
                        return self._normalize_responses_api_result(response, request)
                    except Exception as exc:
                        if not self._should_retry_responses_with_json_object(
                            request, exc
                        ):
                            raise
                        self._log_responses_schema_rejection(
                            request=request,
                            params=params,
                            exc=exc,
                        )
                        self.logger.warning(
                            "[LLMClient] Responses rejected json_schema response_format. Retrying once with text.format=json_object."
                        )
                        self._mark_schema_for_json_object_mode(request)
                        retry_params = self._build_responses_request_params(request)
                        response = await self.client.responses.create(**retry_params)
                        return self._normalize_responses_api_result(response, request)

                params = self._build_request_params(request)
                response = await self.client.chat.completions.create(**params)
                return ChatCompletionResponse(**response.model_dump(mode="json"))

            except Exception as e:
                self.logger.error(
                    f"[LLMClient] Error in non-stream completion: {e}", exc_info=True
                )
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

    async def summarize_conversation(
        self,
        conversation: List[Message],
        base_request: ChatCompletionRequest,
        access_token: str,
        outer_span,
    ):
        """
        Summarize the conversation so far using the LLM.

        Args:
            base_request: Base request to use for summarization
            access_token: Access token for authentication
            outer_span: Parent span for tracing

        Returns:
            Summarized text
        """

        def stringify_messages(messages: List[Message]) -> str:
            return "\n".join([f"{m.role}: {m.content}" for m in messages])

        return await self.ask(
            base_prompt="# System Prompt: Summarization Expert\n\nYou are a **summarization expert**. Your task is to read the **user's question** and the **assistant's reply** (which may include tool outputs), and then produce a concise, accurate summary of the reply that directly addresses the user's question.  \n\n---\n\n## Key Instructions\n\n### 1. Stay Aligned with the User's Question\n- Only summarize the information that is relevant to what the user asked.  \n- If the assistant's reply contains extraneous content (e.g., HTML markup in emails, raw metadata, or formatting noise), **ignore it** unless the user explicitly requested it.  \n\n### 2. Context-Sensitive Relevance\n- If the user asked about the *content* (e.g., “Summarize the email”), focus only on the meaningful text.  \n- If the user asked about *structure or metadata* (e.g., “Which senders use HTML emails?”), then the presence of HTML or metadata details is essential and should be included in the summary.  \n\n### 3. Clarity & Brevity\n- Rewrite in plain, natural language.  \n- Strip out technical noise, boilerplate, or irrelevant tool artifacts.  \n- Preserve essential details (facts, names, actions, outcomes).  \n\n### 4. Prioritization\n- Always privilege the **user's intent** over the assistant's full reply.  \n- Keep summaries **short but complete**: capture the key points, not every detail.  \n\n---\n\n## Examples\n\n- **User asks:** “Summarize this email.”  \n  - **Assistant reply (tool output):** Includes full HTML source.  \n  - **Your summary:** Only the human-readable body text of the email.  \n\n- **User asks:** “Which senders use HTML emails?”  \n  - **Assistant reply:** Includes headers and HTML details.\n  - **Your summary:** Mention the senders and the fact they use HTML formatting.\n",
            base_request=base_request,
            question=stringify_messages(conversation),
            access_token=access_token,
            outer_span=outer_span,
        )

    async def summarize_text(
        self,
        base_request: ChatCompletionRequest,
        content: str,
        access_token: str,
        outer_span,
    ) -> str:
        """
        Summarize text using the LLM.

        Args:
            base_request: Base request to use for summarization
            content: Content to summarize
            access_token: Access token for authentication
            outer_span: Parent span for tracing

        Returns:
            Summarized text
        """
        return await self.ask(
            base_prompt=(
                "# System Prompt: Content Summarization\n\n"
                "You are a **summarization expert**. Summarize the content provided by "
                "the user. Preserve key facts, identifiers, lists, tables, and any "
                "structured data. If the content includes JSON or other structured "
                "data, keep the structure and retain all items; do not invent new "
                "data or claim data is missing. Use concise, plain language when "
                "summarizing narrative text.\n"
            ),
            base_request=base_request,
            question=content,
            access_token=access_token,
            outer_span=outer_span,
        )

    async def ask(
        self,
        base_prompt: str,
        base_request: ChatCompletionRequest,
        outer_span,
        question: str = None,
        assistant_statement: str = None,
        access_token: str = None,
    ) -> str:
        """
        Ask a question, get a reply

        Args:
            base_prompt: A base system prompt to use for the question
            base_request: Base request to use for summarization
            question: The question to ask
            access_token: Access token for authentication
            outer_span: Parent span for tracing

        Returns:
            Summarized text
        """

        messages_history = [
            Message(
                role=MessageRole.SYSTEM,
                content=base_prompt,
            ),
        ]
        if question:
            messages_history.append(Message(role=MessageRole.USER, content=question))
        if assistant_statement:
            messages_history.append(
                Message(role=MessageRole.ASSISTANT, content=assistant_statement)
            )

        request = ChatCompletionRequest(
            model=base_request.model,
            messages=messages_history,
            stream=False,
        )

        response = await self.non_stream_completion(request, access_token, outer_span)

        # Handle response from ChatCompletionResponse or raw dict
        if isinstance(response, ChatCompletionResponse):
            choices = response.choices or []
            if choices:
                message = choices[0].message
                if message and message.content is not None:
                    return message.content
                delta = choices[0].delta
                if delta and delta.content is not None:
                    return delta.content
            return ""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        return ""
