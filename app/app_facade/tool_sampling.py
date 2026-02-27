"""Tool sampling and dummy-data generation pipeline.

The ``ToolSampler`` class encapsulates all methods for building sample
arguments, invoking tools (or dry-runs), analysing sampling errors via
LLM, deriving output schemas, and orchestrating the full dummy-data
generation pipeline.
"""

import asyncio
import contextlib
import copy
import json
import logging
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from app.session import MCPSessionBase
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.tgi.tool_dry_run.tool_response import get_tool_dry_run_response
from app.app_facade.generated_schemas import generation_response_format
from app.app_facade.generated_dummy_data import (
    DummyDataGenerator,
    SCHEMA_DERIVATION_RESPONSE_SCHEMA,
    SCHEMA_DERIVATION_SYSTEM_PROMPT,
)
from app.app_facade.prompt_helpers import (
    extract_json_block,
    parse_json,
    prompt_with_runtime_context,
    to_json_value,
)
from app.vars import EFFECT_TOOLS
from app.app_facade.generated_types import Scope

logger = logging.getLogger("uvicorn.error")

UI_MODEL_HEADERS = {"x-inxm-model-capability": "code-generation"}


def _generation_response_format(schema=None, name: str = "generated_ui"):
    return generation_response_format(schema=schema, name=name)


_DUMMY_DATA_SAMPLING_EXCLUDED_TOOLS = {"describe_tool", "select-from-tool-response"}
_DUMMY_DATA_ERROR_RECOVERY_SYSTEM_PROMPT = (
    "You are diagnosing a failed tool call used only for generating dummy test data. "
    "Classify whether the failure is recoverable by changing only input arguments. "
    "Always provide retry_arguments as a JSON object. "
    "If recoverable, provide retry_arguments that match the input schema exactly. "
    "If not recoverable, set retry_arguments to an empty object. "
    "If not recoverable (auth, permissions, server outage, missing integration, etc.), "
    "set recoverable to false."
)
_DUMMY_DATA_ERROR_RECOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "recoverable": {"type": "boolean"},
        "reason": {"type": "string"},
        "retry_arguments": {"type": "object"},
    },
    "required": ["recoverable", "reason", "retry_arguments"],
    "additionalProperties": False,
}


class ToolSampler:
    """Builds sample args, invokes tools, and generates dummy data."""

    def __init__(
        self,
        *,
        tgi_service: ProxiedTGIService,
        dummy_data_generator: DummyDataGenerator,
    ):
        self.tgi_service = tgi_service
        self.dummy_data_generator = dummy_data_generator

    def _build_sample_value_from_schema(
        self, field_name: str, schema: Any, depth: int = 0
    ) -> Any:
        if depth > 2:
            return "sample"
        if not isinstance(schema, dict):
            return "sample"

        if "const" in schema:
            return schema.get("const")

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return enum_values[0]

        if "default" in schema:
            return schema.get("default")

        examples = schema.get("examples")
        if isinstance(examples, list) and examples:
            return examples[0]

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            non_null = [item for item in schema_type if item != "null"]
            schema_type = (
                non_null[0] if non_null else (schema_type[0] if schema_type else None)
            )

        if schema_type == "string":
            fmt = str(schema.get("format") or "").lower()
            if fmt == "date":
                return datetime.now(timezone.utc).date().isoformat()
            if fmt == "date-time":
                return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            if fmt == "email":
                return "user@example.com"
            if fmt in {"uri", "url"}:
                return "https://example.com"
            if fmt == "uuid":
                return "00000000-0000-4000-8000-000000000000"
            return "sample"

        if schema_type == "integer":
            minimum = schema.get("minimum")
            return int(minimum) if isinstance(minimum, (int, float)) else 1

        if schema_type == "number":
            minimum = schema.get("minimum")
            return float(minimum) if isinstance(minimum, (int, float)) else 1.0

        if schema_type == "boolean":
            return False

        if schema_type == "array":
            item_schema = schema.get("items")
            return [
                self._build_sample_value_from_schema(field_name, item_schema, depth + 1)
            ]

        if schema_type == "object":
            properties = schema.get("properties")
            if isinstance(properties, dict):
                required = schema.get("required")
                if not isinstance(required, list):
                    required = list(properties.keys())[:1]
                value: Dict[str, Any] = {}
                for req_name in required:
                    if req_name in properties:
                        value[req_name] = self._build_sample_value_from_schema(
                            req_name, properties.get(req_name), depth + 1
                        )
                return value
            additional = schema.get("additionalProperties")
            if isinstance(additional, dict):
                return {
                    "value": self._build_sample_value_from_schema(
                        "value", additional, depth + 1
                    )
                }
            return {}

        if schema_type == "null":
            return None

        return "sample"

    def _build_sample_args_for_tool(self, input_schema: Any) -> Dict[str, Any]:
        if not isinstance(input_schema, dict):
            return {}
        properties = input_schema.get("properties")
        if not isinstance(properties, dict):
            return {}

        required = input_schema.get("required")
        required_fields: List[str] = []
        if isinstance(required, list):
            required_fields = [field for field in required if field in properties]

        if not required_fields:
            for candidate in ("city", "query", "id", "name"):
                if candidate in properties:
                    required_fields.append(candidate)
                    break

        args: Dict[str, Any] = {}
        for field_name in required_fields:
            args[field_name] = self._build_sample_value_from_schema(
                field_name, properties.get(field_name)
            )
        return args

    async def _derive_sample_args_with_llm(
        self,
        *,
        tool_name: str,
        tool_description: Optional[str],
        input_schema: Any,
        prompt: str,
    ) -> Optional[Dict[str, Any]]:
        llm_client = getattr(self.tgi_service, "llm_client", None)
        if not llm_client:
            return None
        non_stream = getattr(llm_client, "non_stream_completion", None)
        if not callable(non_stream):
            return None
        if not isinstance(input_schema, dict) or not input_schema:
            return None

        schema = copy.deepcopy(input_schema)
        if not isinstance(schema, dict):
            return None
        if schema.get("type") is None:
            schema["type"] = "object"

        request = ChatCompletionRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "Generate realistic sample JSON arguments for a tool call. "
                        "Return only a JSON object that matches the provided schema."
                    ),
                ),
                Message(
                    role=MessageRole.USER,
                    content=json.dumps(
                        {
                            "tool": {
                                "name": tool_name,
                                "description": tool_description,
                            },
                            "task_prompt": prompt,
                            "input_schema": schema,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
            tools=None,
            stream=False,
            response_format=_generation_response_format(
                schema=schema,
                name=f"{tool_name}_sample_input",
            ),
            extra_headers=UI_MODEL_HEADERS,
        )

        try:
            maybe_response = non_stream(request, "", None)
            if asyncio.iscoroutine(maybe_response):
                response = await maybe_response
            else:
                response = maybe_response
        except Exception as exc:
            logger.warning(
                "[GeneratedUI] Failed to derive sample args via LLM for tool '%s': %s",
                tool_name,
                exc,
            )
            return None

        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return None
            choice = choices[0]
            content = getattr(getattr(choice, "message", None), "content", None) or ""
            if not content and getattr(
                getattr(choice, "message", None), "tool_calls", None
            ):
                content = choice.message.tool_calls[0].function.arguments
            if not content:
                return None
            parsed = parse_json(content)
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            logger.warning(
                "[GeneratedUI] Failed to parse LLM sample args for tool '%s': %s",
                tool_name,
                exc,
            )
            return None

    async def _analyze_sampling_error_with_llm(
        self,
        *,
        tool_name: str,
        tool_description: Optional[str],
        input_schema: Any,
        prompt: str,
        attempted_args: Dict[str, Any],
        error_detail: Any,
    ) -> Dict[str, Any]:
        llm_client = getattr(self.tgi_service, "llm_client", None)
        if not llm_client:
            return {"recoverable": False, "retry_arguments": None, "reason": ""}
        non_stream = getattr(llm_client, "non_stream_completion", None)
        if not callable(non_stream):
            return {"recoverable": False, "retry_arguments": None, "reason": ""}

        retry_arguments_schema: Dict[str, Any]
        if isinstance(input_schema, dict) and input_schema:
            retry_arguments_schema = copy.deepcopy(input_schema)
        else:
            retry_arguments_schema = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        recovery_schema = copy.deepcopy(_DUMMY_DATA_ERROR_RECOVERY_SCHEMA)
        recovery_schema["properties"]["retry_arguments"] = retry_arguments_schema

        request = ChatCompletionRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=_DUMMY_DATA_ERROR_RECOVERY_SYSTEM_PROMPT,
                ),
                Message(
                    role=MessageRole.USER,
                    content=json.dumps(
                        {
                            "tool": {
                                "name": tool_name,
                                "description": tool_description,
                            },
                            "task_prompt": prompt,
                            "input_schema": input_schema,
                            "attempted_args": attempted_args,
                            "error": error_detail,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
            tools=None,
            stream=False,
            response_format=_generation_response_format(
                schema=recovery_schema,
                name=f"{tool_name}_sample_error_recovery",
            ),
            extra_headers=UI_MODEL_HEADERS,
        )

        try:
            maybe_response = non_stream(request, "", None)
            if asyncio.iscoroutine(maybe_response):
                response = await maybe_response
            else:
                response = maybe_response
        except Exception as exc:
            logger.warning(
                "[GeneratedUI] Failed to analyze sampling error via LLM for tool '%s': %s",
                tool_name,
                exc,
            )
            return {"recoverable": False, "retry_arguments": None, "reason": ""}

        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return {"recoverable": False, "retry_arguments": None, "reason": ""}
            choice = choices[0]
            content = getattr(getattr(choice, "message", None), "content", None) or ""
            if not content and getattr(
                getattr(choice, "message", None), "tool_calls", None
            ):
                content = choice.message.tool_calls[0].function.arguments
            if not content:
                return {"recoverable": False, "retry_arguments": None, "reason": ""}
            parsed = parse_json(content)
            if not isinstance(parsed, dict):
                return {"recoverable": False, "retry_arguments": None, "reason": ""}
            retry_arguments = parsed.get("retry_arguments")
            if not isinstance(retry_arguments, dict):
                retry_arguments = None
            reason = parsed.get("reason")
            if not isinstance(reason, str):
                reason = ""
            recoverable = (
                bool(parsed.get("recoverable")) and retry_arguments is not None
            )
            return {
                "recoverable": recoverable,
                "retry_arguments": retry_arguments,
                "reason": reason,
            }
        except Exception as exc:
            logger.warning(
                "[GeneratedUI] Failed to parse LLM sampling-error analysis for tool '%s': %s",
                tool_name,
                exc,
            )
            return {"recoverable": False, "retry_arguments": None, "reason": ""}

    async def _call_tool_for_dummy_sampling(
        self,
        *,
        session: MCPSessionBase,
        tool_name: str,
        tool_def: Dict[str, Any],
        sample_args: Dict[str, Any],
        access_token: Optional[str],
    ) -> Any:
        if tool_name in EFFECT_TOOLS:
            maybe_result = get_tool_dry_run_response(session, tool_def, sample_args)
            return (
                await maybe_result
                if asyncio.iscoroutine(maybe_result)
                else maybe_result
            )
        # Gateway-routed tool: invoke via call_tool with server_id
        gateway = tool_def.get("_gateway") if isinstance(tool_def, dict) else None
        if gateway and isinstance(gateway, dict):
            server_id = gateway.get("server_id")
            via_tool = gateway.get("via_tool", "call_tool")
            if server_id:
                logger.info(
                    "[_call_tool_for_dummy_sampling] Routing '%s' via gateway "
                    "call_tool(server_id=%s)",
                    tool_name,
                    server_id,
                )
                return await session.call_tool(
                    via_tool,
                    {
                        "server_id": server_id,
                        "tool_name": tool_name,
                        "input_data": sample_args,
                    },
                    access_token,
                )
        return await session.call_tool(tool_name, sample_args, access_token)

    def _payload_has_error_markers(self, payload: Any) -> bool:
        if isinstance(payload, dict):
            if payload.get("isError") is True:
                return True
            if payload.get("success") is False:
                return True
            if payload.get("error") or payload.get("errors"):
                return True
            return any(
                self._payload_has_error_markers(value) for value in payload.values()
            )
        if isinstance(payload, list):
            return any(self._payload_has_error_markers(item) for item in payload)
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                return False
            if stripped[0] in ("{", "["):
                with contextlib.suppress(Exception):
                    parsed = json.loads(stripped)
                    return self._payload_has_error_markers(parsed)
            lowered = stripped.lower()
            return "error" in lowered or "failed" in lowered or "not found" in lowered
        return False

    def _extract_error_from_tool_result_content(
        self, content_items: Any
    ) -> Optional[Any]:
        if not isinstance(content_items, list):
            return None
        for item in content_items:
            text: Optional[str] = None
            if isinstance(item, dict):
                text = item.get("text")
            elif hasattr(item, "text"):
                text = getattr(item, "text", None)
            if not isinstance(text, str) or not text.strip():
                continue
            stripped = text.strip()
            parsed: Any = None
            try:
                parsed = json.loads(stripped)
            except Exception:
                candidate = extract_json_block(stripped)
                if candidate:
                    with contextlib.suppress(Exception):
                        parsed = json.loads(candidate)
            if parsed is not None:
                if self._payload_has_error_markers(parsed):
                    return parsed
                continue
            lowered = stripped.lower()
            if "error" in lowered or "failed" in lowered or "not found" in lowered:
                return {"error": stripped}
        return None

    def _result_has_sampling_error(self, result: Any) -> bool:
        is_error = getattr(result, "isError", None)
        if is_error is None and isinstance(result, dict):
            is_error = result.get("isError")
        if is_error is True:
            return True

        success = getattr(result, "success", None)
        if success is None and isinstance(result, dict):
            success = result.get("success")
        if success is False:
            return True

        explicit_error = (
            result.get("error")
            if isinstance(result, dict)
            else getattr(result, "error", None)
        )
        if explicit_error:
            return True

        structured = (
            result.get("structuredContent")
            if isinstance(result, dict)
            else getattr(result, "structuredContent", None)
        )
        if self._payload_has_error_markers(structured):
            return True

        content_items = (
            result.get("content")
            if isinstance(result, dict)
            else getattr(result, "content", None)
        )
        return self._extract_error_from_tool_result_content(content_items) is not None

    def _extract_sampling_error_detail(self, result: Any) -> Any:
        explicit_error = (
            result.get("error")
            if isinstance(result, dict)
            else getattr(result, "error", None)
        )
        if explicit_error:
            return to_json_value(explicit_error)

        structured = (
            result.get("structuredContent")
            if isinstance(result, dict)
            else getattr(result, "structuredContent", None)
        )
        if self._payload_has_error_markers(structured):
            return to_json_value(structured)

        content_items = (
            result.get("content")
            if isinstance(result, dict)
            else getattr(result, "content", None)
        )
        content_error = self._extract_error_from_tool_result_content(content_items)
        if content_error is not None:
            return to_json_value(content_error)

        return to_json_value(result)

    def _build_error_dummy_sample(
        self,
        *,
        tool_name: str,
        attempted_args: Dict[str, Any],
        error_detail: Any,
        analysis_reason: str,
    ) -> Dict[str, Any]:
        normalized_error = to_json_value(error_detail)
        if isinstance(normalized_error, dict):
            payload = copy.deepcopy(normalized_error)
        else:
            payload = {"error": normalized_error}
        payload["_dummy_data_error"] = True
        payload["tool"] = tool_name
        payload["attempted_args"] = to_json_value(attempted_args)
        if analysis_reason:
            payload["analysis"] = analysis_reason
        return payload

    def _extract_sample_from_success_tool_result(self, result: Any) -> Optional[Any]:
        structured = (
            result.get("structuredContent")
            if isinstance(result, dict)
            else getattr(result, "structuredContent", None)
        )
        resolved = self._resolve_sample_value(structured)
        if resolved is not None:
            return resolved
        content_items = (
            result.get("content")
            if isinstance(result, dict)
            else getattr(result, "content", None)
        )
        return self._extract_sample_from_tool_result_content(content_items)

    def _extract_sample_from_tool_result_content(
        self, content_items: Any
    ) -> Optional[Any]:
        if not isinstance(content_items, list):
            return None
        for item in content_items:
            text: Optional[str] = None
            if isinstance(item, dict):
                text = item.get("text")
            elif hasattr(item, "text"):
                text = getattr(item, "text", None)
            if not isinstance(text, str) or not text.strip():
                continue
            stripped = text.strip()
            parsed: Any = None
            try:
                parsed = json.loads(stripped)
            except Exception:
                candidate = extract_json_block(stripped)
                if candidate:
                    with contextlib.suppress(Exception):
                        parsed = json.loads(candidate)
            if parsed is None:
                continue
            resolved = self._resolve_sample_value(parsed)
            if resolved is not None:
                return resolved
        return None

    def _resolve_sample_value(self, payload: Any) -> Optional[Any]:
        if isinstance(payload, dict):
            structured = payload.get("structuredContent")
            if structured is not None:
                return structured
            nested_content = payload.get("content")
            nested = self._extract_sample_from_tool_result_content(nested_content)
            if nested is not None:
                return nested
            envelope_keys = {"isError", "content", "structuredContent", "error"}
            if not set(payload.keys()).issubset(envelope_keys):
                return payload
            return None
        if isinstance(payload, (list, str, int, float, bool)) or payload is None:
            return payload
        return None

    async def _sample_tool_output_for_dummy_data(
        self,
        *,
        session: MCPSessionBase,
        tool_name: str,
        tool_description: Optional[str],
        input_schema: Any,
        output_schema: Any,
        prompt: str,
        access_token: Optional[str],
        gateway_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        sample_args = await self._derive_sample_args_with_llm(
            tool_name=tool_name,
            tool_description=tool_description,
            input_schema=input_schema,
            prompt=prompt,
        )
        if sample_args is None:
            sample_args = self._build_sample_args_for_tool(input_schema)
        tool_def: Dict[str, Any] = {
            "name": tool_name,
            "description": tool_description,
            "inputSchema": input_schema or {"type": "object"},
            "outputSchema": output_schema,
        }
        if gateway_info:
            tool_def["_gateway"] = gateway_info
        result: Any = None
        error_detail: Any = None
        try:
            result = await self._call_tool_for_dummy_sampling(
                session=session,
                tool_name=tool_name,
                tool_def=tool_def,
                sample_args=sample_args,
                access_token=access_token,
            )
        except Exception as exc:
            logger.warning(
                "[GeneratedUI] Failed to sample tool '%s' for dummy data: %s",
                tool_name,
                exc,
            )
            error_detail = {
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }
        else:
            if not self._result_has_sampling_error(result):
                return self._extract_sample_from_success_tool_result(result)
            error_detail = self._extract_sampling_error_detail(result)

        analysis = await self._analyze_sampling_error_with_llm(
            tool_name=tool_name,
            tool_description=tool_description,
            input_schema=input_schema,
            prompt=prompt,
            attempted_args=sample_args,
            error_detail=error_detail,
        )
        retry_args = analysis.get("retry_arguments")
        if analysis.get("recoverable") and isinstance(retry_args, dict):
            logger.info(
                "[GeneratedUI] Retrying sampled call for tool '%s' with LLM-corrected args",
                tool_name,
            )
            try:
                retry_result = await self._call_tool_for_dummy_sampling(
                    session=session,
                    tool_name=tool_name,
                    tool_def=tool_def,
                    sample_args=retry_args,
                    access_token=access_token,
                )
            except Exception as retry_exc:
                logger.warning(
                    "[GeneratedUI] Retry failed while sampling tool '%s' for dummy data: %s",
                    tool_name,
                    retry_exc,
                )
                error_detail = {
                    "initial_error": to_json_value(error_detail),
                    "retry_error": {
                        "error": str(retry_exc),
                        "exception_type": type(retry_exc).__name__,
                    },
                }
            else:
                if not self._result_has_sampling_error(retry_result):
                    retry_sample = self._extract_sample_from_success_tool_result(
                        retry_result
                    )
                    if retry_sample is not None:
                        return retry_sample
                error_detail = {
                    "initial_error": to_json_value(error_detail),
                    "retry_error": self._extract_sampling_error_detail(retry_result),
                }

        return self._build_error_dummy_sample(
            tool_name=tool_name,
            attempted_args=sample_args,
            error_detail=error_detail,
            analysis_reason=str(analysis.get("reason") or ""),
        )

    def _extract_dummy_data_payload_from_module(
        self, module_source: Optional[str]
    ) -> Dict[str, Any]:
        text = (module_source or "").strip()
        if not text:
            return {}

        marker = "export const dummyData ="
        start = text.find(marker)
        if start < 0:
            return {}

        remainder = text[start + len(marker) :].lstrip()
        if not remainder.startswith("{"):
            return {}

        end_marker = "export const dummyDataSchemaHints"
        end = remainder.find(end_marker)
        candidate = remainder[:end].strip() if end >= 0 else remainder
        candidate = candidate.rstrip()
        if candidate.endswith(";"):
            candidate = candidate[:-1].rstrip()

        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    async def _derive_output_schema_from_sample_with_llm(
        self,
        *,
        tool_name: str,
        tool_description: Optional[str],
        input_schema: Any,
        sample_payload: Any,
    ) -> Optional[Dict[str, Any]]:
        llm_client = getattr(self.tgi_service, "llm_client", None)
        if not llm_client:
            return None
        non_stream = getattr(llm_client, "non_stream_completion", None)
        if not callable(non_stream):
            return None

        request = ChatCompletionRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=SCHEMA_DERIVATION_SYSTEM_PROMPT,
                ),
                Message(
                    role=MessageRole.USER,
                    content=(
                        "Derive output schema for the resolved service value from this observed tool data:\n"
                        + json.dumps(
                            {
                                "name": tool_name,
                                "description": tool_description,
                                "inputSchema": input_schema,
                                "sampleResolvedValue": sample_payload,
                            },
                            ensure_ascii=False,
                        )
                        + "\nReturn only JSON."
                    ),
                ),
            ],
            tools=None,
            stream=False,
            response_format=_generation_response_format(
                schema=SCHEMA_DERIVATION_RESPONSE_SCHEMA,
                name=f"{tool_name}_derived_output_schema",
            ),
            extra_headers=UI_MODEL_HEADERS,
        )

        try:
            response = await non_stream(request, None, None)
        except Exception as exc:
            logger.warning(
                "[GeneratedUI] Failed deriving output schema for '%s' from dummy data: %s",
                tool_name,
                exc,
            )
            return None

        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return None
            choice = choices[0]
            content = getattr(getattr(choice, "message", None), "content", None) or ""
            if not content and getattr(
                getattr(choice, "message", None), "tool_calls", None
            ):
                content = choice.message.tool_calls[0].function.arguments
            parsed = json.loads(content) if content else {}
            schema = parsed.get("schema") if isinstance(parsed, dict) else None
            if isinstance(schema, dict) and schema:
                sanitizer = getattr(
                    self.dummy_data_generator, "_sanitize_output_schema", None
                )
                if callable(sanitizer):
                    sanitized = sanitizer(schema)
                    if isinstance(sanitized, dict) and sanitized:
                        return sanitized
                return schema
        except Exception as exc:
            logger.warning(
                "[GeneratedUI] Failed parsing derived output schema for '%s': %s",
                tool_name,
                exc,
            )
        return None

    async def _augment_tools_with_derived_output_schemas(
        self,
        *,
        allowed_tools: Optional[List[Dict[str, Any]]],
        dummy_data_module: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], int]:
        tools = copy.deepcopy(list(allowed_tools or []))
        if not tools:
            return [], 0

        dummy_payload = self._extract_dummy_data_payload_from_module(dummy_data_module)
        if not dummy_payload:
            return tools, 0

        derived_count = 0
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = (
                tool.get("function") if isinstance(tool.get("function"), dict) else {}
            )
            tool_name = function.get("name") or tool.get("name")
            if not tool_name:
                continue
            if function.get("outputSchema") or tool.get("outputSchema"):
                continue

            sample_payload = dummy_payload.get(tool_name)
            if sample_payload is None:
                continue
            if isinstance(sample_payload, dict) and sample_payload.get(
                "_dummy_data_error"
            ):
                continue
            if self._payload_has_error_markers(sample_payload):
                continue

            derived_schema = await self._derive_output_schema_from_sample_with_llm(
                tool_name=str(tool_name),
                tool_description=function.get("description") or tool.get("description"),
                input_schema=function.get("parameters")
                or tool.get("inputSchema")
                or {},
                sample_payload=sample_payload,
            )
            if not derived_schema:
                continue

            tool["outputSchema"] = derived_schema
            if isinstance(function, dict):
                function["outputSchema"] = derived_schema
            derived_count += 1

        return tools, derived_count

    async def _generate_dummy_data(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        ui_id: str,
        name: str,
        prompt: str,
        allowed_tools: Optional[List[Dict[str, Any]]],
        access_token: Optional[str],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not allowed_tools:
            return (
                "export const dummyData = {};\nexport const dummyDataSchemaHints = {};"
            )

        async def process_tool(tool: Any) -> Optional[Dict[str, Any]]:
            tool_name: Optional[str] = None
            tool_description: Optional[str] = None
            input_schema: Any = {}
            output_schema: Any = None
            if isinstance(tool, dict):
                function = tool.get("function", {})
                tool_name = function.get("name")
                tool_description = function.get("description") or tool.get(
                    "description"
                )
                input_schema = (
                    function.get("parameters") or tool.get("inputSchema") or {}
                )
                output_schema = function.get("outputSchema") or tool.get("outputSchema")
            else:
                function = getattr(tool, "function", None)
                output_schema = getattr(tool, "outputSchema", None)
                input_schema = getattr(tool, "inputSchema", None) or {}
                if function and hasattr(function, "name"):
                    tool_name = function.name
                if function and hasattr(function, "description"):
                    tool_description = function.description
                if function and hasattr(function, "parameters"):
                    input_schema = getattr(function, "parameters") or input_schema
                if function and hasattr(function, "outputSchema"):
                    output_schema = getattr(function, "outputSchema")
            if not tool_name:
                return None
            if tool_name in _DUMMY_DATA_SAMPLING_EXCLUDED_TOOLS:
                return None

            # Propagate gateway metadata for indirect tool invocation
            gateway_info: Optional[Dict[str, Any]] = None
            if isinstance(tool, dict) and tool.get("_gateway"):
                gateway_info = tool["_gateway"]

            tool_spec: Dict[str, Any] = {
                "name": tool_name,
                "description": tool_description,
                "inputSchema": input_schema,
                "outputSchema": output_schema,
            }
            if gateway_info and isinstance(gateway_info, dict):
                server_id = str(gateway_info.get("server_id") or "").strip()
                via_tool = str(gateway_info.get("via_tool") or "call_tool").strip()
                if server_id:
                    tool_spec["gatewayHint"] = {
                        "mcp_server_id": f"/api/{server_id}/tools/{tool_name}",
                        "server_id": server_id,
                        "tool_name": tool_name,
                        "via_tool": via_tool or "call_tool",
                    }
            if not output_schema:
                sampled = await self._sample_tool_output_for_dummy_data(
                    session=session,
                    tool_name=tool_name,
                    tool_description=tool_description,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    prompt=prompt,
                    access_token=access_token,
                    gateway_info=gateway_info,
                )
                if sampled is not None:
                    if isinstance(sampled, dict) and sampled.get("_dummy_data_error"):
                        logger.info(
                            "[GeneratedUI] Ignoring error-shaped sampled output for tool '%s' while building dummy data",
                            tool_name,
                        )
                    else:
                        tool_spec["sampleStructuredContent"] = sampled
            return tool_spec

        tool_specs_results = await asyncio.gather(
            *(process_tool(tool) for tool in allowed_tools or [])
        )
        tool_specs = [spec for spec in tool_specs_results if spec is not None]

        enriched_prompt = prompt_with_runtime_context(
            prompt=prompt,
            runtime_context=runtime_context,
            purpose="dummy_data",
        )

        return await self.dummy_data_generator.generate_dummy_data(
            prompt=enriched_prompt,
            tool_specs=tool_specs,
            ui_model_headers=UI_MODEL_HEADERS,
        )
