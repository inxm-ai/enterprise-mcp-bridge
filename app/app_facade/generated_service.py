import logging
import os
import re
import asyncio
import json
import contextlib
import subprocess
import tempfile
import shutil
import copy
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from typing import AsyncIterator

from fastapi import HTTPException

from app.session import MCPSessionBase
from app.vars import (
    MCP_BASE_PATH,
    GENERATED_UI_PROMPT_DUMP,
    APP_UI_SESSION_TTL_MINUTES,
    GENERATED_UI_FIX_CODE_FIRST,
    EFFECT_TOOLS,
)
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.protocols.chunk_reader import chunk_reader
from app.tgi.services.proxied_tgi_service import ProxiedTGIService
from app.app_facade.generated_output_factory import (
    GeneratedUIOutputFactory,
    MCP_SERVICE_TEST_HELPER_SCRIPT,
)
from app.app_facade.generated_phase1 import run_phase1_attempt
from app.app_facade.generated_phase2 import run_phase2_attempt
from app.app_facade.generated_dummy_data import DummyDataGenerator
from app.app_facade.generated_schemas import (
    generation_response_format,
)
from app.app_facade.generated_storage import GeneratedUIStorage
from app.app_facade.generated_types import (
    Actor,
    Scope,  # re-exported for route/tests import compatibility
    validate_identifier,  # noqa: F401  # re-exported for route/tests import compatibility
)
from app.app_facade.test_fix_tools import _parse_tap_output, run_tool_driven_test_fix
from app.tgi.tool_dry_run.tool_response import get_tool_dry_run_response


logger = logging.getLogger("uvicorn.error")

DEFAULT_DESIGN_PROMPT = (
    "Use lightweight, responsive layouts. Prefer utility-first styling via Tailwind "
    "CSS conventions when no explicit design system guidance is provided."
)

UI_MODEL_HEADERS = {"x-inxm-model-capability": "code-generation"}

SCRIPT_KEYS = ("service_script", "components_script", "test_script", "dummy_data")
MAX_TEST_EVENTS = 200
TEST_ACTIONS = ("run", "fix_code", "adjust_test", "delete_test", "add_test")
TestAction = Literal["run", "fix_code", "adjust_test", "delete_test", "add_test"]
MAX_RUNTIME_CONTEXT_ENTRIES = 20
MAX_RUNTIME_CONTEXT_DEPTH = 3
MAX_RUNTIME_CONTEXT_TEXT = 2000
MAX_RUNTIME_CONSOLE_EVENTS = 20
_DUMMY_DATA_SAMPLING_EXCLUDED_TOOLS = {"describe_tool", "select-from-tool-response"}
_DUMMY_DATA_TEST_USAGE_GUIDANCE = (
    "Dummy data module for tests is available as ./dummy_data.js. "
    "Tests MUST import { dummyData, dummyDataSchemaHints } from './dummy_data.js' and use "
    "svc.test.addResolved(toolName, dummyData[toolName]) for final resolved results, "
    "or globalThis.fetch.addRoute(...) when validating raw transport/extraction paths. "
    "If dummyDataSchemaHints[toolName] exists, that tool is missing output schema; "
    "the client should ask for schema and regenerate dummy data before relying on that fixture. "
    "Never import './dummy_data.js' in service_script/components_script; "
    "it is test-only and not browser-delivered at runtime. "
    "Do NOT inject fetched domain data directly via component initial state or "
    "test-only event payloads; components must fetch/refetch themselves. "
    "When asserting concrete field values in tests, derive expectations from a normalized shape (e.g. "
    "const normalized = data.current_air_quality || data; const pm25 = normalized.pm2_5) instead of assuming flat paths. "
    "Do NOT hardcode dynamic time/value literals when fixture payload already provides source-of-truth fields; "
    "assert against transformed fixture values."
)
_DUMMY_DATA_ERROR_RECOVERY_SYSTEM_PROMPT = (
    "You are diagnosing a failed tool call used only for generating dummy test data. "
    "Classify whether the failure is recoverable by changing only input arguments. "
    "If recoverable, provide retry_arguments that match the input schema exactly. "
    "If not recoverable (auth, permissions, server outage, missing integration, etc.), "
    "set recoverable to false."
)
_DUMMY_DATA_ERROR_RECOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "recoverable": {"type": "boolean"},
        "reason": {"type": "string"},
        "retry_arguments": {
            "anyOf": [
                {"type": "object", "additionalProperties": True},
                {"type": "null"},
            ]
        },
    },
    "required": ["recoverable"],
    "additionalProperties": False,
}


def _positive_int_env(name: str, default: int) -> int:
    try:
        parsed = int(os.environ.get(name, str(default)))
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


NODE_TEST_TIMEOUT_MS = _positive_int_env("GENERATED_UI_NODE_TEST_TIMEOUT_MS", 8000)


def _sse_event(event: str, payload: Dict[str, Any]) -> bytes:
    return (
        f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode(
            "utf-8"
        )
    )


def _load_pfusch_prompt() -> str:
    """Load the pfusch ui prompt from the markdown file and replace placeholders."""
    prompt_path = os.path.join(os.path.dirname(__file__), "pfusch_ui_prompt.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        # Replace the MCP_BASE_PATH placeholder
        return prompt_content.replace("{{MCP_BASE_PATH}}", MCP_BASE_PATH)
    except Exception as e:
        logger.error(f"Error loading pfusch prompt: {e}")
        raise e


def _load_pfusch_phase2_prompt() -> str:
    """Load the pfusch phase 2 presentation prompt."""
    prompt_path = os.path.join(os.path.dirname(__file__), "pfusch_ui_phase2_prompt.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        return prompt_content
    except Exception as e:
        logger.error(f"Error loading pfusch phase 2 prompt: {e}")
        raise e


def _generation_response_format(
    schema: Optional[Dict[str, Any]] = None, name: str = "generated_ui"
) -> Dict[str, Any]:
    return generation_response_format(schema=schema, name=name)


class GeneratedUIService:
    def __init__(
        self,
        *,
        storage: GeneratedUIStorage,
        tgi_service: Optional[ProxiedTGIService] = None,
    ):
        self.storage = storage
        self.tgi_service = tgi_service or ProxiedTGIService()
        self.output_factory = GeneratedUIOutputFactory()
        self.dummy_data_generator = DummyDataGenerator(self.tgi_service)
        self._test_event_subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._test_run_tasks: Dict[str, asyncio.Task] = {}
        self._test_run_locks: Dict[str, asyncio.Lock] = {}
        self._test_run_seq = 0

    def _current_version(self, metadata: Dict[str, Any]) -> int:
        raw = metadata.get("version")
        try:
            parsed = int(raw)
            return parsed if parsed > 0 else 1
        except (TypeError, ValueError):
            return 1

    def _ensure_version_metadata(self, metadata: Dict[str, Any]) -> None:
        version = self._current_version(metadata)
        metadata["version"] = version
        if not metadata.get("published_at"):
            metadata["published_at"] = (
                metadata.get("updated_at") or metadata.get("created_at") or self._now()
            )
        if not metadata.get("published_by"):
            metadata["published_by"] = metadata.get("updated_by") or metadata.get(
                "created_by"
            )

    def _run_tests(
        self,
        service_code: str,
        components_code: str,
        test_code: str,
        dummy_data: Optional[str] = None,
        test_name: Optional[str] = None,
    ) -> Tuple[bool, str]:
        helpers_dir = os.path.join(os.path.dirname(__file__), "node_test_helpers")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy helpers including pfusch.js
            for filename in ["domstubs.js", "pfusch.js"]:
                src = os.path.join(helpers_dir, filename)
                dst = os.path.join(tmpdir, filename)
                if os.path.exists(src):
                    shutil.copy(src, dst)

            with open(os.path.join(tmpdir, "package.json"), "w", encoding="utf-8") as f:
                f.write('{"type":"module"}\n')

            mocked_components_code = (
                (components_code or "")
                .replace(
                    "https://matthiaskainer.github.io/pfusch/pfusch.min.js",
                    "./pfusch.js",
                )
                .replace(
                    "https://matthiaskainer.github.io/pfusch/pfusch.js", "./pfusch.js"
                )
            )
            resolved_service_code = service_code or ""
            # Always inject the global McpService/service safety prelude in tests.
            # It writes only to globalThis and avoids top-level symbol collisions.
            service_prelude = MCP_SERVICE_TEST_HELPER_SCRIPT
            combined_code = f"{service_prelude}\n\n{resolved_service_code}\n\n{mocked_components_code}"

            with open(os.path.join(tmpdir, "app.js"), "w", encoding="utf-8") as f:
                f.write(combined_code)

            with open(os.path.join(tmpdir, "user_test.js"), "w", encoding="utf-8") as f:
                f.write(test_code or "")

            if dummy_data is not None:
                with open(
                    os.path.join(tmpdir, "dummy_data.js"), "w", encoding="utf-8"
                ) as f:
                    f.write(dummy_data or "")

            test_wrapper = (
                "import { setupDomStubs, pfuschTest } from './domstubs.js';\n"
                "if (typeof globalThis.HTMLElement === 'undefined') {\n"
                "  setupDomStubs();\n"
                "}\n"
                "await import('./user_test.js');\n"
            )
            with open(os.path.join(tmpdir, "test.js"), "w", encoding="utf-8") as f:
                f.write(test_wrapper)

            # Run node --test
            try:
                # set path to include current directory so imports work if needed
                env = os.environ.copy()
                env["NODE_PATH"] = tmpdir
                cmd = [
                    "node",
                    "--test",
                    "--test-force-exit",
                    "--test-timeout",
                    str(NODE_TEST_TIMEOUT_MS),
                    "test.js",
                ]
                if test_name:
                    cmd.extend(["--test-name-pattern", test_name])
                result = subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30,  # Safety timeout
                    env=env,
                )
                output = result.stdout + "\n" + result.stderr
                if result.returncode != 0:
                    passed, failed, failed_tests = _parse_tap_output(output)
                    output_excerpt = self._trim_output(output)
                    failed_preview = ", ".join(failed_tests[:5]) if failed_tests else ""
                    if len(failed_tests) > 5:
                        failed_preview += f", ... ({len(failed_tests) - 5} more)"
                    logger.error(
                        "[GeneratedUI] Test run failed (passed=%s, failed=%s). "
                        "Failing tests: %s\nOutput excerpt:\n%s",
                        passed,
                        failed,
                        failed_preview or "unknown",
                        output_excerpt,
                    )
                return result.returncode == 0, output
            except subprocess.TimeoutExpired:
                return False, "Tests timed out after 30 seconds."
            except Exception as e:
                return False, f"Error running tests: {str(e)}"

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
        has_non_stream = callable(getattr(llm_client, "non_stream_completion", None))
        if not has_non_stream and not getattr(llm_client, "client", None):
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
            non_stream = getattr(llm_client, "non_stream_completion", None)
            if callable(non_stream):
                maybe_response = non_stream(request, "", None)
                if asyncio.iscoroutine(maybe_response):
                    response = await maybe_response
                else:
                    response = maybe_response
            else:
                response = await llm_client.client.chat.completions.create(
                    **llm_client._build_request_params(request)
                )
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
            parsed = self._parse_json(content)
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
        has_non_stream = callable(getattr(llm_client, "non_stream_completion", None))
        if not has_non_stream and not getattr(llm_client, "client", None):
            return {"recoverable": False, "retry_arguments": None, "reason": ""}

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
                schema=_DUMMY_DATA_ERROR_RECOVERY_SCHEMA,
                name=f"{tool_name}_sample_error_recovery",
            ),
            extra_headers=UI_MODEL_HEADERS,
        )

        try:
            non_stream = getattr(llm_client, "non_stream_completion", None)
            if callable(non_stream):
                maybe_response = non_stream(request, "", None)
                if asyncio.iscoroutine(maybe_response):
                    response = await maybe_response
                else:
                    response = maybe_response
            else:
                response = await llm_client.client.chat.completions.create(
                    **llm_client._build_request_params(request)
                )
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
            parsed = self._parse_json(content)
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

    def _to_json_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_json_value(v) for v in value]
        if hasattr(value, "model_dump"):
            with contextlib.suppress(Exception):
                return self._to_json_value(value.model_dump())
        if hasattr(value, "__dict__"):
            with contextlib.suppress(Exception):
                return self._to_json_value(vars(value))
        return str(value)

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
                candidate = self._extract_json_block(stripped)
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
            return self._to_json_value(explicit_error)

        structured = (
            result.get("structuredContent")
            if isinstance(result, dict)
            else getattr(result, "structuredContent", None)
        )
        if self._payload_has_error_markers(structured):
            return self._to_json_value(structured)

        content_items = (
            result.get("content")
            if isinstance(result, dict)
            else getattr(result, "content", None)
        )
        content_error = self._extract_error_from_tool_result_content(content_items)
        if content_error is not None:
            return self._to_json_value(content_error)

        return self._to_json_value(result)

    def _build_error_dummy_sample(
        self,
        *,
        tool_name: str,
        attempted_args: Dict[str, Any],
        error_detail: Any,
        analysis_reason: str,
    ) -> Dict[str, Any]:
        normalized_error = self._to_json_value(error_detail)
        if isinstance(normalized_error, dict):
            payload = copy.deepcopy(normalized_error)
        else:
            payload = {"error": normalized_error}
        payload["_dummy_data_error"] = True
        payload["tool"] = tool_name
        payload["attempted_args"] = self._to_json_value(attempted_args)
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
                candidate = self._extract_json_block(stripped)
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
    ) -> Optional[Any]:
        sample_args = await self._derive_sample_args_with_llm(
            tool_name=tool_name,
            tool_description=tool_description,
            input_schema=input_schema,
            prompt=prompt,
        )
        if sample_args is None:
            sample_args = self._build_sample_args_for_tool(input_schema)
        tool_def = {
            "name": tool_name,
            "description": tool_description,
            "inputSchema": input_schema or {"type": "object"},
            "outputSchema": output_schema,
        }
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
                    "initial_error": self._to_json_value(error_detail),
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
                    "initial_error": self._to_json_value(error_detail),
                    "retry_error": self._extract_sampling_error_detail(retry_result),
                }

        return self._build_error_dummy_sample(
            tool_name=tool_name,
            attempted_args=sample_args,
            error_detail=error_detail,
            analysis_reason=str(analysis.get("reason") or ""),
        )

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

            tool_spec: Dict[str, Any] = {
                "name": tool_name,
                "description": tool_description,
                "inputSchema": input_schema,
                "outputSchema": output_schema,
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
                )
                if sampled is not None:
                    tool_spec["sampleStructuredContent"] = sampled
            return tool_spec

        tool_specs_results = await asyncio.gather(
            *(process_tool(tool) for tool in allowed_tools or [])
        )
        tool_specs = [spec for spec in tool_specs_results if spec is not None]

        enriched_prompt = self._prompt_with_runtime_context(
            prompt=prompt,
            runtime_context=runtime_context,
            purpose="dummy_data",
        )

        return await self.dummy_data_generator.generate_dummy_data(
            prompt=enriched_prompt,
            tool_specs=tool_specs,
            ui_model_headers=UI_MODEL_HEADERS,
        )

    async def _iterative_test_fix(
        self,
        *,
        service_script: str,
        components_script: str,
        test_script: str,
        dummy_data: Optional[str],
        messages: List[Message],
        allowed_tools: Optional[List[Dict[str, Any]]],
        access_token: Optional[str],
        max_attempts: int = 25,
        event_queue: Optional[asyncio.Queue] = None,
        post_success_validator: Optional[
            Callable[[str, str, str, Optional[str]], Tuple[bool, Optional[str]]]
        ] = None,
    ) -> Tuple[bool, str, str, str, Optional[str], List[Message]]:
        if not GENERATED_UI_FIX_CODE_FIRST:
            return await run_tool_driven_test_fix(
                tgi_service=self.tgi_service,
                service_script=service_script,
                components_script=components_script,
                test_script=test_script,
                dummy_data=dummy_data,
                messages=messages,
                allowed_tools=allowed_tools,
                access_token=access_token,
                max_attempts=max_attempts,
                event_queue=event_queue,
                extra_headers=UI_MODEL_HEADERS,
                post_success_validator=post_success_validator,
            )

        def _score_candidate(
            svc: str, comps: str, tests: str, fixtures: Optional[str]
        ) -> Tuple[bool, int, int]:
            try:
                ok, output = self._run_tests(svc, comps, tests, fixtures)
            except Exception:
                return False, -1, 10**9
            passed, failed, _ = _parse_tap_output(output or "")
            if not ok:
                return False, passed, failed
            if not post_success_validator:
                return True, passed, failed
            try:
                validation_ok, validation_reason = post_success_validator(
                    svc, comps, tests, fixtures
                )
            except Exception as exc:
                logger.error(
                    "[iterative_test_fix] Post-success validator errored while scoring candidate: %s",
                    exc,
                    exc_info=exc,
                )
                return False, passed, max(1, failed)
            if validation_ok:
                return True, passed, failed
            logger.info(
                "[iterative_test_fix] Candidate rejected by post-success validator: %s",
                (validation_reason or "post_success_validation_failed").strip(),
            )
            return False, passed, max(1, failed)

        test_source = test_script or ""
        likely_pfusch_collection_mismatch = bool(
            re.search(r"\bpfuschTest\s*\(", test_source)
            and re.search(r"\bcomp\.(?:state|shadowRoot)\b", test_source)
        )

        stage_a_attempts = min(max_attempts, 12)
        if likely_pfusch_collection_mismatch:
            stage_a_attempts = min(max_attempts, 6)
            logger.info(
                "[iterative_test_fix] Detected pfuschTest collection access via comp.state/comp.shadowRoot; "
                "reducing code-first attempts before adjust-test fallback"
            )
        stage_b_attempts = min(8, max(0, max_attempts - stage_a_attempts))

        best = {
            "service_script": service_script,
            "components_script": components_script,
            "test_script": test_script,
            "dummy_data": dummy_data,
            "messages": list(messages),
            "ok": False,
            "passed": -1,
            "failed": 10**9,
        }

        initial_ok, initial_passed, initial_failed = _score_candidate(
            service_script, components_script, test_script, dummy_data
        )
        best["ok"] = initial_ok
        best["passed"] = initial_passed
        best["failed"] = initial_failed

        logger.info(
            "[iterative_test_fix] Starting code-first strategy (stage_a=%s, stage_b=%s)",
            stage_a_attempts,
            stage_b_attempts,
        )

        stage_a = await run_tool_driven_test_fix(
            tgi_service=self.tgi_service,
            service_script=service_script,
            components_script=components_script,
            test_script=test_script,
            dummy_data=dummy_data,
            messages=messages,
            allowed_tools=allowed_tools,
            access_token=access_token,
            max_attempts=stage_a_attempts,
            event_queue=event_queue,
            extra_headers=UI_MODEL_HEADERS,
            strategy_mode="fix_code",
            post_success_validator=post_success_validator,
        )
        (
            stage_a_success,
            stage_a_service,
            stage_a_components,
            stage_a_test,
            stage_a_dummy_data,
            stage_a_messages,
        ) = stage_a

        stage_a_ok, stage_a_passed, stage_a_failed = _score_candidate(
            stage_a_service, stage_a_components, stage_a_test, stage_a_dummy_data
        )
        if (
            stage_a_passed > best["passed"]
            or (stage_a_passed == best["passed"] and stage_a_failed < best["failed"])
            or (stage_a_ok and not best["ok"])
        ):
            best = {
                "service_script": stage_a_service,
                "components_script": stage_a_components,
                "test_script": stage_a_test,
                "dummy_data": stage_a_dummy_data,
                "messages": stage_a_messages,
                "ok": stage_a_ok,
                "passed": stage_a_passed,
                "failed": stage_a_failed,
            }

        if stage_a_success and stage_a_ok:
            logger.info("[iterative_test_fix] Code-first stage succeeded")
            return stage_a
        if stage_a_success and not stage_a_ok:
            logger.warning(
                "[iterative_test_fix] Code-first stage reported success but failed post-success validation; continuing to fallback stage"
            )

        if stage_b_attempts <= 0:
            logger.warning(
                "[iterative_test_fix] Code-first stage failed and no fallback attempts available; returning best snapshot"
            )
            return (
                bool(best["ok"]),
                str(best["service_script"]),
                str(best["components_script"]),
                str(best["test_script"]),
                best["dummy_data"],
                list(best["messages"]),
            )

        logger.info(
            "[iterative_test_fix] Code-first stage failed, starting fallback adjust-test stage"
        )
        stage_b = await run_tool_driven_test_fix(
            tgi_service=self.tgi_service,
            service_script=stage_a_service,
            components_script=stage_a_components,
            test_script=stage_a_test,
            dummy_data=stage_a_dummy_data,
            messages=stage_a_messages,
            allowed_tools=allowed_tools,
            access_token=access_token,
            max_attempts=stage_b_attempts,
            event_queue=event_queue,
            extra_headers=UI_MODEL_HEADERS,
            strategy_mode="adjust_test",
            post_success_validator=post_success_validator,
        )
        (
            stage_b_success,
            stage_b_service,
            stage_b_components,
            stage_b_test,
            stage_b_dummy_data,
            stage_b_messages,
        ) = stage_b

        stage_b_ok, stage_b_passed, stage_b_failed = _score_candidate(
            stage_b_service, stage_b_components, stage_b_test, stage_b_dummy_data
        )
        if (
            stage_b_passed > best["passed"]
            or (stage_b_passed == best["passed"] and stage_b_failed < best["failed"])
            or (stage_b_ok and not best["ok"])
        ):
            best = {
                "service_script": stage_b_service,
                "components_script": stage_b_components,
                "test_script": stage_b_test,
                "dummy_data": stage_b_dummy_data,
                "messages": stage_b_messages,
                "ok": stage_b_ok,
                "passed": stage_b_passed,
                "failed": stage_b_failed,
            }

        if stage_b_success and stage_b_ok:
            logger.info("[iterative_test_fix] Fallback adjust-test stage succeeded")
            return stage_b
        if stage_b_success and not stage_b_ok:
            logger.warning(
                "[iterative_test_fix] Fallback adjust-test stage reported success but failed post-success validation"
            )

        logger.warning(
            "[iterative_test_fix] All fix stages failed. Returning best snapshot (passed=%s, failed=%s)",
            best["passed"],
            best["failed"],
        )
        return (
            bool(best["ok"]),
            str(best["service_script"]),
            str(best["components_script"]),
            str(best["test_script"]),
            best["dummy_data"],
            list(best["messages"]),
        )

    async def create_ui(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        if self.storage.exists(scope, ui_id, name):
            raise HTTPException(
                status_code=409,
                detail="Ui already exists for this id and name",
            )

        if scope.kind == "user" and actor.user_id != scope.identifier:
            raise HTTPException(
                status_code=403,
                detail="User uis may only be created by the owning user",
            )

        if scope.kind == "group" and scope.identifier not in set(actor.groups or []):
            raise HTTPException(
                status_code=403,
                detail="Group uis may only be created by group members",
            )

        generated = await self._generate_ui_payload(
            session=session,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            tools=list(tools or []),
            access_token=access_token,
            previous=None,
        )

        # TODO: remove after testing
        logger.info(
            "[create_ui] Initial generation starting point:\n"
            "--- Dummy Data ---\n%s\n"
            "--- Service Script ---\n%s\n"
            "--- Components Script ---\n%s\n"
            "--- Test Script ---\n%s\n",
            generated.get("dummy_data", ""),
            generated.get("service_script", ""),
            generated.get("components_script", ""),
            generated.get("test_script", ""),
        )

        timestamp = self._now()
        payload_scripts = self._changed_scripts(generated, None)
        record = {
            "metadata": {
                "id": ui_id,
                "name": name,
                "scope": {"type": scope.kind, "id": scope.identifier},
                "owner": {"type": scope.kind, "id": scope.identifier},
                "created_by": actor.user_id,
                "created_at": timestamp,
                "updated_at": timestamp,
                "version": 1,
                "published_at": timestamp,
                "published_by": actor.user_id,
                "history": [
                    self._history_entry(
                        action="create",
                        prompt=prompt,
                        tools=list(tools or []),
                        user_id=actor.user_id,
                        generated_at=timestamp,
                        payload_metadata=generated.get("metadata", {}),
                        payload_html=generated.get("html", {}),
                        payload_scripts=payload_scripts or None,
                    )
                ],
            },
            "current": generated,
        }

        self.storage.write(scope, ui_id, name, record)
        return record

    async def _phase_1_attempt(
        self,
        *,
        attempt: int,
        max_attempts: int,
        messages: List[Message],
        allowed_tools: List[Dict[str, Any]],
        dummy_data: Optional[str],
        access_token: Optional[str],
    ) -> AsyncIterator[Union[bytes, Dict[str, Any]]]:
        async for item in run_phase1_attempt(
            attempt=attempt,
            max_attempts=max_attempts,
            messages=messages,
            allowed_tools=allowed_tools,
            dummy_data=dummy_data,
            access_token=access_token,
            tgi_service=self.tgi_service,
            parse_json=self._parse_json,
            run_tests=self._run_tests,
            iterative_test_fix=self._iterative_test_fix,
            chunk_reader=chunk_reader,
            ui_model_headers=UI_MODEL_HEADERS,
        ):
            yield item

    async def _phase_2_attempt(
        self,
        *,
        system_prompt: str,
        prompt: str,
        logic_payload: Dict[str, Any],
        access_token: Optional[str],
        instruction: str,
    ) -> AsyncIterator[Union[bytes, Dict[str, Any]]]:
        async for item in run_phase2_attempt(
            system_prompt=system_prompt,
            prompt=prompt,
            logic_payload=logic_payload,
            access_token=access_token,
            instruction=instruction,
            tgi_service=self.tgi_service,
            parse_json=self._parse_json,
            chunk_reader=chunk_reader,
            ui_model_headers=UI_MODEL_HEADERS,
        ):
            yield item

    async def stream_generate_ui(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> AsyncIterator[bytes]:
        """
        Stream UI generation as Server-Sent Events (SSE).

        Yields bytes that are already formatted as SSE messages. The stream
        will emit keepalive comments roughly every 10 seconds during idle
        model streaming to avoid client timeouts.
        """
        logger.info(
            f"[stream_generate_ui] Starting stream for ui_id={ui_id}, name={name}, scope={scope.kind}:{scope.identifier}"
        )
        requested_tools = list(tools or [])

        # Basic existence and permission checks similar to create_ui
        if self.storage.exists(scope, ui_id, name):
            logger.warning(f"[stream_generate_ui] UI already exists: {ui_id}/{name}")
            # SSE error message and stop
            payload = json.dumps({"error": "Ui already exists for this id and name"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        if scope.kind == "user" and actor.user_id != scope.identifier:
            logger.warning(
                f"[stream_generate_ui] Permission denied: user {actor.user_id} cannot create UI for user {scope.identifier}"
            )
            payload = json.dumps(
                {"error": "User uis may only be created by the owning user"}
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        if scope.kind == "group" and scope.identifier not in set(actor.groups or []):
            logger.warning(
                f"[stream_generate_ui] Permission denied: user {actor.user_id} not in group {scope.identifier}"
            )
            payload = json.dumps(
                {"error": "Group uis may only be created by group members"}
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        # Wrap initialization in try-catch to ensure we yield error messages
        # if anything fails before streaming starts
        logger.info("[stream_generate_ui] Building system prompt and selecting tools")

        attempt = 0
        max_attempts = 3
        messages: List[Message] = []
        allowed_tools: List[Dict[str, Any]] = []
        payload_obj: Dict[str, Any] = {}
        logic_payload: Dict[str, Any] = {}
        dummy_data: Optional[str] = None
        phase1_failure_reasons: List[str] = []

        try:
            system_prompt = await self._build_system_prompt(session)
            logger.info(
                f"[stream_generate_ui] System prompt built, length={len(system_prompt)}"
            )

            allowed_tools = await self._select_tools(session, requested_tools, prompt)

            message_payload = {
                "ui": {
                    "id": ui_id,
                    "name": name,
                    "scope": {"type": scope.kind, "id": scope.identifier},
                },
                "request": {
                    "prompt": prompt,
                    "tools": [t["function"]["name"] for t in (allowed_tools or [])],
                    "requested_tools": requested_tools,
                },
            }

            initial_message = Message(
                role=MessageRole.USER,
                content=json.dumps(message_payload, ensure_ascii=False),
            )

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                initial_message,
            ]

            yield f"event: log\ndata: {json.dumps({'message': 'Generating dummy data for tests...'})}\n\n".encode(
                "utf-8"
            )

            dummy_data = await self._generate_dummy_data(
                session=session,
                scope=scope,
                ui_id=ui_id,
                name=name,
                prompt=prompt,
                allowed_tools=allowed_tools,
                access_token=access_token,
            )

            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        f"{_DUMMY_DATA_TEST_USAGE_GUIDANCE} "
                        f"Tools: {[t['function']['name'] for t in (allowed_tools or [])]}"
                    ),
                )
            )
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        "PHASE 1 CONTRACT (STRICT): Return ONLY a JSON object for logic generation "
                        "with keys: components_script (required), test_script (required), "
                        "service_script (optional). Do NOT return template_parts, html, or metadata."
                    ),
                )
            )

            # --- PHASE 1: GENERATE LOGIC AND TESTS ---
            phase1_success = False
            while attempt < max_attempts:
                attempt += 1
                # Clone messages for this attempt to avoid polluting history on retry
                attempt_messages = copy.deepcopy(messages)

                async for item in self._phase_1_attempt(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    messages=attempt_messages,
                    allowed_tools=allowed_tools,
                    dummy_data=dummy_data,
                    access_token=access_token,
                ):
                    if isinstance(item, bytes):
                        yield item
                    elif isinstance(item, dict) and item.get("type") == "result":
                        if item["success"]:
                            phase1_success = True
                            logic_payload = item["payload"]
                            # Update main messages with the successful history
                            messages = item["messages"]
                        else:
                            reason = (
                                item.get("reason")
                                or item.get("error")
                                or "unknown phase 1 failure"
                            )
                            phase1_failure_reasons.append(
                                f"attempt {attempt}: {reason}"
                            )
                            logger.warning(
                                "[stream_generate_ui] Phase 1 attempt %s failed: %s",
                                attempt,
                                reason,
                            )
                            yield f"event: log\ndata: {json.dumps({'message': f'Phase 1 attempt {attempt} failed', 'reason': reason})}\n\n".encode(
                                "utf-8"
                            )
                            # Should we update messages here?
                            # If we update messages here, we are keeping failed attempt history, which defeats "discarding".
                            # The user requested "cleanly discarded once an attempt completed".
                            # So we DO NOT update `messages` on failure.
                            # The next attempt will start from original `messages` state (fresh retry).
                            pass
                        break

                if phase1_success:
                    break

            if not phase1_success:
                detail = (
                    " | ".join(phase1_failure_reasons) if phase1_failure_reasons else ""
                )
                logger.error(
                    "[stream_generate_ui] Failed to generate valid logic after %s attempts. Reasons: %s",
                    max_attempts,
                    detail or "none captured",
                )
                error_message = (
                    f"Failed to generate valid logic after {max_attempts} attempts"
                )
                if detail:
                    error_message = f"{error_message}: {detail}"
                yield f"event: error\ndata: {json.dumps({'error': error_message})}\n\n".encode(
                    "utf-8"
                )
                return

            # --- PHASE 2: GENERATE PRESENTATION ---
            presentation_payload: Dict[str, Any] = {}
            instruction = (
                "Tests passed. Now generate presentation template parts only. "
                "Return `template_parts` and `metadata`. "
                "Do not return `html.page` or `html.snippet`."
            )

            phase2_system_prompt = await self._build_phase2_system_prompt(session)

            async for item in self._phase_2_attempt(
                system_prompt=phase2_system_prompt,
                prompt=prompt,
                logic_payload=logic_payload,
                access_token=access_token,
                instruction=instruction,
            ):
                if isinstance(item, bytes):
                    yield item
                elif isinstance(item, dict) and item.get("type") == "result":
                    if item["success"]:
                        presentation_payload = item["payload"]
                    else:
                        return

            # Merge
            payload_obj = {**logic_payload, **presentation_payload}
            self._normalise_payload(payload_obj, scope, ui_id, name, prompt, None)

        except HTTPException as exc:
            logger.error(
                f"[stream_generate_ui] HTTPException during initialization: {exc.detail}",
                exc_info=exc,
            )
            payload = json.dumps({"error": exc.detail})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return
        except Exception as exc:
            logger.error(
                f"[stream_generate_ui] Exception during initialization: {str(exc)}",
                exc_info=exc,
            )
            payload = json.dumps({"error": "Failed to initialize generation"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

        # Build final record and persist
        logger.info("[stream_generate_ui] Building final record")
        timestamp = self._now()
        payload_scripts = self._changed_scripts(payload_obj, None)
        record = {
            "metadata": {
                "id": ui_id,
                "name": name,
                "scope": {"type": scope.kind, "id": scope.identifier},
                "owner": {"type": scope.kind, "id": scope.identifier},
                "created_by": actor.user_id,
                "created_at": timestamp,
                "updated_at": timestamp,
                "version": 1,
                "published_at": timestamp,
                "published_by": actor.user_id,
                "history": [
                    self._history_entry(
                        action="create",
                        prompt=prompt,
                        tools=list(tools or []),
                        user_id=actor.user_id,
                        generated_at=timestamp,
                        payload_metadata=payload_obj.get("metadata", {}),
                        payload_html=payload_obj.get("html", {}),
                        payload_scripts=payload_scripts or None,
                    )
                ],
            },
            "current": payload_obj,
        }

        # persist
        logger.info("[stream_generate_ui] Persisting record to storage")
        try:
            self.storage.write(scope, ui_id, name, record)
            logger.info("[stream_generate_ui] Record persisted successfully")
        except Exception as e:
            logger.error(
                f"[stream_generate_ui] Failed to persist record: {str(e)}", exc_info=e
            )
            payload = json.dumps({"error": f"Failed to persist generated ui: {str(e)}"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

        # Final event with the created record
        logger.info("[stream_generate_ui] Sending final done event")

        # We want to send the expanded record to the client so they can run it
        expanded_record = record.copy()
        expanded_record["current"] = self._expand_payload(record["current"])

        final_payload = json.dumps(
            {"status": "created", "record": expanded_record}, ensure_ascii=False
        )
        yield f"event: log\ndata: {json.dumps({'message': 'Page successfully generated'})}\n\n".encode(
            "utf-8"
        )
        yield f"event: done\ndata: {final_payload}\n\n".encode("utf-8")
        # Send proper SSE done marker
        yield b"data: [DONE]\n\n"
        logger.info("[stream_generate_ui] Stream completed successfully")

    async def stream_update_ui(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> AsyncIterator[bytes]:
        """
        Stream UI updates as Server-Sent Events (SSE).

        Uses the same generation/test pipeline as create, but applies the
        results to an existing UI record with update history.
        """
        logger.info(
            f"[stream_update_ui] Starting stream for ui_id={ui_id}, name={name}, scope={scope.kind}:{scope.identifier}"
        )
        requested_tools = list(tools or [])

        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError:
            logger.warning(f"[stream_update_ui] UI not found: {ui_id}/{name}")
            payload = json.dumps({"error": "Ui not found"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        try:
            self._assert_scope_consistency(existing, scope, name)
            self._ensure_update_permissions(existing, scope, actor)
        except HTTPException as exc:
            payload = json.dumps({"error": exc.detail})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

        attempt = 0
        max_attempts = 3
        messages: List[Message] = []
        allowed_tools: List[Dict[str, Any]] = []
        payload_obj: Dict[str, Any] = {}
        logic_payload: Dict[str, Any] = {}
        dummy_data: Optional[str] = None
        phase1_failure_reasons: List[str] = []

        try:
            system_prompt = await self._build_system_prompt(session)
            logger.info(
                f"[stream_update_ui] System prompt built, length={len(system_prompt)}"
            )

            allowed_tools = await self._select_tools(session, requested_tools, prompt)

            previous_metadata = existing.get("metadata", {})
            message_payload = {
                "ui": {
                    "id": ui_id,
                    "name": name,
                    "scope": {"type": scope.kind, "id": scope.identifier},
                },
                "request": {
                    "prompt": prompt,
                    "tools": [t["function"]["name"] for t in (allowed_tools or [])],
                    "requested_tools": requested_tools,
                },
                "context": {
                    "original_prompt": self._initial_prompt(previous_metadata),
                    "history": self._history_for_prompt(
                        previous_metadata.get("history", [])
                    ),
                    "current_state": existing.get("current", {}),
                },
            }

            initial_message = Message(
                role=MessageRole.USER,
                content=json.dumps(message_payload, ensure_ascii=False),
            )

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                initial_message,
            ]

            yield f"event: log\ndata: {json.dumps({'message': 'Generating dummy data for tests...'})}\n\n".encode(
                "utf-8"
            )

            dummy_data = await self._generate_dummy_data(
                session=session,
                scope=scope,
                ui_id=ui_id,
                name=name,
                prompt=prompt,
                allowed_tools=allowed_tools,
                access_token=access_token,
            )

            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        f"{_DUMMY_DATA_TEST_USAGE_GUIDANCE} "
                        f"Tools: {[t['function']['name'] for t in (allowed_tools or [])]}"
                    ),
                )
            )
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        "PHASE 1 CONTRACT (STRICT): Return ONLY a JSON object for logic generation "
                        "with keys: components_script (required), test_script (required), "
                        "service_script (optional). Do NOT return template_parts, html, or metadata."
                    ),
                )
            )

            # --- PHASE 1: GENERATE LOGIC AND TESTS ---
            phase1_success = False
            while attempt < max_attempts:
                attempt += 1
                attempt_messages = copy.deepcopy(messages)

                async for item in self._phase_1_attempt(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    messages=attempt_messages,
                    allowed_tools=allowed_tools,
                    dummy_data=dummy_data,
                    access_token=access_token,
                ):
                    if isinstance(item, bytes):
                        yield item
                    elif isinstance(item, dict) and item.get("type") == "result":
                        if item["success"]:
                            phase1_success = True
                            logic_payload = item["payload"]
                            messages = item["messages"]
                        else:
                            reason = (
                                item.get("reason")
                                or item.get("error")
                                or "unknown phase 1 failure"
                            )
                            phase1_failure_reasons.append(
                                f"attempt {attempt}: {reason}"
                            )
                            logger.warning(
                                "[stream_update_ui] Phase 1 attempt %s failed: %s",
                                attempt,
                                reason,
                            )
                            yield f"event: log\ndata: {json.dumps({'message': f'Phase 1 attempt {attempt} failed', 'reason': reason})}\n\n".encode(
                                "utf-8"
                            )
                        break

                if phase1_success:
                    break

            if not phase1_success:
                detail = (
                    " | ".join(phase1_failure_reasons) if phase1_failure_reasons else ""
                )
                logger.error(
                    "[stream_update_ui] Failed to generate valid logic after %s attempts. Reasons: %s",
                    max_attempts,
                    detail or "none captured",
                )
                error_message = (
                    f"Failed to generate valid logic after {max_attempts} attempts"
                )
                if detail:
                    error_message = f"{error_message}: {detail}"
                yield f"event: error\ndata: {json.dumps({'error': error_message})}\n\n".encode(
                    "utf-8"
                )
                return

            # --- PHASE 2: GENERATE PRESENTATION ---
            presentation_payload: Dict[str, Any] = {}
            instruction = (
                "Tests passed. Now update presentation template parts only. "
                "Return `template_parts` and `metadata`. "
                "Do not return `html.page` or `html.snippet`."
            )

            phase2_system_prompt = await self._build_phase2_system_prompt(session)

            async for item in self._phase_2_attempt(
                system_prompt=phase2_system_prompt,
                prompt=prompt,
                logic_payload=logic_payload,
                access_token=access_token,
                instruction=instruction,
            ):
                if isinstance(item, bytes):
                    yield item
                elif isinstance(item, dict) and item.get("type") == "result":
                    if item["success"]:
                        presentation_payload = item["payload"]
                    else:
                        return

            payload_obj = {**logic_payload, **presentation_payload}
            self._normalise_payload(payload_obj, scope, ui_id, name, prompt, existing)

        except HTTPException as exc:
            logger.error(
                f"[stream_update_ui] HTTPException during update: {exc.detail}",
                exc_info=exc,
            )
            payload = json.dumps({"error": exc.detail})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return
        except Exception as exc:
            logger.error(
                f"[stream_update_ui] Exception during update: {str(exc)}",
                exc_info=exc,
            )
            payload = json.dumps({"error": "Failed to update ui"})
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

        # Build final record and persist
        logger.info("[stream_update_ui] Building updated record")
        timestamp = self._now()
        existing.setdefault("metadata", {})
        metadata = existing["metadata"]
        self._ensure_version_metadata(metadata)
        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id
        metadata["version"] = self._current_version(metadata) + 1
        metadata["published_at"] = timestamp
        metadata["published_by"] = actor.user_id
        history = metadata.setdefault("history", [])
        payload_scripts = self._changed_scripts(
            payload_obj, existing.get("current", {})
        )
        history.append(
            self._history_entry(
                action="update",
                prompt=prompt,
                tools=list(tools or []),
                user_id=actor.user_id,
                generated_at=timestamp,
                payload_metadata=payload_obj.get("metadata", {}),
                payload_html=payload_obj.get("html", {}),
                payload_scripts=payload_scripts or None,
            )
        )

        existing["current"] = payload_obj

        logger.info("[stream_update_ui] Persisting record to storage")
        try:
            self.storage.write(scope, ui_id, name, existing)
            logger.info("[stream_update_ui] Record persisted successfully")
        except Exception as exc:
            logger.error(
                f"[stream_update_ui] Failed to persist record: {str(exc)}",
                exc_info=exc,
            )
            payload = json.dumps(
                {"error": f"Failed to persist generated ui: {str(exc)}"}
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            yield f"event: log\ndata: {json.dumps({'message': 'Page creation failed'})}\n\n".encode(
                "utf-8"
            )
            return

        logger.info("[stream_update_ui] Sending final done event")
        expanded_record = existing.copy()
        expanded_record["current"] = self._expand_payload(existing["current"])
        final_payload = json.dumps(
            {"status": "updated", "record": expanded_record}, ensure_ascii=False
        )
        yield f"event: log\ndata: {json.dumps({'message': 'Page successfully generated'})}\n\n".encode(
            "utf-8"
        )
        yield f"event: done\ndata: {final_payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        logger.info("[stream_update_ui] Stream completed successfully")

    async def update_ui(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        prompt: str,
        tools: Optional[Iterable[str]],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)

        generated = await self._generate_ui_payload(
            session=session,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            tools=list(tools or []),
            access_token=access_token,
            previous=existing,
        )

        timestamp = self._now()
        existing.setdefault("metadata", {})
        metadata = existing["metadata"]
        self._ensure_version_metadata(metadata)
        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id
        metadata["version"] = self._current_version(metadata) + 1
        metadata["published_at"] = timestamp
        metadata["published_by"] = actor.user_id
        history = metadata.setdefault("history", [])
        payload_scripts = self._changed_scripts(generated, existing.get("current", {}))
        history.append(
            self._history_entry(
                action="update",
                prompt=prompt,
                tools=list(tools or []),
                user_id=actor.user_id,
                generated_at=timestamp,
                payload_metadata=generated.get("metadata", {}),
                payload_html=generated.get("html", {}),
                payload_scripts=payload_scripts or None,
            )
        )

        existing["current"] = generated

        self.storage.write(scope, ui_id, name, existing)
        return existing

    def _expand_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.output_factory.expand_payload(payload)

    def get_ui(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        expand: bool = False,
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        existing.setdefault("metadata", {})
        self._ensure_version_metadata(existing["metadata"])

        # Only expand payload when explicitly requested (for HTML rendering)
        if expand and "current" in existing:
            existing["current"] = self._expand_payload(existing["current"])

        return existing

    def reset_last_change(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
    ) -> Dict[str, Any]:
        """
        Reset the last change by removing the last history entry and
        restoring the previous state as current.

        Requires update permissions. Will fail if there is only one history entry.
        """
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)

        metadata = existing.get("metadata", {})
        history = metadata.get("history", [])

        if len(history) <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot reset: only one history entry exists (initial creation)",
            )

        # Remove the last history entry
        last_entry = history.pop()

        # Get the new last entry (which becomes current)
        new_last_entry = history[-1]

        # Reconstruct the current state from the new last entry
        new_current = {
            "html": new_last_entry.get("payload_html", {}),
            "metadata": new_last_entry.get("payload_metadata", {}),
        }
        scripts = self._scripts_from_history(history)
        if scripts:
            new_current.update(scripts)

        # Update timestamps
        timestamp = self._now()
        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id

        # Update the record
        existing["current"] = new_current
        existing["metadata"] = metadata

        # Persist the changes
        self.storage.write(scope, ui_id, name, existing)

        logger.info(
            f"Reset UI {ui_id}/{name} - removed history entry from {last_entry.get('generated_at')}"
        )

        return existing

    def create_draft_session(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        tools: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)
        existing.setdefault("metadata", {})
        self._ensure_version_metadata(existing["metadata"])
        self.storage.cleanup_expired_sessions(scope, ui_id, name)

        ttl_minutes = int(
            os.environ.get(
                "APP_UI_SESSION_TTL_MINUTES", str(APP_UI_SESSION_TTL_MINUTES)
            )
        )
        created_at = self._now()
        expires_at = datetime.now(timezone.utc).timestamp() + (ttl_minutes * 60)
        expires_iso = datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat()
        session_id = validate_identifier(uuid.uuid4().hex, "session id")
        base_version = self._current_version(existing["metadata"])

        payload = {
            "session_id": session_id,
            "editor_user_id": actor.user_id,
            "created_at": created_at,
            "updated_at": created_at,
            "expires_at": expires_iso,
            "base_version": base_version,
            "draft_version": 1,
            "draft_payload": copy.deepcopy(existing.get("current", {})),
            "messages": [],
            "last_tools": list(tools or []),
            "metadata_snapshot": copy.deepcopy(existing.get("metadata", {})),
            "test_state": "idle",
            "test_run_id": "",
            "test_summary": {
                "passed": 0,
                "failed": 0,
                "failed_tests": [],
                "message": "No tests run yet",
                "trigger": "none",
                "started_at": None,
                "completed_at": None,
            },
            "test_events": [],
            "last_test_output_tail": "",
        }

        self.storage.write_session(scope, ui_id, name, session_id, payload)
        return {
            "session_id": session_id,
            "base_version": base_version,
            "draft_version": 1,
            "expires_at": expires_iso,
        }

    def _session_is_expired(self, session_payload: Dict[str, Any]) -> bool:
        raw = session_payload.get("expires_at")
        if not raw:
            return False
        value = str(raw)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            expires_dt = datetime.fromisoformat(value)
        except ValueError:
            return True
        if expires_dt.tzinfo is None:
            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
        return expires_dt <= datetime.now(timezone.utc)

    def _ensure_test_session_fields(self, session_payload: Dict[str, Any]) -> bool:
        changed = False
        if "test_state" not in session_payload:
            session_payload["test_state"] = "idle"
            changed = True
        if "test_run_id" not in session_payload:
            session_payload["test_run_id"] = ""
            changed = True
        if "last_test_output_tail" not in session_payload:
            session_payload["last_test_output_tail"] = ""
            changed = True
        if "test_events" not in session_payload or not isinstance(
            session_payload.get("test_events"), list
        ):
            session_payload["test_events"] = []
            changed = True

        summary = session_payload.get("test_summary")
        if not isinstance(summary, dict):
            summary = {}
            changed = True
        defaults = {
            "passed": 0,
            "failed": 0,
            "failed_tests": [],
            "message": "No tests run yet",
            "trigger": "none",
            "started_at": None,
            "completed_at": None,
        }
        for key, value in defaults.items():
            if key not in summary:
                summary[key] = value
                changed = True
        session_payload["test_summary"] = summary
        return changed

    def _load_session(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
    ) -> Dict[str, Any]:
        self.storage.cleanup_expired_sessions(scope, ui_id, name)
        try:
            session_payload = self.storage.read_session(scope, ui_id, name, session_id)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Draft session not found"
            ) from exc
        if self._session_is_expired(session_payload):
            self.storage.delete_session(scope, ui_id, name, session_id)
            raise HTTPException(status_code=404, detail="Draft session expired")
        if self._ensure_test_session_fields(session_payload):
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)
        return session_payload

    def _assert_session_owner(
        self, session_payload: Dict[str, Any], actor: Actor
    ) -> None:
        if session_payload.get("editor_user_id") != actor.user_id:
            raise HTTPException(status_code=403, detail="Draft session access denied")

    def _test_stream_key(
        self, *, scope: Scope, ui_id: str, name: str, session_id: str
    ) -> str:
        return f"{scope.kind}:{scope.identifier}:{ui_id}:{name}:{session_id}"

    def _test_lock(self, stream_key: str) -> asyncio.Lock:
        lock = self._test_run_locks.get(stream_key)
        if lock is None:
            lock = asyncio.Lock()
            self._test_run_locks[stream_key] = lock
        return lock

    def _next_test_run_id(self) -> str:
        self._test_run_seq += 1
        return f"run-{self._test_run_seq:08d}"

    def _trim_output(self, output: str, limit: int = 4000) -> str:
        text = output or ""
        if len(text) <= limit:
            return text
        failure_lines: List[str] = []
        seen_failures = set()
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("not ok ") and line not in seen_failures:
                seen_failures.add(line)
                failure_lines.append(line)

        summary_lines: List[str] = []
        for pattern in (
            r"^# tests\s+\d+",
            r"^# suites\s+\d+",
            r"^# pass\s+\d+",
            r"^# fail\s+\d+",
        ):
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                summary_lines.append(match.group(0))

        prefix_parts = [f"...(trimmed {len(text) - limit} chars)"]
        if failure_lines:
            preview = failure_lines[:8]
            if len(failure_lines) > 8:
                preview.append(f"... ({len(failure_lines) - 8} more failing entries)")
            prefix_parts.append(
                "Failed tests:\n" + "\n".join(f"- {line}" for line in preview)
            )
        if summary_lines:
            prefix_parts.append("Summary:\n" + "\n".join(summary_lines))

        prefix = "\n".join(prefix_parts).strip() + "\n\n"
        remaining = max(800, limit - len(prefix))
        head_budget = max(200, remaining // 3)
        tail_budget = max(400, remaining - head_budget)
        head = text[:head_budget]
        tail = text[-tail_budget:]
        return (
            f"{prefix}" f"--- OUTPUT HEAD ---\n{head}\n" f"--- OUTPUT TAIL ---\n{tail}"
        )

    async def _broadcast_test_event(
        self, stream_key: str, event: str, payload: Dict[str, Any]
    ) -> None:
        queues = list(self._test_event_subscribers.get(stream_key) or [])
        for queue in queues:
            try:
                queue.put_nowait((event, payload))
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    _ = queue.get_nowait()
                with contextlib.suppress(Exception):
                    queue.put_nowait((event, payload))

    def _append_test_event_to_session(
        self,
        session_payload: Dict[str, Any],
        *,
        event: str,
        payload: Dict[str, Any],
    ) -> None:
        events = session_payload.setdefault("test_events", [])
        events.append({"event": event, "payload": payload})
        session_payload["test_events"] = events[-MAX_TEST_EVENTS:]

    async def _update_test_state(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: Optional[str],
        state: str,
        trigger: str,
        message: str,
        draft_version: Optional[int] = None,
        completed_at: Optional[str] = None,
        summary_updates: Optional[Dict[str, Any]] = None,
        skip_if_run_mismatch: bool = True,
    ) -> Optional[Dict[str, Any]]:
        payload: Optional[Dict[str, Any]] = None
        lock = self._test_lock(stream_key)
        async with lock:
            try:
                session_payload = self._load_session(
                    scope=scope, ui_id=ui_id, name=name, session_id=session_id
                )
            except HTTPException:
                return None

            current_run_id = str(session_payload.get("test_run_id") or "")
            run_mismatch = bool(run_id and current_run_id and current_run_id != run_id)
            if skip_if_run_mismatch and run_mismatch:
                return None

            now = self._now()
            if draft_version is None:
                draft_version = int(session_payload.get("draft_version") or 1)

            payload = {
                "run_id": run_id or current_run_id,
                "state": state,
                "trigger": trigger,
                "draft_version": draft_version,
                "message": message,
                "started_at": (session_payload.get("test_summary") or {}).get(
                    "started_at"
                ),
                "completed_at": completed_at,
            }

            if not run_mismatch:
                if run_id:
                    session_payload["test_run_id"] = run_id
                session_payload["test_state"] = state
                session_payload["updated_at"] = now
                summary = session_payload.setdefault("test_summary", {})
                summary.update(
                    {
                        "message": message,
                        "trigger": trigger,
                        "completed_at": completed_at,
                    }
                )
                if summary_updates:
                    summary.update(summary_updates)
                payload["started_at"] = summary.get("started_at")
                session_payload["test_summary"] = summary

            self._append_test_event_to_session(
                session_payload,
                event="test_status",
                payload=payload,
            )
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)

        if payload:
            await self._broadcast_test_event(stream_key, "test_status", payload)
        return payload

    async def _append_test_event(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        event: str,
        payload: Dict[str, Any],
        run_id: Optional[str] = None,
        skip_if_run_mismatch: bool = True,
    ) -> bool:
        lock = self._test_lock(stream_key)
        async with lock:
            try:
                session_payload = self._load_session(
                    scope=scope, ui_id=ui_id, name=name, session_id=session_id
                )
            except HTTPException:
                return False

            current_run_id = str(session_payload.get("test_run_id") or "")
            if (
                skip_if_run_mismatch
                and run_id
                and current_run_id
                and current_run_id != run_id
            ):
                return False

            if event == "test_output":
                session_payload["last_test_output_tail"] = str(
                    payload.get("output_tail") or ""
                )
            self._append_test_event_to_session(
                session_payload,
                event=event,
                payload=payload,
            )
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)

        await self._broadcast_test_event(stream_key, event, payload)
        return True

    def get_draft_ui(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
        expand: bool = False,
    ) -> Dict[str, Any]:
        session_payload = self._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self._assert_session_owner(session_payload, actor)

        metadata = copy.deepcopy(session_payload.get("metadata_snapshot", {}))
        metadata["draft_version"] = session_payload.get("draft_version", 1)
        metadata["session_id"] = session_payload.get("session_id")
        self._ensure_version_metadata(metadata)

        current = copy.deepcopy(session_payload.get("draft_payload", {}))
        if expand:
            current = self._expand_payload(current)

        return {"metadata": metadata, "current": current}

    def discard_draft_session(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
    ) -> bool:
        session_payload = self._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self._assert_session_owner(session_payload, actor)
        return self.storage.delete_session(scope, ui_id, name, session_id)

    def publish_draft_session(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
    ) -> Dict[str, Any]:
        try:
            existing = self.storage.read(scope, ui_id, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Ui not found") from exc

        session_payload = self._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self._assert_session_owner(session_payload, actor)

        self._assert_scope_consistency(existing, scope, name)
        self._ensure_update_permissions(existing, scope, actor)

        existing.setdefault("metadata", {})
        metadata = existing["metadata"]
        self._ensure_version_metadata(metadata)
        current_version = self._current_version(metadata)
        base_version = int(session_payload.get("base_version") or 0)

        if base_version != current_version:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "publish_conflict",
                    "detail": "Draft base version is stale. Refresh draft session.",
                    "current_version": current_version,
                    "base_version": base_version,
                },
            )

        timestamp = self._now()
        draft_payload = copy.deepcopy(session_payload.get("draft_payload", {}))
        self._normalise_payload(
            draft_payload,
            scope,
            ui_id,
            name,
            "Publish conversational draft",
            existing,
        )
        payload_scripts = self._changed_scripts(
            draft_payload, existing.get("current", {})
        )

        metadata["updated_at"] = timestamp
        metadata["updated_by"] = actor.user_id
        metadata["version"] = current_version + 1
        metadata["published_at"] = timestamp
        metadata["published_by"] = actor.user_id
        history = metadata.setdefault("history", [])
        history.append(
            self._history_entry(
                action="publish",
                prompt="Publish conversational draft",
                tools=list(session_payload.get("last_tools") or []),
                user_id=actor.user_id,
                generated_at=timestamp,
                payload_metadata=draft_payload.get("metadata", {}),
                payload_html=draft_payload.get("html", {}),
                payload_scripts=payload_scripts or None,
            )
        )

        existing["current"] = draft_payload
        self.storage.write(scope, ui_id, name, existing)
        self.storage.delete_session(scope, ui_id, name, session_id)
        return existing

    async def queue_test_action(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
        action: TestAction,
        test_name: Optional[str],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        if action not in TEST_ACTIONS:
            raise HTTPException(status_code=400, detail="Invalid test action")
        session_payload = self._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self._assert_session_owner(session_payload, actor)
        return await self._queue_test_run(
            scope=scope,
            ui_id=ui_id,
            name=name,
            session_id=session_id,
            action=action,
            trigger=f"manual_{action}",
            test_name=test_name,
            access_token=access_token,
        )

    async def stream_test_events(
        self,
        *,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
    ) -> AsyncIterator[bytes]:
        session_payload = self._load_session(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        self._assert_session_owner(session_payload, actor)
        stream_key = self._test_stream_key(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        subscribers = self._test_event_subscribers.setdefault(stream_key, [])
        subscribers.append(queue)
        try:
            for item in list(session_payload.get("test_events") or []):
                if not isinstance(item, dict):
                    continue
                event = str(item.get("event") or "").strip()
                payload = item.get("payload")
                if not event or not isinstance(payload, dict):
                    continue
                yield _sse_event(event, payload)
            while True:
                try:
                    event, payload = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                if not isinstance(event, str) or not isinstance(payload, dict):
                    continue
                yield _sse_event(event, payload)
        finally:
            current = self._test_event_subscribers.get(stream_key) or []
            self._test_event_subscribers[stream_key] = [
                q for q in current if q != queue
            ]
            if not self._test_event_subscribers[stream_key]:
                self._test_event_subscribers.pop(stream_key, None)

    async def _queue_test_run(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        action: TestAction,
        trigger: str,
        test_name: Optional[str],
        access_token: Optional[str],
    ) -> Dict[str, Any]:
        stream_key = self._test_stream_key(
            scope=scope, ui_id=ui_id, name=name, session_id=session_id
        )
        lock = self._test_lock(stream_key)
        events_to_emit: List[Tuple[str, Dict[str, Any]]] = []
        previous_task: Optional[asyncio.Task] = None
        run_id = self._next_test_run_id()

        async with lock:
            session_payload = self._load_session(
                scope=scope, ui_id=ui_id, name=name, session_id=session_id
            )
            previous_task = self._test_run_tasks.get(stream_key)
            previous_run_id = str(session_payload.get("test_run_id") or "")
            draft_version = int(session_payload.get("draft_version") or 1)

            if previous_task and not previous_task.done() and previous_run_id:
                previous_task.cancel()
                cancelled_payload = {
                    "run_id": previous_run_id,
                    "state": "cancelled",
                    "trigger": trigger,
                    "draft_version": draft_version,
                    "message": "Superseded by a newer test run",
                    "started_at": None,
                    "completed_at": self._now(),
                }
                self._append_test_event_to_session(
                    session_payload,
                    event="test_status",
                    payload=cancelled_payload,
                )
                events_to_emit.append(("test_status", cancelled_payload))

            session_payload["test_run_id"] = run_id
            session_payload["test_state"] = "queued"
            summary = session_payload.setdefault("test_summary", {})
            summary.update(
                {
                    "message": "Test run queued",
                    "trigger": trigger,
                    "passed": int(summary.get("passed") or 0),
                    "failed": int(summary.get("failed") or 0),
                    "failed_tests": list(summary.get("failed_tests") or []),
                    "started_at": None,
                    "completed_at": None,
                }
            )
            session_payload["test_summary"] = summary
            session_payload["updated_at"] = self._now()
            queued_payload = {
                "run_id": run_id,
                "state": "queued",
                "trigger": trigger,
                "draft_version": draft_version,
                "message": "Test run queued",
                "started_at": None,
                "completed_at": None,
            }
            self._append_test_event_to_session(
                session_payload,
                event="test_status",
                payload=queued_payload,
            )
            events_to_emit.append(("test_status", queued_payload))
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)

        for event, payload in events_to_emit:
            await self._broadcast_test_event(stream_key, event, payload)

        task = asyncio.create_task(
            self._execute_test_run(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                action=action,
                trigger=trigger,
                test_name=test_name,
                access_token=access_token,
            )
        )
        self._test_run_tasks[stream_key] = task

        def _cleanup(done_task: asyncio.Task) -> None:
            current = self._test_run_tasks.get(stream_key)
            if current is done_task:
                self._test_run_tasks.pop(stream_key, None)

        task.add_done_callback(_cleanup)
        return {"status": "queued", "run_id": run_id, "trigger": trigger}

    async def _commit_test_action_payload(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: str,
        updated_payload: Dict[str, Any],
        update_mode: str,
    ) -> Optional[int]:
        lock = self._test_lock(stream_key)
        async with lock:
            session_payload = self._load_session(
                scope=scope, ui_id=ui_id, name=name, session_id=session_id
            )
            if str(session_payload.get("test_run_id") or "") != run_id:
                return None
            session_payload["draft_payload"] = updated_payload
            session_payload["updated_at"] = self._now()
            session_payload["draft_version"] = (
                int(session_payload.get("draft_version") or 1) + 1
            )
            draft_version = int(session_payload["draft_version"])
            ui_payload = {
                "session_id": session_id,
                "draft_version": draft_version,
                "update_mode": update_mode,
            }
            self._append_test_event_to_session(
                session_payload,
                event="ui_updated",
                payload=ui_payload,
            )
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)
        await self._broadcast_test_event(stream_key, "ui_updated", ui_payload)
        return draft_version

    async def _emit_tool_or_test_event(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: str,
        event_data: Dict[str, Any],
    ) -> None:
        event = str(event_data.get("event") or "").strip()
        if event not in {"tool_start", "test_result"}:
            return
        payload = dict(event_data)
        payload.pop("event", None)
        payload["run_id"] = run_id
        if event == "tool_start":
            why = payload.get("why") or payload.get("fix_explanation")
            if why:
                payload["fix_explanation"] = str(why)
                payload["why"] = str(why)
                logger.info(
                    "[GeneratedUI] Test run %s why for tool %s: %s",
                    run_id,
                    payload.get("tool") or "unknown",
                    self._trim_output(str(why), limit=600),
                )
        await self._append_test_event(
            scope=scope,
            ui_id=ui_id,
            name=name,
            session_id=session_id,
            stream_key=stream_key,
            event=event,
            payload=payload,
            run_id=run_id,
        )

    def _delete_tests_by_name(self, test_script: str, test_names: Sequence[str]) -> str:
        updated = test_script or ""
        for test_name in test_names:
            clean_name = (test_name or "").strip()
            if not clean_name:
                continue
            escaped = re.escape(clean_name)
            block_pattern = re.compile(
                rf"(?ms)^\s*(?:test|it)\(\s*(['\"])({escaped})\1\s*,\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{{.*?^\s*\}}\s*\);\s*"
            )
            single_line_pattern = re.compile(
                rf"(?m)^\s*(?:test|it)\(\s*(['\"])({escaped})\1\s*,.*\);\s*$"
            )
            updated = block_pattern.sub("", updated)
            updated = single_line_pattern.sub("", updated)
        return updated

    def _append_named_test(self, test_script: str, test_name: str) -> str:
        clean_name = (test_name or "").strip()
        if not clean_name:
            return test_script or ""
        existing_pattern = re.compile(
            rf"(?:test|it)\(\s*(['\"]){re.escape(clean_name)}\1\s*,"
        )
        base = test_script or ""
        if existing_pattern.search(base):
            return base

        if "from 'node:test'" not in base and 'from "node:test"' not in base:
            base = "import { test } from 'node:test';\n" + base
        elif "from 'node:test'" in base or 'from "node:test"' in base:
            import_pattern = re.compile(
                r"import\s*\{([^}]*)\}\s*from\s*['\"]node:test['\"];?"
            )
            match = import_pattern.search(base)
            if match:
                symbols = [
                    part.strip() for part in match.group(1).split(",") if part.strip()
                ]
                if "test" not in symbols:
                    symbols.append("test")
                    replacement = (
                        "import { "
                        + ", ".join(sorted(set(symbols)))
                        + " } from 'node:test';"
                    )
                    base = base[: match.start()] + replacement + base[match.end() :]

        stub = (
            f"\n\ntest({json.dumps(clean_name)}, async () => {{\n"
            "  // User-requested test placeholder.\n"
            "});\n"
        )
        return base.rstrip() + stub

    def _trim_runtime_text(
        self, value: Any, limit: int = MAX_RUNTIME_CONTEXT_TEXT
    ) -> str:
        if value is None:
            return ""
        text = str(value)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...[truncated]"

    def _sanitize_runtime_value(self, value: Any, depth: int = 0) -> Any:
        if value is None:
            return None
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._trim_runtime_text(value)
        if depth >= MAX_RUNTIME_CONTEXT_DEPTH:
            with contextlib.suppress(Exception):
                return self._trim_runtime_text(
                    json.dumps(value, ensure_ascii=False, default=str)
                )
            return self._trim_runtime_text(value)
        if isinstance(value, list):
            return [
                self._sanitize_runtime_value(item, depth + 1)
                for item in value[:MAX_RUNTIME_CONTEXT_ENTRIES]
            ]
        if isinstance(value, dict):
            cleaned: Dict[str, Any] = {}
            for idx, (key, item) in enumerate(value.items()):
                if idx >= 40:
                    break
                cleaned[str(key)] = self._sanitize_runtime_value(item, depth + 1)
            return cleaned
        return self._trim_runtime_text(value)

    def _sanitize_runtime_action(
        self, draft_action: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(draft_action, dict):
            return None
        action_type = self._trim_runtime_text(draft_action.get("type"))
        if action_type != "runtime_service_exchanges":
            return None

        raw_entries = draft_action.get("entries")
        if not isinstance(raw_entries, list):
            raw_entries = []

        entries: List[Dict[str, Any]] = []
        for raw in raw_entries[:MAX_RUNTIME_CONTEXT_ENTRIES]:
            if not isinstance(raw, dict):
                continue
            tool = self._trim_runtime_text(raw.get("tool"), limit=160)
            error = self._trim_runtime_text(raw.get("error"))
            request_body = self._sanitize_runtime_value(raw.get("request_body"))
            request_options = self._sanitize_runtime_value(raw.get("request_options"))
            response_payload = self._sanitize_runtime_value(raw.get("response_payload"))
            cursor_raw = raw.get("cursor")
            timestamp_raw = raw.get("timestamp")

            cursor: Optional[int]
            try:
                parsed_cursor = int(cursor_raw)
                cursor = parsed_cursor if parsed_cursor >= 0 else None
            except (TypeError, ValueError):
                cursor = None

            timestamp: Optional[Union[int, str]] = None
            if isinstance(timestamp_raw, (int, float)):
                timestamp = int(timestamp_raw)
            elif timestamp_raw is not None:
                timestamp = self._trim_runtime_text(timestamp_raw, limit=80)

            if not tool and not error and response_payload is None:
                continue

            entry: Dict[str, Any] = {
                "tool": tool,
                "request_body": request_body,
                "request_options": request_options,
                "response_payload": response_payload,
                "error": error,
                "mocked": bool(raw.get("mocked")),
            }
            if cursor is not None:
                entry["cursor"] = cursor
            if timestamp is not None:
                entry["timestamp"] = timestamp
            entries.append(entry)

        raw_console_events = draft_action.get("console_events")
        if not isinstance(raw_console_events, list):
            raw_console_events = []
        console_events: List[Dict[str, Any]] = []
        for raw_event in raw_console_events[:MAX_RUNTIME_CONSOLE_EVENTS]:
            if not isinstance(raw_event, dict):
                continue
            kind = self._trim_runtime_text(raw_event.get("kind"), limit=80)
            message = self._trim_runtime_text(raw_event.get("message"))
            stack = self._trim_runtime_text(raw_event.get("stack"))
            filename = self._trim_runtime_text(raw_event.get("filename"), limit=200)
            line_value = raw_event.get("line")
            column_value = raw_event.get("column")
            timestamp_value = raw_event.get("timestamp")
            line: Optional[int] = None
            column: Optional[int] = None
            timestamp: Optional[int] = None
            with contextlib.suppress(TypeError, ValueError):
                parsed = int(line_value)
                line = parsed if parsed >= 0 else None
            with contextlib.suppress(TypeError, ValueError):
                parsed = int(column_value)
                column = parsed if parsed >= 0 else None
            with contextlib.suppress(TypeError, ValueError):
                parsed = int(timestamp_value)
                timestamp = parsed if parsed >= 0 else None
            if not kind and not message and not stack:
                continue
            event: Dict[str, Any] = {
                "kind": kind,
                "message": message,
                "stack": stack,
                "filename": filename,
            }
            if line is not None:
                event["line"] = line
            if column is not None:
                event["column"] = column
            if timestamp is not None:
                event["timestamp"] = timestamp
            console_events.append(event)

        if not entries and not console_events:
            return None

        cursor_raw = draft_action.get("cursor")
        cursor = 0
        with contextlib.suppress(TypeError, ValueError):
            cursor = max(0, int(cursor_raw))

        captured_at = self._trim_runtime_text(draft_action.get("captured_at"), limit=80)
        return {
            "type": action_type,
            "cursor": cursor,
            "captured_at": captured_at,
            "entries": entries,
            "console_events": console_events,
        }

    def _runtime_context_for_prompt(
        self, runtime_context: Optional[Dict[str, Any]], *, limit: int = 8
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(runtime_context, dict):
            return None
        entries = runtime_context.get("entries")
        console_events = runtime_context.get("console_events")

        prompt_entries: List[Dict[str, Any]] = []
        if isinstance(entries, list):
            for entry in entries[-limit:]:
                if not isinstance(entry, dict):
                    continue
                prompt_entries.append(
                    {
                        "tool": entry.get("tool"),
                        "request_body": entry.get("request_body"),
                        "request_options": entry.get("request_options"),
                        "response_payload": entry.get("response_payload"),
                        "error": entry.get("error"),
                        "mocked": bool(entry.get("mocked")),
                    }
                )

        prompt_console_events: List[Dict[str, Any]] = []
        if isinstance(console_events, list):
            for event in console_events[-MAX_RUNTIME_CONSOLE_EVENTS:]:
                if not isinstance(event, dict):
                    continue
                prompt_console_events.append(
                    {
                        "kind": event.get("kind"),
                        "message": event.get("message"),
                        "stack": event.get("stack"),
                        "filename": event.get("filename"),
                        "line": event.get("line"),
                        "column": event.get("column"),
                    }
                )

        payload: Dict[str, Any] = {}
        if prompt_entries:
            payload["service_exchanges"] = prompt_entries
        if prompt_console_events:
            payload["console_events"] = prompt_console_events
        return payload or None

    def _prompt_with_runtime_context(
        self,
        *,
        prompt: str,
        runtime_context: Optional[Dict[str, Any]],
        purpose: str,
    ) -> str:
        prompt_entries = self._runtime_context_for_prompt(runtime_context)
        if not prompt_entries:
            return prompt

        guidance = (
            "Use these observed runtime service exchanges to align data shapes, edge cases, "
            "and test realism."
        )
        if purpose == "dummy_data":
            guidance = (
                "Use these observed runtime service exchanges to shape realistic dummy data. "
                "Match field names and nesting from observed response_payload values. "
                "Also incorporate console warnings/errors as edge-case hints."
            )

        return (
            f"{prompt}\n\n"
            "Observed runtime context (single-use for this request):\n"
            f"{json.dumps(prompt_entries, ensure_ascii=False)}\n\n"
            f"{guidance}"
        )

    def _to_chat_history_messages(
        self, history: Sequence[Dict[str, Any]]
    ) -> List[Message]:
        messages: List[Message] = []
        for item in history or []:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "")
            if role == MessageRole.USER.value:
                messages.append(Message(role=MessageRole.USER, content=content))
            elif role == MessageRole.ASSISTANT.value:
                messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        return messages

    async def _execute_strategy_fix_action(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: str,
        session_payload: Dict[str, Any],
        action: TestAction,
        access_token: Optional[str],
    ) -> Tuple[bool, str, int, int, List[str], Optional[Dict[str, Any]]]:
        draft_payload = copy.deepcopy(session_payload.get("draft_payload", {}))
        service_script = str(draft_payload.get("service_script") or "")
        components_script = str(draft_payload.get("components_script") or "")
        test_script = str(draft_payload.get("test_script") or "")
        dummy_data = draft_payload.get("dummy_data")
        history_messages = self._to_chat_history_messages(
            session_payload.get("messages", [])
        )

        event_queue: asyncio.Queue = asyncio.Queue()
        strategy_mode: Literal["fix_code", "adjust_test"]
        strategy_mode = "fix_code" if action == "fix_code" else "adjust_test"
        fix_task = asyncio.create_task(
            run_tool_driven_test_fix(
                tgi_service=self.tgi_service,
                service_script=service_script,
                components_script=components_script,
                test_script=test_script,
                dummy_data=dummy_data,
                messages=history_messages,
                allowed_tools=list(session_payload.get("last_tools") or []),
                access_token=access_token,
                max_attempts=15,
                event_queue=event_queue,
                extra_headers=UI_MODEL_HEADERS,
                strategy_mode=strategy_mode,
            )
        )
        try:
            while not fix_task.done():
                try:
                    event_data = event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)
                    continue
                await self._emit_tool_or_test_event(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    session_id=session_id,
                    stream_key=stream_key,
                    run_id=run_id,
                    event_data=event_data,
                )
            (
                fix_success,
                fixed_service,
                fixed_components,
                fixed_test,
                fixed_dummy_data,
                _updated_messages,
            ) = await fix_task
        except asyncio.CancelledError:
            fix_task.cancel()
            raise

        while not event_queue.empty():
            event_data = event_queue.get_nowait()
            await self._emit_tool_or_test_event(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                event_data=event_data,
            )

        final_success, final_output = await asyncio.to_thread(
            self._run_tests,
            fixed_service,
            fixed_components,
            fixed_test,
            fixed_dummy_data,
            None,
        )
        passed, failed, failed_tests = _parse_tap_output(final_output or "")
        updated_payload = None
        if fix_success and action in {"fix_code", "adjust_test"}:
            updated_payload = copy.deepcopy(draft_payload)
            updated_payload["service_script"] = fixed_service
            updated_payload["components_script"] = fixed_components
            updated_payload["test_script"] = fixed_test
            updated_payload["dummy_data"] = fixed_dummy_data
        return (
            final_success,
            final_output,
            int(passed),
            int(failed),
            list(failed_tests),
            updated_payload,
        )

    async def _execute_test_run(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        session_id: str,
        stream_key: str,
        run_id: str,
        action: TestAction,
        trigger: str,
        test_name: Optional[str],
        access_token: Optional[str],
    ) -> None:
        started_at = self._now()
        await self._update_test_state(
            scope=scope,
            ui_id=ui_id,
            name=name,
            session_id=session_id,
            stream_key=stream_key,
            run_id=run_id,
            state="running",
            trigger=trigger,
            message="Running tests...",
            summary_updates={"started_at": started_at, "completed_at": None},
        )

        try:
            session_payload = self._load_session(
                scope=scope, ui_id=ui_id, name=name, session_id=session_id
            )
            if str(session_payload.get("test_run_id") or "") != run_id:
                return

            draft_payload = copy.deepcopy(session_payload.get("draft_payload", {}))
            service_script = str(draft_payload.get("service_script") or "")
            components_script = str(draft_payload.get("components_script") or "")
            test_script = str(draft_payload.get("test_script") or "")
            dummy_data = draft_payload.get("dummy_data")

            if not components_script or not test_script:
                completed_at = self._now()
                await self._update_test_state(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    session_id=session_id,
                    stream_key=stream_key,
                    run_id=run_id,
                    state="error",
                    trigger=trigger,
                    message="Draft is missing components_script or test_script",
                    completed_at=completed_at,
                    summary_updates={
                        "started_at": started_at,
                        "completed_at": completed_at,
                    },
                )
                return

            updated_payload: Optional[Dict[str, Any]] = None
            success = False
            output = ""
            passed = 0
            failed = 0
            failed_tests: List[str] = []

            if action == "run":
                success, output = await asyncio.to_thread(
                    self._run_tests,
                    service_script,
                    components_script,
                    test_script,
                    dummy_data,
                    test_name,
                )
                passed, failed, failed_tests = _parse_tap_output(output or "")
            elif action in {"fix_code", "adjust_test"}:
                (
                    success,
                    output,
                    passed,
                    failed,
                    failed_tests,
                    updated_payload,
                ) = await self._execute_strategy_fix_action(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    session_id=session_id,
                    stream_key=stream_key,
                    run_id=run_id,
                    session_payload=session_payload,
                    action=action,
                    access_token=access_token,
                )
            elif action == "delete_test":
                target_tests = list(
                    (session_payload.get("test_summary") or {}).get("failed_tests")
                    or []
                )
                if test_name and test_name.strip():
                    target_tests = [test_name.strip()]
                updated_test_script = self._delete_tests_by_name(
                    test_script, target_tests
                )
                if updated_test_script == test_script:
                    output = "No matching tests found to delete."
                    success = False
                else:
                    updated_payload = copy.deepcopy(draft_payload)
                    updated_payload["test_script"] = updated_test_script
                    success, output = await asyncio.to_thread(
                        self._run_tests,
                        service_script,
                        components_script,
                        updated_test_script,
                        dummy_data,
                        None,
                    )
                    passed, failed, failed_tests = _parse_tap_output(output or "")
            elif action == "add_test":
                if not (test_name or "").strip():
                    output = "test_name is required for add_test action."
                    success = False
                else:
                    updated_test_script = self._append_named_test(
                        test_script, test_name or ""
                    )
                    updated_payload = copy.deepcopy(draft_payload)
                    updated_payload["test_script"] = updated_test_script
                    success, output = await asyncio.to_thread(
                        self._run_tests,
                        service_script,
                        components_script,
                        updated_test_script,
                        dummy_data,
                        None,
                    )
                    passed, failed, failed_tests = _parse_tap_output(output or "")

            if updated_payload is not None:
                await self._commit_test_action_payload(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    session_id=session_id,
                    stream_key=stream_key,
                    run_id=run_id,
                    updated_payload=updated_payload,
                    update_mode=f"tests_{action}",
                )

            result_payload = {
                "run_id": run_id,
                "status": "passed" if success else "failed",
                "passed": int(passed),
                "failed": int(failed),
                "failed_tests": list(failed_tests),
                "message": "All tests passed" if success else "Tests failed",
            }
            await self._append_test_event(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                event="test_result",
                payload=result_payload,
                run_id=run_id,
            )

            output_tail = self._trim_output(output)
            await self._append_test_event(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                event="test_output",
                payload={"run_id": run_id, "output_tail": output_tail},
                run_id=run_id,
            )

            completed_at = self._now()
            await self._update_test_state(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                state="passed" if success else "failed",
                trigger=trigger,
                message="Tests passing" if success else "Tests failing",
                completed_at=completed_at,
                summary_updates={
                    "passed": int(passed),
                    "failed": int(failed),
                    "failed_tests": list(failed_tests),
                    "started_at": started_at,
                    "completed_at": completed_at,
                },
            )
        except asyncio.CancelledError:
            completed_at = self._now()
            await self._update_test_state(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                state="cancelled",
                trigger=trigger,
                message="Test run cancelled",
                completed_at=completed_at,
                summary_updates={"completed_at": completed_at},
                skip_if_run_mismatch=False,
            )
            raise
        except Exception as exc:
            logger.error("[GeneratedUI] Test run failed: %s", exc, exc_info=exc)
            completed_at = self._now()
            await self._update_test_state(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                stream_key=stream_key,
                run_id=run_id,
                state="error",
                trigger=trigger,
                message="Test run errored",
                completed_at=completed_at,
                summary_updates={
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "error": str(exc),
                },
            )

    async def stream_chat_update(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        actor: Actor,
        ui_id: str,
        name: str,
        session_id: str,
        message: str,
        tools: Optional[Sequence[str]],
        tool_choice: Optional[Any],
        draft_action: Optional[Dict[str, Any]] = None,
        access_token: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        if not message.strip():
            yield _sse_event("error", {"error": "message must not be empty"})
            return

        try:
            session_payload = self._load_session(
                scope=scope, ui_id=ui_id, name=name, session_id=session_id
            )
            self._assert_session_owner(session_payload, actor)
            runtime_context = self._sanitize_runtime_action(draft_action)

            requested_tools = (
                list(tools)
                if tools is not None
                else list(session_payload.get("last_tools") or [])
            )
            selected_tools = await self._select_tools(session, requested_tools, message)
            selected_tool_names = [
                t.get("function", {}).get("name")
                for t in (selected_tools or [])
                if isinstance(t, dict)
            ]
            selected_tool_names = [name for name in selected_tool_names if name]

            draft_payload = copy.deepcopy(session_payload.get("draft_payload", {}))
            assistant_text = await self._run_assistant_message(
                session=session,
                draft_payload=draft_payload,
                history=session_payload.get("messages", []),
                user_message=message,
                selected_tools=selected_tools,
                tool_choice=tool_choice,
                runtime_context=runtime_context,
                access_token=access_token,
            )
            if assistant_text:
                yield _sse_event("assistant", {"delta": assistant_text})

            updated_payload: Optional[Dict[str, Any]] = None
            update_mode = "regenerated_fallback"
            patch_error: Optional[str] = None

            patch_enabled = os.environ.get(
                "APP_UI_PATCH_ENABLED", "true"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if patch_enabled:
                patch_attempt = await self._attempt_patch_update(
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    draft_payload=draft_payload,
                    user_message=message,
                    assistant_message=assistant_text,
                    access_token=access_token,
                    previous_metadata=session_payload.get("metadata_snapshot", {}),
                )
                if patch_attempt:
                    updated_payload = patch_attempt.get("payload")
                    update_mode = "patch_applied"
                else:
                    patch_error = "Patch validation failed, using full regenerate"

            if updated_payload is None:
                previous = {
                    "metadata": copy.deepcopy(
                        session_payload.get("metadata_snapshot", {}) or {}
                    ),
                    "current": draft_payload,
                }
                regenerate_prompt = self._compose_regeneration_prompt(
                    user_message=message,
                    assistant_message=assistant_text,
                    history=session_payload.get("messages", []),
                    runtime_context=runtime_context,
                )
                updated_payload = await self._generate_ui_payload(
                    session=session,
                    scope=scope,
                    ui_id=ui_id,
                    name=name,
                    prompt=regenerate_prompt,
                    tools=selected_tool_names,
                    access_token=access_token,
                    previous=previous,
                    runtime_context=runtime_context,
                )

            messages_history = list(session_payload.get("messages") or [])
            messages_history.append({"role": "user", "content": message})
            messages_history.append({"role": "assistant", "content": assistant_text})

            session_payload["messages"] = messages_history
            session_payload["draft_payload"] = updated_payload
            session_payload["last_tools"] = selected_tool_names
            session_payload["updated_at"] = self._now()
            session_payload["draft_version"] = (
                int(session_payload.get("draft_version") or 1) + 1
            )
            self.storage.write_session(scope, ui_id, name, session_id, session_payload)

            ui_event_payload = {
                "session_id": session_id,
                "draft_version": session_payload["draft_version"],
                "update_mode": update_mode,
                "tools": selected_tool_names,
            }
            if patch_error:
                ui_event_payload["warning"] = patch_error
            yield _sse_event("ui_updated", ui_event_payload)
            queued = await self._queue_test_run(
                scope=scope,
                ui_id=ui_id,
                name=name,
                session_id=session_id,
                action="run",
                trigger="post_update",
                test_name=None,
                access_token=access_token,
            )
            yield _sse_event(
                "tests_queued",
                {
                    "run_id": queued.get("run_id"),
                    "trigger": queued.get("trigger"),
                    "draft_version": session_payload["draft_version"],
                },
            )
            yield _sse_event(
                "done",
                {
                    "session_id": session_id,
                    "draft_version": session_payload["draft_version"],
                    "update_mode": update_mode,
                },
            )
            yield b"data: [DONE]\n\n"
        except HTTPException as exc:
            yield _sse_event("error", {"error": exc.detail})
        except Exception as exc:
            logger.error(
                "[GeneratedUI] Conversational update failed: %s", exc, exc_info=exc
            )
            yield _sse_event(
                "error", {"error": "Failed to process conversational update"}
            )
        finally:
            runtime_context = None

    def _compose_regeneration_prompt(
        self,
        *,
        user_message: str,
        assistant_message: str,
        history: Sequence[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        tail = list(history or [])[-6:]
        prompt = (
            "Update the existing UI using the conversational request.\n\n"
            f"User request:\n{user_message}\n\n"
            f"Assistant analysis:\n{assistant_message}\n\n"
            "Recent conversation history (JSON):\n"
            f"{json.dumps(tail, ensure_ascii=False)}\n\n"
            "Component data-loading constraints:\n"
            "- Keep data fetching component-owned by default.\n"
            "- Avoid root-level fan-out Promise.all() across unrelated UI blocks.\n"
            "- Avoid a single global loading gate that blocks the whole screen when sections are independent.\n"
            "- Keep per-component loading/error/data states with local placeholders.\n"
            "- Use namespaced public events plus targeted refetch in affected components only.\n"
            "- In runtime catch blocks, always log with console.error including component/service context.\n"
            "- Keep tests deterministic with mocked service/fetch responses and never seed fetched domain data directly via test state/event payloads."
        )
        return self._prompt_with_runtime_context(
            prompt=prompt,
            runtime_context=runtime_context,
            purpose="regeneration",
        )

    async def _run_assistant_message(
        self,
        *,
        session: MCPSessionBase,
        draft_payload: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        user_message: str,
        selected_tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any],
        runtime_context: Optional[Dict[str, Any]] = None,
        access_token: Optional[str],
    ) -> str:
        system_prompt = (
            "You are an assistant helping a user iteratively edit a generated web UI. "
            "Keep answers concise and implementation-focused. If tools are available, "
            "use them to gather facts before proposing UI changes. "
            "When this workflow will continue automatically, do not ask for permission to proceed. "
            "State the next action assertively with phrasing that starts with 'I will ...'. "
            "Preserve component-owned data loading and partial rendering: each data-owning "
            "component should manage its own loading/error/data states. "
            "Do not introduce root-level Promise.all() fan-out for unrelated components or a "
            "global blocking loading state unless the whole view is intentionally atomic. "
            "Use namespaced public events for targeted refetch in affected components only. "
            "In runtime catch blocks, always log with console.error and include useful context. "
            "Keep tests deterministic with mocked service/fetch responses and avoid direct test seeding of fetched domain data. "
            "If runtime service exchange context is provided, use it to narrow solutioning, "
            "match real payload shapes, and improve proposed test data quality."
        )
        messages: List[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt)
        ]
        messages.append(
            Message(
                role=MessageRole.USER,
                content=(
                    "Current draft context:\n"
                    + json.dumps(
                        {
                            "html": (draft_payload.get("html") or {}),
                            "metadata": (draft_payload.get("metadata") or {}),
                        },
                        ensure_ascii=False,
                    )
                ),
            )
        )
        runtime_prompt_context = self._runtime_context_for_prompt(runtime_context)
        if runtime_prompt_context:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        "Observed runtime context for this request only:\n"
                        + json.dumps(runtime_prompt_context, ensure_ascii=False)
                    ),
                )
            )
        for item in history or []:
            role = str(item.get("role") or "").lower()
            content = str(item.get("content") or "")
            if role == MessageRole.USER.value:
                messages.append(Message(role=MessageRole.USER, content=content))
            elif role == MessageRole.ASSISTANT.value:
                messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        messages.append(Message(role=MessageRole.USER, content=user_message))

        request = ChatCompletionRequest(
            messages=messages,
            stream=False,
            tools=selected_tools if selected_tools else None,
            tool_choice=tool_choice if tool_choice is not None else "auto",
            extra_headers=UI_MODEL_HEADERS,
        )

        response = await self.tgi_service._non_stream_chat_with_tools(
            session,
            messages,
            selected_tools or [],
            request,
            access_token,
            None,
        )
        return self._assistant_text_from_response(response)

    def _assistant_text_from_response(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message and getattr(message, "content", None):
                return str(message.content)

        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                content = message.get("content")
                if content is not None:
                    return str(content)
        return ""

    async def _attempt_patch_update(
        self,
        *,
        scope: Scope,
        ui_id: str,
        name: str,
        draft_payload: Dict[str, Any],
        user_message: str,
        assistant_message: str,
        access_token: Optional[str],
        previous_metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            system_prompt = (
                "You are a UI patch planner. Return valid JSON only in this shape: "
                '{"patch":{"html":{"page":"...","snippet":"..."},"service_script":"...","components_script":"...","metadata":{...}}}. '
                "Only include fields that need changes. Do not include markdown fences. "
                "Preserve component-owned data loading and partial rendering. Do not rewrite to "
                "one root-level Promise.all() fan-out or a single full-screen blocking loader "
                "for independent components. Keep targeted event-driven refetch behavior. "
                "In runtime catch blocks, always log with console.error and include component/service context. "
                "Keep tests deterministic with mocked service/fetch responses and avoid test-only seeding of fetched domain data."
            )
            payload = {
                "user_message": user_message,
                "assistant_message": assistant_message,
                "current": {
                    "html": draft_payload.get("html"),
                    "service_script": draft_payload.get("service_script"),
                    "components_script": draft_payload.get("components_script"),
                    "metadata": draft_payload.get("metadata"),
                },
            }
            request = ChatCompletionRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content=system_prompt),
                    Message(
                        role=MessageRole.USER,
                        content=json.dumps(payload, ensure_ascii=False),
                    ),
                ],
                stream=False,
                extra_headers=UI_MODEL_HEADERS,
            )

            response = await self.tgi_service.llm_client.non_stream_completion(
                request, access_token or "", None
            )
            content = self._assistant_text_from_response(response)
            if not content:
                return None

            parsed = self._parse_json(content)
            patch = parsed.get("patch")
            if not isinstance(patch, dict):
                return None

            candidate = copy.deepcopy(draft_payload)
            html_patch = patch.get("html")
            if isinstance(html_patch, dict):
                html_target = candidate.setdefault("html", {})
                for key in ("page", "snippet"):
                    value = html_patch.get(key)
                    if isinstance(value, str) and value.strip():
                        html_target[key] = value

            for key in (
                "service_script",
                "components_script",
                "test_script",
                "dummy_data",
            ):
                value = patch.get(key)
                if isinstance(value, str):
                    candidate[key] = value

            metadata_patch = patch.get("metadata")
            if isinstance(metadata_patch, dict):
                current_metadata = candidate.get("metadata")
                if not isinstance(current_metadata, dict):
                    current_metadata = {}
                merged_metadata = {**current_metadata, **metadata_patch}
                candidate["metadata"] = merged_metadata

            previous = {"metadata": previous_metadata, "current": draft_payload}
            self._normalise_payload(
                candidate,
                scope,
                ui_id,
                name,
                user_message,
                previous,
            )

            test_script = candidate.get("test_script") or draft_payload.get(
                "test_script"
            )
            service_script = candidate.get("service_script") or draft_payload.get(
                "service_script"
            )
            components_script = candidate.get("components_script") or draft_payload.get(
                "components_script"
            )
            if (
                isinstance(service_script, str)
                and isinstance(components_script, str)
                and isinstance(test_script, str)
                and test_script.strip()
            ):
                success, _ = self._run_tests(
                    service_script,
                    components_script,
                    test_script,
                    candidate.get("dummy_data"),
                )
                if not success:
                    return None

            return {"payload": candidate}
        except Exception:
            return None

    async def _generate_ui_payload(
        self,
        *,
        session: MCPSessionBase,
        scope: Scope,
        ui_id: str,
        name: str,
        prompt: str,
        tools: List[str],
        access_token: Optional[str],
        previous: Optional[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        system_prompt = await self._build_system_prompt(session)
        prompt_with_runtime = self._prompt_with_runtime_context(
            prompt=prompt,
            runtime_context=runtime_context,
            purpose="generation",
        )
        message_payload = {
            "ui": {
                "id": ui_id,
                "name": name,
                "scope": {"type": scope.kind, "id": scope.identifier},
            },
            "request": {
                "prompt": prompt_with_runtime,
                "tools": tools,
            },
        }

        if previous:
            previous_metadata = previous.get("metadata", {})
            message_payload["context"] = {
                "original_prompt": self._initial_prompt(previous_metadata),
                "history": self._history_for_prompt(
                    previous_metadata.get("history", [])
                ),
                "current_state": previous.get("current", {}),
            }
        runtime_prompt_context = self._runtime_context_for_prompt(runtime_context)
        if runtime_prompt_context:
            context_obj = message_payload.setdefault("context", {})
            if runtime_prompt_context.get("service_exchanges"):
                context_obj["runtime_service_exchanges"] = runtime_prompt_context.get(
                    "service_exchanges"
                )
            if runtime_prompt_context.get("console_events"):
                context_obj["runtime_console_events"] = runtime_prompt_context.get(
                    "console_events"
                )

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(
                role=MessageRole.USER,
                content=json.dumps(message_payload, ensure_ascii=False),
            ),
        ]

        allowed_tools = await self._select_tools(session, tools, prompt)

        dummy_data = await self._generate_dummy_data(
            session=session,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt_with_runtime,
            allowed_tools=allowed_tools,
            access_token=access_token,
            runtime_context=runtime_context,
        )
        messages.append(
            Message(
                role=MessageRole.USER,
                content=(
                    f"{_DUMMY_DATA_TEST_USAGE_GUIDANCE} "
                    f"Tools: {[t['function']['name'] for t in (allowed_tools or [])]}"
                ),
            )
        )

        chat_request = ChatCompletionRequest(
            messages=messages,
            tools=allowed_tools if allowed_tools else None,
            stream=True,
            response_format=_generation_response_format(),
            extra_headers=UI_MODEL_HEADERS,
        )
        self._maybe_dump_chat_request(
            chat_request=chat_request,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            tools=tools,
            message_payload=message_payload,
        )

        # Use streaming to collect the response
        content = ""
        stream_source = self.tgi_service.llm_client.stream_completion(
            chat_request, access_token or "", None
        )

        async with chunk_reader(stream_source) as reader:
            async for parsed in reader.as_parsed():
                if parsed.is_done:
                    break
                if parsed.content:
                    content += parsed.content

        if not content:
            raise HTTPException(status_code=502, detail="Generation response was empty")

        payload = self._parse_json(content)
        payload["dummy_data"] = payload.get("dummy_data") or dummy_data
        self._normalise_payload(payload, scope, ui_id, name, prompt, previous)
        return payload

    async def _build_system_prompt(self, session: MCPSessionBase) -> str:
        """Build the legacy all-in-one generation prompt used outside phase 1/2 flow."""
        prompt_service = self.tgi_service.prompt_service
        design_prompt_content = ""
        try:
            design_prompt = await prompt_service.find_prompt_by_name_or_role(
                session, prompt_name="design-system"
            )
            if design_prompt:
                design_prompt_content = await prompt_service.get_prompt_content(
                    session, design_prompt
                )
        except Exception:
            design_prompt_content = ""

        combined_design = design_prompt_content or DEFAULT_DESIGN_PROMPT

        pfusch_prompt = _load_pfusch_prompt()
        return pfusch_prompt.replace("{{DESIGN_SYSTEM_PROMPT}}", combined_design)

    async def _build_phase2_system_prompt(self, session: MCPSessionBase) -> str:
        prompt_service = self.tgi_service.prompt_service
        design_prompt_content = ""
        try:
            design_prompt = await prompt_service.find_prompt_by_name_or_role(
                session, prompt_name="design-system"
            )
            if design_prompt:
                design_prompt_content = await prompt_service.get_prompt_content(
                    session, design_prompt
                )
        except Exception:
            design_prompt_content = ""

        combined_design = design_prompt_content or DEFAULT_DESIGN_PROMPT

        # Load the pfusch phase 2 prompt from file and replace the design system placeholder
        pfusch_prompt = _load_pfusch_phase2_prompt()
        return pfusch_prompt.replace("{{DESIGN_SYSTEM_PROMPT}}", combined_design)

    async def _select_tools(
        self, session: MCPSessionBase, requested_tools: Sequence[str], prompt: str = ""
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Select relevant tools for ui generation.

        If specific tools are requested, use those. Otherwise, intelligently
        filter tools based on the prompt to reduce context size.
        """
        # Get all tools with output schema for UI generation
        available = await self.tgi_service.tool_service.get_all_mcp_tools(
            session, include_output_schema=True
        )

        if not available:
            return None

        # If specific tools requested, filter to those
        if requested_tools:
            selected: List[Dict[str, Any]] = []
            for tool in available:
                tool_name: Optional[str] = None
                if isinstance(tool, dict):
                    tool_name = tool.get("function", {}).get("name")
                else:
                    tool_name = getattr(tool, "function", None)
                    if tool_name and hasattr(tool_name, "name"):
                        tool_name = tool_name.name
                if tool_name and tool_name in requested_tools:
                    selected.append(tool)
            return selected if selected else None

        # Otherwise, intelligently pre-select most relevant tools
        return self._filter_relevant_tools(available, prompt)

    def _filter_relevant_tools(
        self, tools: List[Dict[str, Any]], prompt: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Filter tools to most relevant ones based on prompt keywords.

        Uses simple keyword matching to reduce context size. If the prompt
        mentions specific domain terms, prioritize tools with those terms.
        """
        if not prompt or len(tools) <= 10:
            # If prompt is empty or tool count is manageable, return all
            return tools

        prompt_lower = prompt.lower()
        scored_tools = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            function = tool.get("function", {})
            name = function.get("name", "")
            description = function.get("description", "")

            # Skip the meta tool "describe_tool"
            if name == "describe_tool":
                scored_tools.append((tool, 100))  # Always include
                continue

            # Score based on keyword matches
            score = 0

            # Check if tool name appears in prompt
            if name.lower() in prompt_lower:
                score += 50

            # Check for partial name matches (e.g., "absence" matches "list_absence_types")
            name_parts = name.lower().replace("_", " ").split()
            for part in name_parts:
                if len(part) > 3 and part in prompt_lower:
                    score += 10

            # Check if description keywords appear in prompt
            desc_words = description.lower().replace("_", " ").split()
            for word in desc_words:
                if len(word) > 4 and word in prompt_lower:
                    score += 5

            # Prioritize list/get operations for uis
            if any(
                prefix in name.lower()
                for prefix in ["list_", "get_", "fetch_", "retrieve_"]
            ):
                score += 3

            # Deprioritize create/update/delete operations unless explicitly mentioned
            if any(
                prefix in name.lower()
                for prefix in ["create_", "update_", "delete_", "remove_"]
            ):
                if not any(
                    word in prompt_lower
                    for word in [
                        "create",
                        "update",
                        "delete",
                        "edit",
                        "modify",
                        "remove",
                    ]
                ):
                    score -= 10

            scored_tools.append((tool, score))

        # Sort by score descending
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        # Take top tools, ensuring we include describe_tool
        max_tools = 15  # Reasonable limit for context size
        selected = [tool for tool, score in scored_tools[:max_tools] if score > 0]

        # If we filtered too aggressively, include some more
        if len(selected) < 5 and len(scored_tools) > len(selected):
            selected = [tool for tool, score in scored_tools[:10]]

        logger.info(
            f"[GeneratedUI] Filtered {len(tools)} tools to {len(selected)} based on prompt relevance"
        )

        return selected if selected else tools

    def _extract_content(self, response: Any) -> str:
        if response is None:
            raise HTTPException(status_code=502, detail="Generation response was empty")

        if isinstance(response, dict):
            content = response.get("content")
            if isinstance(content, str):
                return content

        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message and getattr(message, "content", None):
                return message.content
            delta = getattr(first, "delta", None)
            if delta and getattr(delta, "content", None):
                return delta.content

        raise HTTPException(
            status_code=502,
            detail="Unable to extract content from generation response",
        )

    def _parse_json(self, payload_str: str) -> Dict[str, Any]:
        try:
            return json.loads(payload_str)
        except json.JSONDecodeError:
            candidate = self._extract_json_block(payload_str)
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=502,
                        detail="Generated content is not valid JSON",
                    ) from exc
            raise HTTPException(
                status_code=502,
                detail="Generated content is not valid JSON",
            )

    def _extract_json_block(self, text: str) -> Optional[str]:
        start = None
        depth = 0
        in_string = False
        escape = False
        for idx, char in enumerate(text):
            if start is None:
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

            if char == '"' and not escape:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : idx + 1]
        return None

    def _normalise_payload(
        self,
        payload: Dict[str, Any],
        scope: Scope,
        ui_id: str,
        name: str,
        prompt: str,
        previous: Optional[Dict[str, Any]],
    ) -> None:
        self.output_factory.normalise_payload(
            payload=payload,
            scope=scope,
            ui_id=ui_id,
            name=name,
            prompt=prompt,
            previous=previous,
        )

    def _maybe_dump_chat_request(
        self,
        *,
        chat_request: ChatCompletionRequest,
        scope: Scope,
        ui_id: str,
        name: str,
        prompt: str,
        tools: List[str],
        message_payload: Dict[str, Any],
    ) -> None:
        dump_target = GENERATED_UI_PROMPT_DUMP
        if not dump_target:
            return
        try:
            path = Path(dump_target)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

            def _safe(value: str) -> str:
                return re.sub(r"[^A-Za-z0-9_-]", "_", value or "")

            file_name = (
                f"{timestamp}_{_safe(scope.kind)}-{_safe(scope.identifier)}_"
                f"{_safe(ui_id)}_{_safe(name)}.json"
            )

            if path.exists() and path.is_dir():
                target_path = path / file_name
            elif path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                target_path = path
            else:
                path.mkdir(parents=True, exist_ok=True)
                target_path = path / file_name

            payload = {
                "timestamp": timestamp,
                "scope": {"type": scope.kind, "id": scope.identifier},
                "ui_id": ui_id,
                "name": name,
                "prompt": prompt,
                "requested_tools": tools,
                "message_payload": message_payload,
                "chat_request": chat_request.model_dump(exclude_none=True),
            }

            with target_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

            logger.info("[GeneratedUI] Chat request preview dumped to %s", target_path)
        except Exception as exc:  # pragma: no cover - debug helper
            logger.error(
                "[GeneratedUI] Failed to dump chat request preview: %s",
                exc,
                exc_info=exc,
            )

    def _extract_body(self, html: str) -> Optional[str]:
        return self.output_factory.extract_body(html)

    def _wrap_snippet(self, snippet: str) -> str:
        return self.output_factory.wrap_snippet(snippet)

    def _assert_scope_consistency(
        self, existing: Dict[str, Any], scope: Scope, name: str
    ) -> None:
        metadata = existing.get("metadata", {})
        stored_scope = metadata.get("scope", {})
        if (
            stored_scope.get("type") != scope.kind
            or stored_scope.get("id") != scope.identifier
        ):
            raise HTTPException(
                status_code=403,
                detail="Scope mismatch for stored ui",
            )
        if metadata.get("name") and metadata.get("name") != name:
            raise HTTPException(status_code=403, detail="Ui name mismatch")

    def _ensure_update_permissions(
        self, existing: Dict[str, Any], scope: Scope, actor: Actor
    ) -> None:
        if not actor.is_owner(scope):
            raise HTTPException(status_code=403, detail="Access denied for update")

    def _initial_prompt(self, metadata: Dict[str, Any]) -> Optional[str]:
        history = metadata.get("history") or []
        if history:
            first = history[0]
            if isinstance(first, dict):
                return first.get("prompt")
        return None

    def _changed_scripts(
        self,
        new_payload: Dict[str, Any],
        previous_payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        changed: Dict[str, Any] = {}
        previous_payload = previous_payload or {}
        for key in SCRIPT_KEYS:
            if key not in new_payload and key not in previous_payload:
                continue
            new_value = new_payload.get(key)
            old_value = previous_payload.get(key)
            if new_value != old_value:
                changed[key] = new_value
        return changed

    def _scripts_from_history(self, history_entries: Sequence[Any]) -> Dict[str, Any]:
        scripts: Dict[str, Any] = {}
        for entry in reversed(history_entries or []):
            if not isinstance(entry, dict):
                continue
            payload_scripts = entry.get("payload_scripts")
            if not isinstance(payload_scripts, dict):
                continue
            for key in SCRIPT_KEYS:
                if key in payload_scripts and key not in scripts:
                    scripts[key] = payload_scripts[key]
            if len(scripts) == len(SCRIPT_KEYS):
                break
        return scripts

    def _history_entry(
        self,
        *,
        action: str,
        prompt: str,
        tools: List[str],
        user_id: str,
        generated_at: str,
        payload_metadata: Dict[str, Any],
        payload_html: Optional[Dict[str, Any]] = None,
        payload_scripts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "action": action,
            "prompt": prompt,
            "tools": tools,
            "user_id": user_id,
            "generated_at": generated_at,
            "payload_metadata": payload_metadata,
            "payload_html": payload_html,
        }
        if payload_scripts is not None:
            entry["payload_scripts"] = payload_scripts
        return entry

    def _history_for_prompt(
        self, history_entries: Sequence[Any]
    ) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for entry in history_entries or []:
            if not isinstance(entry, dict):
                continue
            entry_copy = dict(entry)
            entry_copy.pop("payload_html", None)
            entry_copy.pop("payload_scripts", None)
            sanitized.append(entry_copy)
        return sanitized

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
