"""Shared wide-event telemetry for MCP operations.

Every handled ``tools/call``, ``resources/read`` and ``prompts/get`` request —
whether it arrives over the MCP SSE proxy or the REST compatibility routes —
must be routed through :func:`mcp_operation_span` so all transports emit the
same single canonical operation span and cannot drift.

Sensitive-content policy (unconditional): raw tool/prompt arguments, results,
resource bodies, credentials and untrusted exception messages never enter
span attributes, span events or log lines produced by this module. Only safe
metadata is recorded: argument key names/count, result item count, encoded
size, truncation state, bounded error classifications and exception type
names.
"""

import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import jwt
from opentelemetry import baggage, trace
from opentelemetry.propagate import inject
from opentelemetry.trace import SpanKind, Status, StatusCode

from app.utils import token_fingerprint
from app.vars import (
    AUTH_PROVIDER,
    MCP_REMOTE_SERVER,
    MCP_TRACE_BAGGAGE_ALLOWLIST,
)

logger = logging.getLogger("uvicorn.error")

_tracer = trace.get_tracer("app.utils.mcp_operation")

# Canonical MCP JSON-RPC method names covered by the wide-event seam.
MCP_METHOD_TOOLS_CALL = "tools/call"
MCP_METHOD_RESOURCES_READ = "resources/read"
MCP_METHOD_PROMPTS_GET = "prompts/get"

# gen_ai.operation.name values per method (OTel GenAI semantic conventions).
_GEN_AI_OPERATION_BY_METHOD = {
    MCP_METHOD_TOOLS_CALL: "execute_tool",
}

# Bounded error classification values for `error.type`. Free-text error
# messages are untrusted and must never widen this set.
ERROR_TYPE_VALIDATION = "validation_error"
ERROR_TYPE_AUTHORIZATION = "authorization_error"
ERROR_TYPE_UPSTREAM_TIMEOUT = "upstream_timeout"
ERROR_TYPE_DOWNSTREAM_EXECUTION = "downstream_execution_error"
ERROR_TYPE_RESPONSE_TOO_LARGE = "response_too_large"
ERROR_TYPE_BRIDGE_INTERNAL = "bridge_internal_error"

RESULT_STATUS_SUCCESS = "success"
RESULT_STATUS_ERROR = "error"

# Transport values for `enterprise_mcp_bridge.transport`.
TRANSPORT_REST = "rest"
TRANSPORT_SSE = "sse"

_HTTP_STATUS_TO_ERROR_TYPE = {
    401: ERROR_TYPE_AUTHORIZATION,
    403: ERROR_TYPE_AUTHORIZATION,
    504: ERROR_TYPE_UPSTREAM_TIMEOUT,
}


def safe_arg_keys(args: Optional[Dict]) -> list[str]:
    """Argument key names only — values are sensitive and never collected."""
    if not isinstance(args, dict):
        return []
    return sorted(str(k) for k in args.keys())


def extract_opaque_user_id(access_token: Optional[str]) -> Optional[str]:
    """Return the token's `sub` claim — an opaque stable ID — or None.

    Deliberately no fallback to preferred_username/email: those are PII and
    must never become the telemetry user id.
    """
    if not access_token:
        return None
    try:
        payload = jwt.decode(access_token, options={"verify_signature": False})
    except Exception:
        return None
    sub = payload.get("sub")
    return str(sub) if sub else None


def downstream_server_label() -> Optional[str]:
    """Bounded name of the configured downstream MCP server.

    Only hostname (and port) — never netloc or the raw URL, both of which can
    embed user:password@ credentials.
    """
    import os
    from urllib.parse import urlparse

    remote = (MCP_REMOTE_SERVER or "").strip()
    if remote:
        try:
            parsed = urlparse(remote)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return None
        if not hostname:
            return None
        return f"{hostname}:{port}" if port else hostname
    command = os.environ.get("MCP_SERVER_COMMAND", "").strip()
    if command:
        return command.split()[0].rsplit("/", 1)[-1]
    return None


def classify_error_text(error_text: str) -> str:
    """Map a downstream isError result to a bounded error type.

    FastMCP flattens exception classes to prose, so the text is all we have
    to classify on. The text itself is untrusted and is not recorded.
    """
    lowered = (error_text or "").lower()
    if "unknown tool" in lowered or "unknown prompt" in lowered:
        return ERROR_TYPE_VALIDATION
    if "validation error" in lowered:
        return ERROR_TYPE_VALIDATION
    if "timed out" in lowered:
        return ERROR_TYPE_UPSTREAM_TIMEOUT
    return ERROR_TYPE_DOWNSTREAM_EXECUTION


def classify_exception(exc: BaseException) -> str:
    """Map an exception to a bounded error type without trusting its message."""
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        status = exc.status_code
        mapped = _HTTP_STATUS_TO_ERROR_TYPE.get(status)
        if mapped:
            return mapped
        if 400 <= status < 500:
            return ERROR_TYPE_VALIDATION
        return ERROR_TYPE_BRIDGE_INTERNAL

    try:
        from app.oauth.token_exchange import UserLoggedOutException

        if isinstance(exc, UserLoggedOutException):
            return ERROR_TYPE_AUTHORIZATION
    except Exception:  # pragma: no cover - import safety only
        pass

    # Imported lazily: session_context imports this module at load time.
    try:
        from app.session_manager.session_context import (
            ResponseTooLargeError,
            ToolPolicyDeniedError,
        )

        if isinstance(exc, ToolPolicyDeniedError):
            return ERROR_TYPE_AUTHORIZATION
        if isinstance(exc, ResponseTooLargeError):
            return ERROR_TYPE_RESPONSE_TOO_LARGE
    except Exception:  # pragma: no cover - import safety only
        pass

    if isinstance(exc, PermissionError):
        return ERROR_TYPE_AUTHORIZATION
    if isinstance(exc, TimeoutError):
        return ERROR_TYPE_UPSTREAM_TIMEOUT
    exc_name = type(exc).__name__
    if exc_name == "McpError" and "timed out" in str(exc).lower():
        return ERROR_TYPE_UPSTREAM_TIMEOUT
    return ERROR_TYPE_BRIDGE_INTERNAL


def log_sanitized_exception(
    log: logging.Logger, prefix: str, exc: BaseException
) -> None:
    """Log an exception without its message.

    Downstream exception messages can embed request/response content, so on
    MCP request paths only the exception type and bounded classification are
    logged; the full traceback stays out of application logs.
    """
    log.error(
        "%s Exception: type=%s error.type=%s",
        prefix,
        type(exc).__name__,
        classify_exception(exc),
    )


def encoded_result_size(result: Any) -> Optional[int]:
    """Byte size of the JSON-encoded result; None when it cannot be encoded."""
    try:
        if hasattr(result, "model_dump_json"):
            return len(result.model_dump_json().encode("utf-8"))
        return len(json.dumps(result, default=str).encode("utf-8"))
    except Exception:
        return None


def result_item_count(result: Any) -> Optional[int]:
    content = getattr(result, "content", None)
    if isinstance(content, list):
        return len(content)
    messages = getattr(result, "messages", None)
    if isinstance(messages, list):
        return len(messages)
    contents = getattr(result, "contents", None)
    if isinstance(contents, list):
        return len(contents)
    return None


def downstream_call_kwargs(
    call_fn: Any, *, trace_meta: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Optional kwargs (trace meta, timeout) the downstream call supports.

    Trace context travels in the MCP ``_meta`` field: only ``traceparent``,
    ``tracestate`` and allowlisted baggage, never arbitrary headers. Pass
    ``trace_meta`` explicitly when the call happens outside the request's
    span context (e.g. in the persistent session task).
    """
    import inspect
    from datetime import timedelta

    from app import vars as app_vars

    kwargs: Dict[str, Any] = {}
    try:
        params = inspect.signature(call_fn).parameters
        has_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        return kwargs
    if "meta" in params or has_var_kw:
        meta = trace_meta if trace_meta is not None else build_trace_meta()
        if meta:
            kwargs["meta"] = meta
    if app_vars.MCP_TOOL_TIMEOUT_SECONDS > 0 and (
        "read_timeout_seconds" in params or has_var_kw
    ):
        kwargs["read_timeout_seconds"] = timedelta(
            seconds=app_vars.MCP_TOOL_TIMEOUT_SECONDS
        )
    return kwargs


def build_trace_meta() -> Dict[str, str]:
    """W3C trace context plus allowlisted baggage for the MCP ``_meta`` field.

    Only ``traceparent``, ``tracestate`` and explicitly allowlisted baggage
    keys are propagated; arbitrary baggage and wildcard headers never are.
    """
    carrier: Dict[str, str] = {}
    inject(carrier)
    meta = {
        key: value
        for key, value in carrier.items()
        if key in ("traceparent", "tracestate")
    }
    allowed_baggage = {
        key: str(value)
        for key, value in baggage.get_all().items()
        if key in MCP_TRACE_BAGGAGE_ALLOWLIST
    }
    if allowed_baggage:
        meta["baggage"] = ",".join(
            f"{key}={value}" for key, value in sorted(allowed_baggage.items())
        )
    return meta


@dataclass
class MCPOperationRecorder:
    """Handle for the canonical span; records only safe result metadata."""

    span: trace.Span
    _finalized: bool = field(default=False, init=False)

    def set_attribute(self, key: str, value: Any) -> None:
        if value is not None:
            self.span.set_attribute(key, value)

    def record_success(self, result: Any = None, truncated: bool = False) -> None:
        self._finalized = True
        self.set_attribute("enterprise_mcp_bridge.result.status", RESULT_STATUS_SUCCESS)
        self.set_attribute("enterprise_mcp_bridge.result.truncated", truncated)
        if result is None:
            return
        self.set_attribute(
            "enterprise_mcp_bridge.result.item_count", result_item_count(result)
        )
        self.set_attribute(
            "enterprise_mcp_bridge.result.encoded_bytes", encoded_result_size(result)
        )

    def record_error_result(
        self, error_type: str, result: Any = None, truncated: bool = False
    ) -> None:
        """Record a downstream isError result (the request itself completed)."""
        self._finalized = True
        self.set_attribute("enterprise_mcp_bridge.result.status", RESULT_STATUS_ERROR)
        self.set_attribute("error.type", error_type)
        self.set_attribute("enterprise_mcp_bridge.result.truncated", truncated)
        if result is not None:
            self.set_attribute(
                "enterprise_mcp_bridge.result.item_count", result_item_count(result)
            )
            self.set_attribute(
                "enterprise_mcp_bridge.result.encoded_bytes",
                encoded_result_size(result),
            )
        self.span.set_status(Status(StatusCode.ERROR))

    def record_exception(self, exc: BaseException) -> None:
        """Record a failure with a sanitized exception event.

        Only the exception type name and the bounded classification are
        attached; untrusted messages and stack data are omitted.
        """
        self._finalized = True
        error_type = classify_exception(exc)
        self.set_attribute("enterprise_mcp_bridge.result.status", RESULT_STATUS_ERROR)
        self.set_attribute("error.type", error_type)
        self.span.add_event(
            "exception",
            {
                "exception.type": type(exc).__name__,
                "error.type": error_type,
            },
        )
        self.span.set_status(Status(StatusCode.ERROR))


@contextmanager
def mcp_operation_span(
    *,
    method: str,
    target: Optional[str] = None,
    transport: str,
    session_value: Optional[str] = None,
    access_token: Optional[str] = None,
    group: Optional[str] = None,
    arg_keys: Optional[Iterable[str]] = None,
    request_id: Optional[str] = None,
    protocol_version: Optional[str] = None,
    client_name: Optional[str] = None,
    client_version: Optional[str] = None,
):
    """Emit the single canonical wide operation span for one MCP request.

    Missing context is omitted, never fabricated. The session attribute is a
    fingerprint because bridge session values can embed the caller's access
    token. Exceptions are classified, sanitized and re-raised.
    """
    span_name = f"{method} {target}" if target else method
    with _tracer.start_as_current_span(span_name, kind=SpanKind.SERVER) as span:
        recorder = MCPOperationRecorder(span)
        recorder.set_attribute("mcp.method.name", method)
        gen_ai_operation = _GEN_AI_OPERATION_BY_METHOD.get(method)
        recorder.set_attribute("gen_ai.operation.name", gen_ai_operation)
        if method == MCP_METHOD_TOOLS_CALL and target:
            recorder.set_attribute("gen_ai.tool.name", target)
        elif target:
            recorder.set_attribute("enterprise_mcp_bridge.target.name", target)
        recorder.set_attribute("enterprise_mcp_bridge.transport", transport)
        if session_value:
            recorder.set_attribute("mcp.session.id", token_fingerprint(session_value))
        recorder.set_attribute(
            "enterprise_mcp_bridge.request.id", request_id or str(uuid.uuid4())
        )
        recorder.set_attribute("mcp.protocol.version", protocol_version)
        recorder.set_attribute("enterprise_mcp_bridge.client.name", client_name)
        recorder.set_attribute("enterprise_mcp_bridge.client.version", client_version)
        recorder.set_attribute(
            "enterprise_mcp_bridge.downstream.server", downstream_server_label()
        )
        recorder.set_attribute("user.id", extract_opaque_user_id(access_token))
        recorder.set_attribute("enterprise_mcp_bridge.group.id", group)
        recorder.set_attribute(
            "enterprise_mcp_bridge.auth.mode",
            AUTH_PROVIDER if access_token else "anonymous",
        )
        if arg_keys is not None:
            keys = list(arg_keys)
            recorder.set_attribute("enterprise_mcp_bridge.argument.keys", keys)
            recorder.set_attribute("enterprise_mcp_bridge.argument.count", len(keys))
        try:
            yield recorder
        except BaseException as exc:
            # An already-recorded downstream classification (e.g. an isError
            # result re-raised as an HTTP status) wins over the raise itself.
            if not recorder._finalized:
                recorder.record_exception(exc)
            raise
        else:
            if not recorder._finalized:
                recorder.record_success()
