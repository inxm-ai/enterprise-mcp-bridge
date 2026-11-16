"""Bounded, paginated span discovery over an OTLP JSON trace query endpoint.

Native tracing MCP servers can cap how many spans one call returns (Jaeger's
topology tool stops at twenty). This opt-in overlay reads the whole trace from
the backend's query API as OTLP JSON (``resourceSpans``) and serves it in
deterministic pages under the same authenticated bridge session boundary. It
is vendor-neutral: the deployment chooses the endpoint through a URL template,
and the parser knows only the OpenTelemetry wire format.
"""

from collections.abc import Iterator
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import re

import httpx
import ijson
from ijson.common import ObjectBuilder
from mcp import types

from app import vars as app_vars
from app.multi_server import TRACE_ID_PLACEHOLDER
from app.session_manager.session_context import (
    ResponseTooLargeError,
    _to_tool_list,
    ensure_tool_allowed,
    enforce_response_ceiling,
)

TOOL_NAME = "get_trace_spans"
MAX_QUERY_BYTES = 32 * 1024 * 1024
MAX_PAGE_SIZE = 20
QUERY_TIMEOUT_SECONDS = 30
RESPONSE_TOO_LARGE_CODE = "response_too_large"

# OTLP enum wire values (opentelemetry.proto.trace.v1).
SPAN_KIND_NAMES = {
    0: "Unspecified",
    1: "Internal",
    2: "Server",
    3: "Client",
    4: "Producer",
    5: "Consumer",
}
STATUS_CODE_NAMES = {0: "Unset", 1: "Ok", 2: "Error"}
NANOSECONDS_PER_MICROSECOND = 1_000


class TraceQueryMisconfigured(RuntimeError):
    """The overlay is enabled but its deployment contract is violated."""


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("invalid OTLP trace response")
    return value


def _items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("invalid OTLP trace response")
    return value


def _any_value(value: object) -> object:
    """Unwrap an OTLP ``AnyValue`` (``{"stringValue": ...}`` and friends)."""
    wrapper = _mapping(value)
    if "arrayValue" in wrapper:
        return [
            _any_value(item)
            for item in _items(_mapping(wrapper["arrayValue"]).get("values", []))
        ]
    if "kvlistValue" in wrapper:
        return _attributes(_mapping(wrapper["kvlistValue"]).get("values", []))
    if "intValue" in wrapper:
        # Protobuf JSON encodes 64-bit integers as decimal strings.
        return int(str(wrapper["intValue"]))
    if isinstance(wrapper.get("doubleValue"), Decimal):
        # The streaming parser keeps non-integer numbers exact; report floats.
        return float(wrapper["doubleValue"])
    for key in ("stringValue", "boolValue", "doubleValue", "bytesValue"):
        if key in wrapper:
            return wrapper[key]
    return None


def _attributes(value: object) -> dict[str, object]:
    return {
        str(item["key"]): _any_value(item.get("value", {}))
        for raw in _items(value)
        if (item := _mapping(raw)) and isinstance(item.get("key"), str)
    }


def _unix_nano(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("invalid OTLP span timestamp")
    return int(value)


def _timestamp(unix_nano: int) -> str:
    return datetime.fromtimestamp(unix_nano / 1_000_000_000, timezone.utc).isoformat()


INVALID_RESPONSE = "invalid OTLP trace response"
RESOURCE_SPANS = "resourceSpans"
RESOURCE = f"{RESOURCE_SPANS}.item.resource"
SCOPE_SPANS = f"{RESOURCE_SPANS}.item.scopeSpans"
SCOPE = f"{SCOPE_SPANS}.item.scope"
SPANS = f"{SCOPE_SPANS}.item.spans"
CAPTURED = {RESOURCE, SCOPE, f"{SPANS}.item"}
DEPTH_CHANGE = {"start_map": 1, "start_array": 1, "end_map": -1, "end_array": -1}
# Containers the walk descends through: (prefix, open event, close event).
CONTAINERS = {
    RESOURCE_SPANS: ("start_array", "end_array"),
    f"{RESOURCE_SPANS}.item": ("start_map", "end_map"),
    SCOPE_SPANS: ("start_array", "end_array"),
    f"{SCOPE_SPANS}.item": ("start_map", "end_map"),
    SPANS: ("start_array", "end_array"),
}


def _relative(prefix: str) -> str | None:
    """Path below the document root; Jaeger's API v3 wraps it in ``result``."""
    if prefix == RESOURCE_SPANS or prefix.startswith(f"{RESOURCE_SPANS}."):
        return prefix
    if prefix.startswith(f"result.{RESOURCE_SPANS}"):
        return prefix.removeprefix("result.")
    return None


def _otlp_spans(body: bytes) -> Iterator[tuple[dict, dict, dict]]:
    """Stream (span, resource attributes, scope) for each span.

    Only one span is materialized at a time, so memory stays near the size of
    ``body`` instead of the whole trace as Python objects. A resource or scope
    sent after its spans is still attached, by holding those spans as compact
    JSON until the enclosing object ends.
    """
    root_keys: set[str] = set()
    seen_resource_spans = False
    builder: ObjectBuilder | None = None
    resource: dict | None = None
    scope: dict | None = None
    held_by_scope: list[str] = []
    held_by_resource: list[tuple[str, dict]] = []
    # Default number handling: integers of any size stay exact ints.
    events = ijson.parse(io.BytesIO(body))
    try:
        for prefix, event, value in events:
            if builder is not None:
                build(event, value)
                depth += DEPTH_CHANGE.get(event, 0)
                if depth:
                    continue
                captured, builder = _mapping(builder.value), None
                if captured_at == RESOURCE:
                    resource = _attributes(captured.get("attributes", []))
                elif captured_at == SCOPE:
                    scope = captured
                elif resource is not None and scope is not None:
                    yield captured, resource, scope
                else:
                    held_by_scope.append(_compact(captured))
                continue
            if prefix == "":
                if event == "map_key":
                    root_keys.add(value)
                    if {"result", RESOURCE_SPANS} <= root_keys:
                        raise ValueError(INVALID_RESPONSE)
                elif event not in {"start_map", "end_map"}:
                    raise ValueError(INVALID_RESPONSE)
                continue
            if prefix == "result" and event not in {"start_map", "end_map", "map_key"}:
                raise ValueError(INVALID_RESPONSE)
            path = _relative(prefix)
            if path is None:
                continue
            if path in CAPTURED:
                if event not in {"start_map", "start_array"}:
                    raise ValueError(INVALID_RESPONSE)
                captured_at, builder, depth = path, ObjectBuilder(), 1
                build = builder.event
                build(event, value)
                continue
            if path not in CONTAINERS:
                continue
            if event == "map_key":
                continue
            if event not in CONTAINERS[path]:
                raise ValueError(INVALID_RESPONSE)
            seen_resource_spans = True
            if (path, event) == (f"{RESOURCE_SPANS}.item", "start_map"):
                resource = None
            elif (path, event) == (f"{SCOPE_SPANS}.item", "start_map"):
                scope = None
            elif (path, event) == (f"{SCOPE_SPANS}.item", "end_map"):
                for span in held_by_scope:
                    if resource is None:
                        held_by_resource.append((span, scope or {}))
                    else:
                        yield _expand(span), resource, scope or {}
                held_by_scope = []
            elif (path, event) == (f"{RESOURCE_SPANS}.item", "end_map"):
                for span, span_scope in held_by_resource:
                    yield _expand(span), resource or {}, span_scope
                held_by_resource = []
    except ijson.JSONError as error:
        raise ValueError(INVALID_RESPONSE) from error
    if not seen_resource_spans:
        raise ValueError(INVALID_RESPONSE)


def _compact(value: object) -> str:
    """Hold a parsed span as JSON text, close to its wire size."""
    return json.dumps(value, separators=(",", ":"), default=_exact_number)


def _exact_number(value: object) -> float:
    """Non-integer numbers become the float ``json.loads`` would produce."""
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _expand(text: str) -> dict:
    return _mapping(json.loads(text))


def _normalize(
    span: dict[str, object],
    resource_attributes: dict[str, object],
    scope: dict[str, object],
) -> dict[str, object]:
    """Span details without ``trace_id``, which the page adds once validated."""
    start = _unix_nano(span.get("startTimeUnixNano"))
    end = _unix_nano(span.get("endTimeUnixNano", start))
    status = _mapping(span.get("status", {}))
    status_code = status.get("code", 0)
    return {
        "span_id": span["spanId"],
        "parent_span_id": span.get("parentSpanId") or None,
        "service": str(resource_attributes.get("service.name", "")),
        "scope": str(scope.get("name", "")),
        "span_name": span.get("name", ""),
        "kind": SPAN_KIND_NAMES.get(span.get("kind", 0), "Unspecified"),
        "start_time": _timestamp(start),
        "end_time": _timestamp(end),
        "duration_us": max(end - start, 0) // NANOSECONDS_PER_MICROSECOND,
        "status": {
            "code": STATUS_CODE_NAMES.get(status_code, "Unset"),
            **({"message": status["message"]} if status.get("message") else {}),
        },
        "attributes": _attributes(span.get("attributes", [])),
        "resource_attributes": resource_attributes,
        "events": [
            {
                "name": str(event.get("name", "")),
                "timestamp": _timestamp(_unix_nano(event.get("timeUnixNano"))),
                "attributes": _attributes(event.get("attributes", [])),
            }
            for raw in _items(span.get("events", []))
            if (event := _mapping(raw)) is not None
        ],
        "links": [
            {
                "trace_id": link.get("traceId"),
                "span_id": link.get("spanId"),
                "attributes": _attributes(link.get("attributes", [])),
            }
            for raw in _items(span.get("links", []))
            if (link := _mapping(raw)) is not None
        ],
    }


def span_page(body: bytes, trace_id: str, offset: int, limit: int) -> dict[str, object]:
    """Page a complete OTLP JSON trace in one streaming pass; never a prefix.

    Each span is kept only as its compact normalized JSON, which is close to
    its wire size, instead of as Python objects; only the page is decoded.
    """
    spans: list[tuple[int, str, str]] = []
    for span, resource, scope in _otlp_spans(body):
        span_id = span.get("spanId")
        if not isinstance(span_id, str):
            raise ValueError("invalid OTLP span identity")
        if span.get("traceId") != trace_id:
            raise ValueError("trace query returned a different trace")
        normalized = _normalize(span, resource, scope)
        text = _compact(normalized)
        spans.append((_unix_nano(span.get("startTimeUnixNano")), span_id, text))
    if not spans:
        raise ValueError("trace not found")
    spans.sort()
    if len({span[1] for span in spans}) != len(spans):
        raise ValueError("duplicate span IDs in trace")
    # Hash the normalized spans in order so the snapshot is vendor-independent.
    snapshot = hashlib.sha256()
    for _, _, text in spans:
        snapshot.update(text.encode())
        snapshot.update(b"\n")
    page = [
        {"trace_id": trace_id, **json.loads(text)}
        for _, _, text in spans[offset : offset + limit]
    ]
    next_offset = offset + len(page)
    return {
        "trace_id": trace_id,
        "spans": page,
        "total_count": len(spans),
        "next_offset": next_offset if next_offset < len(spans) else None,
        "snapshot_id": snapshot.hexdigest(),
    }


async def get_trace_spans(
    url_template: str, args: dict[str, object]
) -> types.CallToolResult:
    """Fetch a bounded trace, then return a deterministic page of full span details."""
    trace_id = args.get("trace_id")
    offset = args.get("offset", 0)
    limit = args.get("limit", MAX_PAGE_SIZE)
    if (
        not isinstance(trace_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", trace_id)
        or int(trace_id, 16) == 0
    ):
        raise ValueError(
            "trace_id must be a nonzero 32-character lowercase hexadecimal ID"
        )
    if (
        type(offset) is not int
        or offset < 0
        or type(limit) is not int
        or not 1 <= limit <= MAX_PAGE_SIZE
        or set(args) - {"trace_id", "offset", "limit"}
    ):
        raise ValueError(
            "offset must be nonnegative and limit must be between 1 and 20; unknown arguments are refused"
        )
    async with httpx.AsyncClient(
        timeout=QUERY_TIMEOUT_SECONDS, follow_redirects=False
    ) as client:
        async with client.stream(
            "GET", url_template.replace(TRACE_ID_PLACEHOLDER, trace_id)
        ) as response:
            if response.status_code != 200:
                raise ValueError(f"trace query returned HTTP {response.status_code}")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > MAX_QUERY_BYTES:
                    raise ValueError(
                        "trace exceeds the 32 MiB query budget; no partial trace returned"
                    )
                body.extend(chunk)
    page = span_page(bytes(body), trace_id, offset, limit)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(page))],
        structured_content=page,
    )


class TraceQueryDelegate:
    def __init__(self, delegate: object, url_template: str):
        self.delegate = delegate
        self.url_template = url_template

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)

    async def list_tools(self) -> object:
        tools = _to_tool_list(await self.delegate.list_tools())
        if any(getattr(tool, "name", None) == TOOL_NAME for tool in tools):
            return tools
        return [
            *tools,
            types.Tool(
                name=TOOL_NAME,
                description=(
                    "List every span in a trace with full details in bounded pages. "
                    "Each page re-reads the whole trace from the backend, so cost grows "
                    "with pages times trace size. Follow next_offset; compare "
                    "snapshot_id across pages to detect changing traces."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "trace_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_PAGE_SIZE,
                        },
                    },
                    "required": ["trace_id"],
                    "additionalProperties": False,
                },
                annotations=types.ToolAnnotations(
                    read_only_hint=True, destructive_hint=False
                ),
            ),
        ]

    async def call_tool(
        self,
        name: str,
        args: dict[str, object] | None = None,
        *positional: object,
        **kwargs: object,
    ) -> object:
        if name != TOOL_NAME:
            return await self.delegate.call_tool(name, args, *positional, **kwargs)
        ensure_tool_allowed(name)
        try:
            return enforce_response_ceiling(
                await get_trace_spans(self.url_template, args or {})
            )
        except ResponseTooLargeError as error:
            # Retrying the same page cannot help; callers must reduce its limit.
            # Keep the typed error small and omit the rejected span content.
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(error))],
                structured_content={
                    "error": {
                        "code": RESPONSE_TOO_LARGE_CODE,
                        "response_bytes": error.size,
                        "limit_bytes": error.limit,
                        "retryable": False,
                    }
                },
                is_error=True,
            )
        except (ValueError, httpx.HTTPError) as error:
            message = (
                str(error)
                if isinstance(error, ValueError)
                else "trace query unavailable or invalid"
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=message)], is_error=True
            )

    async def call_tool_with_progress(
        self,
        name: str,
        args: dict[str, object] | None = None,
        *positional: object,
        **kwargs: object,
    ) -> object:
        if name == TOOL_NAME:
            return await self.call_tool(name, args)
        return await self.delegate.call_tool_with_progress(
            name, args, *positional, **kwargs
        )

    async def call_tool_streaming(
        self,
        name: str,
        args: dict[str, object] | None = None,
        *positional: object,
        **kwargs: object,
    ) -> object:
        if name != TOOL_NAME:
            return await self.delegate.call_tool_streaming(
                name, args, *positional, **kwargs
            )
        result = await self.call_tool(name, args)

        async def stream():
            yield {
                "type": "result",
                "data": result.structured_content
                or result.model_dump(by_alias=True, exclude_none=True),
            }

        return stream()


def with_trace_query_tools(delegate: object) -> object:
    url_template = app_vars.per_server_str("TRACE_QUERY_URL", app_vars.TRACE_QUERY_URL)
    if not url_template:
        return delegate
    if not app_vars.per_server_list(
        "BRIDGE_REQUIRED_GROUPS", app_vars.BRIDGE_REQUIRED_GROUPS
    ):
        raise TraceQueryMisconfigured(
            "TRACE_QUERY_URL requires the bridge caller group boundary"
        )
    return TraceQueryDelegate(delegate, url_template)
