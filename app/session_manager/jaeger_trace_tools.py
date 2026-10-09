"""Bounded, paginated span discovery for the group-gated Jaeger deployment.

Native Jaeger MCP topology is capped at twenty spans. This opt-in overlay reads
the internal query API through the same authenticated bridge session boundary.
It does not change the other MCP tools or accept caller-supplied upstream URLs.
"""

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Protocol

import httpx
from mcp import types

from app import vars as app_vars

TOOL_NAME = "get_trace_spans"
MAX_QUERY_BYTES = 32 * 1024 * 1024
MAX_PAGE_SIZE = 20
QUERY_TIMEOUT_SECONDS = 30
RESPONSE_TOO_LARGE_CODE = "response_too_large"


class Delegate(Protocol):
    async def list_tools(self) -> object: ...
    async def call_tool(
        self,
        name: str,
        args: dict[str, object] | None,
        *positional: object,
        **kwargs: object,
    ) -> object: ...


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("invalid Jaeger trace response")
    return value


def _items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("invalid Jaeger trace response")
    return value


def _attributes(value: object) -> dict[str, object]:
    return {
        str(item["key"]): item.get("value")
        for raw in _items(value)
        if (item := _mapping(raw)) and isinstance(item.get("key"), str)
    }


def _timestamp(microseconds: object) -> str:
    if type(microseconds) is not int:
        raise ValueError("invalid Jaeger span timestamp")
    return datetime.fromtimestamp(microseconds / 1_000_000, timezone.utc).isoformat()


def span_page(
    payload: object, trace_id: str, offset: int, limit: int
) -> dict[str, object]:
    """Normalize a complete Jaeger query response; never silently return a prefix."""
    traces = _items(_mapping(payload).get("data"))
    if len(traces) != 1:
        raise ValueError("trace not found or ambiguous Jaeger response")
    trace = _mapping(traces[0])
    if trace.get("traceID") != trace_id:
        raise ValueError("Jaeger returned a different trace")
    spans = [_mapping(span) for span in _items(trace.get("spans"))]
    if any(
        not isinstance(span.get("spanID"), str)
        or type(span.get("startTime")) is not int
        for span in spans
    ):
        raise ValueError("invalid Jaeger span identity")
    spans.sort(key=lambda span: (span["startTime"], span["spanID"]))
    ids = [str(span["spanID"]) for span in spans]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate Jaeger span IDs")
    processes = _mapping(trace.get("processes", {}))
    page: list[dict[str, object]] = []
    for span in spans[offset : offset + limit]:
        attributes = _attributes(span.get("tags", []))
        references = [
            _mapping(reference) for reference in _items(span.get("references", []))
        ]
        parent = next(
            (
                reference.get("spanID")
                for reference in references
                if reference.get("refType") == "CHILD_OF"
                and reference.get("traceID") == trace_id
            ),
            None,
        )
        process = _mapping(processes.get(str(span.get("processID")), {}))
        events = []
        for raw in _items(span.get("logs", [])):
            log = _mapping(raw)
            fields = _attributes(log.get("fields", []))
            events.append(
                {
                    "name": str(fields.pop("event", "event")),
                    "timestamp": _timestamp(log.get("timestamp")),
                    "attributes": fields,
                }
            )
        error = (
            attributes.get("error") is True
            or attributes.get("otel.status_code") == "ERROR"
        )
        page.append(
            {
                "trace_id": trace_id,
                "span_id": span["spanID"],
                "parent_span_id": parent,
                "service": process.get("serviceName", ""),
                "span_name": span.get("operationName", ""),
                "start_time": _timestamp(span["startTime"]),
                "duration_us": span.get("duration", 0),
                "status": {"code": "Error" if error else "Unset"},
                "attributes": attributes,
                "resource_attributes": _attributes(process.get("tags", [])),
                "events": events,
                "links": [
                    {
                        "trace_id": reference.get("traceID"),
                        "span_id": reference.get("spanID"),
                        "attributes": {},
                    }
                    for reference in references
                    if reference.get("refType") == "FOLLOWS_FROM"
                ],
            }
        )
    next_offset = offset + len(page)
    return {
        "trace_id": trace_id,
        "spans": page,
        "total_count": len(spans),
        "next_offset": next_offset if next_offset < len(spans) else None,
        "snapshot_id": hashlib.sha256("|".join(ids).encode()).hexdigest(),
    }


async def get_trace_spans(
    base_url: str, args: dict[str, object]
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
            "GET", f"{base_url.rstrip('/')}/api/traces/{trace_id}"
        ) as response:
            if response.status_code != 200:
                raise ValueError(f"Jaeger query returned HTTP {response.status_code}")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > MAX_QUERY_BYTES:
                    raise ValueError(
                        "Jaeger trace exceeds the 32 MiB query budget; no partial trace returned"
                    )
                body.extend(chunk)
    page = span_page(json.loads(body), trace_id, offset, limit)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(page))],
        structured_content=page,
    )


class JaegerTraceDelegate:
    def __init__(self, delegate: Delegate, base_url: str):
        self.delegate = delegate
        self.base_url = base_url

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)

    async def list_tools(self) -> object:
        from app.session_manager.session_context import _to_tool_list

        tools = _to_tool_list(await self.delegate.list_tools())
        if any(getattr(tool, "name", None) == TOOL_NAME for tool in tools):
            return tools
        return [
            *tools,
            types.Tool(
                name=TOOL_NAME,
                description="List every span in a trace with full details in bounded pages. Follow next_offset; compare snapshot_id across pages to detect changing traces.",
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
        from app.session_manager.session_context import (
            ResponseTooLargeError,
            ensure_tool_allowed,
            enforce_response_ceiling,
        )

        ensure_tool_allowed(name)
        try:
            return enforce_response_ceiling(
                await get_trace_spans(self.base_url, args or {})
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
        except (ValueError, httpx.HTTPError, json.JSONDecodeError) as error:
            message = (
                str(error)
                if isinstance(error, ValueError)
                and not isinstance(error, json.JSONDecodeError)
                else "Jaeger trace query unavailable or invalid"
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=message)], is_error=True
            )


def with_jaeger_trace_tools(delegate: Delegate) -> Delegate | JaegerTraceDelegate:
    base_url = app_vars.per_server_str("JAEGER_QUERY_URL", app_vars.JAEGER_QUERY_URL)
    if not base_url:
        return delegate
    if not app_vars.per_server_list(
        "BRIDGE_REQUIRED_GROUPS", app_vars.BRIDGE_REQUIRED_GROUPS
    ):
        raise ValueError("JAEGER_QUERY_URL requires the bridge caller group boundary")
    return JaegerTraceDelegate(delegate, base_url)
