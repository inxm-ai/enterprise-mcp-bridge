import asyncio
import copy

import pytest
from mcp import types

from app.session_manager import trace_query_tools as trace_tools

TRACE = "1" * 32
# One second of unix-nano as a decimal string, the protobuf JSON encoding.
BASE_NANO = 1_791_460_884_000_000_000


def otlp_span(index, count, trace_id=TRACE, **overrides):
    span = {
        "traceId": trace_id,
        "spanId": f"{index + 1:016x}",
        "parentSpanId": "" if index == 0 else f"{1:016x}",
        "name": "update" if index == count - 1 else "chat",
        "kind": 2,
        "startTimeUnixNano": str(BASE_NANO + index * 1_000),
        "endTimeUnixNano": str(BASE_NANO + index * 1_000 + 10_000),
        "attributes": [],
        "events": [],
        "links": [],
        "status": {},
    }
    span.update(overrides)
    return span


def trace_payload(count=344, resource_attributes=None, wrap_result=True):
    """Mirror Jaeger query API v3 output: OTLP ``resourceSpans`` under ``result``."""
    resource = {
        "attributes": [
            {"key": "service.name", "value": {"stringValue": "agent-hub"}},
            *(resource_attributes or []),
        ]
    }
    document = {
        "resourceSpans": [
            {
                "resource": resource,
                "scopeSpans": [
                    {
                        "scope": {"name": "agent-hub"},
                        "spans": [otlp_span(index, count) for index in range(count)],
                    }
                ],
            }
        ]
    }
    return {"result": document} if wrap_result else document


def spans_of(payload):
    document = payload.get("result", payload)
    return document["resourceSpans"][0]["scopeSpans"][0]["spans"]


@pytest.mark.parametrize(
    "field",
    ["attributes", "startTimeUnixNano", "endTimeUnixNano", "events", "links", "status"],
)
def test_snapshot_detects_content_changes_with_unchanged_span_ids(field):
    payload = trace_payload(2)
    initial = trace_tools.span_page(payload, TRACE, 0, 1)["snapshot_id"]
    changed = copy.deepcopy(payload)
    span = spans_of(changed)[1]
    if field in {"startTimeUnixNano", "endTimeUnixNano"}:
        span[field] = str(int(span[field]) + 1_000)
    elif field == "attributes":
        span[field] = [{"key": "changed", "value": {"boolValue": True}}]
    elif field == "events":
        span[field] = [{"name": "retry", "timeUnixNano": str(BASE_NANO)}]
    elif field == "links":
        span[field] = [{"traceId": "2" * 32, "spanId": "2" * 16}]
    else:
        span[field] = {"code": 2}
    assert trace_tools.span_page(changed, TRACE, 0, 1)["snapshot_id"] != initial
    spans_of(payload).reverse()
    assert trace_tools.span_page(payload, TRACE, 0, 1)["snapshot_id"] == initial


def test_resource_changes_alter_the_snapshot():
    payload = trace_payload(1)
    initial = trace_tools.span_page(payload, TRACE, 0, 1)["snapshot_id"]
    changed = trace_payload(
        1,
        resource_attributes=[
            {"key": "service.version", "value": {"stringValue": "r2"}}
        ],
    )
    assert trace_tools.span_page(changed, TRACE, 0, 1)["snapshot_id"] != initial


def test_progress_and_streaming_calls_route_overlay_and_preserve_native_callbacks(
    monkeypatch,
):
    monkeypatch.setattr(trace_tools, "ensure_tool_allowed", lambda name: None)
    monkeypatch.setattr(trace_tools, "enforce_response_ceiling", lambda result: result)

    async def fetch(url_template, args):
        return types.CallToolResult(content=[], structured_content={"trace_id": TRACE})

    monkeypatch.setattr(trace_tools, "get_trace_spans", fetch)

    class Native:
        async def call_tool_with_progress(self, *args, **kwargs):
            return args, kwargs

        async def call_tool_streaming(self, *args, **kwargs):
            return args, kwargs

    async def exercise():
        overlay = trace_tools.TraceQueryDelegate(Native(), "http://q/{trace_id}")
        callback = object()
        options = {"progress_callback": callback, "log_callback": callback}
        assert await overlay.call_tool_with_progress(
            "native", {}, "token", **options
        ) == (("native", {}, "token"), options)
        assert await overlay.call_tool_streaming("native", {}, "token") == (
            ("native", {}, "token"),
            {},
        )
        result = await overlay.call_tool_with_progress(
            trace_tools.TOOL_NAME, {}, "token", **options
        )
        assert result.structured_content == {"trace_id": TRACE}
        stream = await overlay.call_tool_streaming(trace_tools.TOOL_NAME, {}, "token")
        assert [event async for event in stream] == [
            {"type": "result", "data": {"trace_id": TRACE}}
        ]

    asyncio.run(exercise())


def test_all_344_spans_are_reachable_with_a_stable_snapshot():
    payload = trace_payload()
    pages = [
        trace_tools.span_page(payload, TRACE, offset, 20)
        for offset in range(0, 344, 20)
    ]
    spans = [span for page in pages for span in page["spans"]]
    assert len(spans) == 344
    assert len({span["span_id"] for span in spans}) == 344
    assert spans[-1]["span_name"] == "update"
    assert pages[-1]["next_offset"] is None
    assert len({page["snapshot_id"] for page in pages}) == 1
    assert (
        trace_tools.span_page(trace_payload(345), TRACE, 0, 20)["snapshot_id"]
        != pages[0]["snapshot_id"]
    )


def test_accepts_a_bare_otlp_document_without_the_result_envelope():
    page = trace_tools.span_page(trace_payload(3, wrap_result=False), TRACE, 0, 20)
    assert page["total_count"] == 3


def test_normalizes_parent_kind_status_attributes_events_and_links():
    payload = trace_payload(1)
    span = spans_of(payload)[0]
    span["parentSpanId"] = "2" * 16
    span["kind"] = 3
    span["status"] = {"code": 2, "message": "boom"}
    span["attributes"] = [
        {"key": "plan_id", "value": {"stringValue": "p1"}},
        {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "123"}},
        {"key": "sampled", "value": {"boolValue": True}},
        {"key": "ratio", "value": {"doubleValue": 0.5}},
        {
            "key": "models",
            "value": {
                "arrayValue": {
                    "values": [{"stringValue": "haiku"}, {"stringValue": "sonnet"}]
                }
            },
        },
        {
            "key": "nested",
            "value": {
                "kvlistValue": {"values": [{"key": "k", "value": {"intValue": 7}}]}
            },
        },
    ]
    span["events"] = [
        {
            "name": "tool requested",
            "timeUnixNano": str(BASE_NANO + 123_000),
            "attributes": [
                {"key": "gen_ai.tool.name", "value": {"stringValue": "update"}}
            ],
        }
    ]
    span["links"] = [
        {
            "traceId": "3" * 32,
            "spanId": "4" * 16,
            "attributes": [{"key": "kind", "value": {"stringValue": "approval"}}],
        }
    ]
    detail = trace_tools.span_page(payload, TRACE, 0, 20)["spans"][0]
    assert detail["parent_span_id"] == "2" * 16
    assert detail["kind"] == "Client"
    assert detail["status"] == {"code": "Error", "message": "boom"}
    assert detail["attributes"] == {
        "plan_id": "p1",
        "gen_ai.usage.input_tokens": 123,
        "sampled": True,
        "ratio": 0.5,
        "models": ["haiku", "sonnet"],
        "nested": {"k": 7},
    }
    assert detail["events"] == [
        {
            "name": "tool requested",
            "timestamp": "2026-10-08T12:01:24.000123+00:00",
            "attributes": {"gen_ai.tool.name": "update"},
        }
    ]
    assert detail["links"] == [
        {"trace_id": "3" * 32, "span_id": "4" * 16, "attributes": {"kind": "approval"}}
    ]
    assert detail["duration_us"] == 10
    assert detail["end_time"] > detail["start_time"]


def test_root_span_has_no_parent_and_ok_status_maps_to_ok():
    payload = trace_payload(1)
    spans_of(payload)[0]["status"] = {"code": 1}
    detail = trace_tools.span_page(payload, TRACE, 0, 20)["spans"][0]
    assert detail["parent_span_id"] is None
    assert detail["status"] == {"code": "Ok"}
    assert detail["kind"] == "Server"


def test_resource_metadata_is_kept_separate_from_span_attributes():
    """Versions and environments must survive full-span discovery without collisions."""
    payload = trace_payload(
        1,
        resource_attributes=[
            {"key": "service.version", "value": {"stringValue": "revision-123"}},
            {"key": "deployment.environment.name", "value": {"stringValue": "qa"}},
            {"key": "shared", "value": {"stringValue": "resource"}},
        ],
    )
    spans_of(payload)[0]["attributes"] = [
        {"key": "shared", "value": {"stringValue": "span"}}
    ]
    detail = trace_tools.span_page(payload, TRACE, 0, 20)["spans"][0]
    assert detail["service"] == "agent-hub"
    assert detail["scope"] == "agent-hub"
    assert detail["resource_attributes"] == {
        "service.name": "agent-hub",
        "service.version": "revision-123",
        "deployment.environment.name": "qa",
        "shared": "resource",
    }
    assert detail["attributes"] == {"shared": "span"}


def test_spans_across_resources_are_merged_and_ordered_by_start_time():
    payload = trace_payload(2)
    document = payload["result"]
    late = otlp_span(2, 3, startTimeUnixNano=str(BASE_NANO + 500))
    document["resourceSpans"].append(
        {
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "tgi-core"}}
                ]
            },
            "scopeSpans": [{"scope": {"name": "tgi"}, "spans": [late]}],
        }
    )
    page = trace_tools.span_page(payload, TRACE, 0, 20)
    assert [span["service"] for span in page["spans"]] == [
        "agent-hub",
        "tgi-core",
        "agent-hub",
    ]


def test_wide_pages_have_a_typed_size_error_and_smaller_pages_keep_all_spans(
    monkeypatch,
):
    """A response ceiling must be recoverable by lowering limit at the same offset."""
    from app.session_manager import session_context
    from app.utils.mcp_operation import encoded_result_size
    from app.routes import _error_detail

    ceiling = 1_000_000
    span_content_bytes = 60_000
    payload = trace_payload(20)
    for span in spans_of(payload):
        span["attributes"] = [
            {
                "key": "synthetic_metadata",
                "value": {"stringValue": "x" * span_content_bytes},
            }
        ]
    monkeypatch.setattr(
        session_context.app_vars, "per_server_int", lambda *args: ceiling
    )
    monkeypatch.setattr(trace_tools, "ensure_tool_allowed", lambda name: None)

    async def fetch(url_template, args):
        page = trace_tools.span_page(payload, TRACE, args["offset"], args["limit"])
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=trace_tools.json.dumps(page))],
            structured_content=page,
        )

    monkeypatch.setattr(trace_tools, "get_trace_spans", fetch)
    overlay = trace_tools.TraceQueryDelegate(object(), "http://q/{trace_id}")
    for limit in [20, 10]:
        result = asyncio.run(
            overlay.call_tool(
                trace_tools.TOOL_NAME, {"trace_id": TRACE, "offset": 0, "limit": limit}
            )
        )
        assert result.is_error
        error = result.structured_content["error"]
        assert error["code"] == trace_tools.RESPONSE_TOO_LARGE_CODE
        assert error["response_bytes"] > error["limit_bytes"] == ceiling
        assert encoded_result_size(result) < ceiling
        assert "synthetic_metadata" not in result.model_dump_json()
        assert (
            _error_detail(result)["structuredContent"]["error"]["code"]
            == trace_tools.RESPONSE_TOO_LARGE_CODE
        )

    results = [
        asyncio.run(
            overlay.call_tool(
                trace_tools.TOOL_NAME, {"trace_id": TRACE, "offset": offset, "limit": 5}
            )
        )
        for offset in range(0, 20, 5)
    ]
    assert all(
        not result.is_error and encoded_result_size(result) < ceiling
        for result in results
    )
    spans = [span for result in results for span in result.structured_content["spans"]]
    assert len({span["span_id"] for span in spans}) == 20
    assert len({result.structured_content["snapshot_id"] for result in results}) == 1


def test_rejects_duplicate_mismatched_or_empty_trace_identity():
    payload = trace_payload(2)
    spans_of(payload)[1]["spanId"] = spans_of(payload)[0]["spanId"]
    with pytest.raises(ValueError, match="duplicate"):
        trace_tools.span_page(payload, TRACE, 0, 20)
    with pytest.raises(ValueError, match="different trace"):
        trace_tools.span_page(trace_payload(), "2" * 32, 0, 20)
    with pytest.raises(ValueError, match="not found"):
        trace_tools.span_page({"result": {"resourceSpans": []}}, TRACE, 0, 20)
    with pytest.raises(ValueError, match="invalid OTLP"):
        trace_tools.span_page({"data": []}, TRACE, 0, 20)


@pytest.mark.parametrize(
    "arguments",
    [
        {"trace_id": "../secret"},
        {"trace_id": TRACE, "offset": -1},
        {"trace_id": TRACE, "limit": 21},
        {"trace_id": TRACE, "limit": True},
        {"trace_id": TRACE, "url": "http://attacker"},
    ],
)
def test_invalid_arguments_never_contact_the_backend(arguments):
    with pytest.raises(ValueError):
        asyncio.run(trace_tools.get_trace_spans("http://q/{trace_id}", arguments))


def test_query_url_substitutes_the_trace_id_only_after_validation(monkeypatch):
    seen = {}

    class Response:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def aiter_bytes(self):
            yield trace_tools.json.dumps(trace_payload(1)).encode()

    class Client:
        def __init__(self, **kwargs):
            seen["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url):
            seen["url"] = url
            return Response()

    monkeypatch.setattr(trace_tools.httpx, "AsyncClient", Client)
    result = asyncio.run(
        trace_tools.get_trace_spans(
            "http://q/api/v3/traces/{trace_id}", {"trace_id": TRACE}
        )
    )
    assert seen["url"] == f"http://q/api/v3/traces/{TRACE}"
    assert seen["client"]["follow_redirects"] is False
    assert result.structured_content["total_count"] == 1


def test_overlay_requires_the_existing_caller_group_boundary(monkeypatch):
    monkeypatch.setattr(
        trace_tools.app_vars, "per_server_str", lambda *args: "http://q/{trace_id}"
    )
    monkeypatch.setattr(trace_tools.app_vars, "per_server_list", lambda *args: [])
    with pytest.raises(trace_tools.TraceQueryMisconfigured, match="group boundary"):
        trace_tools.with_trace_query_tools(object())


def test_url_template_without_placeholder_is_refused_at_settings_load():
    from app.multi_server import trace_query_url

    assert trace_query_url("") == ""
    assert trace_query_url("http://q/{trace_id}") == "http://q/{trace_id}"
    with pytest.raises(ValueError, match="trace_id"):
        trace_query_url("http://q/api/traces/")


def test_native_tools_keep_their_arguments_and_transport_metadata():
    class Native:
        async def list_tools(self):
            return types.ListToolsResult(tools=[])

        async def call_tool(self, *args, **kwargs):
            return (args, kwargs)

    overlay = trace_tools.TraceQueryDelegate(Native(), "http://q/{trace_id}")
    tools = asyncio.run(overlay.list_tools())
    assert tools[0].name == "get_trace_spans"
    assert tools[0].annotations.read_only_hint
    assert asyncio.run(
        overlay.call_tool("native", {"id": 1}, "token", meta={"traceparent": "context"})
    ) == (("native", {"id": 1}, "token"), {"meta": {"traceparent": "context"}})
