import asyncio

import pytest
from mcp import types

from app.session_manager import jaeger_trace_tools as trace_tools

TRACE = "1" * 32


def trace_payload(count=344):
    return {
        "data": [
            {
                "traceID": TRACE,
                "processes": {"p": {"serviceName": "agent-hub"}},
                "spans": [
                    {
                        "spanID": f"{index + 1:016x}",
                        "startTime": index,
                        "duration": 10,
                        "processID": "p",
                        "operationName": "update" if index == count - 1 else "chat",
                        "tags": [],
                        "logs": [],
                        "references": [],
                    }
                    for index in range(count)
                ],
            }
        ]
    }


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


def test_normalizes_parent_error_attributes_events_and_cross_trace_links():
    payload = trace_payload(1)
    span = payload["data"][0]["spans"][0]
    span["tags"] = [{"key": "error", "value": True}, {"key": "plan_id", "value": "p1"}]
    span["references"] = [
        {"refType": "CHILD_OF", "traceID": TRACE, "spanID": "2" * 16},
        {"refType": "FOLLOWS_FROM", "traceID": "3" * 32, "spanID": "4" * 16},
    ]
    span["logs"] = [
        {
            "timestamp": 123,
            "fields": [
                {"key": "event", "value": "tool requested"},
                {"key": "gen_ai.tool.name", "value": "update"},
            ],
        }
    ]
    detail = trace_tools.span_page(payload, TRACE, 0, 20)["spans"][0]
    assert detail["parent_span_id"] == "2" * 16
    assert detail["status"]["code"] == "Error"
    assert detail["attributes"]["plan_id"] == "p1"
    assert detail["events"][0]["name"] == "tool requested"
    assert detail["links"][0]["trace_id"] == "3" * 32


def test_resource_metadata_is_kept_separate_from_span_attributes():
    """Versions and environments must survive full-span discovery without collisions."""
    payload = trace_payload(1)
    payload["data"][0]["processes"]["p"]["tags"] = [
        {"key": "service.version", "value": "revision-123"},
        {"key": "deployment.environment.name", "value": "qa"},
        {"key": "shared", "value": "resource"},
    ]
    payload["data"][0]["spans"][0]["tags"] = [{"key": "shared", "value": "span"}]
    detail = trace_tools.span_page(payload, TRACE, 0, 20)["spans"][0]
    assert detail["resource_attributes"] == {
        "service.version": "revision-123",
        "deployment.environment.name": "qa",
        "shared": "resource",
    }
    assert detail["attributes"] == {"shared": "span"}


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
    for span in payload["data"][0]["spans"]:
        span["tags"] = [
            {"key": "synthetic_metadata", "value": "x" * span_content_bytes}
        ]
    monkeypatch.setattr(
        session_context.app_vars, "per_server_int", lambda *args: ceiling
    )
    monkeypatch.setattr(session_context, "ensure_tool_allowed", lambda name: None)

    async def fetch(base_url, args):
        page = trace_tools.span_page(payload, TRACE, args["offset"], args["limit"])
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=trace_tools.json.dumps(page))],
            structured_content=page,
        )

    monkeypatch.setattr(trace_tools, "get_trace_spans", fetch)
    overlay = trace_tools.JaegerTraceDelegate(object(), "http://jaeger")
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


def test_rejects_duplicate_or_mismatched_trace_identity():
    payload = trace_payload(2)
    payload["data"][0]["spans"][1]["spanID"] = payload["data"][0]["spans"][0]["spanID"]
    with pytest.raises(ValueError, match="duplicate"):
        trace_tools.span_page(payload, TRACE, 0, 20)
    with pytest.raises(ValueError, match="different trace"):
        trace_tools.span_page(trace_payload(), "2" * 32, 0, 20)


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
def test_invalid_arguments_never_contact_jaeger(arguments):
    with pytest.raises(ValueError):
        asyncio.run(trace_tools.get_trace_spans("http://jaeger", arguments))


def test_overlay_requires_the_existing_caller_group_boundary(monkeypatch):
    monkeypatch.setattr(
        trace_tools.app_vars, "per_server_str", lambda *args: "http://jaeger"
    )
    monkeypatch.setattr(trace_tools.app_vars, "per_server_list", lambda *args: [])
    with pytest.raises(ValueError, match="group boundary"):
        trace_tools.with_jaeger_trace_tools(object())


def test_native_tools_keep_their_arguments_and_transport_metadata():
    class Native:
        async def list_tools(self):
            return types.ListToolsResult(tools=[])

        async def call_tool(self, *args, **kwargs):
            return (args, kwargs)

    overlay = trace_tools.JaegerTraceDelegate(Native(), "http://jaeger")
    tools = asyncio.run(overlay.list_tools())
    assert tools[0].name == "get_trace_spans"
    assert tools[0].annotations.read_only_hint
    assert asyncio.run(
        overlay.call_tool("native", {"id": 1}, "token", meta={"traceparent": "context"})
    ) == (("native", {"id": 1}, "token"), {"meta": {"traceparent": "context"}})
