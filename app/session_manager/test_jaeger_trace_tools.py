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
