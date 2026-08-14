"""SSE proxy: execution-time tool policy and canonical operation spans.

The MCP SSE transport must behave exactly like the REST compatibility routes:
hidden/denied tools fail before any downstream contact, one canonical wide
span is emitted per JSON-RPC request, trace context is injected into _meta,
and canary content never reaches logs or spans.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types

from app.session_manager import session_context
from app.sse.mcp_proxy import _build_proxy_server
from app.utils import mcp_operation
from app.utils_tests.recording_tracer import RecordingTracer, span_payload_text

CANARY_SECRET = "canary-secret-XYZZY-9f8e7d"


def _make_mock_downstream(result_text="result"):
    downstream = AsyncMock()
    tool = types.Tool(
        name="allowed_tool",
        description="A test tool",
        inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    downstream.list_tools.return_value = SimpleNamespace(tools=[tool])
    downstream.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text=result_text)],
        isError=False,
    )
    downstream.get_prompt.return_value = types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="assistant",
                content=types.TextContent(type="text", text="hello"),
            )
        ]
    )
    downstream.read_resource.return_value = types.ReadResourceResult(contents=[])
    return downstream


def _build(downstream):
    return _build_proxy_server(
        downstream,
        access_token=None,
        incoming_headers=None,
        session_key="sess-1",
    )


def _call_tool_request(name, arguments=None):
    return types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments or {}),
    )


@pytest.fixture
def recording_tracer(monkeypatch):
    tracer = RecordingTracer()
    monkeypatch.setattr(mcp_operation, "_tracer", tracer)
    return tracer


@pytest.fixture
def pilot_allowlist(monkeypatch):
    monkeypatch.setattr(session_context, "INCLUDE_TOOLS", ["allowed_tool"])
    monkeypatch.setattr(session_context, "EXCLUDE_TOOLS", [])


@pytest.fixture
def capture_logs(caplog):
    logger = logging.getLogger("uvicorn.error")
    previous = logger.propagate
    logger.propagate = True
    caplog.set_level(logging.DEBUG, logger="uvicorn.error")
    yield caplog
    logger.propagate = previous


class TestExecutionTimePolicy:
    @pytest.mark.asyncio
    async def test_direct_call_to_hidden_tool_fails_before_downstream(
        self, pilot_allowlist, recording_tracer
    ):
        downstream = _make_mock_downstream()
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.CallToolRequest]

        result = await handler(_call_tool_request("hidden_tool", {"x": "v"}))

        assert result.root.isError
        # The SDK refreshes its tool cache (a discovery call) before invoking
        # the handler; what must never happen is the execution itself.
        downstream.call_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_tool_fails_before_downstream(
        self, pilot_allowlist, recording_tracer
    ):
        downstream = _make_mock_downstream()
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.CallToolRequest]

        result = await handler(_call_tool_request("get_span_details"))

        assert result.root.isError
        downstream.call_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allowed_tool_passes(self, pilot_allowlist, recording_tracer):
        downstream = _make_mock_downstream()
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.CallToolRequest]

        result = await handler(_call_tool_request("allowed_tool", {"x": "v"}))

        assert not result.root.isError
        downstream.call_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_denied_call_is_classified_as_authorization(
        self, pilot_allowlist, recording_tracer
    ):
        downstream = _make_mock_downstream()
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.CallToolRequest]

        await handler(_call_tool_request("hidden_tool"))

        span = next(
            s for s in recording_tracer.spans if s.name == "tools/call hidden_tool"
        )
        assert span.attributes["error.type"] == "authorization_error"
        assert span.attributes["enterprise_mcp_bridge.result.status"] == "error"


class TestCanonicalSpan:
    @pytest.mark.asyncio
    async def test_call_tool_emits_one_canonical_span(self, recording_tracer):
        downstream = _make_mock_downstream()
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.CallToolRequest]

        await handler(_call_tool_request("allowed_tool", {"x": "v"}))

        spans = [
            s for s in recording_tracer.spans if s.name == "tools/call allowed_tool"
        ]
        assert len(spans) == 1
        span = spans[0]
        assert span.attributes["mcp.method.name"] == "tools/call"
        assert span.attributes["gen_ai.tool.name"] == "allowed_tool"
        assert span.attributes["enterprise_mcp_bridge.transport"] == "sse"
        assert span.attributes["enterprise_mcp_bridge.argument.keys"] == ["x"]
        assert span.attributes["enterprise_mcp_bridge.result.status"] == "success"

    @pytest.mark.asyncio
    async def test_get_prompt_emits_canonical_span(self, recording_tracer):
        downstream = _make_mock_downstream()
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.GetPromptRequest]

        await handler(
            types.GetPromptRequest(
                method="prompts/get",
                params=types.GetPromptRequestParams(
                    name="my_prompt", arguments={"topic": CANARY_SECRET}
                ),
            )
        )

        span = next(
            s for s in recording_tracer.spans if s.name == "prompts/get my_prompt"
        )
        assert span.attributes["mcp.method.name"] == "prompts/get"
        assert span.attributes["enterprise_mcp_bridge.argument.keys"] == ["topic"]
        assert CANARY_SECRET not in span_payload_text(span)

    @pytest.mark.asyncio
    async def test_read_resource_emits_canonical_span(self, recording_tracer):
        downstream = _make_mock_downstream()
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.ReadResourceRequest]

        await handler(
            types.ReadResourceRequest(
                method="resources/read",
                params=types.ReadResourceRequestParams(uri="file:///demo.txt"),
            )
        )

        span = next(s for s in recording_tracer.spans if s.name == "resources/read")
        assert span.attributes["mcp.method.name"] == "resources/read"
        assert span.attributes["enterprise_mcp_bridge.result.status"] == "success"


class TestSseCanary:
    @pytest.mark.asyncio
    async def test_canary_args_and_results_stay_out_of_logs_and_spans(
        self, recording_tracer, capture_logs
    ):
        downstream = _make_mock_downstream(result_text=f"payload {CANARY_SECRET}")
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.CallToolRequest]

        result = await handler(_call_tool_request("allowed_tool", {"x": CANARY_SECRET}))

        # API contract: the client still receives the result.
        assert CANARY_SECRET in result.root.content[0].text
        assert CANARY_SECRET not in capture_logs.text
        for span in recording_tracer.spans:
            assert CANARY_SECRET not in span_payload_text(span)

    @pytest.mark.asyncio
    async def test_downstream_exception_is_sanitized(
        self, recording_tracer, capture_logs
    ):
        downstream = _make_mock_downstream()
        downstream.call_tool.side_effect = RuntimeError(f"boom {CANARY_SECRET}")
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.CallToolRequest]

        result = await handler(_call_tool_request("allowed_tool"))

        assert result.root.isError
        assert CANARY_SECRET not in capture_logs.text
        span = next(
            s for s in recording_tracer.spans if s.name == "tools/call allowed_tool"
        )
        assert span.attributes["error.type"] == "bridge_internal_error"
        assert CANARY_SECRET not in span_payload_text(span)


class TestTraceMetaPropagation:
    @pytest.mark.asyncio
    async def test_trace_context_reaches_downstream_in_meta(
        self, recording_tracer, monkeypatch
    ):
        def fake_inject(carrier):
            carrier["traceparent"] = "00-feedface-cafebabe-01"

        monkeypatch.setattr(mcp_operation, "inject", fake_inject)
        monkeypatch.setattr(mcp_operation, "MCP_TRACE_BAGGAGE_ALLOWLIST", ["group.id"])
        monkeypatch.setattr(
            mcp_operation.baggage,
            "get_all",
            lambda: {"group.id": "obs", "evil.baggage": CANARY_SECRET},
        )

        downstream = _make_mock_downstream()
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.CallToolRequest]

        await handler(_call_tool_request("allowed_tool", {"x": "v"}))

        meta = downstream.call_tool.await_args.kwargs.get("meta")
        assert meta["traceparent"] == "00-feedface-cafebabe-01"
        assert meta["baggage"] == "group.id=obs"
        assert CANARY_SECRET not in str(meta)

    @pytest.mark.asyncio
    async def test_oversized_prompt_is_rejected(self, recording_tracer, monkeypatch):
        from app import vars as app_vars

        monkeypatch.setattr(app_vars, "MCP_MAX_RESPONSE_BYTES", 64)
        downstream = _make_mock_downstream()
        downstream.get_prompt.return_value = types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="assistant",
                    content=types.TextContent(type="text", text="p" * 5000),
                )
            ]
        )
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.GetPromptRequest]

        with pytest.raises(Exception) as excinfo:
            await handler(
                types.GetPromptRequest(
                    method="prompts/get",
                    params=types.GetPromptRequestParams(name="my_prompt"),
                )
            )
        assert "ceiling" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_oversized_response_is_rejected(self, recording_tracer, monkeypatch):
        from app import vars as app_vars

        monkeypatch.setattr(app_vars, "MCP_MAX_RESPONSE_BYTES", 32)
        downstream = _make_mock_downstream(result_text="y" * 500)
        proxy = _build(downstream)
        handler = proxy.request_handlers[types.CallToolRequest]

        result = await handler(_call_tool_request("allowed_tool"))

        assert result.root.isError
        assert "y" * 500 not in str(result.root.content)
        span = next(
            s for s in recording_tracer.spans if s.name == "tools/call allowed_tool"
        )
        assert span.attributes["error.type"] == "response_too_large"
