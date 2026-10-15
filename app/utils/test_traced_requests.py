from app.utils.traced_requests import traced_request


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeTracer:
    def __init__(self):
        self.span = _FakeSpan()

    def start_as_current_span(self, name, **kwargs):
        return self

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc, tb):
        return False


def test_traced_request_sets_user_id_when_present():
    tracer = _FakeTracer()

    with traced_request(
        tracer=tracer,
        operation="op",
        session_value="session-1",
        group=None,
        start_message="starting",
        user_id="user-42",
    ):
        pass

    assert tracer.span.attributes.get("user.id") == "user-42"


def test_traced_request_omits_user_id_when_absent():
    tracer = _FakeTracer()

    with traced_request(
        tracer=tracer,
        operation="op",
        session_value="session-1",
        group=None,
        start_message="starting",
    ):
        pass

    assert "user.id" not in tracer.span.attributes


def test_real_otel_failure_event_does_not_capture_upstream_message():
    import pytest
    from mcp.shared.exceptions import MCPError
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with pytest.raises(MCPError):
        with traced_request(
            provider.get_tracer(__name__), "list_tools", None, None, "starting"
        ):
            raise MCPError(-32603, "canary-secret", {"token": "canary-secret"})
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["enterprise_mcp_bridge.protocol_code"] == -32603
    assert "canary-secret" not in str(spans[0].attributes)
    assert "canary-secret" not in str(
        [dict(event.attributes) for event in spans[0].events]
    )
    provider.shutdown()
