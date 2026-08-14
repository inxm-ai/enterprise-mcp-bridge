"""Canary-secret tests for the MCP request paths.

A synthetic credential/PII marker is planted in tool arguments, results,
access tokens, downstream exception messages and downstream log
notifications. It must never appear in captured application logs or in the
attributes/events of recorded spans — on success and failure paths alike.

The API response itself intentionally still carries results and error
details: this suite guards telemetry, not the tool contract.
"""

import logging
from contextlib import asynccontextmanager

import jwt
import pytest
from fastapi.testclient import TestClient

from app.server import app as fastapi_app
from app.session_manager import session_context
from app.utils import mcp_operation
from app.utils_tests.recording_tracer import RecordingTracer, span_payload_text

CANARY_SECRET = "canary-secret-XYZZY-9f8e7d"
CANARY_EMAIL = "canary.user@example.com"
CANARY_DOWNSTREAM_LOG = "canary-downstream-log-payload-31337"
ACCESS_TOKEN = jwt.encode(
    {"sub": "user-42", "email": CANARY_EMAIL, "preferred_username": "canary.user"},
    "canary-signing-key",
    algorithm="HS256",
)
TOKEN_HEADER = "X-Auth-Request-Access-Token"


class FakeContent:
    def __init__(self, text):
        self.text = text
        self.type = "text"

    def __repr__(self):
        return f"FakeContent(text={self.text!r})"


class FakeToolsContainer:
    def __init__(self, tools):
        self.tools = tools


class FakeTool:
    def __init__(self, name, input_schema=None):
        self.name = name
        self.inputSchema = input_schema or {"properties": {"query": {}}}


class FakeCallToolResult:
    def __init__(self, is_error, text):
        self.isError = is_error
        self.content = [FakeContent(text)]
        self.structuredContent = None

    def __repr__(self):
        return f"FakeCallToolResult(isError={self.isError}, content={self.content!r})"


class FakeClientSession:
    """Downstream MCP session double recording what actually reached it."""

    def __init__(self, result=None, call_error=None):
        self.result = result
        self.call_error = call_error
        self.calls = []

    async def list_tools(self):
        return FakeToolsContainer([FakeTool("test_tool")])

    async def call_tool(self, name, arguments=None, *, meta=None):
        self.calls.append({"name": name, "arguments": arguments, "meta": meta})
        if self.call_error is not None:
            raise self.call_error
        return self.result

    async def get_prompt(self, name, arguments=None):
        raise AssertionError("not used in this suite")


@pytest.fixture
def capture_logs(caplog):
    """uvicorn.error has propagate=False in app setup; caplog needs it on."""
    logger = logging.getLogger("uvicorn.error")
    previous = logger.propagate
    logger.propagate = True
    caplog.set_level(logging.DEBUG, logger="uvicorn.error")
    yield caplog
    logger.propagate = previous


@pytest.fixture
def recording_tracer(monkeypatch):
    tracer = RecordingTracer()
    monkeypatch.setattr(mcp_operation, "_tracer", tracer)
    return tracer


@pytest.fixture
def client():
    return TestClient(fastapi_app)


def _install_fake_session(monkeypatch, fake_session):
    @asynccontextmanager
    async def fake_mcp_session(**_kwargs):
        yield fake_session

    monkeypatch.setattr(session_context, "mcp_session", fake_mcp_session)


def _all_span_text(tracer):
    return "\n".join(span_payload_text(span) for span in tracer.spans)


def _assert_canaries_absent(text):
    assert CANARY_SECRET not in text
    assert CANARY_EMAIL not in text
    assert ACCESS_TOKEN not in text


class TestRestToolCallCanary:
    def test_success_path_leaks_nothing_to_logs_or_spans(
        self, client, monkeypatch, capture_logs, recording_tracer
    ):
        fake = FakeClientSession(
            result=FakeCallToolResult(False, f"result with {CANARY_SECRET}")
        )
        _install_fake_session(monkeypatch, fake)

        response = client.post(
            "/tools/test_tool",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={"query": CANARY_SECRET, "email": CANARY_EMAIL},
        )

        assert response.status_code == 200
        # API contract preserved: the caller still receives the result.
        assert CANARY_SECRET in response.text
        # The canary arguments reached the downstream server unchanged.
        assert fake.calls[0]["arguments"]["query"] == CANARY_SECRET

        _assert_canaries_absent(capture_logs.text)
        _assert_canaries_absent(_all_span_text(recording_tracer))

    def test_canonical_span_shape_on_success(
        self, client, monkeypatch, capture_logs, recording_tracer
    ):
        fake = FakeClientSession(result=FakeCallToolResult(False, "ok"))
        _install_fake_session(monkeypatch, fake)

        client.post(
            "/tools/test_tool",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={"query": "x"},
        )

        spans = [s for s in recording_tracer.spans if s.name == "tools/call test_tool"]
        assert len(spans) == 1, "exactly one canonical operation span per request"
        span = spans[0]
        assert span.attributes["mcp.method.name"] == "tools/call"
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert span.attributes["gen_ai.tool.name"] == "test_tool"
        assert span.attributes["enterprise_mcp_bridge.transport"] == "rest"
        assert span.attributes["user.id"] == "user-42"
        assert span.attributes["enterprise_mcp_bridge.auth.mode"] == "keycloak"
        assert span.attributes["enterprise_mcp_bridge.request.id"]
        assert span.attributes["enterprise_mcp_bridge.argument.keys"] == ["query"]
        assert span.attributes["enterprise_mcp_bridge.result.status"] == "success"
        assert span.attributes["enterprise_mcp_bridge.result.item_count"] == 1
        assert span.attributes["enterprise_mcp_bridge.result.encoded_bytes"] > 0

    def test_downstream_error_result_is_classified_not_leaked(
        self, client, monkeypatch, capture_logs, recording_tracer
    ):
        fake = FakeClientSession(
            result=FakeCallToolResult(True, f"Execution failed: {CANARY_SECRET}")
        )
        _install_fake_session(monkeypatch, fake)

        response = client.post(
            "/tools/test_tool",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={"query": "x"},
        )

        assert response.status_code == 422  # API contract: terminal tool error
        _assert_canaries_absent(capture_logs.text)
        span_text = _all_span_text(recording_tracer)
        _assert_canaries_absent(span_text)
        span = next(
            s for s in recording_tracer.spans if s.name == "tools/call test_tool"
        )
        assert span.attributes["error.type"] == "downstream_execution_error"
        assert span.attributes["enterprise_mcp_bridge.result.status"] == "error"
        assert span.status is not None

    def test_downstream_exception_is_sanitized(
        self, client, monkeypatch, capture_logs, recording_tracer
    ):
        fake = FakeClientSession(
            call_error=RuntimeError(f"exploded with {CANARY_SECRET} and {CANARY_EMAIL}")
        )
        _install_fake_session(monkeypatch, fake)

        response = client.post(
            "/tools/test_tool",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={"query": "x"},
        )

        assert response.status_code == 500
        _assert_canaries_absent(capture_logs.text)
        span_text = _all_span_text(recording_tracer)
        _assert_canaries_absent(span_text)
        span = next(
            s for s in recording_tracer.spans if s.name == "tools/call test_tool"
        )
        assert span.attributes["error.type"] == "bridge_internal_error"
        assert span.events and span.events[0][1]["exception.type"] == "RuntimeError"


class TestRestPromptCanary:
    def test_prompt_args_never_logged(
        self, client, monkeypatch, capture_logs, recording_tracer
    ):
        class FakePromptResult:
            description = "desc"
            messages = []
            meta = None

        class FakePromptSession(FakeClientSession):
            async def get_prompt(self, name, arguments=None):
                return FakePromptResult()

        _install_fake_session(monkeypatch, FakePromptSession())

        response = client.post(
            "/prompts/test_prompt",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={"topic": CANARY_SECRET},
        )

        assert response.status_code == 200
        _assert_canaries_absent(capture_logs.text)
        _assert_canaries_absent(_all_span_text(recording_tracer))
        span = next(
            s for s in recording_tracer.spans if s.name == "prompts/get test_prompt"
        )
        assert span.attributes["mcp.method.name"] == "prompts/get"
        assert span.attributes["enterprise_mcp_bridge.argument.keys"] == ["topic"]
        assert span.attributes["enterprise_mcp_bridge.result.status"] == "success"


class TestRestPromptErrorMapping:
    def test_prompt_error_maps_through_description_not_attributeerror(
        self, client, monkeypatch, recording_tracer
    ):
        """RunPromptResult has no `content`; the error mapping must read
        `description` instead of raising AttributeError and collapsing every
        prompt failure into a generic 500."""

        class BadMessage:
            role = "assistant"

            @property
            def content(self):
                raise RuntimeError("Unknown prompt: nope")

        class BrokenPromptResult:
            description = "desc"
            meta = None
            # Consuming messages raises inside RunPromptResult.__init__, which
            # records the failure text in `description` with isError=True.
            messages = [BadMessage()]

        class FakePromptSession(FakeClientSession):
            async def get_prompt(self, name, arguments=None):
                return BrokenPromptResult()

        _install_fake_session(monkeypatch, FakePromptSession())

        response = client.post(
            "/prompts/test_prompt",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={},
        )

        # The "Unknown prompt" text in `description` maps to 404 — before the
        # fix this was an AttributeError collapsing into a generic 500.
        assert response.status_code == 404
        assert "Unknown prompt" in response.text


class TestDownstreamLogNotificationCanary:
    @pytest.mark.asyncio
    async def test_notification_payload_is_not_logged(self, capture_logs):
        from app.session.client_strategy import _log_mcp_notification

        class FakeParams:
            level = "error"
            logger = "downstream"
            data = {"secret": CANARY_SECRET, "email": CANARY_EMAIL}

        await _log_mcp_notification(FakeParams())

        assert CANARY_SECRET not in capture_logs.text
        assert CANARY_EMAIL not in capture_logs.text
        # The notification itself is still visible as metadata.
        assert "[MCP][downstream][ERROR]" in capture_logs.text


class TestToolPolicyFailClosed:
    @pytest.fixture
    def pilot_allowlist(self, monkeypatch):
        monkeypatch.setattr(session_context, "INCLUDE_TOOLS", ["test_tool"])
        monkeypatch.setattr(session_context, "EXCLUDE_TOOLS", [])

    def test_direct_call_to_hidden_tool_is_rejected_before_downstream(
        self, client, monkeypatch, pilot_allowlist, recording_tracer
    ):
        fake = FakeClientSession(result=FakeCallToolResult(False, "should not run"))
        _install_fake_session(monkeypatch, fake)

        response = client.post(
            "/tools/hidden_tool",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={"query": "x"},
        )

        assert response.status_code == 404
        assert fake.calls == [], "downstream must never be contacted"

    def test_unknown_tool_is_rejected_before_downstream(
        self, client, monkeypatch, pilot_allowlist
    ):
        fake = FakeClientSession(result=FakeCallToolResult(False, "should not run"))
        _install_fake_session(monkeypatch, fake)

        response = client.post(
            "/tools/get_span_details",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={},
        )

        assert response.status_code == 404
        assert fake.calls == []

    def test_allowed_tool_still_runs(self, client, monkeypatch, pilot_allowlist):
        fake = FakeClientSession(result=FakeCallToolResult(False, "ok"))
        _install_fake_session(monkeypatch, fake)

        response = client.post(
            "/tools/test_tool", headers={TOKEN_HEADER: ACCESS_TOKEN}, json={}
        )

        assert response.status_code == 200
        assert len(fake.calls) == 1

    def test_streaming_route_rejects_hidden_tool_before_downstream(
        self, client, monkeypatch, pilot_allowlist
    ):
        fake = FakeClientSession(result=FakeCallToolResult(False, "should not run"))
        _install_fake_session(monkeypatch, fake)

        with client.stream(
            "POST",
            "/tools/hidden_tool/stream",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={},
        ) as response:
            body = "".join(chunk for chunk in response.iter_text())

        assert "error" in body
        assert fake.calls == []


class TestResponseByteCeiling:
    def test_oversized_response_is_rejected_not_relayed(
        self, client, monkeypatch, recording_tracer
    ):
        from app import vars as app_vars

        monkeypatch.setattr(app_vars, "MCP_MAX_RESPONSE_BYTES", 64)
        fake = FakeClientSession(
            result=FakeCallToolResult(False, "x" * 500 + CANARY_SECRET)
        )
        _install_fake_session(monkeypatch, fake)

        response = client.post(
            "/tools/test_tool", headers={TOKEN_HEADER: ACCESS_TOKEN}, json={}
        )

        assert response.status_code == 422
        assert CANARY_SECRET not in response.text
        assert "ceiling" in response.text

    def test_within_ceiling_passes(self, client, monkeypatch):
        from app import vars as app_vars

        monkeypatch.setattr(app_vars, "MCP_MAX_RESPONSE_BYTES", 10_000)
        fake = FakeClientSession(result=FakeCallToolResult(False, "small"))
        _install_fake_session(monkeypatch, fake)

        response = client.post(
            "/tools/test_tool", headers={TOKEN_HEADER: ACCESS_TOKEN}, json={}
        )

        assert response.status_code == 200


class TestResponseByteCeilingPromptsAndResources:
    """The ceiling applies to every instrumented MCP request type, not only
    tools/call."""

    def test_oversized_prompt_is_rejected(self, client, monkeypatch):
        from app import vars as app_vars

        monkeypatch.setattr(app_vars, "MCP_MAX_RESPONSE_BYTES", 64)

        class BigMessageContent:
            type = "text"
            text = "y" * 5000

        class BigMessage:
            role = "assistant"
            content = BigMessageContent()

        class FakePromptResult:
            description = "desc"
            messages = [BigMessage()]
            meta = None

        class FakePromptSession(FakeClientSession):
            async def get_prompt(self, name, arguments=None):
                return FakePromptResult()

        _install_fake_session(monkeypatch, FakePromptSession())

        response = client.post(
            "/prompts/test_prompt",
            headers={TOKEN_HEADER: ACCESS_TOKEN},
            json={},
        )

        assert response.status_code == 422
        assert "y" * 100 not in response.text
        assert "ceiling" in response.text

    def test_oversized_resource_read_is_rejected(self, client, monkeypatch):
        from app import vars as app_vars

        monkeypatch.setattr(app_vars, "MCP_MAX_RESPONSE_BYTES", 64)

        class BigResourceContent:
            mimeType = "text/plain"
            text = "z" * 5000

        class FakeResourceResult:
            contents = [BigResourceContent()]

            def model_dump_json(self):
                import json

                return json.dumps({"contents": [{"text": "z" * 5000}]})

        class ResourceRef:
            name = "big_resource"
            uri = "file:///big_resource"

        class ResourceList:
            resources = [ResourceRef()]

        class FakeResourceSession(FakeClientSession):
            async def list_resources(self):
                return ResourceList()

            async def read_resource(self, uri):
                return FakeResourceResult()

        _install_fake_session(monkeypatch, FakeResourceSession())

        response = client.get(
            "/resources/big_resource", headers={TOKEN_HEADER: ACCESS_TOKEN}
        )

        assert response.status_code == 422
        assert "z" * 100 not in response.text
        assert "ceiling" in response.text


class TestTraceContextPropagation:
    def test_trace_context_reaches_fake_downstream_in_meta(
        self, client, monkeypatch, recording_tracer
    ):
        def fake_inject(carrier):
            carrier["traceparent"] = "00-feedface-cafebabe-01"

        monkeypatch.setattr(mcp_operation, "inject", fake_inject)
        monkeypatch.setattr(mcp_operation, "MCP_TRACE_BAGGAGE_ALLOWLIST", ["tenant.id"])
        monkeypatch.setattr(
            mcp_operation.baggage,
            "get_all",
            lambda: {"tenant.id": "tenant-1", "not.allowlisted": CANARY_SECRET},
        )

        fake = FakeClientSession(result=FakeCallToolResult(False, "ok"))
        _install_fake_session(monkeypatch, fake)

        response = client.post(
            "/tools/test_tool", headers={TOKEN_HEADER: ACCESS_TOKEN}, json={}
        )

        assert response.status_code == 200
        meta = fake.calls[0]["meta"]
        assert meta["traceparent"] == "00-feedface-cafebabe-01"
        assert meta["baggage"] == "tenant.id=tenant-1"
        assert CANARY_SECRET not in str(meta)
