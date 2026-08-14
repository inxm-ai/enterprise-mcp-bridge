import jwt
import pytest
from fastapi import HTTPException

from app.session_manager.session_context import (
    ResponseTooLargeError,
    ToolPolicyDeniedError,
)
from app.utils import mcp_operation
from app.utils.mcp_operation import (
    ERROR_TYPE_AUTHORIZATION,
    ERROR_TYPE_BRIDGE_INTERNAL,
    ERROR_TYPE_DOWNSTREAM_EXECUTION,
    ERROR_TYPE_RESPONSE_TOO_LARGE,
    ERROR_TYPE_UPSTREAM_TIMEOUT,
    ERROR_TYPE_VALIDATION,
    build_trace_meta,
    classify_error_text,
    classify_exception,
    downstream_call_kwargs,
    extract_opaque_user_id,
    mcp_operation_span,
    safe_arg_keys,
)
from app.utils_tests.recording_tracer import RecordingTracer, span_payload_text

CANARY = "canary-secret-XYZZY-9f8e7d"


@pytest.fixture
def recording_tracer(monkeypatch):
    tracer = RecordingTracer()
    monkeypatch.setattr(mcp_operation, "_tracer", tracer)
    return tracer


class TestClassification:
    def test_error_text_unknown_tool_is_validation(self):
        assert classify_error_text("Unknown tool: nope") == ERROR_TYPE_VALIDATION

    def test_error_text_validation_error(self):
        assert classify_error_text("1 validation error for x") == ERROR_TYPE_VALIDATION

    def test_error_text_timeout(self):
        assert (
            classify_error_text("Request timed out after 60 seconds.")
            == ERROR_TYPE_UPSTREAM_TIMEOUT
        )

    def test_error_text_default_is_downstream_execution(self):
        assert classify_error_text("boom") == ERROR_TYPE_DOWNSTREAM_EXECUTION

    @pytest.mark.parametrize(
        "status,expected",
        [
            (400, ERROR_TYPE_VALIDATION),
            (401, ERROR_TYPE_AUTHORIZATION),
            (403, ERROR_TYPE_AUTHORIZATION),
            (404, ERROR_TYPE_VALIDATION),
            (504, ERROR_TYPE_UPSTREAM_TIMEOUT),
            (500, ERROR_TYPE_BRIDGE_INTERNAL),
        ],
    )
    def test_http_exception_mapping(self, status, expected):
        assert classify_exception(HTTPException(status_code=status)) == expected

    def test_tool_policy_denied_is_authorization(self):
        assert (
            classify_exception(ToolPolicyDeniedError("hidden"))
            == ERROR_TYPE_AUTHORIZATION
        )

    def test_response_too_large(self):
        assert (
            classify_exception(ResponseTooLargeError(10, 5))
            == ERROR_TYPE_RESPONSE_TOO_LARGE
        )

    def test_timeout_error(self):
        assert classify_exception(TimeoutError()) == ERROR_TYPE_UPSTREAM_TIMEOUT

    def test_generic_exception_is_bridge_internal(self):
        assert classify_exception(RuntimeError(CANARY)) == ERROR_TYPE_BRIDGE_INTERNAL


class TestSafeMetadata:
    def test_safe_arg_keys_returns_sorted_keys_only(self):
        assert safe_arg_keys({"b": CANARY, "a": 1}) == ["a", "b"]

    def test_safe_arg_keys_handles_non_dict(self):
        assert safe_arg_keys(None) == []
        assert safe_arg_keys("string") == []

    def test_user_id_is_sub_claim_only(self):
        token = jwt.encode(
            {"sub": "user-42", "email": "alice@example.com"},
            "secret",
            algorithm="HS256",
        )
        assert extract_opaque_user_id(token) == "user-42"

    def test_user_id_never_falls_back_to_email(self):
        token = jwt.encode(
            {"email": "alice@example.com", "preferred_username": "alice"},
            "secret",
            algorithm="HS256",
        )
        assert extract_opaque_user_id(token) is None

    def test_user_id_of_garbage_token_is_none(self):
        assert extract_opaque_user_id("not-a-jwt") is None
        assert extract_opaque_user_id(None) is None


class TestTraceMeta:
    def test_default_allowlist_forwards_no_baggage(self, monkeypatch):
        monkeypatch.setattr(
            mcp_operation.baggage, "get_all", lambda: {"tenant.id": "tenant-1"}
        )
        monkeypatch.setattr(mcp_operation, "inject", lambda carrier: None)
        assert "baggage" not in build_trace_meta()

    def test_only_traceparent_tracestate_and_allowlisted_baggage(self, monkeypatch):
        monkeypatch.setattr(mcp_operation, "MCP_TRACE_BAGGAGE_ALLOWLIST", ["tenant.id"])

        def fake_inject(carrier):
            carrier["traceparent"] = "00-abc-def-01"
            carrier["tracestate"] = "vendor=1"
            carrier["x-arbitrary-header"] = "nope"

        monkeypatch.setattr(mcp_operation, "inject", fake_inject)
        monkeypatch.setattr(
            mcp_operation.baggage,
            "get_all",
            lambda: {
                "tenant.id": "tenant-1",
                "session.token": CANARY,
                "user.email": "alice@example.com",
            },
        )

        meta = build_trace_meta()

        assert meta["traceparent"] == "00-abc-def-01"
        assert meta["tracestate"] == "vendor=1"
        assert "x-arbitrary-header" not in meta
        assert meta["baggage"] == "tenant.id=tenant-1"
        assert CANARY not in str(meta)

    def test_downstream_call_kwargs_passes_meta_when_supported(self, monkeypatch):
        async def call_with_meta(name, args, *, meta=None, read_timeout_seconds=None):
            return None

        monkeypatch.setattr(
            mcp_operation, "build_trace_meta", lambda: {"traceparent": "00-abc"}
        )
        kwargs = downstream_call_kwargs(call_with_meta)
        assert kwargs["meta"] == {"traceparent": "00-abc"}

    def test_downstream_call_kwargs_skips_meta_when_unsupported(self):
        async def plain_call(name, args):
            return None

        assert downstream_call_kwargs(plain_call) == {}

    def test_downstream_call_kwargs_applies_timeout(self, monkeypatch):
        from app import vars as app_vars

        async def call_with_timeout(
            name, args, *, meta=None, read_timeout_seconds=None
        ):
            return None

        monkeypatch.setattr(app_vars, "MCP_TOOL_TIMEOUT_SECONDS", 7.0)
        kwargs = downstream_call_kwargs(
            call_with_timeout, trace_meta={"traceparent": "00-abc"}
        )
        assert kwargs["read_timeout_seconds"].total_seconds() == 7.0
        assert kwargs["meta"] == {"traceparent": "00-abc"}


class TestDownstreamServerLabel:
    def test_remote_url_credentials_never_recorded(self, monkeypatch):
        from app.utils.mcp_operation import downstream_server_label

        monkeypatch.setattr(
            mcp_operation,
            "MCP_REMOTE_SERVER",
            f"http://user:{CANARY}@mcp.internal:16686/api/mcp/",
        )
        label = downstream_server_label()
        assert label == "mcp.internal:16686"
        assert CANARY not in label

    def test_remote_url_without_port(self, monkeypatch):
        from app.utils.mcp_operation import downstream_server_label

        monkeypatch.setattr(
            mcp_operation, "MCP_REMOTE_SERVER", "https://mcp.internal/mcp"
        )
        assert downstream_server_label() == "mcp.internal"

    def test_unparseable_remote_is_omitted(self, monkeypatch):
        from app.utils.mcp_operation import downstream_server_label

        monkeypatch.setattr(
            mcp_operation, "MCP_REMOTE_SERVER", f"user:{CANARY}@nowhere"
        )
        label = downstream_server_label()
        assert label is None or CANARY not in str(label)


class _FakeResult:
    def __init__(self, is_error=False, texts=("ok",)):
        self.isError = is_error
        self.content = [type("C", (), {"text": t})() for t in texts]


class TestOperationSpan:
    def test_success_records_wide_attributes(self, recording_tracer):
        token = jwt.encode({"sub": "user-42"}, "secret", algorithm="HS256")
        with mcp_operation_span(
            method="tools/call",
            target="get_services",
            transport="rest",
            session_value=f"session-1:{token}",
            access_token=token,
            group="observability",
            arg_keys=["a", "b"],
        ) as op:
            op.record_success(_FakeResult())

        (span,) = recording_tracer.spans
        assert span.name == "tools/call get_services"
        assert span.attributes["mcp.method.name"] == "tools/call"
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert span.attributes["gen_ai.tool.name"] == "get_services"
        assert span.attributes["enterprise_mcp_bridge.transport"] == "rest"
        assert span.attributes["user.id"] == "user-42"
        assert span.attributes["enterprise_mcp_bridge.group.id"] == "observability"
        assert span.attributes["enterprise_mcp_bridge.result.status"] == "success"
        assert span.attributes["enterprise_mcp_bridge.result.item_count"] == 1
        assert span.attributes["enterprise_mcp_bridge.result.truncated"] is False
        assert span.attributes["enterprise_mcp_bridge.argument.count"] == 2
        assert span.attributes["enterprise_mcp_bridge.request.id"]
        # The session attribute is a fingerprint, never the raw value.
        assert token not in str(span.attributes["mcp.session.id"])

    def test_error_result_sets_status_and_bounded_type(self, recording_tracer):
        with mcp_operation_span(
            method="tools/call", target="t", transport="rest"
        ) as op:
            op.record_error_result(
                ERROR_TYPE_DOWNSTREAM_EXECUTION, _FakeResult(is_error=True)
            )

        (span,) = recording_tracer.spans
        assert span.attributes["enterprise_mcp_bridge.result.status"] == "error"
        assert span.attributes["error.type"] == ERROR_TYPE_DOWNSTREAM_EXECUTION
        assert span.status is not None

    def test_exception_is_sanitized_and_reraised(self, recording_tracer):
        with pytest.raises(RuntimeError):
            with mcp_operation_span(method="tools/call", target="t", transport="rest"):
                raise RuntimeError(f"downstream blew up: {CANARY}")

        (span,) = recording_tracer.spans
        assert span.attributes["error.type"] == ERROR_TYPE_BRIDGE_INTERNAL
        assert span.status is not None
        event_name, event_attrs = span.events[0]
        assert event_name == "exception"
        assert event_attrs["exception.type"] == "RuntimeError"
        # The untrusted exception message never enters the span.
        assert CANARY not in span_payload_text(span)

    def test_exception_after_recorded_result_keeps_first_classification(
        self, recording_tracer
    ):
        with pytest.raises(HTTPException):
            with mcp_operation_span(
                method="tools/call", target="t", transport="rest"
            ) as op:
                op.record_error_result(
                    ERROR_TYPE_UPSTREAM_TIMEOUT, _FakeResult(is_error=True)
                )
                raise HTTPException(status_code=504, detail="timed out")

        (span,) = recording_tracer.spans
        assert span.attributes["error.type"] == ERROR_TYPE_UPSTREAM_TIMEOUT
        assert not span.events  # no second, exception-based classification

    def test_missing_context_is_omitted_not_fabricated(self, recording_tracer):
        with mcp_operation_span(method="prompts/get", transport="sse"):
            pass

        (span,) = recording_tracer.spans
        assert "user.id" not in span.attributes
        assert "enterprise_mcp_bridge.group.id" not in span.attributes
        assert "mcp.protocol.version" not in span.attributes
        assert "gen_ai.tool.name" not in span.attributes
        assert span.attributes["enterprise_mcp_bridge.result.status"] == "success"
