import asyncio
import time
from contextlib import asynccontextmanager

import pytest

import app.session.client_strategy as client_strategy
from mcp import StdioServerParameters, types


class DummyClientSession:
    def __init__(self, *args, **kwargs):
        self.initialized = False
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        self.initialized = True


@pytest.mark.asyncio
async def test_remote_strategy_uses_token_exchange(monkeypatch):
    SENTINEL = object()
    captured = {
        "stream_url": SENTINEL,
        "stream_headers": SENTINEL,
        "stream_auth": SENTINEL,
    }

    class DummyRetriever:
        def retrieve_token(self, token: str):
            assert token == "kc-token"
            return {
                "access_token": "provider-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "a b",
                "refresh_token": "refresh-me",
            }

    class DummyFactory:
        def get(self):
            return DummyRetriever()

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        captured["stream_url"] = url
        captured["stream_headers"] = headers
        captured["stream_auth"] = auth
        yield object(), object()

    monkeypatch.setattr(
        client_strategy, "TokenRetrieverFactory", lambda: DummyFactory()
    )
    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)

    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "demo-scope")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "client123")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "secret123")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")
    import app.vars as vars_module

    monkeypatch.setattr(
        vars_module,
        "MCP_REMOTE_SERVER_FORWARD_HEADERS",
        ["Authorization"],
    )

    strategy = client_strategy.build_mcp_client_strategy(
        access_token="kc-token",
        requested_group="group1",
        incoming_headers={"Authorization": "Bearer original-token"},
    )

    assert isinstance(strategy, client_strategy.RemoteMCPClientStrategy)
    assert strategy._token_storage is None
    assert strategy.headers["Authorization"] == "Bearer provider-token"
    assert strategy._auth_provider is None

    async with strategy.session() as session:
        assert isinstance(session, DummyClientSession)
        assert session.initialized is True
        assert (
            session.kwargs.get("logging_callback")
            is client_strategy._log_mcp_notification
        )
    assert captured["stream_auth"] is None
    assert captured["stream_url"] == "https://remote.example"
    assert captured["stream_headers"]["Authorization"] == "Bearer provider-token"


@pytest.mark.asyncio
async def test_remote_strategy_anon_prefers_bearer_token(monkeypatch):
    SENTINEL = object()
    headers_seen = {"headers": SENTINEL, "auth": SENTINEL}

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        headers_seen["headers"] = headers
        headers_seen["auth"] = auth
        yield object(), object()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-token")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    strategy = client_strategy.build_mcp_client_strategy(
        access_token=None, requested_group=None, anon=True
    )

    assert isinstance(strategy, client_strategy.RemoteMCPClientStrategy)
    assert strategy.headers["Authorization"] == "Bearer static-token"
    async with strategy.session():
        pass
    assert headers_seen["headers"]["Authorization"] == "Bearer static-token"
    assert headers_seen["auth"] is None


def test_remote_strategy_conflicting_env(monkeypatch):
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "python server.py")

    with pytest.raises(ValueError):
        client_strategy.build_mcp_client_strategy(
            access_token=None, requested_group=None
        )


@pytest.mark.asyncio
async def test_remote_strategy_anon_uses_access_token_header(monkeypatch):
    SENTINEL = object()
    headers_seen = {"headers": SENTINEL, "auth": SENTINEL}

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        headers_seen["headers"] = headers
        headers_seen["auth"] = auth
        yield object(), object()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    strategy = client_strategy.build_mcp_client_strategy(
        access_token="incoming-token", requested_group=None, anon=True
    )

    assert strategy.headers["Authorization"] == "Bearer incoming-token"
    async with strategy.session():
        pass
    assert headers_seen["headers"]["Authorization"] == "Bearer incoming-token"
    assert headers_seen["auth"] is None


@pytest.mark.asyncio
async def test_local_strategy_session(monkeypatch):
    captured = {}

    @asynccontextmanager
    async def fake_stdio_client(params, errlog=None):
        captured["params"] = params
        captured["errlog"] = errlog
        yield object(), object()

    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "")
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    monkeypatch.setattr(client_strategy, "stdio_client", fake_stdio_client)

    def fake_get_server_params(**kwargs):
        captured["server_params_kwargs"] = kwargs
        return StdioServerParameters(command="python", args=["app.py"], env={})

    monkeypatch.setattr(client_strategy, "get_server_params", fake_get_server_params)

    strategy = client_strategy.build_mcp_client_strategy(
        access_token="token", requested_group="group"
    )

    assert isinstance(strategy, client_strategy.LocalMCPClientStrategy)
    async with strategy.session() as session:
        assert isinstance(session, DummyClientSession)
        assert session.initialized is True
        assert (
            session.kwargs.get("logging_callback")
            is client_strategy._log_mcp_notification
        )
    assert captured["server_params_kwargs"] == {
        "access_token": "token",
        "requested_group": "group",
        "anon": False,
    }
    assert isinstance(captured["params"], StdioServerParameters)
    assert captured["errlog"] is not None


@pytest.mark.asyncio
async def test_remote_strategy_forwards_allowed_headers(monkeypatch):
    """Test that RemoteMCPClientStrategy forwards headers configured in MCP_REMOTE_SERVER_FORWARD_HEADERS."""
    captured = {"stream_headers": None}

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        captured["stream_headers"] = headers
        yield object(), object()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-token")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    # Mock the vars module to set MCP_REMOTE_SERVER_FORWARD_HEADERS
    import app.vars as vars_module

    monkeypatch.setattr(
        vars_module,
        "MCP_REMOTE_SERVER_FORWARD_HEADERS",
        ["X-Request-ID", "X-Correlation-ID"],
    )

    # Create incoming headers
    incoming_headers = {
        "X-Request-ID": "req-123",
        "X-Correlation-ID": "corr-456",
        "Authorization": "Bearer user-token",  # Should not be forwarded (not in allow list)
        "User-Agent": "test-agent",  # Should not be forwarded (not in allow list)
    }

    strategy = client_strategy.RemoteMCPClientStrategy(
        "https://remote.example",
        access_token=None,
        requested_group=None,
        anon=False,
        incoming_headers=incoming_headers,
    )

    async with strategy.session() as session:
        assert isinstance(session, DummyClientSession)
        assert session.initialized is True

    # Check that only allowed headers were forwarded
    assert captured["stream_headers"] is not None
    assert "X-Request-ID" in captured["stream_headers"]
    assert captured["stream_headers"]["X-Request-ID"] == "req-123"
    assert "X-Correlation-ID" in captured["stream_headers"]
    assert captured["stream_headers"]["X-Correlation-ID"] == "corr-456"
    # Authorization header should use the fallback bearer token, not the incoming one
    assert captured["stream_headers"]["Authorization"] == "Bearer static-token"
    # User-Agent should not be forwarded (not in allow list)
    assert "User-Agent" not in captured["stream_headers"]


@pytest.mark.asyncio
async def test_remote_strategy_headers_case_insensitive(monkeypatch):
    """Test that header forwarding is case-insensitive."""
    captured = {"stream_headers": None}

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        captured["stream_headers"] = headers
        yield object(), object()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-token")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    import app.vars as vars_module

    monkeypatch.setattr(
        vars_module,
        "MCP_REMOTE_SERVER_FORWARD_HEADERS",
        ["x-request-id"],  # lowercase in config
    )

    # Incoming headers with different case
    incoming_headers = {
        "X-Request-ID": "req-789",  # Different case
    }

    strategy = client_strategy.RemoteMCPClientStrategy(
        "https://remote.example",
        access_token=None,
        requested_group=None,
        anon=False,
        incoming_headers=incoming_headers,
    )

    async with strategy.session() as session:
        assert isinstance(session, DummyClientSession)

    # Should match case-insensitively and forward
    assert captured["stream_headers"] is not None
    assert "x-request-id" in captured["stream_headers"]
    assert captured["stream_headers"]["x-request-id"] == "req-789"


@pytest.mark.asyncio
async def test_remote_strategy_forward_all_headers(monkeypatch):
    captured = {"stream_headers": None}

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        captured["stream_headers"] = headers
        yield object(), object()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-token")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    import app.vars as vars_module

    monkeypatch.setattr(
        vars_module,
        "MCP_REMOTE_SERVER_FORWARD_HEADERS",
        ["*"],
    )

    incoming_headers = {
        "X-Test-Header": "test-value",
        "Host": "qa.example.local",
        "Content-Length": "42",
    }

    strategy = client_strategy.RemoteMCPClientStrategy(
        "https://remote.example",
        access_token=None,
        requested_group=None,
        anon=False,
        incoming_headers=incoming_headers,
    )

    async with strategy.session() as session:
        assert isinstance(session, DummyClientSession)

    assert captured["stream_headers"] is not None
    assert captured["stream_headers"]["X-Test-Header"] == "test-value"
    assert "Host" not in captured["stream_headers"]
    assert "Content-Length" not in captured["stream_headers"]
    assert captured["stream_headers"]["Authorization"] == "Bearer static-token"


@pytest.mark.asyncio
async def test_mcp_log_notifications_forwarded(monkeypatch):
    class FakeLogger:
        def __init__(self):
            self.messages = []

        def debug(self, msg, *args, **kwargs):
            self.messages.append(("debug", msg))

        def info(self, msg, *args, **kwargs):
            self.messages.append(("info", msg))

        def warning(self, msg, *args, **kwargs):
            self.messages.append(("warning", msg))

        def error(self, msg, *args, **kwargs):
            self.messages.append(("error", msg))

        def critical(self, msg, *args, **kwargs):
            self.messages.append(("critical", msg))

    fake_logger = FakeLogger()
    monkeypatch.setattr(client_strategy, "logger", fake_logger)

    params_info = types.LoggingMessageNotificationParams(
        level="info", data="hello", logger="tools"
    )
    params_error = types.LoggingMessageNotificationParams(
        level="error", data={"detail": "boom"}, logger=None
    )

    await client_strategy._log_mcp_notification(params_info)
    await client_strategy._log_mcp_notification(params_error)

    # Downstream log payloads are arbitrary untrusted content; only their
    # shape is recorded, never the content itself.
    assert fake_logger.messages[0][0] == "info"
    assert fake_logger.messages[0][1].startswith("[MCP][tools][INFO]")
    assert "hello" not in fake_logger.messages[0][1]
    assert "type=str chars=5" in fake_logger.messages[0][1]
    assert fake_logger.messages[-1][0] == "error"
    assert "[MCP][MCP][ERROR]" in fake_logger.messages[-1][1]
    assert "boom" not in fake_logger.messages[-1][1]
    assert "type=dict" in fake_logger.messages[-1][1]


@pytest.mark.asyncio
async def test_remote_strategy_closes_on_cancellation(monkeypatch):
    closed = {"stream": False, "session": False}

    class SlowExitSession(DummyClientSession):
        async def __aexit__(self, exc_type, exc, tb):
            await asyncio.sleep(0)
            closed["session"] = True
            return False

    class FakeStreamableContext:
        async def __aenter__(self):
            return object(), object()

        async def __aexit__(self, exc_type, exc, tb):
            await asyncio.sleep(0)
            closed["stream"] = True
            return False

    def fake_streamable_client(url, headers=None, auth=None):
        return FakeStreamableContext()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", SlowExitSession)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-token")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    strategy = client_strategy.build_mcp_client_strategy(
        access_token=None, requested_group=None
    )

    with pytest.raises(asyncio.CancelledError):
        async with strategy.session():
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)

    assert closed["session"] is True
    assert closed["stream"] is True


@pytest.mark.asyncio
async def test_remote_strategy_teardown_bounded_when_close_hangs(monkeypatch):
    """A transport whose close never returns must not wedge the session exit.

    Regression test for the anyio _deliver_cancellation busy-loop: cancelling a
    request while the remote transport hangs in __aexit__ used to leave the
    event loop spinning forever. Teardown must give up after the configured
    timeout and let the cancellation propagate.
    """

    class HangingStreamableContext:
        async def __aenter__(self):
            return object(), object()

        async def __aexit__(self, exc_type, exc, tb):
            await asyncio.Event().wait()

    def fake_streamable_client(url, headers=None, auth=None):
        return HangingStreamableContext()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-token")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_TEARDOWN_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    strategy = client_strategy.build_mcp_client_strategy(
        access_token=None, requested_group=None
    )

    async def cancelled_session():
        async with strategy.session():
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)

    started = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        # wait_for bounds the test itself: without the teardown timeout this
        # hangs forever instead of failing an assertion.
        await asyncio.wait_for(asyncio.shield(cancelled_session()), timeout=5)
    elapsed = time.monotonic() - started
    # 10x the 0.1s teardown timeout as scheduling tolerance: asserts the
    # configured bound is honored, not merely that we exit before wait_for.
    assert elapsed < 1.0, f"teardown took {elapsed:.2f}s, expected ~0.1s"


@pytest.mark.asyncio
async def test_remote_strategy_custom_auth_header_name_and_template(monkeypatch):
    """Provider token lands in a custom header without a Bearer prefix."""

    class DummyRetriever:
        def retrieve_token(self, token: str):
            return {"access_token": "user-api-key-123", "token_type": "Bearer"}

    class DummyFactory:
        def get(self):
            return DummyRetriever()

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        yield object(), object()

    monkeypatch.setattr(
        client_strategy, "TokenRetrieverFactory", lambda: DummyFactory()
    )
    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "")
    monkeypatch.setattr(
        client_strategy, "MCP_REMOTE_AUTH_HEADER_NAME", "esb-subscription-key"
    )
    monkeypatch.setattr(
        client_strategy, "MCP_REMOTE_AUTH_HEADER_VALUE_TEMPLATE", "{token}"
    )
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    strategy = client_strategy.build_mcp_client_strategy(
        access_token="kc-token", requested_group=None
    )

    assert strategy.headers["esb-subscription-key"] == "user-api-key-123"
    assert "Authorization" not in strategy.headers


@pytest.mark.asyncio
async def test_remote_strategy_custom_auth_header_fallback_token(monkeypatch):
    """Fallback bearer token also honors the custom header name/template."""

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        yield object(), object()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-key")
    monkeypatch.setattr(
        client_strategy, "MCP_REMOTE_AUTH_HEADER_NAME", "esb-subscription-key"
    )
    monkeypatch.setattr(
        client_strategy, "MCP_REMOTE_AUTH_HEADER_VALUE_TEMPLATE", "{token}"
    )
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    strategy = client_strategy.build_mcp_client_strategy(
        access_token=None, requested_group=None, anon=True
    )

    assert strategy.headers["esb-subscription-key"] == "static-key"
    assert "Authorization" not in strategy.headers


def test_format_auth_header_value_default(monkeypatch):
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_AUTH_HEADER_VALUE_TEMPLATE", "")
    assert (
        client_strategy.RemoteMCPClientStrategy._format_auth_header_value("tok")
        == "Bearer tok"
    )


@pytest.mark.asyncio
async def test_user_api_key_mode_never_falls_back_to_shared_token(monkeypatch):
    """AUTH_PROVIDER=user-api-key must fail closed on retrieval errors:
    neither MCP_REMOTE_BEARER_TOKEN nor the incoming Keycloak token may
    replace the user's identity."""

    class FailingRetriever:
        def retrieve_token(self, token):
            return {"success": False, "error": "connection_key_lookup_failed"}

    class FailingFactory:
        def get(self):
            return FailingRetriever()

    monkeypatch.setattr(client_strategy, "TokenRetrieverFactory", FailingFactory)
    monkeypatch.setattr(client_strategy, "AUTH_PROVIDER", "user-api-key")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    # _prepare_auth runs in the strategy constructor, so the failure is
    # expected at build time already.
    # Case 1: shared bearer token configured — must NOT be used.
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-token")
    with pytest.raises(client_strategy.UserLoggedOutException):
        client_strategy.build_mcp_client_strategy(
            access_token="user-jwt", requested_group=None
        )

    # Case 2: no shared token — the incoming token must NOT be forwarded.
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "")
    with pytest.raises(client_strategy.UserLoggedOutException):
        client_strategy.build_mcp_client_strategy(
            access_token="user-jwt", requested_group=None
        )


@pytest.mark.parametrize("provider", ["gcp-metadata", "azure-metadata", "aws-metadata"])
@pytest.mark.parametrize(
    "access_token,anon", [(None, False), (None, True), ("user-jwt", False)]
)
def test_ambient_identity_providers_set_header_regardless_of_caller_identity(
    monkeypatch, provider, access_token, anon
):
    """Ambient-identity providers self-fetch a token and must win over both
    the anon path and the "no caller token" early-return in _prepare_auth."""

    class DummyRetriever:
        def retrieve_token(self, token):
            return {"access_token": "ambient-token", "token_type": "Bearer"}

    class DummyFactory:
        def get(self):
            return DummyRetriever()

    monkeypatch.setattr(client_strategy, "TokenRetrieverFactory", DummyFactory)
    monkeypatch.setattr(client_strategy, "AUTH_PROVIDER", provider)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    strategy = client_strategy.build_mcp_client_strategy(
        access_token=access_token, requested_group=None, anon=anon
    )

    assert strategy.headers["Authorization"] == "Bearer ambient-token"


@pytest.mark.parametrize("provider", ["gcp-metadata", "azure-metadata", "aws-metadata"])
def test_ambient_identity_providers_fail_closed_never_fall_back(monkeypatch, provider):
    """A retriever failure must never fall back to MCP_REMOTE_BEARER_TOKEN
    or an incoming caller token for an ambient-identity provider."""

    class FailingRetriever:
        def retrieve_token(self, token):
            raise client_strategy.UserLoggedOutException("platform unreachable")

    class FailingFactory:
        def get(self):
            return FailingRetriever()

    monkeypatch.setattr(client_strategy, "TokenRetrieverFactory", FailingFactory)
    monkeypatch.setattr(client_strategy, "AUTH_PROVIDER", provider)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-token")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    with pytest.raises(client_strategy.UserLoggedOutException):
        client_strategy.build_mcp_client_strategy(
            access_token="user-jwt", requested_group=None
        )


@pytest.mark.parametrize("provider", ["gcp-metadata", "azure-metadata", "aws-metadata"])
def test_ambient_identity_providers_fail_closed_on_empty_token(monkeypatch, provider):
    """A retriever returning a dict with no usable access_token must fail
    closed, not send an "Bearer None" Authorization header."""

    class EmptyTokenRetriever:
        def retrieve_token(self, token):
            return {"success": True, "access_token": None, "token_type": "Bearer"}

    class EmptyTokenFactory:
        def get(self):
            return EmptyTokenRetriever()

    monkeypatch.setattr(client_strategy, "TokenRetrieverFactory", EmptyTokenFactory)
    monkeypatch.setattr(client_strategy, "AUTH_PROVIDER", provider)
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    with pytest.raises(client_strategy.UserLoggedOutException):
        client_strategy.build_mcp_client_strategy(
            access_token=None, requested_group=None
        )


def _remote_without_oauth(monkeypatch, bearer_token):
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", bearer_token)
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")


class _ExplodingFactory:
    def get(self):
        raise AssertionError("no token exchange with forwarding disabled")


@pytest.mark.parametrize(
    "bearer_token, expected_authorization",
    [("static-api-key", "Bearer static-api-key"), ("", None)],
)
def test_forwarding_disabled_by_env_never_sends_caller_token(
    monkeypatch, bearer_token, expected_authorization
):
    """MCP_FORWARD_ACCESS_TOKEN=false on a standalone bridge: an API-key
    remote gets only its key, a no-auth remote gets no Authorization."""
    monkeypatch.setattr(client_strategy, "TokenRetrieverFactory", _ExplodingFactory)
    monkeypatch.setenv("MCP_FORWARD_ACCESS_TOKEN", "false")
    _remote_without_oauth(monkeypatch, bearer_token)

    strategy = client_strategy.build_mcp_client_strategy(
        access_token="caller-jwt", requested_group=None
    )

    assert strategy.headers.get("Authorization") == expected_authorization
    assert all("caller-jwt" not in value for value in strategy.headers.values())


def test_provider_alias_fails_closed_when_exchange_returns_no_token(monkeypatch):
    """With a provider alias, an exchange that returns no token must not fall
    back to the caller's own token."""

    class EmptyRetriever:
        def retrieve_token(self, token):
            return {"success": False, "error": "no stored provider token"}

    class EmptyFactory:
        def get(self):
            return EmptyRetriever()

    monkeypatch.setattr(client_strategy, "TokenRetrieverFactory", EmptyFactory)
    monkeypatch.setattr(client_strategy, "AUTH_PROVIDER", "keycloak")
    monkeypatch.setattr(client_strategy, "KEYCLOAK_PROVIDER_ALIAS", "github")
    _remote_without_oauth(monkeypatch, "")

    with pytest.raises(client_strategy.UserLoggedOutException):
        client_strategy.build_mcp_client_strategy(
            access_token="caller-jwt", requested_group=None
        )


def _proxy_server(proxy):
    from app import multi_server

    (server,) = multi_server.parse_servers(
        __import__("json").dumps(
            [
                {
                    "id": "mcp-home-server",
                    "base_path": "/api/mcp-home-server",
                    "url": "https://home.tail1234.ts.net/mcp",
                    "settings": {"MCP_REMOTE_PROXY_URL": proxy},
                }
            ]
        )
    )
    return server


@pytest.mark.asyncio
async def test_remote_strategy_sends_a_servers_traffic_through_its_proxy(
    monkeypatch,
):
    from app import multi_server

    captured = {}

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None, proxy=None):
        captured["url"] = url
        captured["proxy"] = proxy
        yield object(), object()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    proxy = "http://tailnet-abc.orchestrator.svc.cluster.local:1055"
    token = multi_server.bind_server(_proxy_server(proxy))
    try:
        strategy = client_strategy.RemoteMCPClientStrategy(
            "https://home.tail1234.ts.net/mcp",
            access_token=None,
            requested_group=None,
            anon=True,
        )
    finally:
        multi_server.reset_server(token)
    async with strategy.session():
        pass
    assert captured == {"url": "https://home.tail1234.ts.net/mcp", "proxy": proxy}


@pytest.mark.asyncio
async def test_remote_strategy_without_proxy_connects_directly(monkeypatch):
    captured = {}

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None, **kwargs):
        captured.update(kwargs)
        yield object(), object()

    monkeypatch.setattr(client_strategy, "_streamable_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    strategy = client_strategy.RemoteMCPClientStrategy(
        "https://remote.example/mcp",
        access_token=None,
        requested_group=None,
        anon=True,
    )
    assert strategy.proxy is None
    async with strategy.session():
        pass
    assert captured == {}


@pytest.mark.asyncio
async def test_proxied_client_sends_requests_to_the_proxy_only(monkeypatch):
    seen = []

    async def proxy(reader, writer):
        seen.append((await reader.readline()).decode().strip())
        while line := (await reader.readline()).strip():
            name, _, value = line.decode().partition(":")
            if name.lower() == "proxy-authorization":
                seen.append(value.strip())
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(proxy, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv("HTTP_PROXY", "http://env-proxy.invalid:3128")
    try:
        client = client_strategy._proxied_client_factory(
            f"http://inxm:t0ken@127.0.0.1:{port}"
        )(headers={"x": "y"})
        async with client:
            assert client.timeout.read == 300.0
            response = await client.get("http://home.tail1234.ts.net:8080/mcp")
    finally:
        server.close()
        await server.wait_closed()
    assert response.text == "ok"
    # The credential in the URL goes to the proxy as Basic auth.
    assert seen == [
        "GET http://home.tail1234.ts.net:8080/mcp HTTP/1.1",
        "Basic " + __import__("base64").b64encode(b"inxm:t0ken").decode(),
    ]


@pytest.mark.asyncio
async def test_remote_strategy_sends_sse_traffic_through_its_proxy(monkeypatch):
    from app import multi_server

    captured = {}

    @asynccontextmanager
    async def fake_sse_client(url, headers=None, httpx_client_factory=None, **kw):
        captured["url"] = url
        client = httpx_client_factory(headers=headers)
        captured["proxied"] = client._mounts
        await client.aclose()
        yield object(), object()

    monkeypatch.setattr(client_strategy, "sse_client", fake_sse_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)
    proxy = "http://tailnet-abc.orchestrator.svc.cluster.local:1055"
    token = multi_server.bind_server(_proxy_server(proxy))
    try:
        strategy = client_strategy.RemoteMCPClientStrategy(
            "https://home.tail1234.ts.net/sse",
            access_token=None,
            requested_group=None,
            anon=True,
        )
    finally:
        multi_server.reset_server(token)
    async with strategy.session():
        pass
    assert captured["url"] == "https://home.tail1234.ts.net/sse"
    assert captured["proxied"], "the SSE client must be built with the proxy"


def test_a_proxy_from_settings_from_is_validated(monkeypatch):
    from app import multi_server

    monkeypatch.setenv("MCPS_HOME__PROXY", "http://user:pw@proxy:1055/path")
    # (credentials are fine; the path is not)
    (server,) = multi_server.parse_servers(
        __import__("json").dumps(
            [
                {
                    "id": "mcp-home-server",
                    "base_path": "/api/mcp-home-server",
                    "url": "https://home.tail1234.ts.net/mcp",
                    "settings_from": {"MCP_REMOTE_PROXY_URL": "MCPS_HOME__PROXY"},
                }
            ]
        )
    )
    token = multi_server.bind_server(server)
    try:
        with pytest.raises(ValueError):
            client_strategy.RemoteMCPClientStrategy(
                "https://home.tail1234.ts.net/mcp",
                access_token=None,
                requested_group=None,
                anon=True,
            )
    finally:
        multi_server.reset_server(token)


def test_an_unavailable_token_service_is_503_never_a_shared_credential(monkeypatch):
    """Keycloak mode with a provider served by INXM's OAuth broker: a broker
    outage must not fall back to MCP_REMOTE_BEARER_TOKEN or the caller's
    token; it is 503 with Retry-After, like the ambient-identity providers."""
    from fastapi import HTTPException

    class DownRetriever:
        def retrieve_token(self, token):
            raise client_strategy.TokenServiceUnavailableError("broker down")

    class DownFactory:
        def get(self):
            return DownRetriever()

    monkeypatch.setattr(client_strategy, "TokenRetrieverFactory", DownFactory)
    monkeypatch.setattr(client_strategy, "AUTH_PROVIDER", "keycloak")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "static-token")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    with pytest.raises(HTTPException) as raised:
        client_strategy.build_mcp_client_strategy(
            access_token="user-jwt", requested_group=None
        )
    assert raised.value.status_code == 503
    assert raised.value.headers["Retry-After"]
