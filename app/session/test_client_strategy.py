import asyncio
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
        yield object(), object(), lambda: "remote-session-id"

    monkeypatch.setattr(
        client_strategy, "TokenRetrieverFactory", lambda: DummyFactory()
    )
    monkeypatch.setattr(
        client_strategy, "streamablehttp_client", fake_streamable_client
    )
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
        assert hasattr(session, "get_remote_session_id")
        assert session.get_remote_session_id() == "remote-session-id"
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
        yield object(), object(), lambda: None

    monkeypatch.setattr(
        client_strategy, "streamablehttp_client", fake_streamable_client
    )
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
        yield object(), object(), lambda: None

    monkeypatch.setattr(
        client_strategy, "streamablehttp_client", fake_streamable_client
    )
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
        yield object(), object(), lambda: "remote-session-id"

    monkeypatch.setattr(
        client_strategy, "streamablehttp_client", fake_streamable_client
    )
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
        yield object(), object(), lambda: "remote-session-id"

    monkeypatch.setattr(
        client_strategy, "streamablehttp_client", fake_streamable_client
    )
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
        yield object(), object(), lambda: "remote-session-id"

    monkeypatch.setattr(
        client_strategy, "streamablehttp_client", fake_streamable_client
    )
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

    assert ("info", "[MCP][tools][INFO] hello") in fake_logger.messages
    # Second call should log to error with JSON rendered payload
    assert fake_logger.messages[-1][0] == "error"
    assert "[MCP][MCP][ERROR]" in fake_logger.messages[-1][1]
    assert '"detail": "boom"' in fake_logger.messages[-1][1]


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
            return object(), object(), lambda: None

        async def __aexit__(self, exc_type, exc, tb):
            await asyncio.sleep(0)
            closed["stream"] = True
            return False

    def fake_streamable_client(url, headers=None, auth=None):
        return FakeStreamableContext()

    monkeypatch.setattr(
        client_strategy, "streamablehttp_client", fake_streamable_client
    )
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
