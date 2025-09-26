import contextlib
from contextlib import asynccontextmanager

import pytest

import app.session.client_strategy as client_strategy
from mcp import StdioServerParameters


class DummyClientSession:
    def __init__(self, *args, **kwargs):
        self.initialized = False

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

    class DummyAuthProvider:
        def __init__(self, **kwargs):
            captured["auth_kwargs"] = kwargs
            captured["auth_instance"] = self

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

    monkeypatch.setattr(client_strategy, "OAuthClientProvider", DummyAuthProvider)
    monkeypatch.setattr(client_strategy, "TokenRetrieverFactory", lambda: DummyFactory())
    monkeypatch.setattr(client_strategy, "streamablehttp_client", fake_streamable_client)
    monkeypatch.setattr(client_strategy, "ClientSession", DummyClientSession)

    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "demo-scope")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "client123")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "secret123")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")

    strategy = client_strategy.build_mcp_client_strategy(
        access_token="kc-token", requested_group="group1"
    )

    assert isinstance(strategy, client_strategy.RemoteMCPClientStrategy)
    assert captured["auth_kwargs"]["server_url"] == "https://remote.example"
    assert strategy._token_storage is not None

    async with strategy.session() as session:
        assert isinstance(session, DummyClientSession)
        assert session.initialized is True
        assert hasattr(session, "get_remote_session_id")
        assert session.get_remote_session_id() == "remote-session-id"
    assert captured["stream_auth"] is captured["auth_instance"]
    assert captured["stream_url"] == "https://remote.example"
    assert captured["stream_headers"] is None


@pytest.mark.asyncio
async def test_remote_strategy_anon_prefers_bearer_token(monkeypatch):
    SENTINEL = object()
    headers_seen = {"headers": SENTINEL, "auth": SENTINEL}

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        headers_seen["headers"] = headers
        headers_seen["auth"] = auth
        yield object(), object(), lambda: None

    monkeypatch.setattr(client_strategy, "streamablehttp_client", fake_streamable_client)
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
        client_strategy.build_mcp_client_strategy(access_token=None, requested_group=None)


@pytest.mark.asyncio
async def test_remote_strategy_anon_uses_access_token_header(monkeypatch):
    SENTINEL = object()
    headers_seen = {"headers": SENTINEL, "auth": SENTINEL}

    @asynccontextmanager
    async def fake_streamable_client(url, headers=None, auth=None):
        headers_seen["headers"] = headers
        headers_seen["auth"] = auth
        yield object(), object(), lambda: None

    monkeypatch.setattr(client_strategy, "streamablehttp_client", fake_streamable_client)
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
    async def fake_stdio_client(params):
        captured["params"] = params
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
    assert captured["server_params_kwargs"] == {
        "access_token": "token",
        "requested_group": "group",
        "anon": False,
    }
    assert isinstance(captured["params"], StdioServerParameters)
