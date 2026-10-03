import json
from pathlib import Path

import pytest

from app import multi_server


def _servers():
    return multi_server.parse_servers(
        json.dumps(
            [
                {
                    "id": "alpha",
                    "base_path": "/api/mcp/alpha",
                    "command": "python alpha.py",
                    "env": {"ALPHA_ONLY": "1"},
                    "include_tools": ["alpha_*"],
                    "sessionless": True,
                },
                {
                    "id": "nested",
                    "base_path": "/api/mcp/alpha/nested",
                    "url": "https://example.invalid/mcp",
                    "exclude_tools": ["admin_*"],
                },
            ]
        )
    )


def test_parse_servers_supports_local_and_remote_entries():
    alpha, nested = _servers()

    assert alpha.command == "python alpha.py"
    assert alpha.remote_url is None
    assert alpha.env == {"ALPHA_ONLY": "1"}
    assert alpha.include_tools == ("alpha_*",)
    assert alpha.sessionless is True

    assert nested.command is None
    assert nested.remote_url == "https://example.invalid/mcp"
    assert nested.exclude_tools == ("admin_*",)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{"id": "x", "base_path": "relative", "command": "x"}],
        [{"id": "x", "base_path": "/", "command": "x"}],
        [{"id": "x", "base_path": "/x", "command": "x", "url": "https://x"}],
        [{"id": "x", "base_path": "/x"}],
        [{"id": "x", "base_path": "/x", "command": "x", "include_tools": {}}],
        [{"id": "x", "base_path": "/x", "command": "x", "include_tools": ""}],
        [{"id": "x", "base_path": "/x", "command": "x", "exclude_tools": {}}],
        [{"id": "x", "base_path": "/x", "command": "x", "exclude_tools": ""}],
        [{"id": "x", "base_path": "/x", "command": "x", "env": []}],
        [{"id": "x", "base_path": "/x", "command": "x", "auth_provider": 1}],
        [{"id": "x", "base_path": "/x", "command": "x", "auth_provider": []}],
        [
            {
                "id": "x",
                "base_path": "/x",
                "command": "x",
                "keycloak_provider_alias": False,
            }
        ],
        [
            {
                "id": "x",
                "base_path": "/x",
                "command": "x",
                "keycloak_provider_alias": ["a"],
            }
        ],
        [{"id": "x", "base_path": "/x", "command": "x", "effect_tools": "auto"}],
        [{"id": "x", "base_path": "/x", "command": "x", "effect_tools": {}}],
        [{"id": "x", "base_path": "/x", "command": "x", "effect_tools": [1]}],
        [
            {
                "id": "x",
                "base_path": "/x",
                "command": "x",
                "effect_tools": ["auto", None],
            }
        ],
        [
            {"id": "x", "base_path": "/x", "command": "x"},
            {"id": "x", "base_path": "/y", "command": "y"},
        ],
        [
            {"id": "x", "base_path": "/x", "command": "x"},
            {"id": "y", "base_path": "/x", "command": "y"},
        ],
    ],
)
def test_parse_servers_rejects_invalid_config(payload):
    with pytest.raises(ValueError):
        multi_server.parse_servers(json.dumps(payload))


def _remote_host_servers():
    """A central remote host: several remote MCPs, each with its own IdP."""
    return multi_server.parse_servers(
        json.dumps(
            [
                {
                    "id": "mcp-cloudflare-server",
                    "base_path": "/api/mcp-cloudflare-server",
                    "url": "https://mcp.cloudflare.com/mcp",
                    "sessionless": True,
                    "auth_provider": "keycloak",
                    "keycloak_provider_alias": "cloudflare",
                    "effect_tools": ["auto"],
                },
                {
                    "id": "mcp-notion-server",
                    "base_path": "/api/mcp-notion-server",
                    "url": "https://mcp.notion.com/mcp",
                    "sessionless": True,
                    "auth_provider": " Keycloak ",
                    "keycloak_provider_alias": "notion",
                    "effect_tools": ["create_*", "auto"],
                },
                {
                    "id": "passthrough",
                    "base_path": "/api/passthrough",
                    "url": "https://passthrough.example/mcp",
                    "sessionless": True,
                    "keycloak_provider_alias": "",
                    "effect_tools": ["create_*"],
                    # Unknown fields keep being ignored.
                    "some_future_field": {"x": 1},
                },
                {
                    "id": "inherit",
                    "base_path": "/api/inherit",
                    "url": "https://inherit.example/mcp",
                    "sessionless": True,
                },
            ]
        )
    )


def test_parse_servers_supports_auth_and_effect_overrides():
    cloudflare, notion, passthrough, inherit = _remote_host_servers()

    assert cloudflare.auth_provider == "keycloak"
    assert cloudflare.keycloak_provider_alias == "cloudflare"
    assert cloudflare.effect_tools == ("auto",)

    # Normalized like the global AUTH_PROVIDER.
    assert notion.auth_provider == "keycloak"
    assert notion.keycloak_provider_alias == "notion"
    assert notion.effect_tools == ("create_*", "auto")

    # "" is an explicit "no alias", distinct from "not configured".
    assert passthrough.auth_provider is None
    assert passthrough.keycloak_provider_alias == ""
    assert passthrough.effect_tools == ("create_*",)

    assert inherit.auth_provider is None
    assert inherit.keycloak_provider_alias is None
    assert inherit.effect_tools is None


def test_parse_servers_allows_empty_effect_tools_and_auth_provider():
    (server,) = multi_server.parse_servers(
        json.dumps(
            [
                {
                    "id": "x",
                    "base_path": "/x",
                    "command": "x",
                    "auth_provider": "",
                    "effect_tools": [],
                }
            ]
        )
    )
    # Empty auth_provider inherits the global; [] disables effect tools.
    assert server.auth_provider is None
    assert server.effect_tools == ()


def test_auth_and_effect_helpers_prefer_server_overrides():
    cloudflare, notion, passthrough, inherit = _remote_host_servers()

    expectations = [
        (cloudflare, "keycloak", "cloudflare", ["auto"]),
        (notion, "keycloak", "notion", ["create_*", "auto"]),
        (passthrough, "user-api-key", "", ["create_*"]),
        (inherit, "user-api-key", "global-alias", ["global_*"]),
    ]
    for server, provider, alias, effect_tools in expectations:
        token = multi_server.bind_server(server)
        try:
            assert multi_server.current_auth_provider("user-api-key") == provider
            assert multi_server.current_keycloak_provider_alias("global-alias") == (
                alias
            )
            assert multi_server.current_effect_tools(["global_*"]) == effect_tools
        finally:
            multi_server.reset_server(token)


def test_auth_and_effect_helpers_default_outside_server_context(monkeypatch):
    monkeypatch.setattr(multi_server, "SERVERS", ())

    assert multi_server.current_auth_provider("keycloak") == "keycloak"
    assert multi_server.current_keycloak_provider_alias("legacy") == "legacy"
    assert multi_server.current_keycloak_provider_alias("") == ""
    assert multi_server.current_effect_tools(["create_*"]) == ["create_*"]
    assert multi_server.current_effect_tools([]) == []


def _fake_keycloak_broker(monkeypatch, token_exchange):
    """Fake Keycloak broker answering with an alias-specific provider token."""
    requested = []

    class Response:
        status_code = 200

        def __init__(self, alias):
            self.alias = alias

        def json(self):
            # Opaque (non-JWT) token: never triggers a refresh.
            return {"access_token": f"{self.alias}-token", "token_type": "Bearer"}

    def fake_get(url, **kwargs):
        requested.append(url)
        alias = url.split("/broker/", 1)[1].split("/", 1)[0]
        return Response(alias)

    monkeypatch.setattr(token_exchange.requests, "get", fake_get)
    return requested


def test_keycloak_alias_resolves_per_server(monkeypatch):
    from app.oauth import token_exchange

    monkeypatch.setattr(token_exchange, "AUTH_PROVIDER", "user-api-key")
    monkeypatch.setattr(token_exchange, "KEYCLOAK_PROVIDER_ALIAS", "global-alias")
    monkeypatch.setattr(token_exchange, "AUTH_BASE_URL", "https://kc.example")
    requested = _fake_keycloak_broker(monkeypatch, token_exchange)
    cloudflare, notion, passthrough, _inherit = _remote_host_servers()

    # Built outside any request: the alias must still follow the request.
    shared = token_exchange.KeyCloakTokenRetriever()

    for server, alias in ((cloudflare, "cloudflare"), (notion, "notion")):
        token = multi_server.bind_server(server)
        try:
            retriever = token_exchange.TokenRetrieverFactory().get()
            assert isinstance(retriever, token_exchange.KeyCloakTokenRetriever)
            result = retriever.retrieve_token("kc-token")
            assert result["access_token"] == f"{alias}-token"
            assert requested[-1] == (
                f"https://kc.example/realms/{token_exchange.KEYCLOAK_REALM}"
                f"/broker/{alias}/token"
            )
            assert shared.provider_alias == alias
            assert shared.retrieve_token("kc-token")["access_token"] == (
                f"{alias}-token"
            )
        finally:
            multi_server.reset_server(token)

    # Explicit empty alias: pass-through, no broker call; the global
    # AUTH_PROVIDER (user-api-key) still applies to this server.
    requested.clear()
    token = multi_server.bind_server(passthrough)
    try:
        assert isinstance(
            token_exchange.TokenRetrieverFactory().get(),
            token_exchange.UserApiKeyTokenRetriever,
        )
        assert shared.retrieve_token("kc-token")["access_token"] == "kc-token"
        forced = shared.force_token_refresh("kc-token")
        assert forced == {"success": True, "access_token": "kc-token"}
        assert requested == []
    finally:
        multi_server.reset_server(token)

    # Outside a server context the globals apply, exactly as before.
    assert shared.provider_alias == "global-alias"
    assert isinstance(
        token_exchange.TokenRetrieverFactory().get(),
        token_exchange.UserApiKeyTokenRetriever,
    )


def test_pinned_keycloak_alias_is_not_overridden_by_server():
    from app.oauth import token_exchange

    cloudflare, *_ = _remote_host_servers()
    token = multi_server.bind_server(cloudflare)
    try:
        assert token_exchange.KeyCloakTokenRetriever("pinned").provider_alias == (
            "pinned"
        )
    finally:
        multi_server.reset_server(token)


def test_oauth_decorator_uses_server_auth_settings(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from app.oauth import decorator

    monkeypatch.setattr(decorator, "AUTH_PROVIDER", "keycloak")
    monkeypatch.setattr(decorator, "KEYCLOAK_PROVIDER_ALIAS", "")
    exchanged = []

    class Retriever:
        def retrieve_token(self, access_token):
            alias = multi_server.current_keycloak_provider_alias("")
            exchanged.append(alias)
            return {"access_token": f"{alias}-token"}

    monkeypatch.setattr(
        decorator,
        "TokenRetrieverFactory",
        lambda: SimpleNamespace(get=lambda: Retriever()),
    )
    tools = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name="needs_token",
                inputSchema={"properties": {"oauth_token": {}}},
            )
        ]
    )

    def decorate():
        return asyncio.run(
            decorator.decorate_args_with_oauth_token(tools, "needs_token", {}, "kc")
        )["oauth_token"]

    cloudflare, notion, passthrough, inherit = _remote_host_servers()
    for server, expected in (
        (cloudflare, "cloudflare-token"),
        (notion, "notion-token"),
        (passthrough, "kc"),
        (inherit, "kc"),
    ):
        token = multi_server.bind_server(server)
        try:
            assert decorate() == expected
        finally:
            multi_server.reset_server(token)

    # Single-server mode: unchanged pass-through with no global alias.
    assert decorate() == "kc"
    assert exchanged == ["cloudflare", "notion"]


def test_remote_strategy_exchanges_token_with_server_alias(monkeypatch):
    from app.oauth import token_exchange
    from app.session import client_strategy

    monkeypatch.setattr(client_strategy, "AUTH_PROVIDER", "gcp-metadata")
    monkeypatch.setattr(token_exchange, "AUTH_PROVIDER", "gcp-metadata")
    monkeypatch.setattr(token_exchange, "KEYCLOAK_PROVIDER_ALIAS", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "")
    _fake_keycloak_broker(monkeypatch, token_exchange)
    cloudflare, notion, *_ = _remote_host_servers()

    # Each server overrides the (ambient) global provider with keycloak and
    # exchanges the caller's token through its own identity provider.
    for server, alias in ((cloudflare, "cloudflare"), (notion, "notion")):
        token = multi_server.bind_server(server)
        try:
            strategy = client_strategy.build_mcp_client_strategy(
                access_token="kc-token", requested_group=None
            )
            assert isinstance(strategy, client_strategy.RemoteMCPClientStrategy)
            assert strategy.url == server.remote_url
            assert strategy.headers["Authorization"] == f"Bearer {alias}-token"
        finally:
            multi_server.reset_server(token)


def test_match_server_uses_longest_base_path(monkeypatch):
    servers = _servers()
    monkeypatch.setattr(multi_server, "SERVERS", servers)

    assert multi_server.match_server("/api/mcp/alpha/tools").id == "alpha"
    assert multi_server.match_server("/api/mcp/alpha/nested/tools").id == "nested"
    assert multi_server.match_server("/api/mcp/unknown/tools") is None


def test_server_context_isolates_session_cache_filters_and_env():
    alpha, nested = _servers()
    default_cache = Path("/tmp/tools.json")
    default_lock = Path("/tmp/tools.json.lock")

    alpha_token = multi_server.bind_server(alpha)
    try:
        assert multi_server.session_storage_key("same") == "alpha:same"
        assert multi_server.tools_cache_paths(default_cache, default_lock) == (
            Path("/tmp/tools.json.alpha"),
            Path("/tmp/tools.json.lock.alpha"),
        )
        assert multi_server.current_tool_filters(["default"], [])[0] == ["alpha_*"]
        assert multi_server.current_env({"BASE": "1"}) == {
            "BASE": "1",
            "ALPHA_ONLY": "1",
        }
        assert multi_server.current_sessionless(False) is True
    finally:
        multi_server.reset_server(alpha_token)

    nested_token = multi_server.bind_server(nested)
    try:
        assert multi_server.session_storage_key("same") == "nested:same"
        assert multi_server.tools_cache_paths(default_cache, default_lock) == (
            Path("/tmp/tools.json.nested"),
            Path("/tmp/tools.json.lock.nested"),
        )
        include, exclude = multi_server.current_tool_filters(["default"], [])
        assert include == []
        assert exclude == ["admin_*"]
        assert multi_server.current_sessionless(True) is True
    finally:
        multi_server.reset_server(nested_token)


def test_single_server_mode_preserves_defaults(monkeypatch):
    monkeypatch.setattr(multi_server, "SERVERS", ())

    assert multi_server.is_multi_server_mode() is False
    assert multi_server.match_server("/anything") is None
    legacy_remote = multi_server.current_remote_url("https://legacy/mcp")
    assert legacy_remote == "https://legacy/mcp"
    assert multi_server.current_command("python legacy.py") == "python legacy.py"
    assert multi_server.current_tool_filters(["a"], ["b"]) == (["a"], ["b"])
    assert multi_server.current_sessionless(True) is True
    assert multi_server.session_storage_key("legacy") == "legacy"
    assert multi_server.tools_cache_paths(
        Path("/tmp/tools.json"), Path("/tmp/tools.lock")
    ) == (Path("/tmp/tools.json"), Path("/tmp/tools.lock"))


def test_backend_selection_can_mix_local_and_remote():
    from app.session.client_strategy import (
        LocalMCPClientStrategy,
        RemoteMCPClientStrategy,
        build_mcp_client_strategy,
    )

    alpha, nested = _servers()

    token = multi_server.bind_server(alpha)
    try:
        strategy = build_mcp_client_strategy(
            access_token=None,
            requested_group=None,
            anon=True,
        )
        assert isinstance(strategy, LocalMCPClientStrategy)
        assert strategy.server_params.command == "python"
        assert strategy.server_params.args == ["alpha.py"]
        assert strategy.server_params.env["ALPHA_ONLY"] == "1"
    finally:
        multi_server.reset_server(token)

    token = multi_server.bind_server(nested)
    try:
        strategy = build_mcp_client_strategy(
            access_token=None,
            requested_group=None,
            anon=True,
        )
        assert isinstance(strategy, RemoteMCPClientStrategy)
        assert strategy.url == "https://example.invalid/mcp"
    finally:
        multi_server.reset_server(token)


def test_tool_filters_are_request_local(monkeypatch):
    from app.session_manager import session_context

    monkeypatch.setattr(session_context, "INCLUDE_TOOLS", [])
    monkeypatch.setattr(session_context, "EXCLUDE_TOOLS", [])
    alpha, nested = _servers()

    token = multi_server.bind_server(alpha)
    try:
        assert session_context.tool_allowed("alpha_read")
        assert not session_context.tool_allowed("beta_read")
    finally:
        multi_server.reset_server(token)

    token = multi_server.bind_server(nested)
    try:
        assert session_context.tool_allowed("alpha_read")
        assert not session_context.tool_allowed("admin_delete")
    finally:
        multi_server.reset_server(token)


def test_multi_server_routes_are_isolated(monkeypatch):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import routes
    from app.session_manager import session_context

    servers = multi_server.parse_servers(
        json.dumps(
            [
                {
                    "id": "one",
                    "base_path": "/api/mcp/one",
                    "command": "python one.py",
                    "include_tools": ["one_*"],
                    "sessionless": True,
                },
                {
                    "id": "two",
                    "base_path": "/api/mcp/two",
                    "command": "python two.py",
                    "include_tools": ["two_*"],
                    "sessionless": True,
                },
            ]
        )
    )
    monkeypatch.setattr(multi_server, "SERVERS", servers)
    monkeypatch.setattr(routes.router, "prefix", "")

    class FakeSession:
        async def list_tools(self):
            return [
                {
                    "name": "one_echo",
                    "description": "one",
                    "inputSchema": {"type": "object"},
                },
                {
                    "name": "two_echo",
                    "description": "two",
                    "inputSchema": {"type": "object"},
                },
            ]

        async def call_tool(self, tool_name, args, access_token):
            session_context.ensure_tool_allowed(tool_name)
            return {
                "isError": False,
                "content": [{"type": "text", "text": tool_name}],
            }

    @asynccontextmanager
    async def fake_context(*args, **kwargs):
        yield FakeSession()

    monkeypatch.setattr(routes, "mcp_session_context", fake_context)

    app = FastAPI()
    app.add_middleware(multi_server.MultiServerContextMiddleware)
    for server in servers:
        app.include_router(routes.router, prefix=server.base_path)

    with TestClient(app) as client:
        one_tools = client.get("/api/mcp/one/tools")
        two_tools = client.get("/api/mcp/two/tools")
        assert one_tools.status_code == 200
        assert [tool["name"] for tool in one_tools.json()] == ["one_echo"]
        assert two_tools.status_code == 200
        assert [tool["name"] for tool in two_tools.json()] == ["two_echo"]

        one_call = client.post("/api/mcp/one/tools/one_echo", json={})
        two_call = client.post("/api/mcp/two/tools/two_echo", json={})
        cross_call = client.post("/api/mcp/one/tools/two_echo", json={})
        unknown = client.get("/api/mcp/unknown/tools")

        assert one_call.status_code == 200
        assert two_call.status_code == 200
        assert cross_call.status_code == 404
        assert unknown.status_code == 404


def _dry_run_app(monkeypatch, global_effect_tools):
    """Mount the routes per server with a fake session that records calls."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from app import routes
    from app.sse import routes as sse_routes
    from app.tgi.tool_dry_run import tool_response

    servers = _remote_host_servers()
    monkeypatch.setattr(multi_server, "SERVERS", servers)
    monkeypatch.setattr(routes.router, "prefix", "")
    monkeypatch.setattr(routes, "EFFECT_TOOLS", global_effect_tools)
    monkeypatch.setattr(sse_routes, "EFFECT_TOOLS", global_effect_tools)

    calls = {"list_tools": [], "real": [], "dry_run": []}
    tools = [
        {"name": "list_zones", "inputSchema": {"type": "object"}},
        {"name": "delete_zone", "inputSchema": {"type": "object"}},
        {"name": "create_page", "inputSchema": {"type": "object"}},
        {
            "name": "purge_preview",
            "inputSchema": {"type": "object"},
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "get_and_reset",
            "inputSchema": {"type": "object"},
            "annotations": {"readOnlyHint": False},
        },
    ]

    def _result(text):
        return {"isError": False, "content": [{"type": "text", "text": text}]}

    class FakeSession:
        async def list_tools(self):
            calls["list_tools"].append(multi_server.current_server_id())
            return tools

        async def call_tool(self, tool_name, args, access_token):
            calls["real"].append((multi_server.current_server_id(), tool_name))
            return _result("real")

        async def call_tool_with_progress(
            self, tool_name, args, access_token, progress_callback=None, **_kwargs
        ):
            calls["real"].append((multi_server.current_server_id(), tool_name))
            return _result("real")

    @asynccontextmanager
    async def fake_context(*args, **kwargs):
        yield FakeSession()

    async def fake_dry_run_response(session, tool, tool_input):
        calls["dry_run"].append(
            (multi_server.current_server_id(), (tool or {}).get("name"))
        )
        return {
            "isError": False,
            "content": [],
            "structuredContent": {"result": "dry-run"},
        }

    monkeypatch.setattr(routes, "mcp_session_context", fake_context)
    monkeypatch.setattr(sse_routes, "mcp_session_context", fake_context)
    monkeypatch.setattr(
        tool_response, "get_tool_dry_run_response", fake_dry_run_response
    )

    app = FastAPI()
    app.add_middleware(multi_server.MultiServerContextMiddleware)
    for server in servers:
        app.include_router(routes.router, prefix=server.base_path)
    return app, calls


def _call_tool(client, base_path, tool_name, *, stream=False, dry_run=True):
    headers = {"X-Inxm-Dry-Run": "true"} if dry_run else {}
    if not stream:
        response = client.post(
            f"{base_path}/tools/{tool_name}", headers=headers, json={}
        )
        assert response.status_code == 200, response.text
        return response.json()
    response = client.post(
        f"{base_path}/tools/{tool_name}/stream", headers=headers, json={}
    )
    assert response.status_code == 200, response.text
    events = [
        json.loads(line[6:])
        for line in response.content.decode("utf-8").split("\n")
        if line.startswith("data: ")
    ]
    assert events and events[-1]["type"] == "result", events
    return events[-1]["data"]


@pytest.mark.parametrize("stream", [False, True], ids=["rest", "sse"])
def test_dry_run_uses_per_server_effect_tools(monkeypatch, stream):
    from fastapi.testclient import TestClient

    app, calls = _dry_run_app(monkeypatch, global_effect_tools=["delete_*"])

    def outcome(base_path, tool_name):
        dry_before = len(calls["dry_run"])
        real_before = len(calls["real"])
        _call_tool(client, base_path, tool_name, stream=stream)
        if len(calls["dry_run"]) > dry_before:
            assert len(calls["real"]) == real_before
            return "dry-run"
        assert len(calls["real"]) == real_before + 1
        return "real"

    with TestClient(app) as client:
        cloudflare = "/api/mcp-cloudflare-server"  # ["auto"]
        notion = "/api/mcp-notion-server"  # ["create_*", "auto"]
        passthrough = "/api/passthrough"  # ["create_*"]
        inherit = "/api/inherit"  # global ["delete_*"]

        expectations = [
            (cloudflare, "list_zones", "real"),
            (cloudflare, "delete_zone", "dry-run"),
            (cloudflare, "create_page", "dry-run"),
            (cloudflare, "purge_preview", "real"),  # readOnlyHint=True
            (cloudflare, "get_and_reset", "dry-run"),  # readOnlyHint=False
            (cloudflare, "unknown_tool", "dry-run"),
            (notion, "create_page", "dry-run"),
            (notion, "purge_preview", "real"),
            (notion, "delete_zone", "dry-run"),
            (passthrough, "create_page", "dry-run"),
            # The server's globs replace the global ones, and no auto.
            (passthrough, "delete_zone", "real"),
            (passthrough, "get_and_reset", "real"),
            (inherit, "delete_zone", "dry-run"),
            (inherit, "create_page", "real"),
            (inherit, "get_and_reset", "real"),
        ]
        for base_path, tool_name, expected in expectations:
            assert outcome(base_path, tool_name) == expected, (base_path, tool_name)

        # Each dry run got the tool definition, from the server it targeted.
        assert ("mcp-cloudflare-server", "get_and_reset") in calls["dry_run"]
        assert ("passthrough", "create_page") in calls["dry_run"]

        # Auto decision + dry-run result share one tool listing.
        calls["list_tools"].clear()
        assert outcome(cloudflare, "delete_zone") == "dry-run"
        assert calls["list_tools"] == ["mcp-cloudflare-server"]

        # Without the header, auto never lists tools and nothing is simulated.
        calls["list_tools"].clear()
        dry_before = len(calls["dry_run"])
        for base_path in (cloudflare, notion):
            _call_tool(client, base_path, "delete_zone", stream=stream, dry_run=False)
        assert calls["list_tools"] == []
        assert len(calls["dry_run"]) == dry_before


def test_single_server_dry_run_with_global_auto(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app import routes

    _app, calls = _dry_run_app(monkeypatch, global_effect_tools=["auto"])
    monkeypatch.setattr(multi_server, "SERVERS", ())
    app = FastAPI()
    app.add_middleware(multi_server.MultiServerContextMiddleware)
    app.include_router(routes.router)

    with TestClient(app) as client:
        _call_tool(client, "", "list_zones")
        _call_tool(client, "", "purge_preview")
        _call_tool(client, "", "delete_zone")
        _call_tool(client, "", "create_page", stream=True)

    assert calls["real"] == [(None, "list_zones"), (None, "purge_preview")]
    assert calls["dry_run"] == [(None, "delete_zone"), (None, "create_page")]


def test_request_local_advertised_paths(monkeypatch):
    from types import SimpleNamespace

    from app.app_facade import route as app_route
    from app.well_known import oauth_metadata

    server = multi_server.ServerConfig(
        id="alpha",
        base_path="/api/mcp/alpha",
        command="python alpha.py",
    )
    token = multi_server.bind_server(server)
    try:
        assert app_route._proxy_prefix() == "/api/mcp/alpha/app"

        class Headers(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        request = SimpleNamespace(
            headers=Headers({"host": "bridge.example"}),
            url=SimpleNamespace(scheme="https"),
        )
        monkeypatch.setattr(oauth_metadata, "MCP_OAUTH_RESOURCE_URL", None)
        monkeypatch.setattr(
            oauth_metadata, "MCP_OAUTH_ISSUER", "https://issuer.example"
        )

        import asyncio

        response = asyncio.run(oauth_metadata.get_protected_resource_metadata(request))
        payload = json.loads(response.body)
        assert payload["resource"] == "https://bridge.example/api/mcp/alpha"
    finally:
        multi_server.reset_server(token)


def test_a2a_routes_can_be_mounted_per_server():
    from app.tgi.a2a_runtime import build_a2a_app

    app = build_a2a_app(base_path="/api/mcp/alpha")
    paths = {getattr(route, "path", None) for route in app.routes}

    assert "/api/mcp/alpha/tgi/v1/a2a" in paths
    assert "/api/mcp/alpha/.well-known/agent-card.json" in paths
    assert "/api/mcp/alpha/.well-known/agent.json" in paths


def test_overlapping_paths_use_server_specific_session_cookies():
    from http.cookies import SimpleCookie

    alpha, nested = _servers()
    scope = {
        "headers": [
            (
                b"cookie",
                (
                    "x-inxm-mcp-session.alpha=parent-session; "
                    "x-inxm-mcp-session.nested=nested-session"
                ).encode("latin-1"),
            )
        ]
    }

    multi_server._rewrite_session_cookie_header(scope, nested)

    cookie = SimpleCookie()
    cookie.load(scope["headers"][0][1].decode("latin-1"))
    assert cookie["x-inxm-mcp-session"].value == "nested-session"
    assert cookie["x-inxm-mcp-session.alpha"].value == "parent-session"


def test_generated_ui_artifacts_use_request_local_base_path():
    from app.app_facade.generated_output_factory import _mcp_service_class_source
    from app.app_facade.generated_service import _load_pfusch_prompt

    server = multi_server.ServerConfig(
        id="alpha",
        base_path="/api/mcp/alpha",
        command="python alpha.py",
    )
    token = multi_server.bind_server(server)
    try:
        service_source = _mcp_service_class_source()
        prompt = _load_pfusch_prompt()
        assert "/api/mcp/alpha/tools" in service_source
        assert "/api/mcp/alpha/tgi/v1/chat/completions" in service_source
        # The prompt is path-free: generated UIs reach tools via McpService.
        assert "{{MCP_BASE_PATH}}" not in prompt
        assert "/api/mcp/alpha" not in prompt
    finally:
        multi_server.reset_server(token)


def test_metadata_identity_uses_request_local_remote_url(monkeypatch):
    from app.oauth import token_exchange

    captured = {}

    class Response:
        text = "token"

        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "token"}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return Response()

    monkeypatch.setattr(token_exchange.requests, "get", fake_get)
    monkeypatch.setattr(token_exchange, "GCP_METADATA_IDENTITY_AUDIENCE", "")
    monkeypatch.setattr(token_exchange, "AZURE_METADATA_IDENTITY_RESOURCE", "")
    monkeypatch.setattr(
        token_exchange, "MCP_REMOTE_SERVER", "https://legacy.invalid/mcp"
    )

    server = multi_server.ServerConfig(
        id="remote",
        base_path="/api/mcp/remote",
        remote_url="https://remote.example/mcp",
    )
    token = multi_server.bind_server(server)
    try:
        token_exchange.GcpMetadataTokenRetriever()._fetch_token()
        assert captured["params"]["audience"] == "https://remote.example/mcp"

        token_exchange.AzureManagedIdentityTokenRetriever()._fetch_token()
        assert captured["params"]["resource"] == "https://remote.example/mcp"
    finally:
        multi_server.reset_server(token)
