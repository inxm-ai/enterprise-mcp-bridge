"""
Tests for the MCP-compliant SSE proxy endpoint.

Verifies that the SSE proxy correctly:
- Creates routes with proper paths
- Extracts access tokens from headers/cookies
- Builds a proxy MCP server with correct handlers
- Validates group access before opening SSE
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from mcp import types

from app.sse.mcp_proxy import (
    _build_proxy_server,
    _extract_access_token,
    get_sse_proxy_routes,
    sse_transport,
    _SSEConnectionApp,
    _SSEMessagesApp,
)


# ---------------------------------------------------------------------------
# Route / transport configuration
# ---------------------------------------------------------------------------


class TestRouteConfiguration:
    """Verify route paths and transport setup."""

    def test_routes_created(self):
        routes = get_sse_proxy_routes()
        assert len(routes) == 2

    def test_sse_route_path(self):
        routes = get_sse_proxy_routes()
        paths = [r.path for r in routes]
        assert "/sse" in paths

    def test_messages_route_path(self):
        routes = get_sse_proxy_routes()
        paths = [r.path for r in routes]
        assert "/sse/messages" in paths

    def test_messages_route_post_only(self):
        routes = get_sse_proxy_routes()
        msg_route = next(r for r in routes if r.path.endswith("/messages"))
        assert msg_route.methods == {"POST"}

    def test_sse_route_endpoint_is_asgi_app(self):
        routes = get_sse_proxy_routes()
        sse_route = next(r for r in routes if r.path == "/sse")
        # Should be a callable class instance, not wrapped by request_response
        assert isinstance(sse_route.endpoint, _SSEConnectionApp)

    def test_messages_route_endpoint_is_asgi_app(self):
        routes = get_sse_proxy_routes()
        msg_route = next(r for r in routes if r.path.endswith("/messages"))
        assert isinstance(msg_route.endpoint, _SSEMessagesApp)

    def test_message_endpoint_path_no_base(self):
        with patch("app.sse.mcp_proxy.MCP_BASE_PATH", ""):
            from app.sse.mcp_proxy import _message_endpoint_path

            assert _message_endpoint_path() == "/sse/messages"

    def test_message_endpoint_path_with_base(self):
        with patch("app.sse.mcp_proxy.MCP_BASE_PATH", "/api"):
            from app.sse.mcp_proxy import _message_endpoint_path

            assert _message_endpoint_path() == "/api/sse/messages"

    def test_transport_endpoint(self):
        assert sse_transport._endpoint == "/sse/messages"


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


class TestTokenExtraction:
    def test_extract_from_header(self):
        request = MagicMock()
        request.headers = {"X-Auth-Request-Access-Token": "tok_123"}
        request.cookies = {}
        with patch("app.sse.mcp_proxy.TOKEN_SOURCE", "header"), patch(
            "app.sse.mcp_proxy.TOKEN_NAME", "X-Auth-Request-Access-Token"
        ):
            assert _extract_access_token(request) == "tok_123"

    def test_extract_from_cookie(self):
        request = MagicMock()
        request.headers = {}
        request.cookies = {"_oauth2_proxy": "cookie_tok"}
        with patch("app.sse.mcp_proxy.TOKEN_SOURCE", "cookie"), patch(
            "app.sse.mcp_proxy.TOKEN_COOKIE_NAME", "_oauth2_proxy"
        ):
            assert _extract_access_token(request) == "cookie_tok"

    def test_extract_returns_none_when_missing(self):
        request = MagicMock()
        request.headers = {}
        request.cookies = {}
        with patch("app.sse.mcp_proxy.TOKEN_SOURCE", "header"), patch(
            "app.sse.mcp_proxy.TOKEN_NAME", "X-Auth-Request-Access-Token"
        ):
            assert _extract_access_token(request) is None


# ---------------------------------------------------------------------------
# Proxy server builder
# ---------------------------------------------------------------------------


def _make_mock_downstream():
    """Create a mock downstream MCP client session."""
    downstream = AsyncMock()

    # list_tools
    tool = types.Tool(
        name="my_tool",
        description="A test tool",
        inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    downstream.list_tools.return_value = SimpleNamespace(tools=[tool])

    # call_tool
    downstream.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="result")],
        isError=False,
    )

    # list_prompts
    prompt = types.Prompt(name="my_prompt", description="A test prompt")
    downstream.list_prompts.return_value = SimpleNamespace(prompts=[prompt])

    # get_prompt
    downstream.get_prompt.return_value = types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="assistant",
                content=types.TextContent(type="text", text="hello"),
            )
        ]
    )

    # list_resources
    downstream.list_resources.return_value = SimpleNamespace(resources=[])

    # read_resource
    downstream.read_resource.return_value = types.ReadResourceResult(contents=[])

    return downstream


class TestProxyServerBuilder:
    """Verify the proxy Server is correctly wired."""

    def test_proxy_has_handlers(self):
        downstream = _make_mock_downstream()
        proxy = _build_proxy_server(
            downstream, access_token=None, incoming_headers=None
        )
        # Should have handlers for standard MCP request types
        assert types.ListToolsRequest in proxy.request_handlers
        assert types.CallToolRequest in proxy.request_handlers
        assert types.ListPromptsRequest in proxy.request_handlers
        assert types.GetPromptRequest in proxy.request_handlers
        assert types.ListResourcesRequest in proxy.request_handlers
        assert types.ReadResourceRequest in proxy.request_handlers

    def test_proxy_server_name(self):
        downstream = _make_mock_downstream()
        proxy = _build_proxy_server(
            downstream, access_token=None, incoming_headers=None
        )
        assert proxy.name == "enterprise-mcp-bridge"

    @pytest.mark.asyncio
    async def test_list_tools_proxies_to_downstream(self):
        downstream = _make_mock_downstream()
        proxy = _build_proxy_server(
            downstream, access_token=None, incoming_headers=None
        )

        handler = proxy.request_handlers[types.ListToolsRequest]
        req = types.ListToolsRequest(method="tools/list")
        result = await handler(req)

        downstream.list_tools.assert_awaited_once()
        assert result.root.tools is not None

    @pytest.mark.asyncio
    async def test_call_tool_proxies_to_downstream(self):
        downstream = _make_mock_downstream()
        proxy = _build_proxy_server(downstream, access_token="tok", incoming_headers={})

        handler = proxy.request_handlers[types.CallToolRequest]
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="my_tool", arguments={"x": "val"}),
        )
        result = await handler(req)

        downstream.call_tool.assert_awaited_once()
        assert not result.root.isError

    @pytest.mark.asyncio
    async def test_list_prompts_proxies_to_downstream(self):
        downstream = _make_mock_downstream()
        proxy = _build_proxy_server(
            downstream, access_token=None, incoming_headers=None
        )

        handler = proxy.request_handlers[types.ListPromptsRequest]
        req = types.ListPromptsRequest(method="prompts/list")
        result = await handler(req)

        downstream.list_prompts.assert_awaited_once()
        assert result.root.prompts is not None

    @pytest.mark.asyncio
    async def test_get_prompt_proxies_to_downstream(self):
        downstream = _make_mock_downstream()
        proxy = _build_proxy_server(
            downstream, access_token=None, incoming_headers=None
        )

        handler = proxy.request_handlers[types.GetPromptRequest]
        req = types.GetPromptRequest(
            method="prompts/get",
            params=types.GetPromptRequestParams(name="my_prompt"),
        )
        await handler(req)

        downstream.get_prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_read_resource_proxies_to_downstream(self):
        downstream = _make_mock_downstream()
        proxy = _build_proxy_server(
            downstream, access_token=None, incoming_headers=None
        )

        handler = proxy.request_handlers[types.ReadResourceRequest]
        req = types.ReadResourceRequest(
            method="resources/read",
            params=types.ReadResourceRequestParams(uri="file:///test.txt"),
        )
        await handler(req)

        downstream.read_resource.assert_awaited_once()


# ---------------------------------------------------------------------------
# SSE connection app – group validation
# ---------------------------------------------------------------------------


class TestSSEConnectionAppGroupValidation:
    """Verify group access checks run before SSE stream opens."""

    @pytest.mark.asyncio
    async def test_group_permission_denied_returns_403(self):
        app = _SSEConnectionApp()

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/sse",
            "query_string": b"group=secret",
            "headers": [
                (b"x-auth-request-access-token", b"tok"),
            ],
            "root_path": "",
        }
        receive = AsyncMock()
        sent_responses = []

        async def capture_send(msg):
            sent_responses.append(msg)

        mock_data_mgr = MagicMock()
        mock_data_mgr.resolve_data_resource.side_effect = PermissionError("denied")

        with patch("app.sse.mcp_proxy.TOKEN_SOURCE", "header"), patch(
            "app.sse.mcp_proxy.TOKEN_NAME", "x-auth-request-access-token"
        ), patch(
            "app.oauth.user_info.get_data_access_manager", return_value=mock_data_mgr
        ):
            await app(scope, receive, capture_send)

        # Should have sent a 403 response
        status_msg = next(
            (m for m in sent_responses if m.get("type") == "http.response.start"),
            None,
        )
        assert status_msg is not None
        assert status_msg["status"] == 403
