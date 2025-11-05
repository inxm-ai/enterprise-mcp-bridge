import types
import pytest
from unittest.mock import AsyncMock

import app.session_manager.session_context as sc


@pytest.mark.asyncio
async def test_sessionless_call_tool_injects_headers(monkeypatch):
    # Prepare fake tools: toolA expects userId
    def dict_to_obj(d):
        filled = {
            k: d.get(k, None)
            for k in [
                "id",
                "name",
                "title",
                "description",
                "inputSchema",
                "outputSchema",
                "annotations",
                "meta",
            ]
        }
        filled.update(d)
        if not filled.get("id"):
            filled["id"] = d.get("id") or d.get("name")
        return types.SimpleNamespace(**filled)

    tools_list = [
        dict_to_obj(
            {
                "name": "toolA",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "userId": {"type": "string"},
                        "keep": {"type": "string"},
                    },
                },
            }
        )
    ]

    class _AsyncMCPContext:
        def __init__(self, tools):
            self._tools = types.SimpleNamespace(tools=tools)

        async def __aenter__(self):
            obj = types.SimpleNamespace()
            obj.list_tools = AsyncMock(return_value=self._tools)

            async def _call_tool(name, args):
                item = types.SimpleNamespace(
                    text="ok", structuredContent={"args": args}
                )
                return types.SimpleNamespace(
                    content=[item], structuredContent={"args": args}
                )

            obj.call_tool = AsyncMock(side_effect=_call_tool)
            return obj

        async def __aexit__(self, exc_type, exc, tb):
            return False

    # Patch mcp_session used inside session_context
    monkeypatch.setattr(sc, "mcp_session", lambda *a, **k: _AsyncMCPContext(tools_list))

    monkeypatch.setattr(
        sc,
        "decorate_args_with_oauth_token",
        AsyncMock(side_effect=lambda *a, **k: a[2]),
    )

    # Set mapping: userId comes from header x-auth-user-id
    monkeypatch.setattr(sc, "MCP_MAP_HEADER_TO_INPUT", {"userId": "x-auth-user-id"})

    incoming_headers = {"X-Auth-User-Id": "user-123"}

    async with sc.mcp_session_context(
        sessions=None,
        x_inxm_mcp_session=None,
        access_token=None,
        group=None,
        incoming_headers=incoming_headers,
    ) as delegate:
        tools = await delegate.list_tools()
        assert isinstance(tools, list)

        # call_tool should inject the header value for userId
        result = await delegate.call_tool("toolA", {"keep": "value"}, None)
        assert hasattr(result, "structuredContent")
        assert result.structuredContent["args"]["userId"] == "user-123"
        assert result.structuredContent["args"]["keep"] == "value"
