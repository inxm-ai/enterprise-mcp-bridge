"""Tests for the tool exploration feature (GENERATED_UI_EXPLORE_TOOLS)."""

import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from app.app_facade.generated_service import (
    GeneratedUIService,
    GeneratedUIStorage,
    Scope,
)
from app.app_facade.gateway_explorer import (
    _TOOL_EXPLORE_NAME_PATTERNS,
    _TOOL_EXPLORE_EXCLUDED,
    _TOOL_EXPLORE_PLAN_SCHEMA,
    _TOOL_EXPLORE_PARSE_SCHEMA,
)


class DummyTGIService:
    def __init__(self):
        self.llm_client = None
        self.prompt_service = None
        self.tool_service = None


def _make_service():
    storage = GeneratedUIStorage(os.getcwd())
    return GeneratedUIService(storage=storage, tgi_service=DummyTGIService())


def _tool(name, description="", parameters=None, output_schema=None):
    """Helper to build a tool dict in OpenAI function-calling format."""
    t = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }
    if output_schema:
        t["function"]["outputSchema"] = output_schema
    return t


def _gateway_tools(
    include_get_servers=True,
    include_get_tool=True,
):
    """Build a standard set of gateway MCP tools."""
    tools = []
    if include_get_servers:
        tools.append(
            _tool(
                "get_servers",
                "List available MCP servers",
                {
                    "type": "object",
                    "properties": {"filter": {"type": "string"}},
                },
            )
        )
    tools.append(
        _tool(
            "get_tools",
            "List tools for a server",
            {
                "type": "object",
                "properties": {"server_id": {"type": "string"}},
                "required": ["server_id"],
            },
        )
    )
    if include_get_tool:
        tools.append(
            _tool(
                "get_tool",
                "Get full tool definition",
                {
                    "type": "object",
                    "properties": {
                        "server_id": {"type": "string"},
                        "tool_name": {"type": "string"},
                    },
                    "required": ["server_id", "tool_name"],
                },
            )
        )
    tools.append(
        _tool(
            "call_tool",
            "Call a tool on a server",
            {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "input_data": {"type": "object"},
                },
                "required": ["server_id", "tool_name", "input_data"],
            },
        )
    )
    return tools


def _make_tool_result(data: Any) -> SimpleNamespace:
    """Build a fake MCP tool-call result."""
    text = json.dumps(data) if not isinstance(data, str) else data
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        structuredContent=None,
    )


class GatewaySession:
    """Fake session that simulates a gateway MCP server."""

    def __init__(
        self,
        servers: Optional[List[Dict[str, Any]]] = None,
        tools_by_server: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        tool_details: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.servers = servers or []
        self.tools_by_server = tools_by_server or {}
        self.tool_details = tool_details or {}
        self.calls: List[Dict[str, Any]] = []

    async def call_tool(self, name, args, access_token):
        self.calls.append({"name": name, "args": args})
        if name == "get_servers":
            return _make_tool_result(self.servers)
        if name == "get_tools":
            sid = args.get("server_id", "")
            return _make_tool_result(self.tools_by_server.get(sid, []))
        if name == "get_tool":
            sid = args.get("server_id", "")
            tname = args.get("tool_name", "")
            key = f"{sid}/{tname}"
            return _make_tool_result(self.tool_details.get(key, {}))
        if name == "call_tool":
            return _make_tool_result({"result": "ok"})
        raise RuntimeError(f"Unknown tool: {name}")


# ===========================================================================
# _is_exploration_candidate
# ===========================================================================


class TestIsExplorationCandidate:
    def test_matches_list_tools_name(self):
        service = _make_service()
        assert service.gateway_explorer._is_exploration_candidate(
            _tool("list_tools", "List all tools")
        )

    def test_matches_get_tools_name(self):
        service = _make_service()
        assert service.gateway_explorer._is_exploration_candidate(
            _tool("get-tools", "Get tools")
        )

    def test_matches_discover_name(self):
        service = _make_service()
        assert service.gateway_explorer._is_exploration_candidate(
            _tool("discover_capabilities")
        )

    def test_matches_catalog(self):
        service = _make_service()
        assert service.gateway_explorer._is_exploration_candidate(_tool("tool_catalog"))

    def test_matches_by_description_keywords(self):
        service = _make_service()
        assert service.gateway_explorer._is_exploration_candidate(
            _tool("something", "List available tools and capabilities")
        )

    def test_rejects_normal_tool(self):
        service = _make_service()
        assert not service.gateway_explorer._is_exploration_candidate(
            _tool("get_weather", "Get weather data for a city")
        )

    def test_excludes_describe_tool(self):
        service = _make_service()
        assert not service.gateway_explorer._is_exploration_candidate(
            _tool("describe_tool", "Return the full schema for a tool")
        )

    def test_excludes_select_from_tool_response(self):
        service = _make_service()
        assert not service.gateway_explorer._is_exploration_candidate(
            _tool("select-from-tool-response", "Select from tool response")
        )


# ===========================================================================
# _normalize_discovered_tool
# ===========================================================================


class TestNormalizeDiscoveredTool:
    def test_basic_normalization(self):
        service = _make_service()
        raw = {
            "name": "sub_tool_1",
            "description": "A sub-tool",
            "inputSchema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        }
        result = service.gateway_explorer._normalize_discovered_tool(raw)
        assert result["type"] == "function"
        assert result["function"]["name"] == "sub_tool_1"
        assert result["function"]["description"] == "A sub-tool"
        assert result["function"]["parameters"]["type"] == "object"

    def test_with_output_schema(self):
        service = _make_service()
        raw = {
            "name": "tool_x",
            "outputSchema": {"type": "object"},
        }
        result = service.gateway_explorer._normalize_discovered_tool(raw)
        assert result["function"]["outputSchema"] == {"type": "object"}

    def test_without_output_schema(self):
        service = _make_service()
        raw = {"name": "tool_y"}
        result = service.gateway_explorer._normalize_discovered_tool(raw)
        assert "outputSchema" not in result["function"]

    def test_alt_key_names(self):
        service = _make_service()
        raw = {
            "name": "alt_tool",
            "input_schema": {"type": "object", "properties": {"a": {"type": "number"}}},
            "output_schema": {"type": "array"},
        }
        result = service.gateway_explorer._normalize_discovered_tool(raw)
        assert result["function"]["parameters"]["properties"]["a"]["type"] == "number"
        assert result["function"]["outputSchema"] == {"type": "array"}


# ===========================================================================
# _parse_discovered_tools_from_result
# ===========================================================================


class TestParseDiscoveredToolsFromResult:
    def test_parses_tools_array_from_content(self):
        service = _make_service()
        tools_list = [
            {"name": "sub_a", "description": "Sub A"},
            {"name": "sub_b", "description": "Sub B"},
        ]
        result = _make_tool_result(tools_list)
        discovered = service.gateway_explorer._parse_discovered_tools_from_result(
            result
        )
        assert len(discovered) == 2
        names = {d["function"]["name"] for d in discovered}
        assert names == {"sub_a", "sub_b"}

    def test_parses_tools_from_wrapper_object(self):
        service = _make_service()
        wrapper = {
            "tools": [
                {"name": "tool_1", "description": "T1"},
                {"name": "tool_2", "description": "T2"},
            ]
        }
        result = _make_tool_result(wrapper)
        discovered = service.gateway_explorer._parse_discovered_tools_from_result(
            result
        )
        assert len(discovered) == 2

    def test_parses_single_tool_from_top_level(self):
        service = _make_service()
        single = {"name": "only_tool", "description": "I am the only one"}
        result = _make_tool_result(single)
        discovered = service.gateway_explorer._parse_discovered_tools_from_result(
            result
        )
        assert len(discovered) == 1
        assert discovered[0]["function"]["name"] == "only_tool"

    def test_returns_empty_for_non_tool_data(self):
        service = _make_service()
        result = _make_tool_result({"temperature": 22, "city": "Berlin"})
        discovered = service.gateway_explorer._parse_discovered_tools_from_result(
            result
        )
        assert discovered == []

    def test_returns_empty_for_empty_content(self):
        service = _make_service()
        result = SimpleNamespace(content=[], structuredContent=None)
        discovered = service.gateway_explorer._parse_discovered_tools_from_result(
            result
        )
        assert discovered == []

    def test_parses_from_structured_content(self):
        service = _make_service()
        structured = [
            {"name": "struct_tool", "description": "From structured"},
        ]
        result = SimpleNamespace(content=None, structuredContent=structured)
        discovered = service.gateway_explorer._parse_discovered_tools_from_result(
            result
        )
        assert len(discovered) == 1
        assert discovered[0]["function"]["name"] == "struct_tool"

    def test_dict_result_with_items_key(self):
        service = _make_service()
        wrapper = {
            "items": [
                {"name": "item_tool", "description": "Found via items key"},
            ]
        }
        result = _make_tool_result(wrapper)
        discovered = service.gateway_explorer._parse_discovered_tools_from_result(
            result
        )
        assert len(discovered) == 1
        assert discovered[0]["function"]["name"] == "item_tool"

    def test_dict_result_with_capabilities_key(self):
        service = _make_service()
        wrapper = {
            "capabilities": [
                {"name": "cap_tool", "description": "A capability"},
            ]
        }
        result = _make_tool_result(wrapper)
        discovered = service.gateway_explorer._parse_discovered_tools_from_result(
            result
        )
        assert len(discovered) == 1


# ===========================================================================
# _detect_gateway_pattern
# ===========================================================================


class TestDetectGatewayPattern:
    def test_detects_full_gateway(self):
        service = _make_service()
        tools = _gateway_tools()
        result = service.gateway_explorer._detect_gateway_pattern(tools)
        assert result is not None
        assert "list_servers" in result
        assert "list_tools" in result
        assert "get_tool" in result
        assert "call_tool" in result

    def test_detects_minimal_gateway(self):
        """list_tools + call_tool is the minimum viable gateway."""
        service = _make_service()
        tools = _gateway_tools(include_get_servers=False, include_get_tool=False)
        result = service.gateway_explorer._detect_gateway_pattern(tools)
        assert result is not None
        assert "list_tools" in result
        assert "call_tool" in result

    def test_rejects_when_missing_call_tool(self):
        service = _make_service()
        tools = [_tool("get_tools"), _tool("get_servers")]
        result = service.gateway_explorer._detect_gateway_pattern(tools)
        assert result is None

    def test_rejects_when_missing_get_tools(self):
        service = _make_service()
        tools = [_tool("call_tool"), _tool("get_servers")]
        result = service.gateway_explorer._detect_gateway_pattern(tools)
        assert result is None

    def test_rejects_normal_tools(self):
        service = _make_service()
        tools = [_tool("get_weather"), _tool("send_email")]
        result = service.gateway_explorer._detect_gateway_pattern(tools)
        assert result is None

    def test_custom_gateway_names(self, monkeypatch):
        """Custom env-var names are used for gateway detection."""
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_LIST_SERVERS",
            "enumerate_servers",
        )
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_LIST_TOOLS",
            "enumerate_tools",
        )
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_GET_TOOL",
            "describe_tool",
        )
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_CALL_TOOL",
            "execute_tool",
        )
        # Rebuild the maps on a fresh service
        from app.app_facade.gateway_explorer import GatewayExplorer

        role_to_name, name_to_role = GatewayExplorer._build_gateway_maps()
        service = _make_service()
        service.gateway_explorer._gateway_role_to_name = role_to_name
        service.gateway_explorer._gateway_name_to_role = name_to_role

        # Default names should NOT match
        tools_default = _gateway_tools()
        assert service.gateway_explorer._detect_gateway_pattern(tools_default) is None

        # Custom names SHOULD match
        tools_custom = [
            _tool("enumerate_servers", "List servers"),
            _tool("enumerate_tools", "List tools"),
            _tool("describe_tool", "Get tool info"),
            _tool("execute_tool", "Invoke a tool"),
        ]
        result = service.gateway_explorer._detect_gateway_pattern(tools_custom)
        assert result is not None
        assert "list_servers" in result
        assert "list_tools" in result
        assert "get_tool" in result
        assert "call_tool" in result


# ===========================================================================
# _build_gateway_call_args
# ===========================================================================


class TestBuildGatewayCallArgs:
    def test_merges_defaults_with_resolved_template(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {"list_tools": {"prompt": "${prompt}"}},
        )
        service = _make_service()

        args = service.gateway_explorer._build_gateway_call_args(
            role="list_tools",
            defaults={"server_id": "github"},
            context={"prompt": "show open PRs", "server_id": "github"},
        )

        assert args == {"server_id": "github", "prompt": "show open PRs"}

    def test_missing_placeholder_omits_key(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {"list_tools": {"server_id": "${server_id}", "prompt": "${prompt}"}},
        )
        service = _make_service()

        args = service.gateway_explorer._build_gateway_call_args(
            role="list_tools",
            defaults={},
            context={"prompt": "show repos"},
        )

        assert args == {"prompt": "show repos"}
        assert "server_id" not in args

    def test_caps_prompt_before_template_substitution(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_PROMPT_ARG_MAX_CHARS",
            4,
        )
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {"list_tools": {"prompt": "${prompt}"}},
        )
        service = _make_service()

        args = service.gateway_explorer._build_gateway_call_args(
            role="list_tools",
            defaults={},
            context={
                "prompt": service.gateway_explorer._gateway_prompt_for_args(
                    "prompt-value"
                )
            },
        )

        assert args == {"prompt": "prom"}

    def test_substitutes_placeholder_inside_larger_string(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {
                "list_tools": {
                    "prompt": (
                        "${prompt}\n\n" "Note: The result will be used in a web UI."
                    )
                }
            },
        )
        service = _make_service()

        args = service.gateway_explorer._build_gateway_call_args(
            role="list_tools",
            defaults={},
            context={"prompt": "show my github work"},
        )

        assert args == {
            "prompt": "show my github work\n\nNote: The result will be used in a web UI."
        }


# ===========================================================================
# server-id extraction helpers
# ===========================================================================


class TestGatewayServerIdExtraction:
    def test_extracts_server_id_from_url(self):
        service = _make_service()
        summary = {
            "name": "list-chats",
            "url": "/api/mcp-m365-chat-server/tools/list-chats",
        }
        assert (
            service.gateway_explorer._extract_server_id_from_tool_summary(summary)
            == "mcp-m365-chat-server"
        )

    def test_extracts_server_id_from_meta_mcp_server_id(self):
        service = _make_service()
        summary = {
            "name": "list-chats",
            "meta": {"mcp_server_id": "/api/mcp-m365-chat-server/tools/list-chats"},
        }
        assert (
            service.gateway_explorer._extract_server_id_from_tool_summary(summary)
            == "mcp-m365-chat-server"
        )

    def test_respects_custom_server_id_field_order(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_SERVER_ID_FIELDS",
            ["meta.server_ref"],
        )
        service = _make_service()
        summary = {"meta": {"server_ref": "custom-server-id"}}
        assert (
            service.gateway_explorer._extract_server_id_from_tool_summary(summary)
            == "custom-server-id"
        )

    def test_respects_custom_server_id_url_regex(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_SERVER_ID_URL_REGEX",
            r"/tools/(?P<server_id>[^/?#]+)$",
        )
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_SERVER_ID_FIELDS",
            ["url"],
        )
        service = _make_service()
        summary = {"url": "https://example.local/custom/tools/server-x"}
        assert (
            service.gateway_explorer._extract_server_id_from_tool_summary(summary)
            == "server-x"
        )


# ===========================================================================
# _extract_text_from_result / _json_from_result
# ===========================================================================


class TestResultHelpers:
    def test_extract_text_from_content_list(self):
        service = _make_service()
        result = SimpleNamespace(
            content=[
                SimpleNamespace(text="hello"),
                SimpleNamespace(text=" world"),
            ],
            structuredContent=None,
        )
        assert (
            service.gateway_explorer._extract_text_from_result(result)
            == "hello\n world"
        )

    def test_extract_text_from_structured_content(self):
        service = _make_service()
        result = SimpleNamespace(
            content=None,
            structuredContent={"key": "value"},
        )
        text = service.gateway_explorer._extract_text_from_result(result)
        assert '"key"' in text
        assert '"value"' in text

    def test_extract_text_empty_result(self):
        service = _make_service()
        result = SimpleNamespace(content=[], structuredContent=None)
        assert service.gateway_explorer._extract_text_from_result(result) == ""

    def test_json_from_result_parses_json(self):
        service = _make_service()
        data = {"tools": ["a", "b"]}
        result = _make_tool_result(data)
        parsed = service.gateway_explorer._json_from_result(result)
        assert parsed == data

    def test_json_from_result_returns_none_for_non_json(self):
        service = _make_service()
        result = SimpleNamespace(
            content=[SimpleNamespace(text="not json at all")],
            structuredContent=None,
        )
        assert service.gateway_explorer._json_from_result(result) is None

    def test_json_from_result_returns_none_for_empty(self):
        service = _make_service()
        result = SimpleNamespace(content=[], structuredContent=None)
        assert service.gateway_explorer._json_from_result(result) is None


# ===========================================================================
# _explore_gateway
# ===========================================================================


class TestExploreGateway:
    @pytest.mark.asyncio
    async def test_discovers_tools_from_servers(self):
        service = _make_service()
        session = GatewaySession(
            servers=[
                {
                    "id": "github-mcp",
                    "description": "GitHub MCP server",
                    "tools": [
                        {"name": "list_pull_requests", "description": "List PRs"},
                    ],
                },
            ],
            tools_by_server={
                "github-mcp": [
                    {
                        "name": "list_pull_requests",
                        "description": "List PRs",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"repo": {"type": "string"}},
                        },
                    },
                    {"name": "create_issue", "description": "Create an issue"},
                ],
            },
        )
        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, context = await service.gateway_explorer.explore_gateway(
            session=session,
            gateway=gateway,
            prompt="show my PRs",
            access_token=None,
        )

        assert len(discovered) == 2
        names = {t["function"]["name"] for t in discovered}
        assert "list_pull_requests" in names
        assert "create_issue" in names

        # Every discovered tool should have _gateway metadata
        for t in discovered:
            assert "_gateway" in t
            assert t["_gateway"]["server_id"] == "github-mcp"
            assert t["_gateway"]["via_tool"] == "call_tool"

        # Context should have gateway guidance
        assert context["gateway_pattern"] is True
        assert "call_tool" in context["invocation_guidance"]
        assert len(context["discovered_tools"]) == 2

    @pytest.mark.asyncio
    async def test_discovers_from_multiple_servers(self):
        service = _make_service()
        session = GatewaySession(
            servers=[
                {"id": "github", "description": "GitHub", "tools": []},
                {"id": "jira", "description": "Jira", "tools": []},
            ],
            tools_by_server={
                "github": [
                    {"name": "list_prs", "description": "List PRs"},
                ],
                "jira": [
                    {"name": "search_issues", "description": "Search issues"},
                ],
            },
        )
        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, context = await service.gateway_explorer.explore_gateway(
            session=session,
            gateway=gateway,
            prompt="test",
            access_token=None,
        )

        assert len(discovered) == 2
        server_ids = {t["_gateway"]["server_id"] for t in discovered}
        assert server_ids == {"github", "jira"}

    @pytest.mark.asyncio
    async def test_step2_parses_multiple_json_tool_objects(self):
        service = _make_service()

        class MultiJsonToolsSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([{"id": "mcp-github-server"}])
                if name == "get_tools":
                    return SimpleNamespace(
                        isError=False,
                        content=[
                            SimpleNamespace(
                                text=json.dumps(
                                    {
                                        "name": "list_pull_requests",
                                        "description": "List PRs",
                                    }
                                )
                            ),
                            SimpleNamespace(
                                text=json.dumps(
                                    {
                                        "name": "list_issues",
                                        "description": "List issues",
                                    }
                                )
                            ),
                        ],
                        structuredContent=None,
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(
            _gateway_tools(include_get_tool=False)
        )
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=MultiJsonToolsSession(),
            gateway=gateway,
            prompt="show github work",
            access_token=None,
        )
        names = {t["function"]["name"] for t in discovered}
        assert "list_pull_requests" in names
        assert "list_issues" in names

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_servers(self):
        """When get_servers AND get_tools fallback both return empty."""
        service = _make_service()

        class EmptySession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, context = await service.gateway_explorer.explore_gateway(
            session=EmptySession(),
            gateway=gateway,
            prompt="test",
            access_token=None,
        )

        assert discovered == []
        assert context == {}

    @pytest.mark.asyncio
    async def test_handles_get_servers_failure(self):
        """When get_servers fails, fallback to get_tools() without server_id."""
        service = _make_service()

        class FailingSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    raise RuntimeError("Network error")
                if name == "get_tools":
                    return _make_tool_result([])  # fallback also empty
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, context = await service.gateway_explorer.explore_gateway(
            session=FailingSession(),
            gateway=gateway,
            prompt="test",
            access_token=None,
        )
        assert discovered == []

    @pytest.mark.asyncio
    async def test_handles_get_tools_failure_gracefully(self):
        service = _make_service()

        class PartialFailSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([{"id": "srv1"}])
                if name == "get_tools":
                    raise RuntimeError("Timeout")
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, context = await service.gateway_explorer.explore_gateway(
            session=PartialFailSession(),
            gateway=gateway,
            prompt="test",
            access_token=None,
        )
        assert discovered == []

    @pytest.mark.asyncio
    async def test_fallback_get_tools_without_server_id(self):
        """When get_servers returns empty, try get_tools() without server_id."""
        service = _make_service()

        class FallbackSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([])  # empty servers
                if name == "get_tools" and not args.get("server_id"):
                    return _make_tool_result(
                        [
                            {
                                "name": "list_prs",
                                "description": "List PRs",
                                "server_id": "github",
                            },
                            {
                                "name": "search_issues",
                                "description": "Search",
                                "server_id": "linear",
                            },
                        ]
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, context = await service.gateway_explorer.explore_gateway(
            session=FallbackSession(),
            gateway=gateway,
            prompt="test",
            access_token=None,
        )
        assert len(discovered) == 2
        names = {t["function"]["name"] for t in discovered}
        assert "list_prs" in names
        assert "search_issues" in names

    @pytest.mark.asyncio
    async def test_fallback_get_tools_derives_server_id_from_url(self):
        service = _make_service()

        class FallbackUrlSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([])
                if name == "get_tools" and not args.get("server_id"):
                    return _make_tool_result(
                        [
                            {
                                "name": "list_chats",
                                "description": "List chats",
                                "url": "/api/mcp-m365-chat-server/tools/list-chats",
                            },
                            {
                                "name": "list_issues",
                                "description": "List issues",
                                "url": "/api/mcp-linear-server/tools/list-issues",
                            },
                        ]
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=FallbackUrlSession(),
            gateway=gateway,
            prompt="show my work",
            access_token=None,
        )
        assert len(discovered) == 2
        server_ids = {t["_gateway"]["server_id"] for t in discovered}
        assert server_ids == {"mcp-m365-chat-server", "mcp-linear-server"}

    @pytest.mark.asyncio
    async def test_fallback_get_tools_derives_server_id_from_meta_mcp_server_id(self):
        service = _make_service()

        class FallbackMetaSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([])
                if name == "get_tools" and not args.get("server_id"):
                    return _make_tool_result(
                        [
                            {
                                "name": "list_chats",
                                "description": "List chats",
                                "meta": {
                                    "mcp_server_id": "/api/mcp-m365-chat-server/tools/list-chats"
                                },
                            }
                        ]
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=FallbackMetaSession(),
            gateway=gateway,
            prompt="show my work",
            access_token=None,
        )
        assert len(discovered) == 1
        assert discovered[0]["_gateway"]["server_id"] == "mcp-m365-chat-server"

    @pytest.mark.asyncio
    async def test_list_tools_receives_prompt_template_without_server_id(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {"list_tools": {"prompt": "${prompt}"}},
        )
        service = _make_service()
        seen_args: list = []

        class PromptFallbackSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([])
                if name == "get_tools":
                    seen_args.append(dict(args))
                    return _make_tool_result(
                        [{"name": "list_prs", "description": "List PRs"}]
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=PromptFallbackSession(),
            gateway=gateway,
            prompt="show github pull requests",
            access_token=None,
        )
        assert len(discovered) == 1
        assert seen_args[0] == {"prompt": "show github pull requests"}

    @pytest.mark.asyncio
    async def test_list_tools_receives_server_id_and_prompt(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {"list_tools": {"prompt": "${prompt}"}},
        )
        service = _make_service()
        seen_args: list = []

        class PromptPerServerSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([{"id": "github"}])
                if name == "get_tools":
                    seen_args.append(dict(args))
                    return _make_tool_result(
                        [{"name": "list_prs", "description": "List PRs"}]
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=PromptPerServerSession(),
            gateway=gateway,
            prompt="show github pull requests",
            access_token=None,
        )
        assert len(discovered) == 1
        assert seen_args[0] == {
            "server_id": "github",
            "prompt": "show github pull requests",
        }

    @pytest.mark.asyncio
    async def test_missing_server_id_placeholder_is_omitted(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {"list_tools": {"server_id": "${server_id}", "prompt": "${prompt}"}},
        )
        service = _make_service()
        seen_args: list = []

        class OmitServerIdSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([])
                if name == "get_tools":
                    seen_args.append(dict(args))
                    return _make_tool_result(
                        [{"name": "list_prs", "description": "List PRs"}]
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=OmitServerIdSession(),
            gateway=gateway,
            prompt="show github pull requests",
            access_token=None,
        )
        assert len(discovered) == 1
        assert seen_args[0] == {"prompt": "show github pull requests"}
        assert "server_id" not in seen_args[0]

    @pytest.mark.asyncio
    async def test_get_tool_template_resolves_server_and_tool_name(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {"get_tool": {"sid": "${server_id}", "name": "${tool_name}"}},
        )
        service = _make_service()
        get_tool_args: list = []

        class GetToolTemplateSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([{"id": "github"}])
                if name == "get_tools":
                    return _make_tool_result([{"name": "list_prs"}])
                if name == "get_tool":
                    get_tool_args.append(dict(args))
                    return _make_tool_result(
                        {
                            "name": "list_prs",
                            "description": "List pull requests",
                            "inputSchema": {"type": "object"},
                        }
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=GetToolTemplateSession(),
            gateway=gateway,
            prompt="show github pull requests",
            access_token=None,
        )
        assert len(discovered) == 1
        assert len(get_tool_args) == 1
        assert get_tool_args[0]["sid"] == "github"
        assert get_tool_args[0]["name"] == "list_prs"

    @pytest.mark.asyncio
    async def test_prompt_template_is_capped(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_PROMPT_ARG_MAX_CHARS",
            6,
        )
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {"list_tools": {"prompt": "${prompt}"}},
        )
        service = _make_service()
        seen_args: list = []

        class PromptCappedSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([{"id": "github"}])
                if name == "get_tools":
                    seen_args.append(dict(args))
                    return _make_tool_result([{"name": "list_prs"}])
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=PromptCappedSession(),
            gateway=gateway,
            prompt="show github pull requests",
            access_token=None,
        )
        assert len(discovered) == 1
        assert seen_args[0]["prompt"] == "show g"

    @pytest.mark.asyncio
    async def test_default_gateway_args_unchanged_without_templates(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {},
        )
        service = _make_service()
        seen_args: list = []

        class DefaultArgsSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([{"id": "github"}])
                if name == "get_tools":
                    seen_args.append(dict(args))
                    return _make_tool_result([{"name": "list_prs"}])
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=DefaultArgsSession(),
            gateway=gateway,
            prompt="show github pull requests",
            access_token=None,
        )
        assert len(discovered) == 1
        assert seen_args[0] == {"server_id": "github"}

    @pytest.mark.asyncio
    async def test_fallback_get_tools_no_server_id_in_response(self):
        """Fallback tools without server_id get assigned to _default."""
        service = _make_service()

        class FallbackSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([])  # empty servers
                if name == "get_tools" and not args.get("server_id"):
                    return _make_tool_result(
                        [
                            {"name": "do_stuff", "description": "Do stuff"},
                        ]
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=FallbackSession(),
            gateway=gateway,
            prompt="test",
            access_token=None,
        )
        assert len(discovered) == 1
        assert discovered[0]["_gateway"]["server_id"] == "_default"

    @pytest.mark.asyncio
    async def test_get_servers_error_content_triggers_fallback(self):
        """When get_servers returns error text (not exception), fallback runs."""
        service = _make_service()

        class ErrorContentSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    # Simulates the MCP returning an error message as text
                    return _make_tool_result("Error fetching servers: Expecting value")
                if name == "get_tools" and not args.get("server_id"):
                    return _make_tool_result(
                        [
                            {"name": "my_tool", "description": "Works"},
                        ]
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=ErrorContentSession(),
            gateway=gateway,
            prompt="test",
            access_token=None,
        )
        assert len(discovered) == 1
        assert discovered[0]["function"]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_server_entries_with_name_field(self):
        """Servers identified by 'name' instead of 'id' should work."""
        service = _make_service()
        session = GatewaySession(
            servers=[{"name": "github", "description": "GitHub MCP"}],
            tools_by_server={
                "github": [{"name": "list_prs", "description": "List PRs"}],
            },
        )
        # Patch GatewaySession to respond to server name
        orig_call = session.call_tool

        async def patched_call(name, args, access_token):
            if name == "get_servers":
                return _make_tool_result(
                    [{"name": "github", "description": "GitHub MCP"}]
                )
            return await orig_call(name, args, access_token)

        session.call_tool = patched_call

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=session,
            gateway=gateway,
            prompt="test",
            access_token=None,
        )
        assert len(discovered) == 1
        assert discovered[0]["_gateway"]["server_id"] == "github"

    @pytest.mark.asyncio
    async def test_fetches_full_definition_via_get_tool(self):
        service = _make_service()
        session = GatewaySession(
            servers=[{"id": "srv", "tools": [{"name": "do_thing"}]}],
            tools_by_server={
                "srv": [{"name": "do_thing", "description": "Do something"}],
            },
            tool_details={
                "srv/do_thing": {
                    "name": "do_thing",
                    "description": "Do something amazing",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                    },
                    "outputSchema": {"type": "object"},
                },
            },
        )
        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, context = await service.gateway_explorer.explore_gateway(
            session=session,
            gateway=gateway,
            prompt="test",
            access_token=None,
        )

        assert len(discovered) == 1
        fn = discovered[0]["function"]
        assert fn["description"] == "Do something amazing"
        assert fn["parameters"]["properties"]["x"]["type"] == "integer"
        assert fn["outputSchema"] == {"type": "object"}

        # Verify get_tool was called
        call_names = [c["name"] for c in session.calls]
        assert "get_tool" in call_names

    @pytest.mark.asyncio
    async def test_respects_max_calls_budget(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS_MAX_CALLS", 2
        )
        service = _make_service()
        session = GatewaySession(
            servers=[
                {"id": "srv1", "tools": []},
                {"id": "srv2", "tools": []},
                {"id": "srv3", "tools": []},
            ],
            tools_by_server={
                "srv1": [{"name": "tool_a"}],
                "srv2": [{"name": "tool_b"}],
                "srv3": [{"name": "tool_c"}],
            },
        )
        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        discovered, _ = await service.gateway_explorer.explore_gateway(
            session=session,
            gateway=gateway,
            prompt="test",
            access_token=None,
        )

        # With max_calls=2, we get 1 for get_servers + 1 for first server only
        get_tools_calls = [c for c in session.calls if c["name"] == "get_tools"]
        assert len(get_tools_calls) <= 1

    @pytest.mark.asyncio
    async def test_skips_get_tool_without_budget(self, monkeypatch):
        """When budget is exhausted, get_tool should not be called."""
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS_MAX_CALLS", 2
        )
        service = _make_service()
        session = GatewaySession(
            servers=[{"id": "srv", "tools": []}],
            tools_by_server={
                "srv": [
                    {"name": "tool_a", "description": "A"},
                    {"name": "tool_b", "description": "B"},
                ],
            },
            tool_details={
                "srv/tool_a": {"name": "tool_a", "inputSchema": {"type": "object"}},
            },
        )
        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        await service.gateway_explorer.explore_gateway(
            session=session,
            gateway=gateway,
            prompt="test",
            access_token=None,
        )

        # get_tool should have been skipped due to budget
        get_tool_calls = [c for c in session.calls if c["name"] == "get_tool"]
        assert len(get_tool_calls) == 0

    @pytest.mark.asyncio
    async def test_without_get_servers(self):
        """Without get_servers, fallback to get_tools() without server_id."""
        service = _make_service()

        class FallbackSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_tools" and not args.get("server_id"):
                    return _make_tool_result(
                        [
                            {"name": "my_tool", "description": "A tool"},
                        ]
                    )
                return _make_tool_result([])

        tools = _gateway_tools(include_get_servers=False)
        gateway = service.gateway_explorer._detect_gateway_pattern(tools)
        discovered, context = await service.gateway_explorer.explore_gateway(
            session=FallbackSession(),
            gateway=gateway,
            prompt="test",
            access_token=None,
        )
        assert len(discovered) == 1
        assert discovered[0]["function"]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_step2_logs_non_json_response(self, caplog):
        """Step 2 logs raw content when get_tools returns non-JSON."""
        service = _make_service()

        class NonJsonSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([{"id": "srv1"}])
                if name == "get_tools":
                    return SimpleNamespace(
                        content=[SimpleNamespace(text="<html>Bad Gateway</html>")],
                        structuredContent=None,
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        uvicorn_logger = logging.getLogger("uvicorn.error")
        orig_propagate = uvicorn_logger.propagate
        uvicorn_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
                await service.gateway_explorer.explore_gateway(
                    session=NonJsonSession(),
                    gateway=gateway,
                    prompt="test",
                    access_token=None,
                )
            assert any(
                "non-JSON response" in r.message and "Bad Gateway" in r.message
                for r in caplog.records
            )
        finally:
            uvicorn_logger.propagate = orig_propagate

    @pytest.mark.asyncio
    async def test_step2_logs_error_result(self, caplog):
        """Step 2 logs raw content when get_tools returns isError."""
        service = _make_service()

        class ErrorToolsSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([{"id": "srv1"}])
                if name == "get_tools":
                    return _make_error_tool_result("Expecting value: line 1 column 1")
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        uvicorn_logger = logging.getLogger("uvicorn.error")
        orig_propagate = uvicorn_logger.propagate
        uvicorn_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
                await service.gateway_explorer.explore_gateway(
                    session=ErrorToolsSession(),
                    gateway=gateway,
                    prompt="test",
                    access_token=None,
                )
            assert any(
                "returned error" in r.message and "Expecting value" in r.message
                for r in caplog.records
            )
        finally:
            uvicorn_logger.propagate = orig_propagate

    @pytest.mark.asyncio
    async def test_step1b_logs_non_json_response(self, caplog):
        """Step 1b logs raw content when get_tools() returns non-JSON."""
        service = _make_service()

        class NonJsonFallbackSession:
            async def call_tool(self, name, args, access_token):
                if name == "get_servers":
                    return _make_tool_result([])  # empty servers
                if name == "get_tools":
                    return SimpleNamespace(
                        content=[SimpleNamespace(text="upstream timeout error 503")],
                        structuredContent=None,
                    )
                return _make_tool_result([])

        gateway = service.gateway_explorer._detect_gateway_pattern(_gateway_tools())
        uvicorn_logger = logging.getLogger("uvicorn.error")
        orig_propagate = uvicorn_logger.propagate
        uvicorn_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
                await service.gateway_explorer.explore_gateway(
                    session=NonJsonFallbackSession(),
                    gateway=gateway,
                    prompt="test",
                    access_token=None,
                )
            assert any(
                "non-JSON response" in r.message and "503" in r.message
                for r in caplog.records
            )
        finally:
            uvicorn_logger.propagate = orig_propagate


def _make_error_tool_result(text: str) -> SimpleNamespace:
    """Build a fake MCP tool-call result with isError=True."""
    return SimpleNamespace(
        isError=True,
        content=[SimpleNamespace(text=text)],
        structuredContent=None,
    )


# ===========================================================================
# _result_is_error
# ===========================================================================


class TestResultIsError:
    def test_false_for_normal_result(self):
        service = _make_service()
        r = _make_tool_result({"ok": True})
        assert not service.gateway_explorer._result_is_error(r)

    def test_true_for_error_result(self):
        service = _make_service()
        r = _make_error_tool_result("Something went wrong")
        assert service.gateway_explorer._result_is_error(r)

    def test_true_for_dict_result(self):
        service = _make_service()
        assert service.gateway_explorer._result_is_error(
            {"isError": True, "content": []}
        )

    def test_false_for_dict_without_error(self):
        service = _make_service()
        assert not service.gateway_explorer._result_is_error(
            {"content": [{"text": "hi"}]}
        )

    def test_false_for_none(self):
        service = _make_service()
        assert not service.gateway_explorer._result_is_error(None)


# ===========================================================================
# _gateway_discover_servers
# ===========================================================================


class TestGatewayDiscoverServers:
    @pytest.mark.asyncio
    async def test_discovers_servers_first_attempt(self):
        service = _make_service()
        servers_info: list = []

        class OkSession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result([{"id": "gh", "description": "GitHub"}])

        ids, calls = await service.gateway_explorer.gateway_discover_servers(
            session=OkSession(),
            gw_list_servers="get_servers",
            prompt="test",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["gh"]
        assert calls == 1
        assert len(servers_info) == 1
        assert servers_info[0]["id"] == "gh"

    @pytest.mark.asyncio
    async def test_list_servers_receives_prompt_template(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_GATEWAY_ROLE_ARGS",
            {"list_servers": {"prompt": "${prompt}"}},
        )
        service = _make_service()
        servers_info: list = []
        seen_args: list = []

        class PromptServersSession:
            async def call_tool(self, name, args, access_token):
                seen_args.append(dict(args))
                return _make_tool_result([{"id": "gh"}])

        ids, calls = await service.gateway_explorer.gateway_discover_servers(
            session=PromptServersSession(),
            gw_list_servers="get_servers",
            prompt="show my tools",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["gh"]
        assert calls == 1
        assert seen_args[0] == {"prompt": "show my tools"}

    @pytest.mark.asyncio
    async def test_retries_on_error_result(self):
        """isError on first call triggers a retry."""
        service = _make_service()
        servers_info: list = []
        call_count = 0

        class RetrySession:
            async def call_tool(self, name, args, access_token):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _make_error_tool_result("Expecting value: line 1 column 1")
                return _make_tool_result([{"id": "srv1"}])

        # Patch asyncio.sleep to avoid waiting
        import asyncio

        orig_sleep = asyncio.sleep
        asyncio.sleep = AsyncMock()
        try:
            ids, calls = await service.gateway_explorer.gateway_discover_servers(
                session=RetrySession(),
                gw_list_servers="get_servers",
                prompt="test",
                servers_info=servers_info,
                access_token=None,
            )
        finally:
            asyncio.sleep = orig_sleep

        assert ids == ["srv1"]
        assert calls == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_exception(self):
        """Exception on first call triggers retry."""
        service = _make_service()
        servers_info: list = []
        call_count = 0

        class ExcSession:
            async def call_tool(self, name, args, access_token):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("Connection refused")
                return _make_tool_result([{"id": "recovered"}])

        import asyncio

        orig_sleep = asyncio.sleep
        asyncio.sleep = AsyncMock()
        try:
            ids, calls = await service.gateway_explorer.gateway_discover_servers(
                session=ExcSession(),
                gw_list_servers="get_servers",
                prompt="test",
                servers_info=servers_info,
                access_token=None,
            )
        finally:
            asyncio.sleep = orig_sleep

        assert ids == ["recovered"]
        assert calls == 2  # 1 exception-counted + 1 success
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_valid_empty_list(self):
        """A valid empty [] response should NOT trigger retry."""
        service = _make_service()
        servers_info: list = []
        call_count = 0

        class EmptySession:
            async def call_tool(self, name, args, access_token):
                nonlocal call_count
                call_count += 1
                return _make_tool_result([])

        ids, calls = await service.gateway_explorer.gateway_discover_servers(
            session=EmptySession(),
            gw_list_servers="get_servers",
            prompt="test",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == []
        assert calls == 1  # Only one call, no retry
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_unparseable_text(self):
        """Non-JSON text should trigger retry."""
        service = _make_service()
        servers_info: list = []
        call_count = 0

        class TextSession:
            async def call_tool(self, name, args, access_token):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _make_tool_result("Error fetching servers: Expecting value")
                return _make_tool_result([{"name": "fallback-srv"}])

        import asyncio

        orig_sleep = asyncio.sleep
        asyncio.sleep = AsyncMock()
        try:
            ids, calls = await service.gateway_explorer.gateway_discover_servers(
                session=TextSession(),
                gw_list_servers="get_servers",
                prompt="test",
                servers_info=servers_info,
                access_token=None,
            )
        finally:
            asyncio.sleep = orig_sleep

        assert ids == ["fallback-srv"]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_accepts_name_field_for_server_id(self):
        service = _make_service()
        servers_info: list = []

        class NameSession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result([{"name": "my-server"}])

        ids, calls = await service.gateway_explorer.gateway_discover_servers(
            session=NameSession(),
            gw_list_servers="get_servers",
            prompt="test",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["my-server"]

    @pytest.mark.asyncio
    async def test_parses_dict_with_servers_key(self):
        service = _make_service()
        servers_info: list = []

        class DictSession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result({"servers": [{"id": "a"}, {"id": "b"}]})

        ids, _ = await service.gateway_explorer.gateway_discover_servers(
            session=DictSession(),
            gw_list_servers="get_servers",
            prompt="test",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["a", "b"]

    @pytest.mark.asyncio
    async def test_compacts_server_records_for_context(self):
        service = _make_service()
        servers_info: list = []

        class VerboseServerSession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result(
                    [
                        {
                            "id": "gh",
                            "description": "GitHub",
                            "tools": [{"name": "very_large_tool_blob"}],
                            "extra": {"nested": "value"},
                        }
                    ]
                )

        ids, calls = await service.gateway_explorer.gateway_discover_servers(
            session=VerboseServerSession(),
            gw_list_servers="get_servers",
            prompt="test",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["gh"]
        assert calls == 1
        assert len(servers_info) == 1
        assert servers_info[0]["id"] == "gh"
        assert "tools" not in servers_info[0]
        assert "extra" not in servers_info[0]

    @pytest.mark.asyncio
    async def test_parses_single_server_object_payload(self):
        service = _make_service()
        servers_info: list = []

        class SingleServerSession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result(
                    {
                        "id": "mcp-m365-chat-server",
                        "description": "Microsoft 365 chat server",
                        "tools": [],
                    }
                )

        ids, calls = await service.gateway_explorer.gateway_discover_servers(
            session=SingleServerSession(),
            gw_list_servers="get_servers",
            prompt="test",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["mcp-m365-chat-server"]
        assert calls == 1
        assert len(servers_info) == 1
        assert servers_info[0]["id"] == "mcp-m365-chat-server"

    @pytest.mark.asyncio
    async def test_parses_multiple_content_entries_each_with_json_object(self):
        service = _make_service()
        servers_info: list = []

        class MultiContentSession:
            async def call_tool(self, name, args, access_token):
                return SimpleNamespace(
                    isError=False,
                    content=[
                        SimpleNamespace(
                            text=json.dumps(
                                {
                                    "id": "mcp-m365-chat-server",
                                    "description": "Microsoft 365 chat server",
                                }
                            )
                        ),
                        SimpleNamespace(
                            text=json.dumps(
                                {
                                    "id": "mcp-github-server",
                                    "description": "GitHub server",
                                }
                            )
                        ),
                    ],
                    structuredContent=None,
                )

        ids, calls = await service.gateway_explorer.gateway_discover_servers(
            session=MultiContentSession(),
            gw_list_servers="get_servers",
            prompt="test",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["mcp-m365-chat-server", "mcp-github-server"]
        assert calls == 1
        assert len(servers_info) == 2

    @pytest.mark.asyncio
    async def test_parses_newline_delimited_json_payload(self):
        service = _make_service()
        servers_info: list = []

        class NdjsonSession:
            async def call_tool(self, name, args, access_token):
                return SimpleNamespace(
                    isError=False,
                    content=[
                        SimpleNamespace(
                            text="\n".join(
                                [
                                    json.dumps({"id": "first-server"}),
                                    json.dumps({"id": "second-server"}),
                                ]
                            )
                        )
                    ],
                    structuredContent=None,
                )

        ids, calls = await service.gateway_explorer.gateway_discover_servers(
            session=NdjsonSession(),
            gw_list_servers="get_servers",
            prompt="test",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["first-server", "second-server"]
        assert calls == 1

    @pytest.mark.asyncio
    async def test_deduplicates_server_ids_preserving_order(self):
        service = _make_service()
        servers_info: list = []

        class DuplicateSession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result(
                    [
                        {"id": "dup"},
                        {"id": "dup"},
                        {"id": "unique"},
                    ]
                )

        ids, calls = await service.gateway_explorer.gateway_discover_servers(
            session=DuplicateSession(),
            gw_list_servers="get_servers",
            prompt="test",
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["dup", "unique"]
        assert calls == 1
        assert len(servers_info) == 2


# ===========================================================================
# _gateway_probe_servers_from_prompt
# ===========================================================================


class TestGatewayProbeServersFromPrompt:
    @pytest.mark.asyncio
    async def test_probes_github_from_prompt(self):
        service = _make_service()
        tool_summaries: dict = {}
        servers_info: list = []

        class ProbeSession:
            async def call_tool(self, name, args, access_token):
                if args.get("server_id") == "github":
                    return _make_tool_result(
                        [
                            {"name": "list_prs", "description": "List PRs"},
                        ]
                    )
                return _make_tool_result([])

        ids, calls = await service.gateway_explorer.gateway_probe_servers_from_prompt(
            session=ProbeSession(),
            gw_list_tools="get_tools",
            prompt="show me my github pull requests",
            tool_summaries_by_server=tool_summaries,
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == ["github"]
        assert calls == 1
        assert "github" in tool_summaries
        assert len(tool_summaries["github"]) == 1

    @pytest.mark.asyncio
    async def test_probes_multiple_services(self):
        service = _make_service()
        tool_summaries: dict = {}
        servers_info: list = []

        class MultiSession:
            async def call_tool(self, name, args, access_token):
                sid = args.get("server_id", "")
                if sid == "github":
                    return _make_tool_result([{"name": "list_prs"}])
                if sid == "linear":
                    return _make_tool_result([{"name": "get_issues"}])
                return _make_tool_result([])

        ids, calls = await service.gateway_explorer.gateway_probe_servers_from_prompt(
            session=MultiSession(),
            gw_list_tools="get_tools",
            prompt="show github PRs and linear issues",
            tool_summaries_by_server=tool_summaries,
            servers_info=servers_info,
            access_token=None,
        )
        assert set(ids) == {"github", "linear"}
        assert calls == 2

    @pytest.mark.asyncio
    async def test_no_candidates_in_prompt(self):
        service = _make_service()
        tool_summaries: dict = {}
        servers_info: list = []

        class AnySession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result([{"name": "x"}])

        ids, calls = await service.gateway_explorer.gateway_probe_servers_from_prompt(
            session=AnySession(),
            gw_list_tools="get_tools",
            prompt="show me a nice dashboard",
            tool_summaries_by_server=tool_summaries,
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == []
        assert calls == 0

    @pytest.mark.asyncio
    async def test_skips_error_results(self):
        service = _make_service()
        tool_summaries: dict = {}
        servers_info: list = []

        class ErrorSession:
            async def call_tool(self, name, args, access_token):
                return _make_error_tool_result("Not found")

        ids, calls = await service.gateway_explorer.gateway_probe_servers_from_prompt(
            session=ErrorSession(),
            gw_list_tools="get_tools",
            prompt="check github issues",
            tool_summaries_by_server=tool_summaries,
            servers_info=servers_info,
            access_token=None,
        )
        assert ids == []
        assert calls == 1

    @pytest.mark.asyncio
    async def test_respects_probe_budget(self):
        """Should not make more than 4 probe calls."""
        service = _make_service()
        tool_summaries: dict = {}
        servers_info: list = []
        call_count = 0

        class CountSession:
            async def call_tool(self, name, args, access_token):
                nonlocal call_count
                call_count += 1
                return _make_tool_result([])

        # Use a prompt with many known service names
        ids, calls = await service.gateway_explorer.gateway_probe_servers_from_prompt(
            session=CountSession(),
            gw_list_tools="get_tools",
            prompt="github gitlab jira slack notion linear confluence atlassian",
            tool_summaries_by_server=tool_summaries,
            servers_info=servers_info,
            access_token=None,
        )
        assert call_count <= 4

    @pytest.mark.asyncio
    async def test_logs_non_json_probe_result(self, caplog):
        """Non-JSON responses from probing are logged."""
        service = _make_service()
        tool_summaries: dict = {}
        servers_info: list = []

        class NonJsonSession:
            async def call_tool(self, name, args, access_token):
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(text="<html>503 Service Unavailable</html>")
                    ],
                    structuredContent=None,
                )

        uvicorn_logger = logging.getLogger("uvicorn.error")
        orig_propagate = uvicorn_logger.propagate
        uvicorn_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
                await service.gateway_explorer.gateway_probe_servers_from_prompt(
                    session=NonJsonSession(),
                    gw_list_tools="get_tools",
                    prompt="check github PRs",
                    tool_summaries_by_server=tool_summaries,
                    servers_info=servers_info,
                    access_token=None,
                )
            assert any(
                "non-JSON" in r.message and "503" in r.message for r in caplog.records
            )
        finally:
            uvicorn_logger.propagate = orig_propagate


# ===========================================================================
# _build_exploration_context
# ===========================================================================


class TestBuildExplorationContext:
    def test_builds_context_with_tools(self):
        service = _make_service()
        servers_info = [
            {"server_id": "gh", "description": "GitHub", "tool_names": ["list_prs"]}
        ]
        discovered = [
            {
                "type": "function",
                "function": {
                    "name": "list_prs",
                    "description": "List PRs",
                    "parameters": {"type": "object"},
                },
                "_gateway": {"server_id": "gh", "via_tool": "call_tool"},
            }
        ]
        ctx = service.gateway_explorer._build_exploration_context(
            servers_info=servers_info, discovered=discovered
        )
        assert ctx["gateway_pattern"] is True
        assert "call_tool" in ctx["invocation_guidance"]
        assert len(ctx["servers"]) == 1
        assert ctx["servers"][0]["server_id"] == "gh"
        assert len(ctx["discovered_tools"]) == 1
        assert ctx["discovered_tools"][0]["name"] == "list_prs"
        assert ctx["discovered_tools"][0]["server_id"] == "gh"

    def test_empty_discovered_tools(self):
        service = _make_service()
        ctx = service.gateway_explorer._build_exploration_context(
            servers_info=[], discovered=[]
        )
        assert ctx["gateway_pattern"] is True
        assert ctx["discovered_tools"] == []
        assert ctx["servers"] == []


# ===========================================================================
# _explore_tools – integration tests (returns tuple now)
# ===========================================================================


class TestExploreTools:
    @pytest.mark.asyncio
    async def test_noop_when_flag_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", False
        )
        service = _make_service()
        tools = [_tool("get_weather")]
        result_tools, result_ctx = await service.gateway_explorer.explore_tools(
            session=None, tools=tools, prompt="weather", access_token=None
        )
        assert result_tools is tools
        assert result_ctx is None

    @pytest.mark.asyncio
    async def test_noop_when_no_tools(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()
        result_tools, result_ctx = await service.gateway_explorer.explore_tools(
            session=None, tools=[], prompt="weather", access_token=None
        )
        assert result_tools == []
        assert result_ctx is None

    @pytest.mark.asyncio
    async def test_gateway_path_discovers_tools(self, monkeypatch):
        """When the gateway pattern is detected, use deterministic exploration."""
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()
        session = GatewaySession(
            servers=[{"id": "gh", "tools": [{"name": "list_prs"}]}],
            tools_by_server={
                "gh": [
                    {"name": "list_prs", "description": "List PRs"},
                    {"name": "create_issue", "description": "Create issue"},
                ],
            },
        )
        tools = _gateway_tools()
        result_tools, result_ctx = await service.gateway_explorer.explore_tools(
            session=session, tools=tools, prompt="show PRs", access_token=None
        )

        # Gateway meta-tools should be stripped; only discovered tools remain
        all_names = {t["function"]["name"] for t in result_tools}
        assert "list_prs" in all_names
        assert "create_issue" in all_names
        # Gateway meta-tools must NOT be in the result
        assert "get_servers" not in all_names
        assert "get_tools" not in all_names
        assert "get_tool" not in all_names
        assert "call_tool" not in all_names

        # Exploration context should be present
        assert result_ctx is not None
        assert result_ctx["gateway_pattern"] is True
        assert "call_tool" in result_ctx["invocation_guidance"]

    @pytest.mark.asyncio
    async def test_gateway_deduplicates_against_existing(self, monkeypatch):
        """Discovered tools that share a name with non-gateway tools are skipped."""
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()
        session = GatewaySession(
            servers=[{"id": "s", "tools": []}],
            tools_by_server={
                "s": [
                    # This name collides with a user-facing tool
                    {"name": "extra_tool", "description": "Duplicate!"},
                    {"name": "unique_tool", "description": "New"},
                ],
            },
        )
        tools = _gateway_tools() + [_tool("extra_tool", "Already here")]
        result_tools, _ = await service.gateway_explorer.explore_tools(
            session=session, tools=tools, prompt="test", access_token=None
        )

        names = [t["function"]["name"] for t in result_tools]
        assert names.count("extra_tool") == 1
        assert "unique_tool" in names
        # Gateway meta-tools should be stripped
        assert "get_servers" not in names
        assert "get_tools" not in names

    @pytest.mark.asyncio
    async def test_gateway_strips_metatools_even_on_failure(self, monkeypatch):
        """When gateway exploration fails, meta-tools are still stripped."""
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()

        class AllFailSession:
            async def call_tool(self, name, args, access_token):
                raise RuntimeError("Everything is broken")

        tools = _gateway_tools() + [_tool("extra_tool", "Non-gateway tool")]
        result_tools, result_ctx = await service.gateway_explorer.explore_tools(
            session=AllFailSession(), tools=tools, prompt="test", access_token=None
        )

        names = [t["function"]["name"] for t in result_tools]
        # Only non-gateway tools should remain
        assert names == ["extra_tool"]
        # Context should be empty since exploration failed
        assert result_ctx == {}

    @pytest.mark.asyncio
    async def test_generic_fallback_when_no_gateway(self, monkeypatch):
        """Non-gateway tools fall through to LLM-planned exploration."""
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()

        service.gateway_explorer._plan_exploration_calls = AsyncMock(
            return_value=[
                {
                    "tool_name": "list_tools",
                    "arguments": {},
                    "reason": "discover sub-tools",
                }
            ]
        )

        sub_tools = [
            {"name": "jira_search", "description": "Search Jira issues"},
            {"name": "jira_create", "description": "Create Jira issue"},
        ]

        class FakeSession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result(sub_tools)

        tools = [
            _tool("list_tools", "List available tools"),
            _tool("get_weather", "Get weather"),
        ]

        result_tools, result_ctx = await service.gateway_explorer.explore_tools(
            session=FakeSession(),
            tools=tools,
            prompt="I need Jira data",
            access_token="tok",
        )

        assert len(result_tools) == 4
        names = {t["function"]["name"] for t in result_tools}
        assert "jira_search" in names
        assert "jira_create" in names
        # Generic exploration returns None context
        assert result_ctx is None

    @pytest.mark.asyncio
    async def test_deduplicates_discovered_tools_generic(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()
        service.gateway_explorer._plan_exploration_calls = AsyncMock(
            return_value=[{"tool_name": "list_tools", "arguments": {}}]
        )

        sub_tools = [
            {"name": "get_weather", "description": "Already exists"},
            {"name": "new_tool", "description": "Brand new"},
        ]

        class FakeSession:
            async def call_tool(self, name, args, access_token):
                return _make_tool_result(sub_tools)

        tools = [
            _tool("list_tools", "List available tools"),
            _tool("get_weather", "Get weather"),
        ]

        result_tools, _ = await service.gateway_explorer.explore_tools(
            session=FakeSession(), tools=tools, prompt="test", access_token=None
        )

        assert len(result_tools) == 3
        names = [t["function"]["name"] for t in result_tools]
        assert names.count("get_weather") == 1
        assert "new_tool" in names

    @pytest.mark.asyncio
    async def test_handles_call_failure_gracefully(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()
        service.gateway_explorer._plan_exploration_calls = AsyncMock(
            return_value=[{"tool_name": "list_tools", "arguments": {}}]
        )

        class FailingSession:
            async def call_tool(self, name, args, access_token):
                raise RuntimeError("Connection refused")

        tools = [_tool("list_tools", "List available tools")]
        result_tools, _ = await service.gateway_explorer.explore_tools(
            session=FailingSession(), tools=tools, prompt="test", access_token=None
        )

        assert len(result_tools) == 1
        assert result_tools[0]["function"]["name"] == "list_tools"

    @pytest.mark.asyncio
    async def test_skips_excluded_tools_in_plan(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()
        service.gateway_explorer._plan_exploration_calls = AsyncMock(
            return_value=[{"tool_name": "describe_tool", "arguments": {"name": "x"}}]
        )

        call_count = 0

        class TrackingSession:
            async def call_tool(self, name, args, access_token):
                nonlocal call_count
                call_count += 1
                return SimpleNamespace(content=[], structuredContent=None)

        tools = [
            _tool("describe_tool", "Describe a tool"),
            _tool("get_weather", "Get weather"),
        ]
        result_tools, _ = await service.gateway_explorer.explore_tools(
            session=TrackingSession(), tools=tools, prompt="test", access_token=None
        )

        assert call_count == 0
        assert len(result_tools) == 2

    @pytest.mark.asyncio
    async def test_skips_tool_not_in_available_list(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()
        service.gateway_explorer._plan_exploration_calls = AsyncMock(
            return_value=[{"tool_name": "nonexistent_tool", "arguments": {}}]
        )

        class FakeSession:
            async def call_tool(self, name, args, access_token):
                raise AssertionError("Should not be called")

        tools = [_tool("get_weather", "Get weather")]
        result_tools, _ = await service.gateway_explorer.explore_tools(
            session=FakeSession(), tools=tools, prompt="test", access_token=None
        )

        assert len(result_tools) == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_llm_parsing(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()
        service.gateway_explorer._plan_exploration_calls = AsyncMock(
            return_value=[{"tool_name": "capabilities", "arguments": {}}]
        )

        class FakeSession:
            async def call_tool(self, name, args, access_token):
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            text="Here are our APIs: search, create, delete"
                        )
                    ],
                    structuredContent=None,
                )

        service.gateway_explorer._parse_discovered_tools_with_llm = AsyncMock(
            return_value=[
                _tool("search_api", "Search API"),
                _tool("create_api", "Create API"),
            ]
        )

        tools = [_tool("capabilities", "Get capabilities")]
        result_tools, _ = await service.gateway_explorer.explore_tools(
            session=FakeSession(), tools=tools, prompt="test", access_token=None
        )

        assert len(result_tools) == 3
        names = {t["function"]["name"] for t in result_tools}
        assert "search_api" in names
        assert "create_api" in names

    @pytest.mark.asyncio
    async def test_no_exploration_when_no_candidates(self, monkeypatch):
        """When no tools look like discovery tools, plan should get no candidates."""
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS", True
        )
        service = _make_service()

        plan_called = False

        async def mock_plan(*, tools, prompt, access_token):
            nonlocal plan_called
            plan_called = True
            return []

        service.gateway_explorer._plan_exploration_calls = mock_plan

        tools = [
            _tool("get_weather", "Get weather for a city"),
            _tool("send_email", "Send an email"),
        ]
        result_tools, _ = await service.gateway_explorer.explore_tools(
            session=None, tools=tools, prompt="weather", access_token=None
        )

        assert plan_called
        assert result_tools is tools


# ===========================================================================
# _plan_exploration_calls – LLM interaction
# ===========================================================================


class TestPlanExplorationCalls:
    @pytest.mark.asyncio
    async def test_returns_empty_without_llm_client(self):
        service = _make_service()
        result = await service.gateway_explorer._plan_exploration_calls(
            tools=[_tool("list_tools")], prompt="test", access_token=None
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_candidates(self):
        service = _make_service()
        service.tgi_service.llm_client = AsyncMock()
        service.tgi_service.llm_client.non_stream_completion = AsyncMock()
        result = await service.gateway_explorer._plan_exploration_calls(
            tools=[_tool("get_weather", "Get weather")],
            prompt="test",
            access_token=None,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_caps_calls_to_max(self, monkeypatch):
        monkeypatch.setattr(
            "app.app_facade.gateway_explorer.GENERATED_UI_EXPLORE_TOOLS_MAX_CALLS", 2
        )
        service = _make_service()

        llm_response = json.dumps(
            {
                "calls": [
                    {"tool_name": "list_tools", "arguments": {}},
                    {"tool_name": "get_tools", "arguments": {}},
                    {"tool_name": "discover", "arguments": {}},
                ]
            }
        )
        service.tgi_service.llm_client = AsyncMock()
        service.tgi_service.llm_client.non_stream_completion = AsyncMock(
            return_value={"content": llm_response}
        )

        tools = [
            _tool("list_tools", "List tools"),
            _tool("get_tools", "Get tools"),
            _tool("discover", "Discover capabilities"),
        ]
        result = await service.gateway_explorer._plan_exploration_calls(
            tools=tools, prompt="test", access_token=None
        )
        assert len(result) <= 2


# ===========================================================================
# _parse_discovered_tools_with_llm
# ===========================================================================


class TestParseDiscoveredToolsWithLLM:
    @pytest.mark.asyncio
    async def test_returns_empty_without_llm_client(self):
        service = _make_service()
        result = await service.gateway_explorer._parse_discovered_tools_with_llm(
            tool_name="list_tools",
            result=SimpleNamespace(content=[], structuredContent=None),
            access_token=None,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_parses_llm_response(self):
        service = _make_service()
        llm_response = json.dumps(
            {
                "tools": [
                    {"name": "parsed_tool", "description": "Found by LLM"},
                ]
            }
        )
        service.tgi_service.llm_client = AsyncMock()
        service.tgi_service.llm_client.non_stream_completion = AsyncMock(
            return_value={"content": llm_response}
        )

        result = await service.gateway_explorer._parse_discovered_tools_with_llm(
            tool_name="list_tools",
            result=SimpleNamespace(
                content=[SimpleNamespace(text="some ambiguous text")],
                structuredContent=None,
            ),
            access_token="tok",
        )

        assert len(result) == 1
        assert result[0]["function"]["name"] == "parsed_tool"

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self):
        service = _make_service()
        service.tgi_service.llm_client = AsyncMock()
        service.tgi_service.llm_client.non_stream_completion = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )

        result = await service.gateway_explorer._parse_discovered_tools_with_llm(
            tool_name="list_tools",
            result=SimpleNamespace(content=[], structuredContent=None),
            access_token=None,
        )
        assert result == []


# ===========================================================================
# _call_tool_for_dummy_sampling – gateway routing
# ===========================================================================


class TestCallToolForDummySamplingGatewayRouting:
    @pytest.mark.asyncio
    async def test_normal_tool_calls_directly(self):
        service = _make_service()
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=_make_tool_result({"data": "ok"}))

        tool_def = {"name": "get_weather", "inputSchema": {"type": "object"}}
        await service.tool_sampler._call_tool_for_dummy_sampling(
            session=session,
            tool_name="get_weather",
            tool_def=tool_def,
            sample_args={"city": "Berlin"},
            access_token="tok",
        )

        session.call_tool.assert_called_once_with(
            "get_weather", {"city": "Berlin"}, "tok"
        )

    @pytest.mark.asyncio
    async def test_gateway_tool_routes_through_call_tool(self):
        service = _make_service()
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=_make_tool_result({"data": "ok"}))

        tool_def = {
            "name": "list_pull_requests",
            "inputSchema": {"type": "object"},
            "_gateway": {
                "server_id": "github-mcp",
                "via_tool": "call_tool",
            },
        }
        await service.tool_sampler._call_tool_for_dummy_sampling(
            session=session,
            tool_name="list_pull_requests",
            tool_def=tool_def,
            sample_args={"repo": "my-repo"},
            access_token="tok",
        )

        session.call_tool.assert_called_once_with(
            "call_tool",
            {
                "server_id": "github-mcp",
                "tool_name": "list_pull_requests",
                "input_data": {"repo": "my-repo"},
            },
            "tok",
        )

    @pytest.mark.asyncio
    async def test_gateway_without_server_id_falls_through(self):
        """If _gateway exists but server_id is missing, call directly."""
        service = _make_service()
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=_make_tool_result({"data": "ok"}))

        tool_def = {
            "name": "some_tool",
            "inputSchema": {"type": "object"},
            "_gateway": {"via_tool": "call_tool"},  # Missing server_id
        }
        await service.tool_sampler._call_tool_for_dummy_sampling(
            session=session,
            tool_name="some_tool",
            tool_def=tool_def,
            sample_args={},
            access_token=None,
        )

        session.call_tool.assert_called_once_with("some_tool", {}, None)


# ===========================================================================
# Regex pattern tests
# ===========================================================================


class TestExploreNamePatterns:
    @pytest.mark.parametrize(
        "name",
        [
            "list_tools",
            "list-tools",
            "listTools",
            "get_tools",
            "get-tools",
            "available_tools",
            "discover",
            "capabilities",
            "tool_catalog",
            "registry",
        ],
    )
    def test_matches_exploration_patterns(self, name):
        assert _TOOL_EXPLORE_NAME_PATTERNS.search(name)

    @pytest.mark.parametrize(
        "name",
        [
            "get_weather",
            "send_email",
            "create_user",
            "search_issues",
        ],
    )
    def test_rejects_non_exploration_names(self, name):
        assert not _TOOL_EXPLORE_NAME_PATTERNS.search(name)
