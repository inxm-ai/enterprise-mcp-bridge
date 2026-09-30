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
        [{"id": "x", "base_path": "/x", "command": "x", "url": "https://x"}],
        [{"id": "x", "base_path": "/x"}],
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
    assert multi_server.current_remote_url("https://legacy/mcp") == "https://legacy/mcp"
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
