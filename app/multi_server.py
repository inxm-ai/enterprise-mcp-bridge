"""Multi-server configuration and request-local server selection.

MCP_SERVERS is an optional JSON array. When unset, the bridge keeps the
existing single-server environment-variable behavior.

Example:
[
  {
    "id": "filesystem",
    "base_path": "/api/mcp/filesystem",
    "command": "python -m my_filesystem_mcp",
    "env": {"ROOT": "/data"},
    "include_tools": ["read_*"]
  },
  {
    "id": "remote",
    "base_path": "/api/mcp/remote",
    "url": "https://mcp.example.com/mcp",
    "exclude_tools": ["admin_*"]
  }
]
"""

from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ServerConfig:
    id: str
    base_path: str
    command: Optional[str] = None
    remote_url: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    include_tools: tuple[str, ...] = ()
    exclude_tools: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ServerConfig":
        server_id = str(raw.get("id") or "").strip()
        if not server_id or not _ID_RE.match(server_id):
            raise ValueError(f"Invalid MCP server id: {server_id!r}")

        base_path = str(raw.get("base_path") or raw.get("path") or "").strip()
        if not base_path.startswith("/"):
            raise ValueError(
                f"MCP server {server_id!r} base_path must start with '/'"
            )
        base_path = base_path.rstrip("/") or "/"

        command = raw.get("command")
        remote_url = raw.get("url", raw.get("remote_url"))
        command = str(command).strip() if command else None
        remote_url = str(remote_url).strip() if remote_url else None
        if bool(command) == bool(remote_url):
            raise ValueError(
                f"MCP server {server_id!r} must define exactly one of command or url"
            )

        env_raw = raw.get("env") or {}
        if not isinstance(env_raw, dict):
            raise ValueError(f"MCP server {server_id!r} env must be an object")
        env = {str(k): str(v) for k, v in env_raw.items()}

        include = raw.get("include_tools") or []
        exclude = raw.get("exclude_tools") or []
        if not isinstance(include, list) or not isinstance(exclude, list):
            raise ValueError(
                f"MCP server {server_id!r} include_tools/exclude_tools must be arrays"
            )

        return cls(
            id=server_id,
            base_path=base_path,
            command=command,
            remote_url=remote_url,
            env=env,
            include_tools=tuple(str(v) for v in include if str(v)),
            exclude_tools=tuple(str(v) for v in exclude if str(v)),
        )


def parse_servers(raw: str) -> tuple[ServerConfig, ...]:
    raw = raw.strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MCP_SERVERS JSON: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("MCP_SERVERS must be a non-empty JSON array")

    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError("MCP_SERVERS entries must be JSON objects")
    servers = tuple(ServerConfig.from_mapping(item) for item in parsed)
    ids = [server.id for server in servers]
    paths = [server.base_path for server in servers]
    if len(ids) != len(set(ids)):
        raise ValueError("MCP_SERVERS contains duplicate server ids")
    if len(paths) != len(set(paths)):
        raise ValueError("MCP_SERVERS contains duplicate base paths")
    if "/" in paths and len(paths) > 1:
        raise ValueError("base_path '/' cannot be combined with other MCP servers")
    return servers


def _load_servers() -> tuple[ServerConfig, ...]:
    return parse_servers(os.environ.get("MCP_SERVERS", ""))


SERVERS = _load_servers()
_CURRENT_SERVER: ContextVar[Optional[ServerConfig]] = ContextVar(
    "enterprise_mcp_bridge_server", default=None
)


def configured_servers() -> tuple[ServerConfig, ...]:
    return SERVERS


def is_multi_server_mode() -> bool:
    return bool(SERVERS)


def current_server() -> Optional[ServerConfig]:
    return _CURRENT_SERVER.get()


def bind_server(server: ServerConfig) -> Token:
    return _CURRENT_SERVER.set(server)


def reset_server(token: Token) -> None:
    _CURRENT_SERVER.reset(token)


def match_server(path: str) -> Optional[ServerConfig]:
    matches = [
        server
        for server in SERVERS
        if path == server.base_path or path.startswith(server.base_path + "/")
    ]
    if not matches:
        return None
    return max(matches, key=lambda server: len(server.base_path))


def current_server_id() -> Optional[str]:
    server = current_server()
    return server.id if server else None


def current_base_path(default: str = "") -> str:
    server = current_server()
    return server.base_path if server else default


def current_remote_url(default: str = "") -> str:
    server = current_server()
    return (server.remote_url or "") if server else default


def current_command(default: str = "") -> str:
    server = current_server()
    return (server.command or "") if server else default


def current_env(base: dict[str, str]) -> dict[str, str]:
    server = current_server()
    if not server:
        return base
    merged = dict(base)
    merged.update(server.env)
    return merged


def current_tool_filters(
    default_include: list[str], default_exclude: list[str]
) -> tuple[list[str], list[str]]:
    server = current_server()
    if not server:
        return default_include, default_exclude
    return list(server.include_tools), list(server.exclude_tools)


def session_storage_key(session_id: Optional[str]) -> Optional[str]:
    if session_id is None:
        return None
    server = current_server()
    if not server:
        return session_id
    return f"{server.id}:{session_id}"


def tools_cache_paths(default_file: Path, default_lock_file: Path) -> tuple[Path, Path]:
    server = current_server()
    if not server:
        return default_file, default_lock_file
    cache_file = default_file.with_name(f"{default_file.name}.{server.id}")
    lock_file = cache_file.with_name(cache_file.name + ".lock")
    return cache_file, lock_file


class MultiServerContextMiddleware:
    """Bind the configured server to the current ASGI request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        server = match_server(scope.get("path", ""))
        if server is None:
            await self.app(scope, receive, send)
            return
        token = bind_server(server)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_server(token)
