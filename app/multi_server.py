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

Per-server overrides of process-wide settings (all optional; when a field is
absent the global environment variable applies):

- ``auth_provider`` (string): overrides ``AUTH_PROVIDER``.
- ``keycloak_provider_alias`` (string): overrides ``KEYCLOAK_PROVIDER_ALIAS``;
  ``""`` means no alias, i.e. the Keycloak token is passed through.
- ``effect_tools`` (array of strings): overrides ``EFFECT_TOOLS``. The entry
  ``"auto"`` classifies tools automatically (see ``app.utils.effect_tools``)
  and may be combined with explicit globs.
- ``forward_access_token`` (boolean, default true): when false, the caller's
  access token never reaches this server: no token exchange, no fallback to
  the incoming token, no forwarded credential headers and no ``oauth_token``
  tool argument. Only explicitly configured credentials are sent. Set it to
  false for every remote that does not use a provider alias (e.g. a public
  third-party MCP), or the caller's platform token leaks to that third party.
- ``tool_output_schemas`` (object): tool name -> output JSON schema, layered
  over ``TOOL_OUTPUT_SCHEMAS`` for this server only. Inline schemas only, no
  file paths.

Central remote host proxying several remote MCP servers, each exchanging the
caller's token through its own Keycloak identity provider:
[
  {
    "id": "mcp-cloudflare-server",
    "base_path": "/api/mcp-cloudflare-server",
    "url": "https://mcp.cloudflare.com/mcp",
    "sessionless": true,
    "auth_provider": "keycloak",
    "keycloak_provider_alias": "cloudflare",
    "effect_tools": ["auto"]
  },
  {
    "id": "mcp-notion-server",
    "base_path": "/api/mcp-notion-server",
    "url": "https://mcp.notion.com/mcp",
    "sessionless": true,
    "auth_provider": "keycloak",
    "keycloak_provider_alias": "notion",
    "effect_tools": ["auto"]
  },
  {
    "id": "mcp-deepwiki-server",
    "base_path": "/api/mcp-deepwiki-server",
    "url": "https://mcp.deepwiki.com/mcp",
    "sessionless": true,
    "forward_access_token": false,
    "effect_tools": ["auto"]
  }
]
"""

from __future__ import annotations

import json
import os
import re
from http.cookies import SimpleCookie
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
    sessionless: Optional[bool] = None
    auth_provider: Optional[str] = None
    keycloak_provider_alias: Optional[str] = None
    effect_tools: Optional[tuple[str, ...]] = None
    forward_access_token: Optional[bool] = None
    tool_output_schemas: Optional[dict[str, Any]] = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ServerConfig":
        server_id = str(raw.get("id") or "").strip()
        if not server_id or not _ID_RE.match(server_id):
            raise ValueError(f"Invalid MCP server id: {server_id!r}")

        base_path = str(raw.get("base_path") or raw.get("path") or "").strip()
        if not base_path.startswith("/"):
            raise ValueError(f"MCP server {server_id!r} base_path must start with '/'")
        base_path = base_path.rstrip("/") or "/"
        if base_path == "/":
            raise ValueError(
                f"MCP server {server_id!r} base_path cannot be root in multi-server mode"
            )

        command = raw.get("command")
        remote_url = raw.get("url", raw.get("remote_url"))
        command = str(command).strip() if command else None
        remote_url = str(remote_url).strip() if remote_url else None
        if bool(command) == bool(remote_url):
            raise ValueError(
                f"MCP server {server_id!r} must define exactly one of command or url"
            )

        env_raw = raw.get("env")
        if env_raw is None:
            env_raw = {}
        if not isinstance(env_raw, dict):
            raise ValueError(f"MCP server {server_id!r} env must be an object")
        env = {str(k): str(v) for k, v in env_raw.items()}

        include = raw.get("include_tools")
        exclude = raw.get("exclude_tools")
        if include is None:
            include = []
        if exclude is None:
            exclude = []
        sessionless = raw.get("sessionless")
        if sessionless is not None and not isinstance(sessionless, bool):
            raise ValueError(f"MCP server {server_id!r} sessionless must be a boolean")
        if not isinstance(include, list) or not isinstance(exclude, list):
            raise ValueError(
                f"MCP server {server_id!r} include_tools/exclude_tools must be arrays"
            )

        auth_provider = raw.get("auth_provider")
        if auth_provider is not None:
            if not isinstance(auth_provider, str):
                raise ValueError(
                    f"MCP server {server_id!r} auth_provider must be a string"
                )
            # Same normalization as the global AUTH_PROVIDER; empty means
            # "not overridden".
            auth_provider = auth_provider.strip().lower() or None

        provider_alias = raw.get("keycloak_provider_alias")
        if provider_alias is not None:
            if not isinstance(provider_alias, str):
                raise ValueError(
                    f"MCP server {server_id!r} keycloak_provider_alias must be a string"
                )
            # Unlike auth_provider, "" is a real override: no alias, pass the
            # Keycloak token through.
            provider_alias = provider_alias.strip()

        effect_tools = raw.get("effect_tools")
        if effect_tools is not None:
            if not isinstance(effect_tools, list) or not all(
                isinstance(pattern, str) for pattern in effect_tools
            ):
                raise ValueError(
                    f"MCP server {server_id!r} effect_tools must be an array of strings"
                )
            effect_tools = tuple(
                pattern.strip() for pattern in effect_tools if pattern.strip()
            )

        forward_access_token = raw.get("forward_access_token")
        if forward_access_token is not None and not isinstance(
            forward_access_token, bool
        ):
            raise ValueError(
                f"MCP server {server_id!r} forward_access_token must be a boolean"
            )

        tool_output_schemas = raw.get("tool_output_schemas")
        if tool_output_schemas is not None and not (
            isinstance(tool_output_schemas, dict)
            and all(isinstance(v, dict) for v in tool_output_schemas.values())
        ):
            raise ValueError(
                f"MCP server {server_id!r} tool_output_schemas must map tool "
                "names to schema objects"
            )

        return cls(
            id=server_id,
            base_path=base_path,
            command=command,
            remote_url=remote_url,
            env=env,
            include_tools=tuple(str(v) for v in include if str(v)),
            exclude_tools=tuple(str(v) for v in exclude if str(v)),
            sessionless=sessionless,
            auth_provider=auth_provider,
            keycloak_provider_alias=provider_alias,
            effect_tools=effect_tools,
            forward_access_token=forward_access_token,
            tool_output_schemas=tool_output_schemas,
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


def current_sessionless(default: bool) -> bool:
    server = current_server()
    if not server or server.sessionless is None:
        return default
    return server.sessionless


def current_auth_provider(default: str) -> str:
    server = current_server()
    if not server or server.auth_provider is None:
        return default
    return server.auth_provider


def current_keycloak_provider_alias(default: str) -> str:
    server = current_server()
    if not server or server.keycloak_provider_alias is None:
        return default
    return server.keycloak_provider_alias


def current_tool_output_schemas(base: dict[str, Any]) -> dict[str, Any]:
    server = current_server()
    if not server or not server.tool_output_schemas:
        return base
    merged = dict(base)
    merged.update(server.tool_output_schemas)
    return merged


def current_forward_access_token(default: bool = True) -> bool:
    server = current_server()
    if not server or server.forward_access_token is None:
        return default
    return server.forward_access_token


def current_effect_tools(default: list[str]) -> list[str]:
    server = current_server()
    if not server or server.effect_tools is None:
        return default
    return list(server.effect_tools)


def session_cookie_name(default_name: str) -> str:
    server = current_server()
    if not server:
        return default_name
    return f"{default_name}.{server.id}"


def session_cookie_value(
    cookies: dict[str, str], default_name: str, legacy_value: Optional[str] = None
) -> Optional[str]:
    server = current_server()
    if not server:
        return legacy_value
    return cookies.get(session_cookie_name(default_name))


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
    lock_file = default_lock_file.with_name(f"{default_lock_file.name}.{server.id}")
    return cache_file, lock_file


def _rewrite_session_cookie_header(scope, server: ServerConfig) -> None:
    cookie_name = os.environ.get("SESSION_FIELD_NAME", "x-inxm-mcp-session")
    server_cookie_name = f"{cookie_name}.{server.id}"
    headers = list(scope.get("headers") or [])
    cookie_index = next(
        (i for i, (name, _value) in enumerate(headers) if name.lower() == b"cookie"),
        None,
    )
    if cookie_index is None:
        return

    raw_cookie = headers[cookie_index][1].decode("latin-1")
    parsed = SimpleCookie()
    try:
        parsed.load(raw_cookie)
    except Exception:
        return

    parsed.pop(cookie_name, None)
    selected = parsed.get(server_cookie_name)
    if selected is not None:
        parsed[cookie_name] = selected.value

    rewritten = "; ".join(morsel.OutputString() for morsel in parsed.values())
    headers[cookie_index] = (b"cookie", rewritten.encode("latin-1"))
    scope["headers"] = headers


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
        _rewrite_session_cookie_header(scope, server)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_server(token)
