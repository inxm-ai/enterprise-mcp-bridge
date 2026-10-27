"""Typed provider scope rejection for personal OAuth grants.

When a tool call fails (``isError``) on a connector that acts with the
caller's own OAuth grant, and the error text says the provider refused for a
missing permission (Slack ``missing_scope``, OAuth ``insufficient_scope``),
the REST route answers 403 ``scope_missing`` with the
``X-Inxm-Connection-Code`` header instead of a generic 422, so the platform
can ask the person to reconnect. The MCP transports cannot change the status:
they keep ``isError`` and start the first text with the same sentence.

``BRIDGE_SCOPE_ERROR_PATTERNS`` (comma-separated regexes) extends the default
codes. Matching is case-insensitive and on word boundaries.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from app.utils import mcp_fields

logger = logging.getLogger("uvicorn.error")

SCOPE_MISSING_CODE = "scope_missing"
DEFAULT_PATTERNS = ("missing_scope", "insufficient_scope")
_MAX_SCOPES = 32
_SCOPE_RE = re.compile(r"^[A-Za-z0-9:._/-]{1,128}$")
_NEEDED_RE = re.compile(r'"needed"\s*:\s*"([^"]*)"', re.IGNORECASE)
_NEEDED_LIST_RE = re.compile(r'"needed_scopes"\s*:\s*\[([^\]]*)\]', re.IGNORECASE)
_SPLIT_RE = re.compile(r"[,\s]+")


def _patterns() -> list[re.Pattern]:
    sources = list(DEFAULT_PATTERNS)
    sources += [
        p.strip()
        for p in os.getenv("BRIDGE_SCOPE_ERROR_PATTERNS", "").split(",")
        if p.strip()
    ]
    compiled = []
    for source in sources:
        try:
            compiled.append(re.compile(rf"\b(?:{source})\b", re.IGNORECASE))
        except re.error:
            logger.warning("[ScopeRejection] Ignoring invalid scope error pattern")
    return compiled


def _is_personal_oauth_grant() -> bool:
    from app.oauth.connection_policy import credential_kind

    return credential_kind() == "oauth"


def _text_items(result) -> list:
    return [
        item
        for item in mcp_fields.content(result)
        if mcp_fields.is_text_item(item) and mcp_fields.read(item, "text", "text")
    ]


def match_code(text: str) -> Optional[str]:
    """The matched provider error code in the text, or None."""
    for pattern in _patterns():
        found = pattern.search(text or "")
        if found:
            return found.group(0)
    return None


def missing_scopes(text: str) -> list[str]:
    """The scopes the upstream named (Slack ``needed`` / ``needed_scopes``), validated."""
    raw: list[str] = []
    for found in _NEEDED_RE.findall(text or ""):
        raw += _SPLIT_RE.split(found)
    for found in _NEEDED_LIST_RE.findall(text or ""):
        raw += [part.strip().strip("\"'") for part in found.split(",")]
    scopes: list[str] = []
    for scope in raw:
        if _SCOPE_RE.match(scope) and scope not in scopes:
            scopes.append(scope)
        if len(scopes) >= _MAX_SCOPES:
            break
    return scopes


def _provider_alias() -> Optional[str]:
    from app import vars as app_vars
    from app.multi_server import current_keycloak_provider_alias

    return current_keycloak_provider_alias(app_vars.KEYCLOAK_PROVIDER_ALIAS) or None


def _detail(provider: Optional[str], tool: str, code: str) -> str:
    name = provider.capitalize() if provider else "The provider"
    return (
        f"{name} needs more permissions for {tool}. Reconnect {name} to grant "
        f"them. (upstream: {code})"
    )


def detect(result, tool: str) -> Optional[dict]:
    """The typed ``scope_missing`` body for a failed result, or None.

    Only a personal OAuth grant can be fixed by reconnecting, so other
    credential kinds never match.
    """
    if not mcp_fields.is_error(result) or not _is_personal_oauth_grant():
        return None
    text = "\n".join(mcp_fields.read(i, "text", "text") for i in _text_items(result))
    code = match_code(text)
    if code is None:
        return None
    provider = _provider_alias()
    logger.warning(
        "[ScopeRejection] Provider scope rejection on server %s, tool %s, code %s",
        _server_name(),
        tool,
        code,
    )
    return {
        "detail": _detail(provider, tool, code),
        "code": SCOPE_MISSING_CODE,
        "provider": provider,
        "tool": tool,
        "missing_scopes": missing_scopes(text),
    }


def _server_name() -> str:
    from app.oauth.connection_policy import server_id

    return server_id()


def prefix_result(result, tool: str):
    """For MCP transports: start the first text with the detail sentence; status unchanged."""
    body = detect(result, tool)
    if body is None:
        return result
    items = _text_items(result)
    if not items:
        return result
    first = items[0]
    text = mcp_fields.read(first, "text", "text")
    if text.startswith(body["detail"]):
        return result
    new_text = f"{body['detail']} {text}"
    if isinstance(first, dict):
        first["text"] = new_text
    else:
        first.text = new_text
    return result
