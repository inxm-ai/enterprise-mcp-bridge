"""OAuth 2.0 discovery endpoints for MCP client compatibility.

Serves RFC 9728 Protected Resource Metadata and RFC 8414 Authorization Server
Metadata so that MCP clients (Claude Code, Cursor, VS Code Copilot, etc.) can
discover how to authenticate with the bridge automatically.

Endpoints
---------
GET  /.well-known/oauth-protected-resource          — RFC 9728
GET  /.well-known/oauth-authorization-server         — RFC 8414

These endpoints are optional and only active when the required environment
variables are configured (see ``MCP_OAUTH_*`` vars).
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.vars import (
    AUTH_BASE_URL,
    KEYCLOAK_REALM,
    MCP_BASE_PATH,
)

logger = logging.getLogger("uvicorn.error")

router = APIRouter()

# ---------------------------------------------------------------------------
# Configuration  (all optional – endpoints return 404 when not configured)
# ---------------------------------------------------------------------------

# The public URL where this bridge is reachable.
# MCP clients need this so they know which "resource" they are requesting
# tokens for (RFC 8707 "resource" parameter).
MCP_OAUTH_RESOURCE_URL: Optional[str] = (
    os.getenv("MCP_OAUTH_RESOURCE_URL", "").strip() or None
)

# The issuer / authorization server URL.  When using Keycloak this should be
# the realm URL, e.g.  https://keycloak.example.com/realms/inxm
# If not set explicitly we derive it from AUTH_BASE_URL + KEYCLOAK_REALM.
MCP_OAUTH_ISSUER: Optional[str] = os.getenv("MCP_OAUTH_ISSUER", "").strip() or None

# OAuth client ID that MCP clients should use when authenticating.
MCP_OAUTH_CLIENT_ID: Optional[str] = (
    os.getenv("MCP_OAUTH_CLIENT_ID", "").strip() or None
)

# Scopes the bridge accepts / the client should request.
MCP_OAUTH_SCOPES: list[str] = [
    s.strip()
    for s in os.getenv("MCP_OAUTH_SCOPES", "openid profile email").split()
    if s.strip()
]

# Whether dynamic client registration is enabled on the auth server.
MCP_OAUTH_REGISTRATION_ENDPOINT: Optional[str] = (
    os.getenv("MCP_OAUTH_REGISTRATION_ENDPOINT", "").strip() or None
)


def _derive_issuer() -> Optional[str]:
    """Derive the OIDC issuer URL from Keycloak settings when not explicit."""
    if MCP_OAUTH_ISSUER:
        return MCP_OAUTH_ISSUER
    if AUTH_BASE_URL and KEYCLOAK_REALM:
        return f"{AUTH_BASE_URL.rstrip('/')}/realms/{KEYCLOAK_REALM}"
    return None


def _is_configured() -> bool:
    """Check whether enough config is present to serve discovery docs."""
    return bool(_derive_issuer())


# ---------------------------------------------------------------------------
# RFC 9728 – Protected Resource Metadata
# ---------------------------------------------------------------------------


@router.get("/.well-known/oauth-protected-resource")
async def get_protected_resource_metadata(request: Request):
    """Return RFC 9728 Protected Resource Metadata.

    MCP clients fetch this first to discover *which* authorization server
    they should talk to and what scopes are needed.
    """
    if not _is_configured():
        raise HTTPException(status_code=404, detail="OAuth metadata not configured")

    issuer = _derive_issuer()

    # Derive the resource URL from the request if not explicitly configured
    resource_url = MCP_OAUTH_RESOURCE_URL
    if not resource_url:
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("x-forwarded-host") or request.headers.get(
            "host", "localhost"
        )
        base = (MCP_BASE_PATH or "").rstrip("/")
        resource_url = f"{scheme}://{host}{base}"

    metadata = {
        "resource": resource_url,
        "authorization_servers": [issuer],
        "scopes_supported": MCP_OAUTH_SCOPES if MCP_OAUTH_SCOPES else None,
        "bearer_methods_supported": ["header"],
    }

    if MCP_OAUTH_CLIENT_ID:
        metadata["resource_name"] = MCP_OAUTH_CLIENT_ID

    return JSONResponse(
        content={k: v for k, v in metadata.items() if v is not None},
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ---------------------------------------------------------------------------
# RFC 8414 – OAuth Authorization Server Metadata
# ---------------------------------------------------------------------------


@router.get("/.well-known/oauth-authorization-server")
async def get_oauth_authorization_server_metadata():
    """Return RFC 8414 OAuth Authorization Server Metadata.

    When this bridge runs behind an IdP (e.g. Keycloak) you normally want
    MCP clients to talk to the IdP's real endpoints.  This endpoint returns
    the IdP's metadata so the client never has to be configured manually.

    If the auth server already exposes ``/.well-known/openid-configuration``
    clients that follow OIDC discovery will find it there.  This endpoint
    covers the OAuth 2.0 specific path that the MCP SDK checks first.
    """
    if not _is_configured():
        raise HTTPException(status_code=404, detail="OAuth metadata not configured")

    issuer = _derive_issuer()

    metadata = {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/protocol/openid-connect/auth",
        "token_endpoint": f"{issuer}/protocol/openid-connect/token",
        "response_types_supported": ["code"],
        "grant_types_supported": [
            "authorization_code",
            "refresh_token",
        ],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_basic",
            "client_secret_post",
            "none",
        ],
        "scopes_supported": MCP_OAUTH_SCOPES if MCP_OAUTH_SCOPES else None,
    }

    if MCP_OAUTH_REGISTRATION_ENDPOINT:
        metadata["registration_endpoint"] = MCP_OAUTH_REGISTRATION_ENDPOINT
    elif issuer:
        # Keycloak's standard dynamic registration endpoint
        metadata["registration_endpoint"] = (
            f"{issuer}/clients-registrations/openid-connect"
        )

    return JSONResponse(
        content={k: v for k, v in metadata.items() if v is not None},
        headers={"Cache-Control": "public, max-age=3600"},
    )
