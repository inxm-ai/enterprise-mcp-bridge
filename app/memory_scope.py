"""Hub-asserted chat memory and group-authorized project memory.

Only the local memory child can accept this header. It is deliberately not a
general data_path override, and never falls back to a personal/shared tenant.
"""

import hmac
import os
from typing import Optional
from urllib.parse import quote_plus
from uuid import UUID

from fastapi import HTTPException

from app.multi_server import current_server, current_setting, is_multi_server_mode
from app.oauth.user_info import get_data_access_manager

SCOPE_HEADER = "x-inxm-memory-scope"


def memory_tenant(
    headers: Optional[dict[str, str]],
    access_token: Optional[str],
    group: Optional[str],
    *,
    persistent: bool = False,
) -> Optional[str]:
    """Authenticate and validate a scope assertion before starting any child.

    The hub checks conversation membership on every call. Projects map one to
    one to a Keycloak group, whose membership the bridge also checks. Scoped
    memory is sessionless so a session can never be reused across scopes.
    """
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    scope = headers.get(SCOPE_HEADER)
    if scope is None:
        return None
    expected = current_setting("INTERNAL_API_SECRET")
    if expected is None:
        expected = os.environ.get("INTERNAL_API_SECRET", "")
    presented = headers.get("x-internal-secret", "")
    if not expected or not hmac.compare_digest(expected.encode(), presented.encode()):
        raise HTTPException(403, "Invalid memory scope assertion")
    if not access_token:
        raise HTTPException(401, "Memory scope requires a caller credential")
    if persistent:
        raise HTTPException(400, "Scoped memory requires sessionless calls")
    # Do not expose a data_path override to other MCP children in multi-server
    # mode. The memory server's tenant env must be configured explicitly.
    server = current_server()
    if server:
        template = server.env.get("MCP_ENV_MEMORY_TENANT")
        source = server.env_from.get("MCP_ENV_MEMORY_TENANT")
        if source:
            template = os.environ.get(source)
    else:
        template = (
            None if is_multi_server_mode() else os.environ.get("MCP_ENV_MEMORY_TENANT")
        )
    if template != "{data_path}":
        raise HTTPException(400, "Memory scope is only supported by the memory server")
    if scope.startswith("c/"):
        try:
            canonical = f"c/{UUID(scope[2:])}"
        except ValueError:
            raise HTTPException(400, "Invalid conversation memory scope")
        if scope != canonical or group is not None:
            raise HTTPException(400, "Invalid conversation memory scope")
        return canonical
    if scope.startswith("p/"):
        if not group or group.startswith("/") or len(group.encode()) > 200:
            raise HTTPException(400, "Project memory requires a canonical group")
        # Match Rust's application/x-www-form-urlencoded byte serialization.
        encoded = quote_plus(group, safe="*").replace("~", "%7E")
        if scope != f"p/{encoded}":
            raise HTTPException(400, "Project scope must match its Keycloak group")
        try:
            get_data_access_manager().resolve_data_resource(access_token, group)
        except PermissionError:
            raise HTTPException(403, "Project group access denied")
        except (AssertionError, ValueError):
            raise HTTPException(401, "Invalid caller credential")
        return scope
    raise HTTPException(400, "Unsupported memory scope")
