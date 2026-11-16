"""Hub-asserted chat memory and group-authorized project memory.

Only the local memory child can accept this header. It is deliberately not a
general data_path override, and never falls back to a personal/shared tenant.

One more scope exists for removing a person: ``u/<user id>`` names another
person's own memory, and is accepted for exactly one tool, ``delete_graph``,
from a platform administrator. Nothing can read or write through it.
"""

import hmac
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional
from urllib.parse import quote_plus
from uuid import UUID

from fastapi import HTTPException

from app.multi_server import current_server, current_setting, is_multi_server_mode
from app.oauth.user_info import get_data_access_manager
from app.vars import MCP_USER_DATA_ACCESS_TEMPLATE

SCOPE_HEADER = "x-inxm-memory-scope"
# The only tool a removed person's memory accepts, and who may call it.
WIPE_TOOL = "delete_graph"
PLATFORM_ADMIN_ROLE = "platform-admin"

# The tool a request is calling, when it is a tool call. Unset anywhere else
# (listings, prompts, sessions), where a person scope is always refused.
_called_tool: ContextVar[Optional[str]] = ContextVar("memory_scope_tool", default=None)


@contextmanager
def calling_tool(tool_name: Optional[str]) -> Iterator[None]:
    """Marks the tool the current request calls, for the scope decision."""
    token = _called_tool.set(tool_name)
    try:
        yield
    finally:
        _called_tool.reset(token)


def _removed_person_tenant(scope: str, access_token: str, group: Optional[str]) -> str:
    """``u/<user id>``: another person's own memory, to delete it and nothing else."""
    if group is not None or _called_tool.get() != WIPE_TOOL:
        raise HTTPException(400, "A person's memory scope only deletes that memory")
    manager = get_data_access_manager()
    user_id = scope[2:]
    if not user_id or manager._sanitize_identifier(user_id) != user_id:
        raise HTTPException(400, "Invalid person memory scope")
    # The tenant a person's own calls resolve to; a deployment that stores
    # personal memory elsewhere has no `u/` scope to delete.
    if MCP_USER_DATA_ACCESS_TEMPLATE.format(user_id=user_id) != scope:
        raise HTTPException(400, "Invalid person memory scope")
    try:
        caller = manager.user_extractor.extract_user_info(access_token)
    except (AssertionError, ValueError):
        raise HTTPException(401, "Invalid caller credential")
    if PLATFORM_ADMIN_ROLE not in (caller.get("roles") or []):
        raise HTTPException(
            403, "Only a platform administrator deletes a person's memory"
        )
    return scope


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
    if scope.startswith("u/"):
        return _removed_person_tenant(scope, access_token, group)
    raise HTTPException(400, "Unsupported memory scope")
