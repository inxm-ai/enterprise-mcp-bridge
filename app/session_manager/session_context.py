import logging
from fastapi import HTTPException
from typing import Optional, Dict
import os
from contextlib import asynccontextmanager

from app.session import mcp_session
from app.session_manager.session_manager import SessionManagerBase
from app.models import RunToolsResult
from app.mcp_server import get_server_params
from fnmatch import fnmatch
from app.oauth.decorator import decorate_args_with_oauth_token
from app.oauth.user_info import get_data_access_manager
from app.utils import mask_token


logger = logging.getLogger("uvicorn.error")

INCLUDE_TOOLS = [t for t in os.environ.get("INCLUDE_TOOLS", "").split(",") if t]
EXCLUDE_TOOLS = [t for t in os.environ.get("EXCLUDE_TOOLS", "").split(",") if t]
MCP_BASE_PATH = os.environ.get("MCP_BASE_PATH", "")


def matches_pattern(value, pattern):
    """Check if a value matches a glob-like pattern."""
    return fnmatch(value, pattern)


def map_tools(tools):
    """Map tools with INCLUDE_TOOLS and EXCLUDE_TOOLS filters applied."""

    filtered_tools = []
    for tool in tools.tools:
        include_match = any(
            matches_pattern(tool.name, pattern) for pattern in INCLUDE_TOOLS if pattern
        )
        exclude_match = any(
            matches_pattern(tool.name, pattern) for pattern in EXCLUDE_TOOLS if pattern
        )
        logger.debug(
            f"[Tools] Filter check -> Tool: {tool.name}, Include Match: {include_match} - {INCLUDE_TOOLS}, Exclude Match: {exclude_match} - {EXCLUDE_TOOLS}"
        )
        if INCLUDE_TOOLS and any(INCLUDE_TOOLS) and not include_match:
            continue
        if EXCLUDE_TOOLS and any(EXCLUDE_TOOLS) and exclude_match:
            continue

        filtered_tools.append(
            {
                "name": tool.name,
                "title": tool.title,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
                "outputSchema": tool.outputSchema,
                "annotations": tool.annotations,
                "meta": tool.meta,
                "url": f"{MCP_BASE_PATH}/tools/{tool.name}",
            }
        )

    return filtered_tools


@asynccontextmanager
async def mcp_session_context(
    sessions: SessionManagerBase,
    x_inxm_mcp_session: Optional[str],
    access_token: Optional[str],
    group: Optional[str],
):
    """Yield a delegate with unified list_tools() and call_tool() across sessionful and sessionless modes."""
    # Sessionless path: validate group access (if present) and open a transient MCP session
    if x_inxm_mcp_session is None:
        if group and access_token:
            data_manager = get_data_access_manager()
            try:
                data_manager.resolve_data_resource(access_token, group)
            except PermissionError as e:
                logger.warning(f"Group access denied: {str(e)}")
                raise HTTPException(status_code=403, detail=str(e))

        async with mcp_session(get_server_params(access_token, group)) as session:

            class SessionDelegate:
                async def list_tools(self):
                    tools = await session.list_tools()
                    return map_tools(tools)

                async def call_tool(
                    self,
                    tool_name: str,
                    args: Optional[Dict],
                    access_token_inner: Optional[str],
                ):
                    tools = await session.list_tools()
                    decorated_args = await decorate_args_with_oauth_token(
                        tools, tool_name, args, access_token_inner
                    )
                    result = await session.call_tool(tool_name, decorated_args)
                    return RunToolsResult(result)

            yield SessionDelegate()
        return

    # Sessionful path: reuse existing task, but surface a common delegate API
    mcp_task = sessions.get(x_inxm_mcp_session)
    if not mcp_task:
        logger.warning(
            mask_token(
                f"[MCP] Session not found: {x_inxm_mcp_session}",
                x_inxm_mcp_session,
            )
        )
        raise HTTPException(
            status_code=404,
            detail="Session not found. It might have expired, please start a new.",
        )

    class TaskDelegate:
        async def list_tools(self):
            tools = await mcp_task.request("list_tools")
            return map_tools(tools)

        async def call_tool(
            self,
            tool_name: str,
            args: Optional[Dict],
            access_token_inner: Optional[str],
        ):
            tools = await mcp_task.request("list_tools")
            decorated_args = await decorate_args_with_oauth_token(
                tools, tool_name, args, access_token_inner
            )
            result = await mcp_task.request(
                {"action": "run_tool", "tool_name": tool_name, "args": decorated_args}
            )
            return RunToolsResult(result)

    try:
        yield TaskDelegate()
    finally:
        # No-op: persistent session remains managed elsewhere
        pass
