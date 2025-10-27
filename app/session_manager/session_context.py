import json
import logging
from fastapi import HTTPException
from typing import Optional, Dict
import os
from contextlib import asynccontextmanager

from app.session import mcp_session
from app.session_manager.prompt_helper import list_prompts, call_prompt
from app.session_manager.session_manager import SessionManagerBase
from app.models import RunPromptResult, RunToolsResult
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


async def list_resources(list_resources: any):
    system_resources = json.loads(os.environ.get("SYSTEM_DEFINED_RESOURCES", "[]"))
    try:
        resources = await list_resources()
        resources.resources += system_resources
    except Exception as e:
        logger.warning(f"[ResourcesHelper] Error listing resources: {str(e)}")
        # Not every MCP has list_resources, so deal with it friendly
        if (
            hasattr(e, "__class__")
            and e.__class__.__name__ == "McpError"
            and "Method not found" in str(e)
        ):
            if len(system_resources) < 1:
                logger.info("[ResourcesHelper] No system resources available")
                raise HTTPException(status_code=404, detail="Method not found")
            else:
                logger.info("[ResourcesHelper] Returning system resources")
                logger.debug(f"[ResourcesHelper] System resources: {system_resources}")
                resources = {"resources": system_resources}
        else:
            raise HTTPException(status_code=500, detail="Loading resources failed")
    return resources


@asynccontextmanager
async def mcp_session_context(
    sessions: SessionManagerBase,
    x_inxm_mcp_session: Optional[str],
    access_token: Optional[str],
    group: Optional[str],
    incoming_headers: Optional[dict[str, str]] = None,
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

        try:
            async with mcp_session(
                access_token=access_token,
                requested_group=group,
                incoming_headers=incoming_headers,
            ) as session:

                class SessionDelegate:
                    async def list_prompts(self):
                        return await list_prompts(session.list_prompts)

                    async def list_resources(self):
                        return await list_resources(session.list_resources)

                    async def read_resource(self, resource_name: str):
                        return await session.read_resource(resource_name)

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

                    async def call_prompt(
                        self,
                        prompt_name: str,
                        args: Optional[Dict],
                    ):
                        result = await call_prompt(
                            session.get_prompt, prompt_name, args
                        )
                        return RunPromptResult(result)

                yield SessionDelegate()
        except ValueError as exc:
            logger.error(f"[MCP] Failed to create session: {exc}")
            raise HTTPException(status_code=400, detail=str(exc))
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

        async def list_prompts(self):
            async def request():
                return await mcp_task.request("list_prompts")

            return await list_prompts(request)

        async def list_resources(self):
            async def request():
                return await mcp_task.request("list_resources")

            return await list_resources(request)

        async def read_resource(self, resource_name: str):
            return await mcp_task.request(
                {"action": "read_resource", "resource_name": resource_name}
            )

        async def call_prompt(
            self,
            prompt_name: str,
            args: Optional[Dict],
        ):
            async def request(prompt_name, args):
                return await mcp_task.request(
                    {"action": "get_prompt", "prompt_name": prompt_name, "args": args}
                )

            return RunPromptResult(await call_prompt(request, prompt_name, args))

    try:
        yield TaskDelegate()
    finally:
        # No-op: persistent session remains managed elsewhere
        pass
