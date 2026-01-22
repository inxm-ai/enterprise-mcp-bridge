import fcntl
import inspect
import json
import logging
from fastapi import HTTPException
from typing import Optional, Dict, Awaitable, Callable, Any
import os
from contextlib import asynccontextmanager
from pathlib import Path

from app.session import mcp_session
from app.session_manager.prompt_helper import list_prompts, call_prompt
from app.session_manager.session_manager import SessionManagerBase
from app.models import RunPromptResult, RunToolsResult
from fnmatch import fnmatch
from app.oauth.decorator import decorate_args_with_oauth_token
from app.oauth.user_info import get_data_access_manager
from app.utils import mask_token
from app.vars import MCP_MAP_HEADER_TO_INPUT


logger = logging.getLogger("uvicorn.error")

INCLUDE_TOOLS = [t for t in os.environ.get("INCLUDE_TOOLS", "").split(",") if t]
EXCLUDE_TOOLS = [t for t in os.environ.get("EXCLUDE_TOOLS", "").split(",") if t]
MCP_BASE_PATH = os.environ.get("MCP_BASE_PATH", "")
TOOLS_CACHE_ENABLED = (
    os.environ.get("MCP_TOOLS_CACHE_ENABLED", "true").lower() == "true"
)
TOOLS_CACHE_FILE = Path(
    os.environ.get("MCP_TOOLS_CACHE_FILE", "/tmp/mcp_tools_cache.json")
)
TOOLS_CACHE_LOCK_FILE = Path(
    os.environ.get("MCP_TOOLS_CACHE_LOCK_FILE", str(TOOLS_CACHE_FILE) + ".lock")
)
CACHE_VERSION = 1


def _to_tool_list(tools: Any) -> list:
    """
    Normalize tool container objects to a plain list for downstream use.
    Accepts objects with a `.tools` attribute or an iterable directly.
    """
    if hasattr(tools, "tools"):
        try:
            return list(getattr(tools, "tools"))
        except Exception:
            return []
    try:
        return list(tools or [])
    except Exception:
        return []


def get_tool_name(tool: Any) -> Optional[str]:
    """Return the tool name from MCP or OpenAI-style tool definitions."""
    if isinstance(tool, dict):
        name = tool.get("name")
        if not name:
            func = tool.get("function") or {}
            name = func.get("name")
        return name

    name = getattr(tool, "name", None)
    if not name and hasattr(tool, "function"):
        name = getattr(getattr(tool, "function"), "name", None)
    return name


def filter_tools(tools: Any) -> list:
    """Filter tool definitions using INCLUDE_TOOLS and EXCLUDE_TOOLS rules."""
    tool_list = _to_tool_list(tools)
    filtered = []
    for tool in tool_list:
        name = get_tool_name(tool)
        if not name:
            continue
        include_match = any(
            matches_pattern(name, pattern) for pattern in INCLUDE_TOOLS if pattern
        )
        exclude_match = any(
            matches_pattern(name, pattern) for pattern in EXCLUDE_TOOLS if pattern
        )
        logger.debug(
            f"[Tools] Filter check -> Tool: {name}, Include Match: {include_match} - {INCLUDE_TOOLS}, Exclude Match: {exclude_match} - {EXCLUDE_TOOLS}"
        )
        if INCLUDE_TOOLS and any(INCLUDE_TOOLS) and not include_match:
            continue
        if EXCLUDE_TOOLS and any(EXCLUDE_TOOLS) and exclude_match:
            continue
        filtered.append(tool)
    return filtered


def _cache_signature() -> dict:
    """
    Build a lightweight signature so the tools cache is invalidated when the
    server configuration or filtering changes. This prevents stale tool lists
    (e.g., from earlier tests) from being reused after configuration changes.
    """
    return {
        "include": INCLUDE_TOOLS,
        "exclude": EXCLUDE_TOOLS,
        "map_header_to_input": MCP_MAP_HEADER_TO_INPUT,
        "server": os.environ.get("MCP_SERVER_COMMAND", "")
        or os.environ.get("MCP_REMOTE_SERVER", ""),
    }


def matches_pattern(value, pattern):
    """Check if a value matches a glob-like pattern."""
    return fnmatch(value, pattern)


def map_tools(tools):
    """Map tools with INCLUDE_TOOLS and EXCLUDE_TOOLS filters applied."""

    tool_list = filter_tools(tools)
    filtered_tools = []
    for tool in tool_list:
        name = get_tool_name(tool)
        if not name:
            continue
        if isinstance(tool, dict):
            title = tool.get("title")
            description = tool.get("description") or (tool.get("function") or {}).get(
                "description"
            )
            input_schema = tool.get("inputSchema")
            if input_schema is None:
                input_schema = (tool.get("function") or {}).get("parameters")
            output_schema = tool.get("outputSchema")
            if output_schema is None:
                output_schema = (tool.get("function") or {}).get("outputSchema")
            annotations = tool.get("annotations")
            meta = tool.get("meta")
        else:
            title = getattr(tool, "title", None)
            description = getattr(tool, "description", None)
            if not description and hasattr(tool, "function"):
                description = getattr(getattr(tool, "function"), "description", None)
            input_schema = getattr(tool, "inputSchema", None)
            if input_schema is None and hasattr(tool, "function"):
                input_schema = getattr(getattr(tool, "function"), "parameters", None)
            output_schema = getattr(tool, "outputSchema", None)
            if output_schema is None and hasattr(tool, "function"):
                output_schema = getattr(getattr(tool, "function"), "outputSchema", None)
            annotations = getattr(tool, "annotations", None)
            meta = getattr(tool, "meta", None)

        if input_schema is None:
            input_schema = {}
        # make a shallow deepcopy to avoid mutating original tool objects
        try:
            import copy

            input_schema_copy = copy.deepcopy(input_schema)
        except Exception:
            input_schema_copy = input_schema

        if isinstance(input_schema_copy, dict) and input_schema_copy.get("properties"):
            props = input_schema_copy.get("properties", {})
            for input_prop in list(props.keys()):
                if input_prop in MCP_MAP_HEADER_TO_INPUT:
                    props.pop(input_prop, None)
                    required = input_schema_copy.get("required")
                    if isinstance(required, list) and input_prop in required:
                        input_schema_copy["required"] = [
                            r for r in required if r != input_prop
                        ]
            if not input_schema_copy.get("properties"):
                input_schema_copy = {}

        filtered_tools.append(
            {
                "name": name,
                "title": title,
                "description": description,
                "inputSchema": input_schema_copy,
                "outputSchema": output_schema,
                "annotations": annotations,
                "meta": meta,
                "url": f"{MCP_BASE_PATH}/tools/{name}",
            }
        )

    return filtered_tools


def _read_tools_cache() -> list[dict[str, Any]] | None:
    """Return cached tools when available and valid."""
    if not TOOLS_CACHE_ENABLED:
        return None

    try:
        with TOOLS_CACHE_FILE.open("r") as cache_file:
            data = json.load(cache_file)
            if not isinstance(data, dict):
                return None

            meta = data.get("__meta__", {})
            if meta.get("version") != CACHE_VERSION:
                return None

            if meta.get("signature") != _cache_signature():
                return None

            tools = data.get("tools")
            return tools if isinstance(tools, list) else None
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"[ToolsCache] Unable to read cached tools: {exc}")
        return None


def _write_tools_cache(tools: list[dict[str, Any]]) -> bool:
    """
    Persist tool list to cache using a non-blocking file lock.
    If the lock is busy, skip caching and return False.
    """
    if not TOOLS_CACHE_ENABLED:
        return False

    lock_fd = None
    lock_acquired = False
    try:
        TOOLS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(TOOLS_CACHE_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
        except BlockingIOError:
            logger.debug("[ToolsCache] Cache lock busy; returning live result.")
            return False

        tmp_path = TOOLS_CACHE_FILE.with_name(TOOLS_CACHE_FILE.name + ".tmp")
        payload = {
            "__meta__": {"version": CACHE_VERSION, "signature": _cache_signature()},
            "tools": tools,
        }
        with tmp_path.open("w") as tmp_file:
            json.dump(payload, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, TOOLS_CACHE_FILE)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[ToolsCache] Unable to write cached tools: {exc}")
        return False
    finally:
        if lock_fd is not None:
            try:
                if lock_acquired:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


async def _cached_mapped_tools(
    fetch_raw_tools: Callable[[], Awaitable[Any]],
) -> list[dict[str, Any]]:
    """
    Return mapped tools from cache when available, otherwise fetch and cache.
    If caching fails due to a locked file, return the live response immediately.
    """
    cached = _read_tools_cache()
    if cached is not None:
        return cached

    tools = await fetch_raw_tools()
    mapped_tools = map_tools(tools)

    if not mapped_tools:
        return mapped_tools

    if _write_tools_cache(mapped_tools):
        cached = _read_tools_cache()
        if cached is not None:
            return cached

    return mapped_tools


def inject_headers_into_args(
    tools, tool_name: str, args: Optional[Dict], incoming_headers: Optional[dict]
) -> Dict:
    """
    Fill missing args for tool_name from incoming_headers according to
    MCP_MAP_HEADER_TO_INPUT mapping. Header matching is case-insensitive.
    """
    if not MCP_MAP_HEADER_TO_INPUT or not incoming_headers:
        return args or {}

    # normalize incoming headers to lowercase keys for case-insensitive lookup
    headers_lc = {
        k.lower(): v for k, v in (incoming_headers.items() if incoming_headers else [])
    }

    # tools can be an object with .tools attribute or a list of tool-like dicts
    tool_list = getattr(tools, "tools", tools) if tools is not None else []
    tool_def = None
    for t in tool_list:
        name = None
        if hasattr(t, "name"):
            name = getattr(t, "name")
        elif isinstance(t, dict):
            name = t.get("name")
        if name == tool_name:
            tool_def = t
            break

    if not tool_def:
        return args or {}

    # Obtain input schema properties
    input_schema = None
    if hasattr(tool_def, "inputSchema"):
        input_schema = getattr(tool_def, "inputSchema")
    elif isinstance(tool_def, dict):
        input_schema = tool_def.get("inputSchema")

    props = (
        (input_schema or {}).get("properties", {})
        if isinstance(input_schema, dict)
        else {}
    )

    out_args = dict(args or {})
    for input_prop, header_name in MCP_MAP_HEADER_TO_INPUT.items():
        # Only consider if tool declares this property
        if input_prop not in props:
            continue
        # Do not overwrite explicit args
        if input_prop in out_args:
            continue
        header_val = headers_lc.get(header_name.lower())
        if header_val is not None:
            out_args[input_prop] = header_val

    return out_args


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
                        return _to_tool_list(tools)

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
                        # Inject header-mapped inputs if available
                        decorated_args = inject_headers_into_args(
                            tools, tool_name, decorated_args, incoming_headers
                        )
                        result = await session.call_tool(tool_name, decorated_args)
                        return RunToolsResult(result)

                    async def call_tool_with_progress(
                        self,
                        tool_name: str,
                        args: Optional[Dict],
                        access_token_inner: Optional[str],
                        progress_callback: Optional[Callable] = None,
                        log_callback: Optional[Callable] = None,
                    ):
                        """
                        Call a tool with progress and log callbacks for streaming updates.

                        Args:
                            tool_name: Name of the tool to call
                            args: Arguments to pass to the tool
                            access_token_inner: OAuth access token
                            progress_callback: Async callback for progress updates.
                                              Signature: (progress: float, total: float | None, message: str | None) -> None
                            log_callback: Async callback for log messages.
                                         Signature: (level: str, data: Any, logger_name: str | None) -> None
                                         Note: Log callbacks are handled via session-level notifications,
                                         so this parameter is currently informational only.

                        Returns:
                            RunToolsResult with the tool execution result
                        """
                        tools = await session.list_tools()
                        decorated_args = await decorate_args_with_oauth_token(
                            tools, tool_name, args, access_token_inner
                        )
                        # Inject header-mapped inputs if available
                        decorated_args = inject_headers_into_args(
                            tools, tool_name, decorated_args, incoming_headers
                        )

                        call_fn = getattr(session, "call_tool_with_progress", None)
                        if not call_fn:
                            call_fn = getattr(session, "call_tool", None)
                        if not call_fn:
                            raise RuntimeError("MCP session missing call_tool")

                        call_kwargs: dict[str, Any] = {}
                        try:
                            sig = inspect.signature(call_fn)
                            if "progress_callback" in sig.parameters:
                                call_kwargs["progress_callback"] = progress_callback
                            if log_callback and "log_callback" in sig.parameters:
                                call_kwargs["log_callback"] = log_callback
                        except Exception:
                            if progress_callback:
                                call_kwargs["progress_callback"] = progress_callback
                            if log_callback:
                                call_kwargs["log_callback"] = log_callback

                        result = await call_fn(tool_name, decorated_args, **call_kwargs)
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
            return _to_tool_list(tools)

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
            # Inject header-mapped inputs if available
            decorated_args = inject_headers_into_args(
                tools, tool_name, decorated_args, incoming_headers
            )
            result = await mcp_task.request(
                {"action": "run_tool", "tool_name": tool_name, "args": decorated_args}
            )
            return RunToolsResult(result)

        async def call_tool_with_progress(
            self,
            tool_name: str,
            args: Optional[Dict],
            access_token_inner: Optional[str],
            progress_callback: Optional[Callable] = None,
            log_callback: Optional[Callable] = None,
        ):
            """
            Call a tool with progress and log callbacks for streaming updates.

            Args:
                tool_name: Name of the tool to call
                args: Arguments to pass to the tool
                access_token_inner: OAuth access token
                progress_callback: Async callback for progress updates.
                log_callback: Async callback for log messages.

            Returns:
                RunToolsResult with the tool execution result
            """
            tools = await mcp_task.request("list_tools")
            decorated_args = await decorate_args_with_oauth_token(
                tools, tool_name, args, access_token_inner
            )
            decorated_args = inject_headers_into_args(
                tools, tool_name, decorated_args, incoming_headers
            )

            return RunToolsResult(
                await mcp_task.request(
                    {
                        "action": "run_tool_with_progress",
                        "tool_name": tool_name,
                        "args": decorated_args,
                        "progress_callback": progress_callback,
                        "log_callback": log_callback,
                    }
                )
            )

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
