"""One discovery deadline for direct and task-backed MCP sessions."""

import logging
from typing import Awaitable, TypeVar

import anyio
from fastapi import HTTPException

from app import vars as app_vars

T = TypeVar("T")
logger = logging.getLogger("uvicorn.error")


async def with_discovery_deadline(operation: Awaitable[T]) -> T:
    timeout = app_vars.per_server_float(
        "MCP_DISCOVERY_TIMEOUT_SECONDS", app_vars.MCP_DISCOVERY_TIMEOUT_SECONDS
    )
    if timeout <= 0:
        return await operation
    try:
        with anyio.fail_after(timeout):
            return await operation
    except TimeoutError:
        logger.warning("[Tools] MCP discovery timed out before tool execution")
        raise HTTPException(
            status_code=504,
            detail={
                "error": "upstream_discovery_timeout",
                "phase": "discovery",
                "retryable": True,
                "message": "MCP tool discovery timed out; no tool was executed",
            },
        ) from None


def is_discovery_timeout(error: BaseException) -> bool:
    return (
        isinstance(error, HTTPException)
        and error.status_code == 504
        and isinstance(error.detail, dict)
        and error.detail.get("error") == "upstream_discovery_timeout"
    )
