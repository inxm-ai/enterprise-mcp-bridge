"""
Utility functions for enhanced exception logging, particularly for TaskGroup exceptions.
"""

import logging
import re
from typing import Any
from contextlib import contextmanager

CLIENT_ERROR_STATUS_CODE_PATTERN = re.compile(r"\b(4\d{2})\b")


def _safe_str(obj) -> str:
    """
    Safely convert an object to string, handling cases where __str__ or __repr__ might fail.

    Args:
        obj: The object to convert to string

    Returns:
        A string representation of the object, falling back to safe alternatives
    """
    try:
        return str(obj)
    except Exception:
        try:
            return repr(obj)
        except Exception:
            try:
                return f"<{type(obj).__name__} object (string conversion failed)>"
            except Exception:
                return "<object (all string conversions failed)>"


def _safe_get_exceptions(exception_group) -> list:
    """
    Safely get the exceptions list from an exception group.

    Args:
        exception_group: The exception group object

    Returns:
        List of exceptions, or empty list if access fails
    """
    try:
        return list(exception_group.exceptions)
    except Exception:
        return []


# Only bounded metadata is safe to send to clients or telemetry. Remote error
# messages, URLs, bodies and MCP error.data can all contain credentials.
UPSTREAM_BAD_GATEWAY = 502
FAILURE_PHASES = {"transport", "initialization", "discovery", "execution"}


def _exception_nodes(exception):
    pending = [exception]
    seen = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(_safe_get_exceptions(current))
        cause = getattr(current, "__cause__", None)
        if cause is not None:
            pending.append(cause)


def _failure_detail(exception, **metadata):
    phase = getattr(exception, "_mcp_failure_phase", "execution")
    return {
        "error": "upstream_failure",
        "phase": phase if phase in FAILURE_PHASES else "execution",
        "exception_type": type(exception).__name__,
        **metadata,
    }


def _retry_after(value):
    from email.utils import parsedate_to_datetime

    if not value or len(value) > 128 or "\r" in value or "\n" in value:
        return None
    if re.fullmatch(r"[0-9]+", value):
        return value
    try:
        if parsedate_to_datetime(value).tzinfo is not None:
            return value
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def find_exception_in_exception_groups(exception: Exception, target_type: Any):
    """Find typed failures before attempting legacy text-only status recovery.

    Searching leaves first prevents a group's string (which may contain a
    random 4xx number) from hiding a real HTTP or MCP protocol failure.
    """
    from fastapi import HTTPException
    from mcp.shared.exceptions import MCPError
    import httpx
    import httpx2

    nodes = list(_exception_nodes(exception))
    for node in nodes:
        if isinstance(node, target_type):
            return node
        if target_type is not HTTPException:
            continue
        if isinstance(node, (httpx.HTTPStatusError, httpx2.HTTPStatusError)):
            status = node.response.status_code
            retry_after = _retry_after(node.response.headers.get("Retry-After"))
            return HTTPException(
                status_code=status,
                detail=_failure_detail(node, upstream_status=status),
                headers={"Retry-After": retry_after} if retry_after else None,
            )
        if isinstance(node, MCPError):
            return HTTPException(
                status_code=UPSTREAM_BAD_GATEWAY,
                detail=_failure_detail(node, protocol_code=node.code),
            )
    if target_type is HTTPException:
        for node in nodes:
            # Compatibility for transports that have already discarded the
            # typed HTTP response. Never copy the untrusted message out.
            match = CLIENT_ERROR_STATUS_CODE_PATTERN.search(_safe_str(node))
            if match:
                status = int(match.group(1))
                return HTTPException(
                    status_code=status,
                    detail=_failure_detail(node, upstream_status=status),
                )
    return None


@contextmanager
def upstream_phase(phase: str):
    """Annotate original errors without changing MCP protocol semantics."""
    try:
        yield
    except Exception as exc:
        from opentelemetry import trace

        for node in _exception_nodes(exc):
            if not hasattr(node, "_mcp_failure_phase"):
                node._mcp_failure_phase = phase
        from fastapi import HTTPException

        converted = find_exception_in_exception_groups(exc, HTTPException)
        span = trace.get_current_span()
        span.set_attribute(
            "enterprise_mcp_bridge.failure.phase", _failure_detail(exc)["phase"]
        )
        span.set_attribute("exception.type", type(exc).__name__)
        log = logging.getLogger("uvicorn.error")
        metadata = (
            converted.detail if converted and isinstance(converted.detail, dict) else {}
        )
        log.warning(
            "[RemoteMCP] Upstream failure phase=%s type=%s status=%s code=%s",
            _failure_detail(exc)["phase"],
            type(exc).__name__,
            metadata.get("upstream_status"),
            metadata.get("protocol_code"),
        )
        if converted and isinstance(converted.detail, dict):
            for key in ("upstream_status", "protocol_code"):
                if key in converted.detail:
                    span.set_attribute(
                        "enterprise_mcp_bridge." + key, converted.detail[key]
                    )
        # Preserve the SDK/framework exception contract; HTTP conversion is
        # performed only by REST routes, never inside the MCP session.
        raise


def log_exception_with_details(
    logger: logging.Logger,
    prefix: str,
    exception: Exception,
    level: int = logging.ERROR,
) -> None:
    """
    Log an exception with detailed information, including sub-exceptions for TaskGroup errors.
    This function is designed to never throw exceptions itself, even when dealing with
    broken exception objects or logger failures.

    Args:
        logger: The logger instance to use
        prefix: Prefix for the log message (e.g., "[Tool-Call]", "[Session]")
        exception: The exception to log
        level: The logging level to use (default: ERROR)
    """
    try:
        # Handle None or non-exception inputs gracefully
        if exception is None:
            safe_exception_str = "None"
        else:
            safe_exception_str = _safe_str(exception)

        # Safely handle prefix
        safe_prefix = _safe_str(prefix) if prefix is not None else ""

        # Check if this might be an exception group
        has_exceptions = False
        sub_exceptions = []

        try:
            if exception is not None and hasattr(exception, "exceptions"):
                sub_exceptions = _safe_get_exceptions(exception)
                has_exceptions = len(sub_exceptions) > 0
        except Exception:
            # If checking for exceptions fails, treat as regular exception
            has_exceptions = False
            sub_exceptions = []

        if has_exceptions:
            # Log the main exception group
            try:
                main_message = f"{safe_prefix} Exception with {len(sub_exceptions)} sub-exceptions: {safe_exception_str}"
                logger.log(level, main_message)
            except Exception:
                # If logging fails, try without formatting
                try:
                    logger.log(
                        level, f"{safe_prefix} Exception group (logging details failed)"
                    )
                except Exception:
                    # If even basic logging fails, give up silently
                    pass

            # Log each sub-exception
            for i, sub_exc in enumerate(sub_exceptions):
                try:
                    sub_exc_type = (
                        type(sub_exc).__name__ if sub_exc is not None else "NoneType"
                    )
                    sub_exc_str = _safe_str(sub_exc)
                    sub_message = f"{safe_prefix} Sub-exception {i+1}: {sub_exc_type}: {sub_exc_str}"
                    logger.log(level, sub_message, exc_info=sub_exc)
                except Exception:
                    # If logging this sub-exception fails, try basic logging
                    try:
                        logger.log(
                            level,
                            f"{safe_prefix} Sub-exception {i+1}: (logging failed)",
                        )
                    except Exception:
                        # If even basic logging fails, continue with next sub-exception
                        continue
        else:
            # Log as regular exception
            try:
                main_message = f"{safe_prefix} Exception: {safe_exception_str}"
                logger.log(
                    level,
                    main_message,
                    exc_info=exception if exception is not None else False,
                )
            except Exception:
                # If logging with exc_info fails, try without it
                try:
                    logger.log(level, f"{safe_prefix} Exception: {safe_exception_str}")
                except Exception:
                    # If even basic logging fails, try minimal logging
                    try:
                        logger.log(level, f"{safe_prefix} Exception (logging failed)")
                    except Exception:
                        # If all logging fails, give up silently
                        pass

    except Exception:
        # If anything in the entire function fails, try one last minimal log attempt
        try:
            if logger is not None:
                logger.log(logging.ERROR, "Exception logging failed")
        except Exception:
            # If even this fails, give up completely (don't propagate the exception)
            pass


def format_exception_message(exception: Exception) -> str:
    """
    Format an exception message, including sub-exceptions for TaskGroup errors.
    This function is designed to never throw exceptions itself, even when dealing with
    broken exception objects.

    Args:
        exception: The exception to format

    Returns:
        A formatted string describing the exception
    """
    try:
        # Handle None or non-exception inputs gracefully
        if exception is None:
            return "None"

        # Check if this might be an exception group
        has_exceptions = False
        sub_exceptions = []

        try:
            if hasattr(exception, "exceptions"):
                sub_exceptions = _safe_get_exceptions(exception)
                has_exceptions = len(sub_exceptions) > 0
        except Exception:
            # If checking for exceptions fails, treat as regular exception
            has_exceptions = False

        if has_exceptions:
            try:
                # Format the main exception
                main_str = _safe_str(exception)

                # Format sub-exceptions safely
                sub_exception_strs = []
                for sub_exc in sub_exceptions:
                    try:
                        sub_exc_type = (
                            type(sub_exc).__name__
                            if sub_exc is not None
                            else "NoneType"
                        )
                        sub_exc_str = _safe_str(sub_exc)
                        sub_exception_strs.append(f"{sub_exc_type}: {sub_exc_str}")
                    except Exception:
                        # If formatting this sub-exception fails, add a placeholder
                        sub_exception_strs.append("(formatting failed)")

                sub_exceptions_joined = "; ".join(sub_exception_strs)
                return f"{main_str} (Sub-exceptions: {sub_exceptions_joined})"

            except Exception:
                # If formatting as exception group fails, fall back to basic formatting
                return _safe_str(exception)
        else:
            # Format as regular exception
            return _safe_str(exception)

    except Exception:
        # If everything fails, return a safe fallback
        try:
            return f"<{type(exception).__name__} (formatting failed)>"
        except Exception:
            return "<exception (all formatting failed)>"
