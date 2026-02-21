"""
Utility functions for enhanced exception logging, particularly for TaskGroup exceptions.
"""

import logging


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


def find_exception_in_exception_groups(exception: Exception, target_type: any):
    """
    Recursively search through an exception and its sub-exceptions to find
    if any exception is of the target type.

    Args:
        exception: The exception to search through
        target_type: The exception type to look for

    Returns:
        The first exception matching the target type, or None if not found
    """
    try:
        if isinstance(exception, target_type):
            return exception

        # Convert upstream HTTP transport status failures into FastAPI HTTP exceptions
        # so callers can pass through the remote status code (e.g. 401) instead of 500.
        converted_http_exception = _convert_http_status_error(exception, target_type)
        if converted_http_exception is not None:
            return converted_http_exception

        # Check for sub-exceptions if this is an exception group
        if hasattr(exception, "exceptions"):
            sub_exceptions = _safe_get_exceptions(exception)
            for sub_exc in sub_exceptions:
                inner_exc = find_exception_in_exception_groups(sub_exc, target_type)
                if inner_exc is not None:
                    return inner_exc

        return None
    except Exception:
        # If anything goes wrong, we assume we didn't find the target type
        return None


def _convert_http_status_error(exception: Exception, target_type: any):
    """
    Convert httpx.HTTPStatusError into FastAPI HTTPException when requested.

    This keeps upstream HTTP status codes visible to clients instead of
    collapsing them into generic 500 responses.
    """
    try:
        from fastapi import HTTPException as FastAPIHTTPException
        import httpx
    except Exception:
        return None

    if target_type is not FastAPIHTTPException:
        return None

    if not isinstance(exception, httpx.HTTPStatusError):
        return None

    status_code = getattr(getattr(exception, "response", None), "status_code", None)
    if not isinstance(status_code, int):
        return None

    return FastAPIHTTPException(status_code=status_code, detail=_safe_str(exception))


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
