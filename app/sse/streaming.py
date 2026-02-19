"""
SSE streaming implementation for MCP tool calls with progress reporting.

This module provides the core streaming functionality that enables real-time
progress updates from MCP tool executions to be sent to clients via Server-Sent Events.

Event Types:
    - progress: Reports progress updates (progress percentage, total, message)
    - log: Reports log messages from the tool (level, data, logger name)
    - result: Contains the final tool execution result
    - error: Reports any errors that occurred during execution

SSE Format:
    Each event is sent as a JSON object prefixed with "data: " and followed by two newlines:

    data: {"type": "progress", "progress": 50, "total": 100, "message": "Processing..."}

    data: {"type": "result", "data": {"content": [...], "isError": false}}

"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from fastapi.responses import StreamingResponse

from app.elicitation import (
    ElicitationRequiredError,
    InvalidUserFeedbackError,
    UnsupportedElicitationSchemaError,
)

logger = logging.getLogger("uvicorn.error")


class SSEEventType(str, Enum):
    """Types of SSE events that can be sent during tool execution."""

    PROGRESS = "progress"
    LOG = "log"
    RESULT = "result"
    ERROR = "error"
    FEEDBACK_REQUIRED = "feedback_required"


@dataclass
class SSEEvent:
    """
    Represents a Server-Sent Event for streaming to clients.

    Attributes:
        type: The type of event (progress, log, result, error)
        progress: Current progress value (for progress events)
        total: Total progress value (for progress events, optional)
        message: Human-readable message (for progress events, optional)
        level: Log level (for log events)
        data: Event payload data (for log, result, and error events)
        logger_name: Name of the logger (for log events)
    """

    type: SSEEventType
    progress: Optional[float] = None
    total: Optional[float] = None
    message: Optional[str] = None
    level: Optional[str] = None
    data: Optional[Any] = None
    logger_name: Optional[str] = None

    def to_sse_string(self) -> str:
        """Convert the event to SSE wire format."""
        # Build a dict with only non-None values
        event_dict = {"type": self.type.value}

        if self.progress is not None:
            event_dict["progress"] = self.progress
        if self.total is not None:
            event_dict["total"] = self.total
        if self.message is not None:
            event_dict["message"] = self.message
        if self.level is not None:
            event_dict["level"] = self.level
        if self.data is not None:
            event_dict["data"] = self.data
        if self.logger_name is not None:
            event_dict["logger_name"] = self.logger_name

        return f"data: {json.dumps(event_dict)}\n\n"

    @classmethod
    def progress_event(
        cls,
        progress: float,
        total: Optional[float] = None,
        message: Optional[str] = None,
    ) -> "SSEEvent":
        """Create a progress event."""
        return cls(
            type=SSEEventType.PROGRESS,
            progress=progress,
            total=total,
            message=message,
        )

    @classmethod
    def log_event(
        cls,
        level: str,
        data: Any,
        logger_name: Optional[str] = None,
    ) -> "SSEEvent":
        """Create a log event."""
        return cls(
            type=SSEEventType.LOG,
            level=level,
            data=data,
            logger_name=logger_name,
        )

    @classmethod
    def result_event(cls, data: Any) -> "SSEEvent":
        """Create a result event."""
        return cls(type=SSEEventType.RESULT, data=data)

    @classmethod
    def error_event(cls, message: str, details: Optional[Any] = None) -> "SSEEvent":
        """Create an error event."""
        error_data = {"message": message}
        if details is not None:
            error_data["details"] = details
        return cls(type=SSEEventType.ERROR, data=error_data)

    @classmethod
    def feedback_required_event(cls, payload: dict[str, Any]) -> "SSEEvent":
        """Create a feedback-required event."""
        return cls(type=SSEEventType.FEEDBACK_REQUIRED, data=payload)


class ProgressNotificationHandler:
    """
    Handles progress notifications and log messages from MCP tool execution.

    This handler is used as the notification callback for MCP client sessions,
    collecting progress and log events and pushing them to an async queue for
    streaming to clients.
    """

    def __init__(self, event_queue: asyncio.Queue):
        """
        Initialize the handler with an event queue.

        Args:
            event_queue: Queue where SSE events will be pushed
        """
        self.event_queue = event_queue

    async def on_progress(
        self,
        progress: float,
        total: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """
        Handle a progress notification from the MCP tool.

        Args:
            progress: Current progress value
            total: Total progress value (optional)
            message: Human-readable progress message (optional)
        """
        event = SSEEvent.progress_event(progress, total, message)
        await self.event_queue.put(event)
        logger.debug(
            f"[SSE] Progress event: {progress}/{total or '?'} - {message or ''}"
        )

    async def on_log(
        self,
        level: str,
        data: Any,
        logger_name: Optional[str] = None,
    ) -> None:
        """
        Handle a log notification from the MCP tool.

        Args:
            level: Log level (e.g., 'info', 'warning', 'error')
            data: Log message or data
            logger_name: Name of the logger that produced the message
        """
        event = SSEEvent.log_event(level, data, logger_name)
        await self.event_queue.put(event)
        logger.debug(f"[SSE] Log event: [{level}] {data}")


async def stream_tool_call(
    call_tool_func: Callable,
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    access_token: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Execute a tool call and stream progress updates as SSE events.

    This generator function executes the given tool call function and yields
    SSE-formatted strings for progress updates and the final result.

    Args:
        call_tool_func: Async function that calls the tool with progress support.
                       Should accept (tool_name, args, access_token, progress_callback, log_callback)
                       Note: log_callback may not be used by all implementations.
        tool_name: Name of the tool to execute
        args: Arguments to pass to the tool
        access_token: Optional access token for authentication

    Yields:
        SSE-formatted strings for each event

    Example:
        async def my_call_tool(name, args, token, progress_cb, log_cb):
            # Your implementation that calls progress_cb
            result = await session.call_tool(name, args, progress_callback=progress_cb)
            return result

        async for event in stream_tool_call(my_call_tool, "my_tool", {"arg": "value"}):
            yield event
    """
    event_queue: asyncio.Queue[Optional[SSEEvent]] = asyncio.Queue()
    handler = ProgressNotificationHandler(event_queue)

    async def run_tool() -> Any:
        """Execute the tool and push the result to the queue."""
        try:
            result = await call_tool_func(
                tool_name,
                args or {},
                access_token,
                handler.on_progress,
                handler.on_log,  # May not be used by all implementations
            )
            await event_queue.put(SSEEvent.result_event(_serialize_result(result)))
        except ElicitationRequiredError as e:
            await event_queue.put(
                SSEEvent.feedback_required_event(e.to_client_payload())
            )
        except InvalidUserFeedbackError as e:
            await event_queue.put(
                SSEEvent.error_event(
                    "Invalid user feedback",
                    {"detail": str(e), "elicitation": e.payload},
                )
            )
        except UnsupportedElicitationSchemaError as e:
            await event_queue.put(
                SSEEvent.error_event(
                    "Unsupported elicitation schema",
                    {"detail": str(e), "elicitation": e.payload},
                )
            )
        except Exception as e:
            logger.exception(f"[SSE] Error executing tool {tool_name}: {e}")
            await event_queue.put(SSEEvent.error_event(str(e)))
        finally:
            # Signal that we're done
            await event_queue.put(None)

    # Start the tool execution in a background task
    task = asyncio.create_task(run_tool())

    try:
        while True:
            event = await event_queue.get()
            if event is None:
                # End of stream
                break
            yield event.to_sse_string()
    except asyncio.CancelledError:
        # Client disconnected
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def _serialize_result(result: Any) -> Any:
    """
    Serialize a tool result for JSON transmission.

    Handles common MCP result types and converts them to JSON-serializable dicts.
    """
    if result is None:
        return None

    # If it's already a dict, return as-is
    if isinstance(result, dict):
        return result

    # If it has a model_dump method (Pydantic model), use it
    if hasattr(result, "model_dump"):
        return result.model_dump()

    # If it has common MCP result attributes, extract them
    result_dict = {}

    if hasattr(result, "isError"):
        result_dict["isError"] = result.isError
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list):
            result_dict["content"] = [_serialize_content_item(item) for item in content]
        else:
            result_dict["content"] = content
    if hasattr(result, "structuredContent"):
        result_dict["structuredContent"] = result.structuredContent

    return result_dict if result_dict else str(result)


def _serialize_content_item(item: Any) -> Any:
    """Serialize a single content item from an MCP result."""
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "text"):
        result = {"text": item.text}
        if hasattr(item, "type"):
            result["type"] = item.type
        return result
    return str(item)


def create_sse_response(
    event_generator: AsyncGenerator[str, None],
) -> StreamingResponse:
    """
    Create a FastAPI StreamingResponse for SSE.

    Args:
        event_generator: Async generator that yields SSE-formatted strings

    Returns:
        A StreamingResponse configured for Server-Sent Events
    """
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )
