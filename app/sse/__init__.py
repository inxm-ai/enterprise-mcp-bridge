"""
SSE (Server-Sent Events) module for streaming progress updates from MCP tool calls.

This module provides functionality to stream real-time progress updates, logs,
and results from long-running MCP tool executions to REST clients using the
Server-Sent Events protocol.

Example usage with curl:
    curl -N -H "Accept: text/event-stream" \\
         -H "Content-Type: application/json" \\
         -d '{"a": 1, "b": 2}' \\
         "http://localhost:8000/tools/my_tool/stream"

Example usage with JavaScript:
    const eventSource = new EventSource('/tools/report_progress_and_logs/stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'progress') {
            console.log(`Progress: ${data.progress}% - ${data.message}`);
        } else if (data.type === 'log') {
            console.log(`[${data.level}] ${data.data}`);
        } else if (data.type === 'result') {
            console.log('Result:', data.data);
            eventSource.close();
        }
    };
"""

from .streaming import (
    SSEEvent,
    SSEEventType,
    stream_tool_call,
    create_sse_response,
)

__all__ = [
    "SSEEvent",
    "SSEEventType",
    "stream_tool_call",
    "create_sse_response",
]
