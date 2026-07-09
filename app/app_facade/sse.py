"""Shared Server-Sent-Events formatting helpers for the app_facade module."""

import json
from typing import Any, Dict


def sse_event(event: str, payload: Dict[str, Any]) -> bytes:
    return (
        f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode(
            "utf-8"
        )
    )


def assistant_status_event(status: str) -> bytes:
    return sse_event(
        "assistant",
        {
            "delta": status,
            "is_status": True,
        },
    )
