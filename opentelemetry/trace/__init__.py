"""Minimal trace API stub used for unit testing."""

from __future__ import annotations
from typing import Any


class _Span:
    """Simple span stub that records attributes."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict]] = []
        self.status: Any = None

    def __enter__(self) -> "_Span":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def end(self) -> None:
        """Terminate the span (no-op in stub)."""
        return None

    def record_exception(self, _exception: Exception) -> None:
        return None

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        self.events.append((name, dict(attributes or {})))

    def set_status(self, status: Any, description: str | None = None) -> None:
        self.status = status

    def is_recording(self) -> bool:
        return False


class _SpanContextManager:
    def __init__(self, name: str, **_kwargs: Any) -> None:
        self._span = _Span(name)

    def __enter__(self) -> _Span:
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class Tracer:
    """Tracer stub exposing minimal tracing surface used in tests."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _SpanContextManager:
        return _SpanContextManager(name, **kwargs)

    def start_span(self, name: str) -> _Span:
        return _Span(name)


Span = _Span


def get_tracer(_name: str) -> Tracer:
    return Tracer()


class SpanKind:
    """Span kind constants (stub)."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class StatusCode:
    """Status code constants (stub)."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class Status:
    """Span status (stub)."""

    def __init__(self, status_code: Any = None, description: str | None = None) -> None:
        self.status_code = status_code
        self.description = description


def get_current_span() -> _Span:
    return _Span()


__all__ = [
    "get_tracer",
    "get_current_span",
    "Tracer",
    "Span",
    "SpanKind",
    "Status",
    "StatusCode",
]
