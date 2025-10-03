"""Minimal trace API stub used for unit testing."""
from __future__ import annotations
from typing import Any


class _Span:
    """Simple span stub that records attributes."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}

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


class _SpanContextManager:
    def __init__(self, name: str) -> None:
        self._span = _Span(name)

    def __enter__(self) -> _Span:
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class Tracer:
    """Tracer stub exposing minimal tracing surface used in tests."""

    def start_as_current_span(self, name: str) -> _SpanContextManager:
        return _SpanContextManager(name)

    def start_span(self, name: str) -> _Span:
        return _Span(name)


Span = _Span


def get_tracer(_name: str) -> Tracer:
    return Tracer()


__all__ = ["get_tracer", "Tracer", "Span"]
