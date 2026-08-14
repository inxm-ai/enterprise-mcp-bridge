"""In-memory tracer double for asserting span content in tests.

Works identically whether the real OpenTelemetry API or the repo-root test
stub is on the path — tests patch ``app.utils.mcp_operation._tracer`` with an
instance and assert on the recorded spans.
"""

from contextlib import contextmanager
from typing import Any


class RecordingSpan:
    def __init__(self, name: str):
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict]] = []
        self.status: Any = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        self.events.append((name, dict(attributes or {})))

    def set_status(self, status: Any, description: str | None = None) -> None:
        self.status = status


class RecordingTracer:
    def __init__(self):
        self.spans: list[RecordingSpan] = []

    @contextmanager
    def start_as_current_span(self, name: str, **_kwargs: Any):
        span = RecordingSpan(name)
        self.spans.append(span)
        yield span


def span_payload_text(span: RecordingSpan) -> str:
    """Every attribute and event value of a span, flattened for canary greps."""
    parts = [span.name]
    for key, value in span.attributes.items():
        parts.append(f"{key}={value!r}")
    for name, attributes in span.events:
        parts.append(name)
        for key, value in attributes.items():
            parts.append(f"{key}={value!r}")
    return "\n".join(parts)
