from app.vars import SERVICE_NAME, OTLP_ENDPOINT, OTLP_HEADERS
from fastapi import FastAPI
from .routes import router
from opentelemetry import trace
from typing import Sequence
import logging

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Info

from app.sse.mcp_proxy import get_sse_proxy_routes


class OTELFormatter(logging.Formatter):
    """Custom formatter that includes ISO 8601 timestamps and OpenTelemetry trace IDs"""

    def format(self, record: logging.LogRecord) -> str:
        # Create ISO 8601 timestamp with milliseconds
        timestamp = self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
        timestamp += f".{int(record.msecs):03d}Z"

        # Get current trace ID from OpenTelemetry context
        trace_id = "<unknown>"
        try:
            span = trace.get_current_span()
            if span and span.is_recording():
                trace_context = span.get_span_context()
                if trace_context:
                    trace_id = f"{trace_context.trace_id:032x}"
        except Exception:
            pass

        # Format the message with timestamp, trace ID, level, and logger name
        msg = record.getMessage()
        return f"{timestamp} [trace_id={trace_id}] {record.levelname} {record.name}: {msg}"


def configure_logging() -> None:
    """Configure logging with timestamps and OTEL trace IDs for uvicorn"""
    formatter = OTELFormatter()

    # Configure uvicorn access and error loggers
    for logger_name in ["uvicorn.error", "uvicorn.access"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add stream handler with custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logging.getLogger("uvicorn.error").propagate = False
    logging.getLogger("uvicorn.access").propagate = False


# Configure logging early before app creation
configure_logging()

app = FastAPI()
instrumentator = Instrumentator()

instrumentator.instrument(app).expose(app)


try:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when OTEL not installed
    Resource = TracerProvider = ReadableSpan = BatchSpanProcessor = SpanExporter = SpanExportResult = None  # type: ignore
    FastAPIInstrumentor = OTLPSpanExporter = None  # type: ignore
    _OTEL_AVAILABLE = False


class FilteringSpanExporter(SpanExporter if _OTEL_AVAILABLE else object):
    """
    Wrapper exporter that filters out noisy ASGI body spans from streaming responses.
    This prevents hundreds of tiny spans from cluttering traces.
    """

    def __init__(self, exporter: SpanExporter):
        self.exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        # Filter out http.response.body spans that clutter streaming traces
        filtered_spans = [
            span
            for span in spans
            if not (
                span.attributes
                and span.attributes.get("asgi.event.type") == "http.response.body"
            )
        ]
        if filtered_spans:
            return self.exporter.export(filtered_spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        return self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000):
        return self.exporter.force_flush(timeout_millis)


# Configure tracing if OpenTelemetry dependencies are available
if _OTEL_AVAILABLE:
    trace.set_tracer_provider(
        TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    )
    tracer_provider = trace.get_tracer_provider()
    if OTLP_ENDPOINT:
        otlp_exporter = OTLPSpanExporter(
            endpoint=OTLP_ENDPOINT,
            headers=((OTLP_HEADERS).split(",") if OTLP_HEADERS else None),
        )

        # Wrap exporter with filtering to remove noisy ASGI body spans
        filtering_exporter = FilteringSpanExporter(otlp_exporter)
        span_processor = BatchSpanProcessor(filtering_exporter)
        tracer_provider.add_span_processor(span_processor)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="",  # We handle URL exclusion elsewhere
        server_request_hook=None,
        client_request_hook=None,
    )

# Add app_name to the metrics
app_info = Info("fastapi_app_info", "Application Info")
app_info.info({"app_name": SERVICE_NAME})

app.include_router(router)

for route in get_sse_proxy_routes():
    app.router.routes.insert(0, route)
