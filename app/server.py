from app.vars import SERVICE_NAME, OTLP_ENDPOINT, OTLP_HEADERS
from fastapi import FastAPI
from .routes import router
from opentelemetry import trace
from typing import Sequence

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Info

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
