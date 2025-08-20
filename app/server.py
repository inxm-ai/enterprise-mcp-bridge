from fastapi import FastAPI
from .routes import router
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Info

app = FastAPI()
instrumentator = Instrumentator()

instrumentator.instrument(app).expose(app)

# Configure the tracer provider for OTLP
SERVICE_NAME = os.getenv("SERVICE_NAME", "mcp-rest-server")
trace.set_tracer_provider(TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME})))
tracer_provider = trace.get_tracer_provider()
otlp_endpoint = os.getenv("OTLP_ENDPOINT")
if otlp_endpoint:
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers=(os.getenv("OTLP_HEADERS", "")).split(",") if os.getenv("OTLP_HEADERS") else None,
    )

    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
FastAPIInstrumentor.instrument_app(app)

# Add app_name to the metrics
app_info = Info('fastapi_app_info', 'Application Info')
app_info.info({'app_name': SERVICE_NAME})

app.include_router(router)