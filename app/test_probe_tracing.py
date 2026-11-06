"""Probes and Prometheus scrapes must not reach the trace store.

Kubernetes liveness/readiness probes and Prometheus scrapes hit every pod
every few seconds; traced, they dominate an idle install's span volume.
"""

import sys
import textwrap
from contextlib import asynccontextmanager

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from mcp import StdioServerParameters
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import app.routes as routes
from app.server import app as fastapi_app
from app.session.session import mcp_session
from app.utils.probe_tracing import probe_excluded_urls, untraced

BASE = "/api/mcp-jaeger-server"


@pytest.fixture(scope="module")
def global_exporter():
    """Record every span finished through the global (SDK) provider."""
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        pytest.skip("OpenTelemetry SDK provider not configured")
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def spans(global_exporter):
    global_exporter.clear()
    yield global_exporter
    global_exporter.clear()


def _names(exporter):
    return [span.name for span in exporter.get_finished_spans()]


@pytest.mark.parametrize(
    "url",
    [
        "http://pod:8000/metrics",
        "http://pod:8000/",
        "http://pod:8000/healthz",
        "http://pod:8000/healthz/child",
        f"http://pod:8000{BASE}",
        f"http://pod:8000{BASE}/",
        f"http://pod:8000{BASE}/healthz",
        f"http://pod:8000{BASE}/healthz/child",
        f"http://pod:8000{BASE}/metrics",
    ],
)
def test_probe_urls_are_excluded(url):
    from opentelemetry.util.http import parse_excluded_urls

    assert parse_excluded_urls(probe_excluded_urls([BASE])).url_disabled(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://pod:8000/tools",
        "http://pod:8000/tools/metrics",
        "http://pod:8000/tools/healthz",
        f"http://pod:8000{BASE}/tools",
        f"http://pod:8000{BASE}/tools/metrics",
        "http://pod:8000/api/other-server/healthz/child",
        "http://pod:8000/metricsx",
    ],
)
def test_real_routes_stay_traced(url):
    from opentelemetry.util.http import parse_excluded_urls

    assert not parse_excluded_urls(probe_excluded_urls([BASE])).url_disabled(url)


def test_user_excluded_urls_are_kept():
    excluded = probe_excluded_urls([""], extra="client/.*/info")
    from opentelemetry.util.http import parse_excluded_urls

    assert parse_excluded_urls(excluded).url_disabled("http://pod/client/1/info")


def test_base_path_with_a_comma_is_one_pattern():
    import re

    urls = probe_excluded_urls(["/api/a,b"])
    assert "," not in urls
    assert re.search(urls, "http://host:8000/api/a,b/healthz")
    assert not re.search(urls, "http://host:8000/api/a,b/tools")


def test_base_path_probes_produce_no_spans():
    """The instrumentation drops the request span and its http send children."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    mini = FastAPI()
    router = APIRouter(prefix=BASE)

    @router.get("/")
    async def home():
        return {"ok": True}

    @router.get("/healthz/child")
    async def child():
        return {"status": "ok"}

    @router.get("/tools")
    async def tools():
        return []

    mini.include_router(router)
    FastAPIInstrumentor.instrument_app(
        mini, tracer_provider=provider, excluded_urls=probe_excluded_urls([BASE])
    )
    try:
        client = TestClient(mini)
        assert client.get(f"{BASE}/").status_code == 200
        assert client.get(f"{BASE}/healthz/child").status_code == 200
        assert _names(exporter) == []

        assert client.get(f"{BASE}/tools").status_code == 200
        assert f"GET {BASE}/tools" in _names(exporter)
    finally:
        FastAPIInstrumentor.uninstrument_app(mini)


def test_bridge_probes_and_scrapes_produce_no_spans(spans, monkeypatch):
    @asynccontextmanager
    async def fake_session(**kwargs):
        async def list_tools():
            return []

        yield type("S", (), {"list_tools": staticmethod(list_tools)})()

    monkeypatch.setattr(routes, "mcp_session", fake_session)
    client = TestClient(fastapi_app)

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "fastapi_app_info" in metrics.text  # Prometheus keeps working
    assert client.get("/").status_code == 200
    assert client.get("/healthz/child").status_code == 200
    assert _names(spans) == []

    assert client.get("/openapi.json").status_code == 200
    assert "GET /openapi.json" in _names(spans)


@pytest.fixture
def child_params(tmp_path):
    server = tmp_path / "probe_server.py"
    server.write_text(textwrap.dedent("""
            from mcp.server import MCPServer

            server = MCPServer("probe")

            @server.tool()
            def ping() -> str:
                return "pong"

            server.run("stdio")
            """))
    return StdioServerParameters(command=sys.executable, args=[str(server)])


@pytest.mark.asyncio
async def test_child_health_check_emits_no_mcp_spans(spans, child_params, monkeypatch):
    @asynccontextmanager
    async def probe_session(**kwargs):
        async with mcp_session(child_params) as session:
            yield session

    monkeypatch.setattr(routes, "mcp_session", probe_session)

    assert await routes.healthz_child() == {"status": "ok", "tools": 1}
    assert not [name for name in _names(spans) if name.startswith("MCP ")]


@pytest.mark.asyncio
async def test_real_session_still_emits_mcp_spans(spans, child_params):
    async with mcp_session(child_params) as session:
        await session.list_tools()

    names = _names(spans)
    assert "MCP send initialize" in names
    assert "MCP send tools/list" in names


@pytest.mark.asyncio
async def test_untraced_does_not_leak_into_later_spans(spans):
    tracer = trace.get_tracer(__name__)
    with untraced():
        with tracer.start_as_current_span("inside"):
            pass
    with tracer.start_as_current_span("after"):
        pass

    assert _names(spans) == ["after"]


_IMPORT_WITHOUT = textwrap.dedent("""
    import importlib.util, sys
    for name in sys.argv[2:]:
        sys.modules[name] = None  # import of a blocked module raises ImportError
    spec = importlib.util.spec_from_file_location("probe_tracing", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ran = []
    with module.untraced():
        ran.append(True)
    assert ran == [True]
    assert module.probe_excluded_urls(["/api/x"]).startswith("^")
    print("ok")
    """)


@pytest.mark.parametrize(
    "blocked",
    [
        ["opentelemetry.instrumentation", "opentelemetry.instrumentation.utils"],
        ["opentelemetry", "opentelemetry.context", "opentelemetry.trace"],
    ],
    ids=["without-instrumentation", "without-otel"],
)
def test_helper_imports_and_untraced_is_a_no_op_without_otel(blocked):
    import subprocess
    from pathlib import Path

    import app.utils.probe_tracing as helper

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT, str(Path(helper.__file__)), *blocked],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
