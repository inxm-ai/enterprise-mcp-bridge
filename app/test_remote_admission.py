"""Remote admission lifecycle and cross-worker regression tests."""

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager

import anyio
import pytest
from fastapi import HTTPException

from app import admission, vars as app_vars
from app.multi_server import ServerConfig, bind_server, reset_server
import app.session.client_strategy as strategy_module


@pytest.fixture
def remote_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_ADMISSION_DIR", str(tmp_path))
    monkeypatch.setattr(app_vars, "MCP_REMOTE_MAX_CONCURRENT_SESSIONS", 5)
    monkeypatch.setattr(app_vars, "MCP_REMOTE_ADMISSION_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(admission, "POLL_SECONDS", 0.001)
    return tmp_path


def strategy(url="https://remote.invalid/mcp"):
    return strategy_module.RemoteMCPClientStrategy(
        url, access_token=None, requested_group=None, anon=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url", ["https://remote.invalid/mcp", "https://remote.invalid/sse"]
)
async def test_100_sessions_bound_transport_initialization_discovery_and_teardown(
    remote_limit, monkeypatch, url
):
    active = peak = completed = 0

    @asynccontextmanager
    async def transport(*args, **kwargs):
        nonlocal active, peak, completed
        active += 1
        peak = max(active, peak)
        await anyio.sleep(0.001)
        try:
            yield object(), object()
        finally:
            await anyio.sleep(0.001)
            active -= 1
            completed += 1

    class ControlledSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            await anyio.sleep(0.001)

        async def list_tools(self):
            await anyio.sleep(0.001)

    monkeypatch.setattr(strategy_module, "_streamable_client", transport)
    monkeypatch.setattr(strategy_module, "sse_client", transport)
    monkeypatch.setattr(strategy_module, "ClientSession", ControlledSession)

    async def request():
        async with strategy(url).session() as session:
            await session.list_tools()

    await asyncio.gather(*(request() for _ in range(100)))
    assert peak == 5
    assert active == 0
    assert completed == 100


@pytest.mark.asyncio
async def test_timeout_makes_no_transport_contact_and_servers_are_independent(
    remote_limit, monkeypatch
):
    monkeypatch.setattr(app_vars, "MCP_REMOTE_MAX_CONCURRENT_SESSIONS", 1)
    monkeypatch.setattr(app_vars, "MCP_REMOTE_ADMISSION_TIMEOUT_SECONDS", 0)

    def forbidden(*args, **kwargs):
        pytest.fail("refused request contacted downstream")

    monkeypatch.setattr(strategy_module, "_streamable_client", forbidden)
    async with admission.remote_slot("https://remote.invalid/mcp"):
        with pytest.raises(HTTPException) as refused:
            async with strategy().session():
                pass
        async with admission.remote_slot("https://other.invalid/mcp"):
            pass
    assert refused.value.status_code == 503
    assert refused.value.headers == {"Retry-After": "5", "X-MCP-Admission": "refused"}


@pytest.mark.asyncio
async def test_cancellation_waiting_and_running_release_slots(
    remote_limit, monkeypatch
):
    monkeypatch.setattr(app_vars, "MCP_REMOTE_MAX_CONCURRENT_SESSIONS", 1)
    entered = asyncio.Event()

    async def hold():
        async with admission.remote_slot("a"):
            entered.set()
            await asyncio.Event().wait()

    owner = asyncio.create_task(hold())
    await entered.wait()
    waiter = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    async with admission.remote_slot("a"):
        pass


@pytest.mark.asyncio
async def test_per_server_settings_override_globals_and_identity_is_not_url(
    remote_limit, monkeypatch
):
    monkeypatch.setattr(app_vars, "MCP_REMOTE_MAX_CONCURRENT_SESSIONS", 0)
    server = ServerConfig(
        id="github",
        base_path="/github",
        remote_url="https://remote.invalid",
        settings={
            "MCP_REMOTE_MAX_CONCURRENT_SESSIONS": "1",
            "MCP_REMOTE_ADMISSION_TIMEOUT_SECONDS": "0",
        },
    )
    token = bind_server(server)
    try:
        async with admission.remote_slot("url-one"):
            with pytest.raises(HTTPException):
                async with admission.remote_slot("url-two"):
                    pass
    finally:
        reset_server(token)
    assert "secret" not in admission.remote_identity(
        "https://user:secret@remote.invalid?token=secret"
    )


def test_other_process_counts_and_crashed_owner_is_reclaimed(remote_limit):
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import asyncio, sys
from app import admission, vars
vars.MCP_REMOTE_MAX_CONCURRENT_SESSIONS = 1
async def main():
    async with admission.remote_slot("a"):
        print("ready", flush=True)
        await asyncio.Event().wait()
asyncio.run(main())
""",
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        assert worker.stdout.readline().strip() == "ready"
        assert admission._try_remote_admit(admission.remote_identity("a"), 1) is None
    finally:
        worker.kill()
        worker.wait(timeout=5)
    slot = admission._try_remote_admit(admission.remote_identity("a"), 1)
    assert slot is not None
    slot.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["initialize", "teardown"])
async def test_initialization_failure_and_bounded_teardown_release(
    remote_limit, monkeypatch, failure
):
    monkeypatch.setattr(app_vars, "MCP_REMOTE_MAX_CONCURRENT_SESSIONS", 1)
    monkeypatch.setattr(strategy_module, "MCP_REMOTE_TEARDOWN_TIMEOUT_SECONDS", 0.01)

    @asynccontextmanager
    async def transport(*args, **kwargs):
        try:
            yield object(), object()
        finally:
            if failure == "teardown":
                await anyio.sleep_forever()

    class ControlledSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            if failure == "initialize":
                raise RuntimeError("initialization failed")

    monkeypatch.setattr(strategy_module, "_streamable_client", transport)
    monkeypatch.setattr(strategy_module, "ClientSession", ControlledSession)
    if failure == "initialize":
        with pytest.raises(RuntimeError):
            async with strategy().session():
                pass
    else:
        with anyio.fail_after(1):
            async with strategy().session():
                pass
    async with admission.remote_slot("https://remote.invalid/mcp"):
        pass


@pytest.mark.asyncio
async def test_100_requests_against_controlled_http_mcp_server(
    remote_limit, monkeypatch
):
    """Exercise the real SDK transport against a loopback JSON-RPC server."""
    monkeypatch.setattr(app_vars, "MCP_REMOTE_ADMISSION_TIMEOUT_SECONDS", 60)
    import json
    import threading
    import uuid
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    lock = threading.Lock()
    sessions = set()
    peak = 0
    discoveries = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload=None, session=None):
            body = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if session:
                self.send_header("Mcp-Session-Id", session)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(405)

        def do_DELETE(self):
            with lock:
                sessions.discard(self.headers.get("Mcp-Session-Id"))
            self.reply(200)

        def do_POST(self):
            nonlocal peak, discoveries
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = request["method"]
            if "id" not in request:
                self.reply(202)
                return
            session = None
            if method == "initialize":
                session = str(uuid.uuid4())
                with lock:
                    sessions.add(session)
                    peak = max(peak, len(sessions))
                result = {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "controlled", "version": "1"},
                }
            elif method == "tools/list":
                with lock:
                    discoveries += 1
                result = {"tools": []}
            else:
                self.reply(
                    200,
                    {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "error": {"code": -32601, "message": "Method not found"},
                    },
                )
                return
            self.reply(
                200, {"jsonrpc": "2.0", "id": request["id"], "result": result}, session
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/mcp"

    async def request():
        async with strategy(url).session() as session:
            await session.list_tools()

    try:
        with anyio.fail_after(120):
            await asyncio.gather(*(request() for _ in range(100)))
        assert peak <= 5
        assert peak > 1
        assert discoveries == 100
        assert not sessions
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
