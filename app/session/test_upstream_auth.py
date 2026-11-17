"""Upstream "sign in first" refusals become a typed, non-retryable auth error.

The fake upstream behaves like https://www.meigen.ai/api/mcp: ``initialize``
and ``tools/list`` work anonymously, ``tools/call`` answers HTTP 401 with a
``WWW-Authenticate`` challenge. The real SDK transports run against it.
"""

import asyncio
import json
import queue
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import HTTPException
from mcp.shared.exceptions import MCPError

from app import admission, vars as app_vars
import app.session.client_strategy as strategy_module
from app.session.session import MCPLocalSessionTask
from app.session.upstream_auth import (
    UpstreamAuthRequiredError,
    parse_www_authenticate,
    sanitize_text,
)
from app.utils.exception_logging import find_exception_in_exception_groups

MEIGEN_CHALLENGE = (
    'Bearer resource_metadata="https://www.meigen.ai/.well-known/'
    'oauth-protected-resource/api/mcp", error_description="Connect your MeiGen '
    'account with OAuth", scope="meigen:generate meigen:read"'
)
MEIGEN_BODY = {
    "error": "authorization_required",
    "message": "Connect your MeiGen account with OAuth.",
}


@pytest.fixture(autouse=True)
def remote_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_ADMISSION_DIR", str(tmp_path))
    monkeypatch.setattr(app_vars, "MCP_REMOTE_ADMISSION_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(admission, "POLL_SECONDS", 0.001)


def _streamable_handler(tool_call_reply):
    """A streamable HTTP MCP server whose ``tools/call`` answers ``tool_call_reply(id)``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload=None, headers=None):
            body = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(405)

        def do_DELETE(self):
            self.reply(200)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "id" not in request:
                self.reply(202)
                return
            method = request["method"]
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "meigen", "version": "1"},
                }
                self.reply(
                    200,
                    {"jsonrpc": "2.0", "id": request["id"], "result": result},
                    {"Mcp-Session-Id": "session-1"},
                )
            elif method == "tools/list":
                tool = {"name": "generate_image", "inputSchema": {"type": "object"}}
                result = {"tools": [tool]}
                self.reply(
                    200, {"jsonrpc": "2.0", "id": request["id"], "result": result}
                )
            else:
                self.reply(*tool_call_reply(request["id"]))

    return Handler


def _meigen_refusal(request_id):
    return 401, MEIGEN_BODY, {"WWW-Authenticate": MEIGEN_CHALLENGE}


@contextmanager
def _upstream(handler, path="/api/mcp"):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}{path}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _strategy(url):
    return strategy_module.RemoteMCPClientStrategy(
        url, access_token=None, requested_group=None, anon=True
    )


async def _call(url, name="generate_image"):
    async with _strategy(url).session() as session:
        tools = await session.list_tools()
        assert [tool.name for tool in tools.tools] == ["generate_image"]
        return await session.call_tool(name, {"prompt": "a cat"})


def _http(exc) -> HTTPException:
    converted = find_exception_in_exception_groups(exc, HTTPException)
    assert converted is not None
    return converted


@pytest.mark.asyncio
async def test_a_401_challenge_on_tools_call_is_a_typed_auth_error():
    with _upstream(_streamable_handler(_meigen_refusal)) as url:
        with pytest.raises(UpstreamAuthRequiredError) as raised:
            await _call(url)

    # The SDK itself only said -32603; the status came from the HTTP answer.
    assert isinstance(raised.value.__cause__, MCPError)
    assert raised.value.__cause__.message == "Server returned an error response"
    converted = _http(raised.value)
    assert converted.status_code == 401
    detail = converted.detail
    assert detail["error"] == "upstream_auth_required"
    assert detail["upstream_status"] == 401
    assert detail["phase"] == "execution"
    assert detail["upstream_error"] == "authorization_required"
    assert detail["upstream_message"] == "Connect your MeiGen account with OAuth"
    assert detail["required_scopes"] == ["meigen:generate", "meigen:read"]
    assert "connect your account (sign in)" in detail["message"]
    assert "Connect your MeiGen account with OAuth" in detail["message"]
    # Never the challenge's URLs or the body wholesale.
    assert "resource_metadata" not in json.dumps(detail)
    assert "well-known" not in json.dumps(detail)


@pytest.mark.asyncio
async def test_an_insufficient_scope_403_stays_a_403():
    def refusal(request_id):
        challenge = 'Bearer error="insufficient_scope", scope="files:write"'
        return 403, {"error": "forbidden"}, {"WWW-Authenticate": challenge}

    with _upstream(_streamable_handler(refusal)) as url:
        with pytest.raises(UpstreamAuthRequiredError) as raised:
            await _call(url)

    converted = _http(raised.value)
    assert converted.status_code == 403
    assert converted.detail["upstream_error"] == "insufficient_scope"
    assert converted.detail["required_scopes"] == ["files:write"]
    assert "lacks access" in converted.detail["message"]


@pytest.mark.asyncio
async def test_a_jsonrpc_error_body_on_a_401_is_still_an_auth_error():
    def refusal(request_id):
        error = {"code": -32001, "message": "Missing bearer token"}
        return 401, {"jsonrpc": "2.0", "id": request_id, "error": error}, {}

    with _upstream(_streamable_handler(refusal)) as url:
        with pytest.raises(UpstreamAuthRequiredError) as raised:
            await _call(url)

    assert raised.value.code == -32001
    converted = _http(raised.value)
    assert converted.status_code == 401
    assert converted.detail["upstream_message"] == "Missing bearer token"


@pytest.mark.asyncio
async def test_a_jsonrpc_error_that_says_authorization_required_is_an_auth_error():
    def refusal(request_id):
        error = {"code": -32001, "message": "Authorization required for this tool"}
        return 200, {"jsonrpc": "2.0", "id": request_id, "error": error}, {}

    with _upstream(_streamable_handler(refusal)) as url:
        with pytest.raises(UpstreamAuthRequiredError) as raised:
            await _call(url)

    converted = _http(raised.value)
    assert converted.status_code == 401
    assert converted.detail["upstream_status"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        lambda request_id: (500, {"error": "boom"}, {}),
        lambda request_id: (
            200,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": "database exploded"},
            },
            {},
        ),
    ],
)
async def test_other_upstream_failures_stay_opaque_502s(reply):
    with _upstream(_streamable_handler(reply)) as url:
        with pytest.raises(MCPError) as raised:
            await _call(url)

    assert not isinstance(raised.value, UpstreamAuthRequiredError)
    converted = _http(raised.value)
    assert converted.status_code == 502
    assert converted.detail["error"] == "upstream_failure"


@pytest.mark.asyncio
async def test_a_persistent_session_reports_the_auth_error_to_its_caller():
    with _upstream(_streamable_handler(_meigen_refusal)) as url:
        task = MCPLocalSessionTask(_strategy(url))
        task.start()
        try:
            await task.wait_started()
            with pytest.raises(UpstreamAuthRequiredError):
                await asyncio.wait_for(
                    task.request(
                        {
                            "action": "run_tool",
                            "tool_name": "generate_image",
                            "args": {},
                        }
                    ),
                    10,
                )
        finally:
            await task.stop()


def _sse_handler():
    """A legacy HTTP+SSE MCP server whose ``tools/call`` POST answers 401."""
    events: "queue.Queue[dict]" = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"event: endpoint\ndata: /messages?session_id=1\n\n")
            self.wfile.flush()
            while True:
                try:
                    message = events.get(timeout=0.2)
                except queue.Empty:
                    try:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    except OSError:
                        return
                    continue
                data = json.dumps(message).encode()
                self.wfile.write(b"event: message\ndata: " + data + b"\n\n")
                self.wfile.flush()

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if request.get("method") == "tools/call":
                body = json.dumps(MEIGEN_BODY).encode()
                self.send_response(401)
                self.send_header("WWW-Authenticate", MEIGEN_CHALLENGE)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            if "id" not in request:
                return
            if request["method"] == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "legacy", "version": "1"},
                }
            else:
                tool = {"name": "generate_image", "inputSchema": {"type": "object"}}
                result = {"tools": [tool]}
            events.put({"jsonrpc": "2.0", "id": request["id"], "result": result})

    return Handler


@pytest.mark.asyncio
async def test_the_legacy_sse_transport_does_not_hang_on_a_refused_call():
    # The SDK's SSE writer dies on the refused POST and never answers the
    # request; without the refusal watch this waits for the read timeout.
    with _upstream(_sse_handler(), path="/sse") as url:
        with pytest.raises(UpstreamAuthRequiredError) as raised:
            await asyncio.wait_for(_call(url), 10)

    assert _http(raised.value).status_code == 401
    assert raised.value.challenge.scopes == ["meigen:generate", "meigen:read"]


@pytest.mark.asyncio
async def test_the_rest_route_answers_a_typed_401_with_a_connection_code(monkeypatch):
    import httpx

    from app.server import app

    with _upstream(_streamable_handler(_meigen_refusal)) as url:
        monkeypatch.setattr(strategy_module, "MCP_REMOTE_SERVER", url)
        # In the test's own event loop: the sync TestClient cannot host the
        # remote transports' cancel scopes.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            response = await client.post(
                "/tools/generate_image", json={"prompt": "a cat"}
            )

    assert response.status_code == 401
    assert response.headers["x-inxm-connection-code"] == "upstream_auth_required"
    detail = response.json()["detail"]
    assert detail["error"] == "upstream_auth_required"
    assert detail["upstream_status"] == 401
    assert "Connect your MeiGen account with OAuth" in detail["message"]
    assert "www-authenticate" not in {name.lower() for name in response.headers}


def test_challenge_parsing_is_bounded_and_handles_quotes():
    params = parse_www_authenticate(
        'Bearer realm="x", error=invalid_token, error_description="say \\"hi\\""'
    )
    assert params == {
        "realm": "x",
        "error": "invalid_token",
        "error_description": 'say "hi"',
    }
    assert parse_www_authenticate("Bearer " + "a=b, " * 2000) == {}
    assert parse_www_authenticate(None) == {}


def test_sanitized_text_is_short_single_line_and_has_no_credentials():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop"
    text = sanitize_text(f"bad\ntoken {jwt} or Bearer abcdef123456789 " + "x" * 500)
    assert "\n" not in text
    assert jwt not in text
    assert "abcdef123456789" not in text
    assert len(text) <= 200
    assert sanitize_text(None) is None
    assert sanitize_text("   ") is None
