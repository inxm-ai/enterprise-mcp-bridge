"""Canonical black-box conformance suite for the bridge itself.

Uses only the public test-support API (``enterprise_mcp_bridge.testing``)
against the repo's demo MCP server (``mcp/server.py``), spawned over the real
stdio child boundary. This proves the generic bridge behaviours once, in the
bridge repo; MCP repositories only add their own tool-specific cases on top.

Runs offline with plain pytest — no Kubernetes, no network.
"""

import sys
from pathlib import Path

import pytest

from enterprise_mcp_bridge.testing import bridge_client

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SERVER_COMMAND = f"{sys.executable} {REPO_ROOT / 'mcp' / 'server.py'}"
SESSION_HEADER = "x-inxm-mcp-session"
# call_counter both proves session isolation (stateful child) and, marked as an
# effect tool, proves dry-run never reaches the real tool.
EFFECT_TOOLS_CONFIG = "call_counter"

HTTP_OK = 200
HTTP_INTERNAL_ERROR = 500
# The externally intended contract: a tool that ran and rejected the request is
# a client-side failure, never a bridge fault. Asserted as a literal on purpose
# — importing the bridge's own constant would keep this test green if the
# constant and the behavior drifted together.
HTTP_TOOL_EXECUTION_ERROR = 422


@pytest.fixture(scope="module")
def bridge():
    with bridge_client(
        DEMO_SERVER_COMMAND, env={"EFFECT_TOOLS": EFFECT_TOOLS_CONFIG}
    ) as client:
        yield client
        client.cookies.clear()


@pytest.fixture(autouse=True)
def _sessionless_by_default(bridge):
    # /session/start sets a session cookie on the shared client; clear it so
    # every test starts sessionless unless it opts in explicitly.
    bridge.cookies.clear()


def _require_effect_tools_gate():
    # EFFECT_TOOLS is read when the app module is first imported. If another
    # test file imported it earlier without the gate configured (full-suite
    # runs from the repo root), the dry-run cases cannot be proven here.
    from app.vars import EFFECT_TOOLS

    if EFFECT_TOOLS_CONFIG not in EFFECT_TOOLS:
        pytest.skip(
            "app was imported before EFFECT_TOOLS could be set; "
            "run `pytest tests/` in its own process for the dry-run cases"
        )


def _start_session(bridge) -> str:
    response = bridge.post("/session/start")
    assert response.status_code == HTTP_OK, response.text
    return response.json()[SESSION_HEADER]


def _close_session(bridge, session_id: str) -> None:
    # Sessions hold a live child process until closed; cookie clearing alone
    # only detaches the client, it does not end the session.
    response = bridge.post("/session/close", headers={SESSION_HEADER: session_id})
    assert response.status_code == HTTP_OK, response.text


def _call_counter(bridge, session_id: str, headers: dict | None = None) -> object:
    response = bridge.post(
        "/tools/call_counter",
        headers={SESSION_HEADER: session_id, **(headers or {})},
        json={},
    )
    assert response.status_code == HTTP_OK, response.text
    return response.json()["structuredContent"]["result"]


def test_startup_lists_the_child_tools(bridge):
    response = bridge.get("/tools")
    assert response.status_code == HTTP_OK, response.text
    tool_names = [tool["name"] for tool in response.json()]
    assert {"add", "hello", "error", "call_counter"} <= set(tool_names)


def test_tool_invocation_returns_structured_result(bridge):
    response = bridge.post("/tools/add", json={"a": 2, "b": 3})
    assert response.status_code == HTTP_OK, response.text
    assert response.json()["structuredContent"]["result"] == 5


def test_sessions_are_isolated_from_each_other(bridge):
    sessions = []
    try:
        first_session = _start_session(bridge)
        sessions.append(first_session)
        bridge.cookies.clear()
        second_session = _start_session(bridge)
        sessions.append(second_session)
        bridge.cookies.clear()

        assert first_session != second_session
        assert _call_counter(bridge, first_session) == 1
        assert _call_counter(bridge, first_session) == 2
        # The second session has its own child process, so its counter is fresh.
        assert _call_counter(bridge, second_session) == 1
    finally:
        for session_id in sessions:
            _close_session(bridge, session_id)


def test_child_tool_exception_maps_to_tool_execution_error(bridge):
    response = bridge.post("/tools/error", json={"message": "boom"})
    assert response.status_code == HTTP_TOOL_EXECUTION_ERROR, response.text


def test_dry_run_header_is_inert_for_non_effect_tools(bridge):
    _require_effect_tools_gate()
    response = bridge.post(
        "/tools/add", headers={"X-Inxm-Dry-Run": "true"}, json={"a": 2, "b": 3}
    )
    assert response.status_code == HTTP_OK, response.text
    assert response.json()["structuredContent"]["result"] == 5


def test_dry_run_never_executes_the_effect_tool(bridge):
    _require_effect_tools_gate()
    session_id = _start_session(bridge)
    try:
        assert _call_counter(bridge, session_id) == 1

        # Offline there is no TGI_URL, so the dry-run path fails server-side —
        # what matters here is that the real tool is never reached.
        dry_run = bridge.post(
            "/tools/call_counter",
            headers={SESSION_HEADER: session_id, "X-Inxm-Dry-Run": "true"},
            json={},
        )
        assert dry_run.status_code == HTTP_INTERNAL_ERROR, dry_run.text

        assert _call_counter(bridge, session_id) == 2
    finally:
        _close_session(bridge, session_id)
