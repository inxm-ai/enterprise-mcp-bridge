from app.session_manager import session_context
from types import SimpleNamespace


def _make_tools_list():
    # tool as simple namespace with inputSchema
    t1 = SimpleNamespace(
        name="toolA",
        inputSchema={
            "type": "object",
            "properties": {"userId": {"type": "string"}, "keep": {"type": "string"}},
        },
    )
    t2 = SimpleNamespace(
        name="toolB",
        inputSchema={"type": "object", "properties": {"email": {"type": "string"}}},
    )
    return SimpleNamespace(tools=[t1, t2])


def test_inject_headers_fills_missing_args(monkeypatch):
    monkeypatch.setattr(
        session_context,
        "MCP_MAP_HEADER_TO_INPUT",
        {"userId": "x-auth-user-id", "email": "x-auth-user-email"},
    )
    tools = _make_tools_list()
    incoming_headers = {
        "X-Auth-User-Id": "user-123",
        "x-auth-user-email": "u@example.com",
    }
    args = {"keep": "value"}
    out = session_context.inject_headers_into_args(
        tools, "toolA", args, incoming_headers
    )
    assert out.get("userId") == "user-123"
    assert out.get("keep") == "value"


def test_inject_headers_replaces_caller_supplied_mapped_input(monkeypatch):
    monkeypatch.setattr(
        session_context, "MCP_MAP_HEADER_TO_INPUT", {"userId": "x-auth-user-id"}
    )
    tools = _make_tools_list()
    incoming_headers = {"x-auth-user-id": "authenticated-user"}
    args = {"userId": "someone-else", "keep": "value"}
    out = session_context.inject_headers_into_args(
        tools, "toolA", args, incoming_headers
    )
    assert out == {"userId": "authenticated-user", "keep": "value"}


def test_inject_headers_drops_caller_supplied_mapped_input_without_header(
    monkeypatch,
):
    # e.g. a user without groups: the groups header is absent, and the caller
    # must not be able to supply groups instead.
    monkeypatch.setattr(
        session_context, "MCP_MAP_HEADER_TO_INPUT", {"userId": "x-auth-user-id"}
    )
    tools = _make_tools_list()
    args = {"userId": "someone-else", "keep": "value"}
    out = session_context.inject_headers_into_args(
        tools, "toolA", args, {"x-other": "1"}
    )
    assert out == {"keep": "value"}


def test_inject_headers_drops_caller_supplied_mapped_input_with_no_headers(
    monkeypatch,
):
    monkeypatch.setattr(
        session_context, "MCP_MAP_HEADER_TO_INPUT", {"userId": "x-auth-user-id"}
    )
    tools = _make_tools_list()
    args = {"userId": "someone-else", "keep": "value"}
    out = session_context.inject_headers_into_args(tools, "toolA", args, None)
    assert out == {"keep": "value"}


def test_inject_headers_leaves_args_alone_for_tools_without_the_input(monkeypatch):
    monkeypatch.setattr(
        session_context, "MCP_MAP_HEADER_TO_INPUT", {"userId": "x-auth-user-id"}
    )
    tools = _make_tools_list()
    out = session_context.inject_headers_into_args(
        tools, "toolB", {"email": "e@example.com"}, {"x-auth-user-id": "u"}
    )
    assert out == {"email": "e@example.com"}


def test_inject_headers_tool_not_found_returns_args(monkeypatch):
    monkeypatch.setattr(
        session_context, "MCP_MAP_HEADER_TO_INPUT", {"userId": "x-auth-user-id"}
    )
    tools = _make_tools_list()
    incoming_headers = {"x-auth-user-id": "user-123"}
    out = session_context.inject_headers_into_args(
        tools, "unknown", {}, incoming_headers
    )
    assert out == {}


def test_inject_headers_case_insensitive_lookup(monkeypatch):
    monkeypatch.setattr(
        session_context, "MCP_MAP_HEADER_TO_INPUT", {"email": "X-Auth-User-Email"}
    )
    tools = _make_tools_list()
    incoming_headers = {"x-auth-user-email": "case@example.com"}
    out = session_context.inject_headers_into_args(
        tools, "toolB", None, incoming_headers
    )
    assert out["email"] == "case@example.com"
