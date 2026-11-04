"""Bridge-level caller allow-list (BRIDGE_ALLOWED_USERS).

The allow-list shares the BRIDGE_REQUIRED_GROUPS seam: it reads the VERIFIED
token's ``sub``, may be set per server in ``MCP_SERVERS`` settings, and a
caller outside a non-empty list gets the same 403 as a caller outside the
required groups. Empty keeps today's behaviour. With groups also set, both
must pass.
"""

import json
from contextlib import asynccontextmanager

import jwt
import pytest
from fastapi.testclient import TestClient

from app import multi_server
from app import vars as app_vars
from app.oauth import token_exchange
from app.oauth.token_exchange import UserLoggedOutException
from app.oauth.user_info import (
    CallerNotAuthorizedError,
    ensure_caller_in_required_groups,
)
from app.server import app as fastapi_app
from app.session_manager import session_context

TOKEN_HEADER = "X-Auth-Request-Access-Token"
OWNER_SUB = "0b6f3c1e-7d1a-4c5e-9a43-2f1b8e6d4c21"


def _token(sub, groups):
    return jwt.encode(
        {"sub": sub, "groups": groups, "azp": "web-client"},
        "test-signing-key",
        algorithm="HS256",
    )


OWNER_TOKEN = _token(OWNER_SUB, ["employees"])
OWNER_OPERATOR_TOKEN = _token(OWNER_SUB, ["operators"])
OTHER_TOKEN = _token("someone-else", ["employees"])
OTHER_OPERATOR_TOKEN = _token("someone-else", ["operators"])
FORGED_OWNER_TOKEN = jwt.encode(
    {"sub": OWNER_SUB, "groups": [], "azp": "web-client"},
    "attacker-key",
    algorithm="HS256",
)
TRUSTED = (OWNER_TOKEN, OWNER_OPERATOR_TOKEN, OTHER_TOKEN, OTHER_OPERATOR_TOKEN)


@pytest.fixture(autouse=True)
def no_global_gates(monkeypatch):
    monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", [])
    monkeypatch.setattr(app_vars, "BRIDGE_ALLOWED_USERS", [])


@pytest.fixture
def fake_verifier(monkeypatch):
    """Trusts the well-known test tokens, rejects everything else."""

    def _verify(token):
        if token in TRUSTED:
            return jwt.decode(token, options={"verify_signature": False})
        raise UserLoggedOutException("Access token failed verification")

    monkeypatch.setattr(token_exchange, "verified_caller_claims", _verify)


@pytest.fixture
def owner_only(monkeypatch):
    monkeypatch.setattr(app_vars, "BRIDGE_ALLOWED_USERS", [OWNER_SUB])


def _status(token):
    with pytest.raises(CallerNotAuthorizedError) as excinfo:
        ensure_caller_in_required_groups(token)
    return excinfo.value.status_code


def _bound(settings):
    (server,) = multi_server.parse_servers(
        json.dumps(
            [
                {
                    "id": "personal",
                    "base_path": "/api/personal",
                    "url": "https://personal.example/mcp",
                    "settings": settings,
                }
            ]
        )
    )
    return multi_server.bind_server(server)


class TestAllowListUnit:
    def test_empty_allow_list_keeps_behaviour(self, fake_verifier):
        ensure_caller_in_required_groups(None)
        ensure_caller_in_required_groups(OTHER_TOKEN)

    def test_listed_sub_passes(self, owner_only, fake_verifier):
        ensure_caller_in_required_groups(OWNER_TOKEN)

    def test_unlisted_sub_is_403(self, owner_only, fake_verifier):
        assert _status(OTHER_TOKEN) == 403

    def test_missing_token_is_401(self, owner_only, fake_verifier):
        assert _status(None) == 401

    def test_unverifiable_token_with_listed_sub_is_401(self, owner_only, fake_verifier):
        assert _status(FORGED_OWNER_TOKEN) == 401

    def test_comma_separated_list(self, monkeypatch, fake_verifier):
        monkeypatch.setattr(app_vars, "BRIDGE_ALLOWED_USERS", ["a", OWNER_SUB])
        ensure_caller_in_required_groups(OWNER_TOKEN)
        assert _status(OTHER_TOKEN) == 403


class TestPerServerOverride:
    def test_server_list_overrides_global(self, monkeypatch, fake_verifier):
        monkeypatch.setattr(app_vars, "BRIDGE_ALLOWED_USERS", ["someone-else"])
        token = _bound({"BRIDGE_ALLOWED_USERS": OWNER_SUB})
        try:
            ensure_caller_in_required_groups(OWNER_TOKEN)
            assert _status(OTHER_TOKEN) == 403
        finally:
            multi_server.reset_server(token)
        ensure_caller_in_required_groups(OTHER_TOKEN)
        assert _status(OWNER_TOKEN) == 403

    def test_server_list_restricts_unrestricted_global(self, fake_verifier):
        token = _bound({"BRIDGE_ALLOWED_USERS": f" {OWNER_SUB} , other "})
        try:
            ensure_caller_in_required_groups(OWNER_TOKEN)
            assert _status(OTHER_TOKEN) == 403
        finally:
            multi_server.reset_server(token)
        ensure_caller_in_required_groups(OTHER_TOKEN)

    def test_server_without_setting_uses_global(self, owner_only, fake_verifier):
        token = _bound({})
        try:
            assert _status(OTHER_TOKEN) == 403
        finally:
            multi_server.reset_server(token)

    def test_empty_server_value_lifts_global(self, owner_only, fake_verifier):
        token = _bound({"BRIDGE_ALLOWED_USERS": ""})
        try:
            ensure_caller_in_required_groups(OTHER_TOKEN)
        finally:
            multi_server.reset_server(token)


class TestGroupsAndAllowList:
    @pytest.fixture
    def both(self, monkeypatch, owner_only):
        monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", ["operators"])

    def test_listed_member_passes(self, both, fake_verifier):
        ensure_caller_in_required_groups(OWNER_OPERATOR_TOKEN)

    def test_listed_non_member_is_403(self, both, fake_verifier):
        assert _status(OWNER_TOKEN) == 403

    def test_unlisted_member_is_403(self, both, fake_verifier):
        assert _status(OTHER_OPERATOR_TOKEN) == 403


class FakeSession:
    def __init__(self):
        self.calls = 0

    async def list_tools(self):
        self.calls += 1

        class Tool:
            name = "test_tool"
            inputSchema = {"properties": {}}

        class Container:
            tools = [Tool()]

        return Container()


@pytest.fixture
def fake_downstream(monkeypatch):
    fake = FakeSession()

    @asynccontextmanager
    async def fake_mcp_session(**_kwargs):
        yield fake

    monkeypatch.setattr(session_context, "mcp_session", fake_mcp_session)
    return fake


class TestTransports:
    def test_rest_listing_denied_before_downstream(
        self, owner_only, fake_verifier, fake_downstream
    ):
        client = TestClient(fastapi_app)
        response = client.get("/tools", headers={TOKEN_HEADER: OTHER_TOKEN})
        assert response.status_code == 403
        assert fake_downstream.calls == 0
        response = client.get("/tools", headers={TOKEN_HEADER: OWNER_TOKEN})
        assert response.status_code == 200

    def test_session_start_denied(self, owner_only, fake_verifier):
        client = TestClient(fastapi_app)
        response = client.post("/session/start", headers={TOKEN_HEADER: OTHER_TOKEN})
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_sse_connection_denied(self, owner_only, fake_verifier):
        from app.sse.mcp_proxy import _SSEConnectionApp

        sent = []

        async def receive():
            return {"type": "http.request"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/sse",
            "query_string": b"",
            "headers": [(TOKEN_HEADER.lower().encode(), OTHER_TOKEN.encode())],
        }
        await _SSEConnectionApp()(scope, receive, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert start["status"] == 403
