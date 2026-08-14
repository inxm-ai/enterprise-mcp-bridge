"""Bridge-level caller authorization (BRIDGE_REQUIRED_GROUPS).

The bridge itself — not only its ingress — must enforce the operator group
boundary, and it must do so on VERIFIED claims: the bridge accepts direct
Bearer tokens, so a forged but syntactically valid JWT claiming
`groups: ["operators"]` must never pass. With the gate configured, a
caller without identity gets 401, an unverifiable token gets 401, an
authenticated caller outside the required groups gets 403 before any
downstream contact, and members pass. Default (unset) keeps today's
behaviour.
"""

from contextlib import asynccontextmanager

import jwt
import pytest
from fastapi.testclient import TestClient

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


def _token(groups):
    return jwt.encode(
        {"sub": "user-42", "groups": groups, "azp": "web-client"},
        "test-signing-key",
        algorithm="HS256",
    )


IN_GROUP_TOKEN = _token(["operators", "employees"])
OUT_OF_GROUP_TOKEN = _token(["marketing"])
# Same group claim as a member, but nothing this bridge can verify — the gate
# must reject it on verification, never trust the embedded groups.
FORGED_TOKEN = jwt.encode(
    {"sub": "attacker", "groups": ["operators"], "azp": "web-client"},
    "attacker-key",
    algorithm="HS256",
)


@pytest.fixture
def fake_verifier(monkeypatch):
    """Stand-in for JWKS verification: trusts the two well-known test tokens,
    rejects everything else — mirroring what real verification guarantees."""

    def _verify(token):
        if token in (IN_GROUP_TOKEN, OUT_OF_GROUP_TOKEN):
            return jwt.decode(token, options={"verify_signature": False})
        raise UserLoggedOutException("Access token failed verification")

    monkeypatch.setattr(token_exchange, "verified_caller_claims", _verify)
    return _verify


class FakeContent:
    def __init__(self, text):
        self.text = text
        self.type = "text"


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

    async def call_tool(self, name, arguments=None, *, meta=None):
        self.calls += 1

        class Result:
            isError = False
            content = [FakeContent("ok")]
            structuredContent = None

        return Result()


@pytest.fixture
def client():
    return TestClient(fastapi_app)


@pytest.fixture
def required_groups(monkeypatch):
    monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", ["operators"])


@pytest.fixture
def fake_downstream(monkeypatch):
    fake = FakeSession()

    @asynccontextmanager
    async def fake_mcp_session(**_kwargs):
        yield fake

    monkeypatch.setattr(session_context, "mcp_session", fake_mcp_session)
    return fake


class TestGateUnit:
    def test_disabled_gate_is_noop(self, monkeypatch):
        monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", [])
        ensure_caller_in_required_groups(None)
        ensure_caller_in_required_groups(OUT_OF_GROUP_TOKEN)

    def test_missing_token_is_401(self, required_groups, fake_verifier):
        with pytest.raises(CallerNotAuthorizedError) as excinfo:
            ensure_caller_in_required_groups(None)
        assert excinfo.value.status_code == 401

    def test_forged_token_with_matching_groups_is_401(
        self, required_groups, monkeypatch
    ):
        """The REAL verifier runs: a signed-format token this realm never
        issued must fail verification before its group claims are read."""
        monkeypatch.setattr(token_exchange, "_jwks_client", None)
        with pytest.raises(CallerNotAuthorizedError) as excinfo:
            ensure_caller_in_required_groups(FORGED_TOKEN)
        assert excinfo.value.status_code == 401

    def test_group_gate_does_not_require_client_allowlist(
        self, monkeypatch, fake_verifier
    ):
        monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", ["operators"])
        ensure_caller_in_required_groups(IN_GROUP_TOKEN)

    def test_out_of_group_is_403(self, required_groups, fake_verifier):
        with pytest.raises(CallerNotAuthorizedError) as excinfo:
            ensure_caller_in_required_groups(OUT_OF_GROUP_TOKEN)
        assert excinfo.value.status_code == 403

    def test_member_passes(self, required_groups, fake_verifier):
        ensure_caller_in_required_groups(IN_GROUP_TOKEN)

    def test_realm_role_membership_counts(self, required_groups, monkeypatch):
        claims = {
            "sub": "u",
            "realm_access": {"roles": ["operators"]},
            "azp": "web-client",
        }
        monkeypatch.setattr(
            token_exchange, "verified_caller_claims", lambda *a, **k: claims
        )
        ensure_caller_in_required_groups("any-token")


class TestVerifiedCallerClaims:
    """The shared verifier itself: signature and issuer verification."""

    def test_returns_verified_claims(self, monkeypatch):
        from app.oauth.token_exchange import verified_caller_claims

        class OkJwks:
            def get_signing_key_from_jwt(self, token):
                class K:
                    key = "irrelevant"

                return K()

        monkeypatch.setattr(token_exchange, "_get_jwks_client", lambda: OkJwks())
        monkeypatch.setattr(
            token_exchange.jwt,
            "decode",
            lambda *a, **k: {
                "iss": f"{token_exchange.AUTH_BASE_URL}/realms/{token_exchange.KEYCLOAK_REALM}",
                "azp": "bridge-client",
            },
        )
        claims = verified_caller_claims("token")
        assert claims["azp"] == "bridge-client"


class TestRestGate:
    def test_out_of_group_caller_is_denied_before_downstream(
        self, client, required_groups, fake_verifier, fake_downstream
    ):
        response = client.post(
            "/tools/test_tool",
            headers={TOKEN_HEADER: OUT_OF_GROUP_TOKEN},
            json={},
        )
        assert response.status_code == 403
        assert fake_downstream.calls == 0

    def test_forged_token_is_denied_before_downstream(
        self, client, required_groups, fake_verifier, fake_downstream
    ):
        response = client.post(
            "/tools/test_tool",
            headers={TOKEN_HEADER: FORGED_TOKEN},
            json={},
        )
        assert response.status_code == 401
        assert fake_downstream.calls == 0

    def test_anonymous_caller_is_denied(
        self, client, required_groups, fake_verifier, fake_downstream
    ):
        response = client.post("/tools/test_tool", json={})
        assert response.status_code == 401
        assert fake_downstream.calls == 0

    def test_member_passes_through(
        self, client, required_groups, fake_verifier, fake_downstream
    ):
        response = client.post(
            "/tools/test_tool",
            headers={TOKEN_HEADER: IN_GROUP_TOKEN},
            json={},
        )
        assert response.status_code == 200
        assert fake_downstream.calls > 0

    def test_listing_is_gated_too(
        self, client, required_groups, fake_verifier, fake_downstream
    ):
        response = client.get("/tools", headers={TOKEN_HEADER: OUT_OF_GROUP_TOKEN})
        assert response.status_code == 403
        assert fake_downstream.calls == 0

    def test_session_start_is_gated(self, client, required_groups, fake_verifier):
        response = client.post(
            "/session/start", headers={TOKEN_HEADER: OUT_OF_GROUP_TOKEN}
        )
        assert response.status_code == 403

    def test_default_unset_keeps_current_behaviour(
        self, client, monkeypatch, fake_downstream
    ):
        monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", [])
        response = client.post(
            "/tools/test_tool",
            headers={TOKEN_HEADER: OUT_OF_GROUP_TOKEN},
            json={},
        )
        assert response.status_code == 200


class TestSseGate:
    async def _connect(self, token):
        from app.sse.mcp_proxy import _SSEConnectionApp

        sent = []

        async def receive():
            return {"type": "http.request"}

        async def send(message):
            sent.append(message)

        headers = []
        if token is not None:
            headers.append((TOKEN_HEADER.lower().encode(), token.encode()))
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/sse",
            "query_string": b"",
            "headers": headers,
        }
        await _SSEConnectionApp()(scope, receive, send)
        return next(m for m in sent if m["type"] == "http.response.start")

    @pytest.mark.asyncio
    async def test_sse_connection_rejected_for_out_of_group(
        self, required_groups, fake_verifier
    ):
        start = await self._connect(OUT_OF_GROUP_TOKEN)
        assert start["status"] == 403

    @pytest.mark.asyncio
    async def test_sse_connection_rejected_for_forged_token(
        self, required_groups, fake_verifier
    ):
        start = await self._connect(FORGED_TOKEN)
        assert start["status"] == 401

    @pytest.mark.asyncio
    async def test_sse_connection_rejected_anonymous(
        self, required_groups, fake_verifier
    ):
        start = await self._connect(None)
        assert start["status"] == 401
