"""Central connection policy (app-mcp-manager) enforced and attested by the bridge.

With CONNECTION_POLICY_URL set, every request asks the authority as the
verified caller before any downstream contact: a denial is 403, a hidden
server 404, an outage 503 (never "allowed"). Allowed responses carry the
attestation headers the manager's probe requires; a downstream server can
never supply them. Unset keeps today's behaviour with no attestation.
"""

from contextlib import asynccontextmanager

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from app import vars as app_vars
from app.oauth import connection_policy, token_exchange
from app.oauth.token_exchange import UserLoggedOutException
from app.oauth.user_info import CallerNotAuthorizedError
from app.server import app as fastapi_app
from app.session_manager import session_context

TOKEN_HEADER = "X-Auth-Request-Access-Token"
REVISION = "a" * 64
TOKEN = jwt.encode(
    {"sub": "ben", "azp": "web-client"}, "test-signing-key", algorithm="HS256"
)


@pytest.fixture
def verifier(monkeypatch):
    def _verify(token):
        if token == TOKEN:
            return jwt.decode(token, options={"verify_signature": False})
        raise UserLoggedOutException("Access token failed verification")

    monkeypatch.setattr(token_exchange, "verified_caller_claims", _verify)


@pytest.fixture
def authority(monkeypatch):
    """Configured authority whose next answer the test controls."""
    state = {"status": 200, "body": None, "error": None, "calls": []}

    async def _decide(url, access_token, server):
        state["calls"].append((url, access_token, server))
        if state["error"]:
            raise state["error"]
        body = state["body"] or {
            "subject": "ben",
            "server": server,
            "allowed": True,
            "policy_version": 7,
            "connector_revision": REVISION,
        }
        return httpx.Response(state["status"], json=body)

    monkeypatch.setenv(
        "CONNECTION_POLICY_URL", "http://manager.test/v2/connections/enforce"
    )
    monkeypatch.setattr(app_vars, "SERVICE_NAME", "mcp-linear-server")
    monkeypatch.setattr(app_vars, "KEYCLOAK_PROVIDER_ALIAS", "linear")
    monkeypatch.setattr(connection_policy, "_decide", _decide)
    monkeypatch.setattr(connection_policy, "_cache", {})
    return state


class FakeSession:
    def __init__(self):
        self.calls = 0

    async def list_tools(self):
        self.calls += 1

        class Tool:
            name = "current_user"
            inputSchema = {"properties": {}}

        class Container:
            tools = [Tool()]

        return Container()

    async def call_tool(self, name, arguments=None, *, meta=None):
        self.calls += 1

        class Content:
            text = "ok"
            type = "text"

        class Result:
            isError = False
            content = [Content()]
            structuredContent = None

        return Result()


@pytest.fixture
def downstream(monkeypatch):
    fake = FakeSession()

    @asynccontextmanager
    async def fake_mcp_session(**_kwargs):
        yield fake

    monkeypatch.setattr(session_context, "mcp_session", fake_mcp_session)
    return fake


@pytest.fixture
def client():
    return TestClient(fastapi_app)


class TestDecision:
    @pytest.mark.asyncio
    async def test_unconfigured_authority_is_a_noop(self, monkeypatch):
        monkeypatch.delenv("CONNECTION_POLICY_URL", raising=False)
        assert await connection_policy.enforce_connection_policy(None) is None

    @pytest.mark.asyncio
    async def test_allowed_decision_attests_the_callers_own_facts(
        self, authority, verifier
    ):
        attestation = await connection_policy.enforce_connection_policy(TOKEN)
        assert attestation == connection_policy.Attestation(7, REVISION, "ben", "oauth")
        assert authority["calls"] == [
            ("http://manager.test/v2/connections/enforce", TOKEN, "mcp-linear-server")
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,body,error,expected",
        [
            (409, {"code": "blocked_by_policy"}, None, 403),
            (404, {"code": "not_found"}, None, 404),
            (401, {}, None, 401),
            (503, {}, None, 503),
            (500, {}, None, 503),
            (200, None, httpx.ConnectError("down"), 503),
            # A decision for somebody else, another server or without a
            # usable version never counts as allowed.
            (
                200,
                {
                    "subject": "anna",
                    "server": "mcp-linear-server",
                    "allowed": True,
                    "policy_version": 7,
                    "connector_revision": REVISION,
                },
                None,
                503,
            ),
            (
                200,
                {
                    "subject": "ben",
                    "server": "mcp-other",
                    "allowed": True,
                    "policy_version": 7,
                    "connector_revision": REVISION,
                },
                None,
                503,
            ),
            (
                200,
                {
                    "subject": "ben",
                    "server": "mcp-linear-server",
                    "allowed": False,
                    "policy_version": 7,
                    "connector_revision": REVISION,
                },
                None,
                503,
            ),
            (
                200,
                {
                    "subject": "ben",
                    "server": "mcp-linear-server",
                    "allowed": True,
                    "policy_version": "7",
                    "connector_revision": REVISION,
                },
                None,
                503,
            ),
            (
                200,
                {
                    "subject": "ben",
                    "server": "mcp-linear-server",
                    "allowed": True,
                    "policy_version": 7,
                    "connector_revision": "a\r\nx: y",
                },
                None,
                503,
            ),
        ],
    )
    async def test_every_other_answer_fails_closed(
        self, authority, verifier, status, body, error, expected
    ):
        authority.update(status=status, body=body, error=error)
        with pytest.raises(CallerNotAuthorizedError) as excinfo:
            await connection_policy.enforce_connection_policy(TOKEN)
        assert excinfo.value.status_code == expected

    @pytest.mark.asyncio
    async def test_unverifiable_or_missing_token_never_reaches_the_authority(
        self, authority, verifier
    ):
        for token in (None, "forged"):
            with pytest.raises(CallerNotAuthorizedError) as excinfo:
                await connection_policy.enforce_connection_policy(token)
            assert excinfo.value.status_code == 401
        assert authority["calls"] == []

    def test_credential_kind_reports_what_this_bridge_sends(self, monkeypatch):
        monkeypatch.setattr(app_vars, "KEYCLOAK_PROVIDER_ALIAS", "")
        monkeypatch.setattr(app_vars, "AUTH_PROVIDER", "keycloak")
        monkeypatch.delenv("MCP_REMOTE_ANON_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("MCP_REMOTE_BEARER_TOKEN", raising=False)
        assert connection_policy.credential_kind() == "public"
        monkeypatch.setenv("MCP_REMOTE_BEARER_TOKEN", "static-key")
        assert connection_policy.credential_kind() == "service_key"
        monkeypatch.delenv("MCP_REMOTE_BEARER_TOKEN")
        monkeypatch.setenv("MCP_REMOTE_ANON_BEARER_TOKEN", "shared")
        assert connection_policy.credential_kind() == "service_key"
        monkeypatch.setattr(app_vars, "AUTH_PROVIDER", "user-api-key")
        assert connection_policy.credential_kind() == "api_key"
        monkeypatch.setattr(app_vars, "KEYCLOAK_PROVIDER_ALIAS", "linear")
        assert connection_policy.credential_kind() == "oauth"


class TestCache:
    @pytest.mark.asyncio
    async def test_allowed_decisions_are_reused_for_thirty_seconds_only(
        self, authority, verifier, monkeypatch
    ):
        now = [1000.0]
        monkeypatch.setattr(connection_policy.time, "monotonic", lambda: now[0])
        await connection_policy.enforce_connection_policy(TOKEN)
        await connection_policy.enforce_connection_policy(TOKEN)
        assert len(authority["calls"]) == 1
        now[0] += 30.0
        authority.update(status=409, body={"code": "blocked_by_policy"})
        with pytest.raises(CallerNotAuthorizedError) as excinfo:
            await connection_policy.enforce_connection_policy(TOKEN)
        assert excinfo.value.status_code == 403
        assert len(authority["calls"]) == 2

    @pytest.mark.asyncio
    async def test_denials_and_outages_are_never_cached(self, authority, verifier):
        authority.update(status=409, body={"code": "blocked_by_policy"})
        for _ in range(2):
            with pytest.raises(CallerNotAuthorizedError):
                await connection_policy.enforce_connection_policy(TOKEN)
        authority.update(status=200, body=None, error=httpx.ConnectError("down"))
        with pytest.raises(CallerNotAuthorizedError):
            await connection_policy.enforce_connection_policy(TOKEN)
        assert len(authority["calls"]) == 3

    @pytest.mark.asyncio
    async def test_cache_still_verifies_the_token_every_time(
        self, authority, verifier, monkeypatch
    ):
        await connection_policy.enforce_connection_policy(TOKEN)
        monkeypatch.setattr(
            token_exchange,
            "verified_caller_claims",
            lambda token: (_ for _ in ()).throw(UserLoggedOutException("expired")),
        )
        with pytest.raises(CallerNotAuthorizedError) as excinfo:
            await connection_policy.enforce_connection_policy(TOKEN)
        assert excinfo.value.status_code == 401


class TestRest:
    @pytest.mark.parametrize(
        "method,path", [("get", "/tools"), ("post", "/tools/current_user")]
    )
    def test_allowed_listing_and_call_are_attested(
        self, client, authority, verifier, downstream, method, path
    ):
        kwargs = {"json": {}} if method == "post" else {}
        response = getattr(client, method)(
            path, headers={TOKEN_HEADER: TOKEN}, **kwargs
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-inxm-connection-policy-version"] == "7"
        assert response.headers["x-inxm-connector-revision"] == REVISION
        assert response.headers["x-inxm-caller-subject"] == "ben"
        assert response.headers["x-inxm-credential-kind"] == "oauth"
        assert downstream.calls >= 1

    def test_denial_is_final_before_any_downstream_contact(
        self, client, authority, verifier, downstream
    ):
        authority.update(status=409, body={"code": "blocked_by_policy"})
        response = client.post(
            "/tools/current_user", headers={TOKEN_HEADER: TOKEN}, json={}
        )
        assert response.status_code == 403
        assert downstream.calls == 0
        assert "x-inxm-connection-policy-version" not in response.headers

    def test_unconfigured_bridge_sends_no_attestation(
        self, client, monkeypatch, verifier, downstream
    ):
        monkeypatch.delenv("CONNECTION_POLICY_URL", raising=False)
        response = client.get("/tools", headers={TOKEN_HEADER: TOKEN})
        assert response.status_code == 200
        assert not any(h.startswith("x-inxm-") for h in response.headers)


class TestAnonymousListing:
    """The manager's tools crawl lists tools with a placeholder bearer."""

    CRAWL = {"Authorization": "Bearer app-mcp-manager-tools-crawl"}

    def test_an_unverifiable_listing_is_served_unattested(
        self, client, authority, verifier, downstream
    ):
        for headers in (self.CRAWL, {}):
            response = client.get("/tools", headers=headers)
            assert response.status_code == 200, response.text
            assert "x-inxm-connection-policy-version" not in response.headers
        assert authority["calls"] == []

    def test_an_unverifiable_tool_call_is_still_refused(
        self, client, authority, verifier, downstream
    ):
        response = client.post("/tools/current_user", headers=self.CRAWL, json={})
        assert response.status_code == 401
        assert downstream.calls == 0

    def test_a_verified_listing_is_still_checked_and_attested(
        self, client, authority, verifier, downstream
    ):
        authority.update(status=409, body={"code": "blocked_by_policy"})
        response = client.get("/tools", headers={TOKEN_HEADER: TOKEN})
        assert response.status_code == 403


class TestServerId:
    @pytest.mark.asyncio
    async def test_a_single_server_bridge_uses_its_default_service_name(
        self, authority, verifier, monkeypatch
    ):
        monkeypatch.delenv("SERVICE_NAME", raising=False)
        monkeypatch.setattr(app_vars, "SERVICE_NAME", "enterprise-mcp-bridge")
        await connection_policy.enforce_connection_policy(TOKEN)
        assert authority["calls"][0][2] == "enterprise-mcp-bridge"


class TestStreaming:
    """A StreamingResponse starts before its generator runs: the policy is
    decided first, so denials keep their status and allowed streams are
    attested."""

    def test_a_denied_stream_is_refused_with_its_status(
        self, client, authority, verifier, downstream
    ):
        authority.update(status=409, body={"code": "blocked_by_policy"})
        response = client.post(
            "/tools/current_user/stream", headers={TOKEN_HEADER: TOKEN}, json={}
        )
        assert response.status_code == 403
        assert downstream.calls == 0

    def test_an_allowed_stream_carries_the_attestation(
        self, client, authority, verifier, downstream
    ):
        response = client.post(
            "/tools/current_user/stream", headers={TOKEN_HEADER: TOKEN}, json={}
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-inxm-caller-subject"] == "ben"
        assert response.headers["x-inxm-connector-revision"] == REVISION


class TestSseMessages:
    """POST /sse/messages has no credential of its own; each message re-checks
    the caller the stream was opened for."""

    @pytest.mark.asyncio
    async def test_a_message_after_a_policy_change_never_reaches_downstream(
        self, authority, verifier
    ):
        from app.sse.mcp_proxy import _build_proxy_handlers
        from app.utils import mcp_fields
        from mcp import types

        fake = FakeSession()
        handlers = _build_proxy_handlers(fake, TOKEN, {}, None)
        allowed = await handlers["on_call_tool"](
            None, types.CallToolRequestParams(name="current_user", arguments={})
        )
        assert not mcp_fields.is_error(allowed)
        reached = fake.calls

        connection_policy._cache.clear()  # the 30 s decision bound passed
        authority.update(status=409, body={"code": "blocked_by_policy"})
        denied = await handlers["on_call_tool"](
            None, types.CallToolRequestParams(name="current_user", arguments={})
        )
        assert mcp_fields.is_error(denied)
        assert fake.calls == reached
        with pytest.raises(CallerNotAuthorizedError):
            await handlers["on_list_tools"](None, None)
