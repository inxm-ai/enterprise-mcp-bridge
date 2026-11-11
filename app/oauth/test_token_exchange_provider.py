"""AUTH_PROVIDER=token-exchange: the remote gets a token meant only for it.

Before calling the remote MCP, the bridge swaps the caller's token at the
identity provider (OAuth 2.0 Token Exchange, RFC 8693) for one whose audience
is TOKEN_EXCHANGE_AUDIENCE, signing in as TOKEN_EXCHANGE_CLIENT_ID. Without a
caller (tool listings, probes) it swaps the client's own token
(client_credentials) the same way. It never falls back to the caller's token or a shared
bearer: a refused exchange is 401, an unreachable token endpoint is 503.

No test reaches the network: requests.post is replaced by a fake that records
every request and answers from a script.
"""

import json

import pytest
import requests
from fastapi import HTTPException

from app import multi_server
from app import vars as app_vars
from app.oauth import token_exchange
from app.oauth.token_exchange import (
    TokenExchangeTokenRetriever,
    TokenRetrieverFactory,
    TokenServiceUnavailableError,
    UserLoggedOutException,
)
from app.session import client_strategy

TOKEN_URL = "https://idp.example.com/realms/main/protocol/openid-connect/token"
CLIENT_ID = "bridge-exchange"
CLIENT_SECRET = "client-secret-value"
AUDIENCE = "remote-tools"
CALLER_TOKEN = "caller-jwt"
EXCHANGED_TOKEN = "exchanged-jwt"
SERVICE_TOKEN = "service-jwt"
EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
TIMEOUT_SECONDS = 5.0


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeTokenEndpoint:
    """Answers each POST with the next scripted response or exception."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def post(self, url, data=None, timeout=None, verify=True, **_kwargs):
        self.requests.append({"url": url, "data": data, "timeout": timeout})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _ok(token):
    return FakeResponse(200, {"access_token": token, "token_type": "Bearer"})


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(app_vars, "TOKEN_EXCHANGE_URL", TOKEN_URL)
    monkeypatch.setattr(app_vars, "TOKEN_EXCHANGE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setattr(app_vars, "TOKEN_EXCHANGE_CLIENT_SECRET", CLIENT_SECRET)
    monkeypatch.setattr(app_vars, "TOKEN_EXCHANGE_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(app_vars, "TOKEN_EXCHANGE_TIMEOUT_SECONDS", TIMEOUT_SECONDS)


def _endpoint(monkeypatch, *answers):
    fake = FakeTokenEndpoint(*answers)
    monkeypatch.setattr(token_exchange.requests, "post", fake.post)
    return fake


def test_factory_returns_the_exchange_retriever(monkeypatch):
    monkeypatch.setattr(token_exchange, "AUTH_PROVIDER", "token-exchange")
    assert isinstance(TokenRetrieverFactory().get(), TokenExchangeTokenRetriever)


class TestRetriever:
    def test_caller_token_is_exchanged_for_the_audience(self, configured, monkeypatch):
        fake = _endpoint(monkeypatch, _ok(EXCHANGED_TOKEN))
        result = TokenExchangeTokenRetriever().retrieve_token(CALLER_TOKEN)
        assert result["access_token"] == EXCHANGED_TOKEN
        assert result["token_type"] == "Bearer"
        [request] = fake.requests
        assert request["url"] == TOKEN_URL
        assert request["timeout"] == TIMEOUT_SECONDS
        assert request["data"] == {
            "grant_type": EXCHANGE_GRANT,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "subject_token": CALLER_TOKEN,
            "subject_token_type": ACCESS_TOKEN_TYPE,
            "requested_token_type": ACCESS_TOKEN_TYPE,
            "audience": AUDIENCE,
        }

    def test_without_a_caller_the_clients_own_token_is_narrowed(
        self, configured, monkeypatch
    ):
        # The client's own token names every audience the client serves, so
        # it is swapped for one meant only for this remote, like a caller's.
        fake = _endpoint(monkeypatch, _ok(SERVICE_TOKEN), _ok(EXCHANGED_TOKEN))
        result = TokenExchangeTokenRetriever().retrieve_token("")
        assert result["access_token"] == EXCHANGED_TOKEN
        own, swap = fake.requests
        assert own["data"] == {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }
        assert swap["data"]["grant_type"] == EXCHANGE_GRANT
        assert swap["data"]["subject_token"] == SERVICE_TOKEN
        assert swap["data"]["audience"] == AUDIENCE

    @pytest.mark.parametrize(
        "answer, error",
        [
            (
                FakeResponse(401, {"error": "unauthorized_client"}),
                UserLoggedOutException,
            ),
            (requests.ConnectionError("refused"), TokenServiceUnavailableError),
        ],
    )
    def test_failing_client_token_stops_before_the_swap(
        self, configured, monkeypatch, answer, error
    ):
        fake = _endpoint(monkeypatch, answer)
        with pytest.raises(error):
            TokenExchangeTokenRetriever().retrieve_token("")
        assert len(fake.requests) == 1

    def test_audience_can_be_set_per_server(self, configured, monkeypatch):
        tools, other = multi_server.parse_servers(
            json.dumps(
                [
                    {
                        "id": "tools",
                        "base_path": "/api/tools",
                        "url": "https://tools.example.com/mcp",
                        "auth_provider": "token-exchange",
                        "settings": {"TOKEN_EXCHANGE_AUDIENCE": "tools-audience"},
                    },
                    {
                        "id": "other",
                        "base_path": "/api/other",
                        "url": "https://other.example.com/mcp",
                        "auth_provider": "token-exchange",
                    },
                ]
            )
        )
        fake = _endpoint(monkeypatch, _ok(EXCHANGED_TOKEN), _ok(EXCHANGED_TOKEN))
        for server in (tools, other):
            bound = multi_server.bind_server(server)
            try:
                TokenExchangeTokenRetriever().retrieve_token(CALLER_TOKEN)
            finally:
                multi_server.reset_server(bound)
        assert [r["data"]["audience"] for r in fake.requests] == [
            "tools-audience",
            AUDIENCE,
        ]

    def test_default_url_is_the_realms_token_endpoint(self, configured, monkeypatch):
        monkeypatch.setattr(app_vars, "TOKEN_EXCHANGE_URL", "")
        monkeypatch.setattr(app_vars, "AUTH_BASE_URL", "https://idp.example.com/")
        monkeypatch.setattr(app_vars, "KEYCLOAK_REALM", "main")
        fake = _endpoint(monkeypatch, _ok(EXCHANGED_TOKEN))
        TokenExchangeTokenRetriever().retrieve_token(CALLER_TOKEN)
        assert fake.requests[0]["url"] == TOKEN_URL

    @pytest.mark.parametrize("status", [400, 401, 403])
    def test_refused_exchange_is_logged_out(self, configured, monkeypatch, status):
        _endpoint(monkeypatch, FakeResponse(status, {"error": "access_denied"}))
        with pytest.raises(UserLoggedOutException):
            TokenExchangeTokenRetriever().retrieve_token(CALLER_TOKEN)

    @pytest.mark.parametrize(
        "answer",
        [
            requests.ConnectionError("refused"),
            requests.Timeout("slow"),
            FakeResponse(502, {}),
            FakeResponse(503, {}),
        ],
    )
    def test_unreachable_token_endpoint_is_unavailable(
        self, configured, monkeypatch, answer
    ):
        _endpoint(monkeypatch, answer)
        with pytest.raises(TokenServiceUnavailableError):
            TokenExchangeTokenRetriever().retrieve_token(CALLER_TOKEN)

    @pytest.mark.parametrize(
        "body", [{}, {"access_token": ""}, ["not", "a", "dict"], ValueError("not json")]
    )
    def test_answer_without_a_token_is_logged_out(self, configured, monkeypatch, body):
        _endpoint(monkeypatch, FakeResponse(200, body))
        with pytest.raises(UserLoggedOutException):
            TokenExchangeTokenRetriever().retrieve_token(CALLER_TOKEN)

    @pytest.mark.parametrize(
        "setting",
        [
            "TOKEN_EXCHANGE_CLIENT_ID",
            "TOKEN_EXCHANGE_CLIENT_SECRET",
            "TOKEN_EXCHANGE_AUDIENCE",
        ],
    )
    def test_missing_setting_refuses_without_a_request(
        self, configured, monkeypatch, setting
    ):
        monkeypatch.setattr(app_vars, setting, "")
        fake = _endpoint(monkeypatch)
        with pytest.raises(UserLoggedOutException):
            TokenExchangeTokenRetriever().retrieve_token(CALLER_TOKEN)
        assert fake.requests == []

    def test_no_url_and_no_base_url_refuses_without_a_request(
        self, configured, monkeypatch
    ):
        monkeypatch.setattr(app_vars, "TOKEN_EXCHANGE_URL", "")
        monkeypatch.setattr(app_vars, "AUTH_BASE_URL", "")
        fake = _endpoint(monkeypatch)
        with pytest.raises(UserLoggedOutException):
            TokenExchangeTokenRetriever().retrieve_token(CALLER_TOKEN)
        assert fake.requests == []

    def test_client_secret_never_reaches_the_logs(
        self, configured, monkeypatch, caplog
    ):
        _endpoint(monkeypatch, FakeResponse(403, {"error": "access_denied"}))
        with pytest.raises(UserLoggedOutException):
            TokenExchangeTokenRetriever().retrieve_token(CALLER_TOKEN)
        assert CLIENT_SECRET not in caplog.text
        assert CALLER_TOKEN not in caplog.text


@pytest.fixture
def remote(monkeypatch, configured):
    monkeypatch.setattr(token_exchange, "AUTH_PROVIDER", "token-exchange")
    monkeypatch.setattr(client_strategy, "AUTH_PROVIDER", "token-exchange")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SERVER", "https://remote.example")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_SCOPE", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_REDIRECT_URI", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_ID", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_CLIENT_SECRET", "")
    monkeypatch.setattr(client_strategy, "MCP_REMOTE_BEARER_TOKEN", "shared-bearer")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "")


def _build(access_token, anon=False):
    return client_strategy.build_mcp_client_strategy(
        access_token=access_token, requested_group=None, anon=anon
    )


class TestRemoteStrategy:
    def test_remote_gets_the_exchanged_token_not_the_callers(self, remote, monkeypatch):
        _endpoint(monkeypatch, _ok(EXCHANGED_TOKEN))
        strategy = _build(CALLER_TOKEN)
        assert strategy.headers["Authorization"] == f"Bearer {EXCHANGED_TOKEN}"

    @pytest.mark.parametrize("anon", [False, True])
    def test_without_a_caller_the_remote_gets_the_clients_token(
        self, remote, monkeypatch, anon
    ):
        _endpoint(monkeypatch, _ok(SERVICE_TOKEN), _ok(EXCHANGED_TOKEN))
        strategy = _build(None, anon=anon)
        assert strategy.headers["Authorization"] == f"Bearer {EXCHANGED_TOKEN}"

    def test_refused_exchange_never_falls_back(self, remote, monkeypatch):
        # Neither the shared bearer nor the caller's own token may stand in.
        _endpoint(monkeypatch, FakeResponse(403, {"error": "access_denied"}))
        with pytest.raises(UserLoggedOutException):
            _build(CALLER_TOKEN)

    def test_unreachable_token_endpoint_is_503_with_retry_after(
        self, remote, monkeypatch
    ):
        _endpoint(monkeypatch, requests.ConnectionError("refused"))
        with pytest.raises(HTTPException) as excinfo:
            _build(CALLER_TOKEN)
        assert excinfo.value.status_code == 503
        assert "Retry-After" in (excinfo.value.headers or {})


class TestRoutes:
    def test_unreachable_token_endpoint_reaches_the_caller_as_503(
        self, remote, monkeypatch
    ):
        from fastapi.testclient import TestClient

        from app.server import app as fastapi_app

        monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", [])
        monkeypatch.setattr(app_vars, "BRIDGE_ALLOWED_USERS", [])
        fake = _endpoint(monkeypatch, requests.ConnectionError("refused"))
        response = TestClient(fastapi_app).post(
            "/tools/any_tool",
            json={},
            headers={"X-Auth-Request-Access-Token": CALLER_TOKEN},
        )
        assert response.status_code == 503
        assert response.headers.get("Retry-After") == "5"
        assert [r["data"]["subject_token"] for r in fake.requests] == [CALLER_TOKEN]

    def test_refused_exchange_reaches_the_caller_as_401(self, remote, monkeypatch):
        from fastapi.testclient import TestClient

        from app.server import app as fastapi_app

        monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", [])
        monkeypatch.setattr(app_vars, "BRIDGE_ALLOWED_USERS", [])
        fake = _endpoint(monkeypatch, FakeResponse(403, {"error": "access_denied"}))
        response = TestClient(fastapi_app).post(
            "/tools/any_tool",
            json={},
            headers={"X-Auth-Request-Access-Token": CALLER_TOKEN},
        )
        assert response.status_code == 401
        assert len(fake.requests) == 1
