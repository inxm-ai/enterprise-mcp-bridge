"""Provider tokens from INXM's OAuth broker (ops-auth-broker), hosted.

A server keeps KEYCLOAK_PROVIDER_ALIAS=<name>. With OPS_AUTH_BROKER_URL set
and <name> in the broker's public catalog, the token comes from the broker,
with the caller's own platform token; Keycloak is not asked. Every other
alias keeps the Keycloak path. No test reaches the network: requests.get and
requests.post are fakes that record calls and answer from a script.
"""

import pytest
import requests

from app import vars as app_vars
from app.oauth import ops_broker
from app.oauth.token_exchange import (
    KeyCloakTokenRetriever,
    ProviderNotConnectedError,
    ProviderTokenExpiredError,
    TokenServiceUnavailableError,
    UserLoggedOutException,
)

BROKER = "https://ops.inxm.cloud/api/auth-broker"
CALLER = "caller-platform-jwt"


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeHttp:
    def __init__(self, catalog=None, token=None):
        self.catalog = catalog
        self.token = token
        self.gets = []
        self.posts = []

    def get(self, url, timeout=None):
        self.gets.append(url)
        if isinstance(self.catalog, Exception):
            raise self.catalog
        return self.catalog

    def post(self, url, headers=None, timeout=None, **_):
        self.posts.append((url, headers))
        if isinstance(self.token, Exception):
            raise self.token
        return self.token


@pytest.fixture
def broker(monkeypatch):
    monkeypatch.setattr(app_vars, "OPS_AUTH_BROKER_URL", BROKER + "/")
    ops_broker._reset_catalog()
    http = FakeHttp(
        catalog=FakeResponse(
            200,
            {
                "providers": [
                    {"name": "microsoft-365", "preset": "microsoft", "scopes": []},
                    {"name": "google", "preset": "google", "scopes": []},
                ]
            },
        ),
        token=FakeResponse(
            200,
            {
                "access_token": "provider-token",
                "token_type": "Bearer",
                "expires_in": 1200,
                "scopes": ["Mail.Read"],
            },
        ),
    )
    monkeypatch.setattr(ops_broker.requests, "get", http.get)
    monkeypatch.setattr(ops_broker.requests, "post", http.post)
    yield http
    ops_broker._reset_catalog()


def test_off_without_an_https_broker_url(monkeypatch, broker):
    for url in ["", "http://ops.inxm.cloud/api/auth-broker"]:
        monkeypatch.setattr(app_vars, "OPS_AUTH_BROKER_URL", url)
        ops_broker._reset_catalog()
        assert not ops_broker.serves("google")
    assert broker.gets == []


def test_the_catalog_is_cached_and_survives_an_outage(broker, monkeypatch):
    assert ops_broker.serves("microsoft-365")
    assert ops_broker.serves("google")
    assert not ops_broker.serves("notion")
    assert broker.gets == [BROKER + "/v1/catalog"]
    # Expired and the broker is down: the last catalog stays.
    monkeypatch.setattr(ops_broker, "_fetched_at", 0.0)
    broker.catalog = requests.ConnectionError("down")
    assert ops_broker.serves("google")


def test_tokens_come_from_the_broker_with_the_callers_token(broker):
    result = ops_broker.access_token("microsoft-365", CALLER)
    assert result == {
        "success": True,
        "access_token": "provider-token",
        "token_type": "Bearer",
        "expires_in": 1200,
    }
    url, headers = broker.posts[0]
    assert url == BROKER + "/v1/providers/microsoft-365/access-token"
    assert headers["Authorization"] == f"Bearer {CALLER}"


@pytest.mark.parametrize(
    "answer, error",
    [
        (FakeResponse(404), ProviderNotConnectedError),
        (FakeResponse(409), ProviderTokenExpiredError),
        (FakeResponse(401), UserLoggedOutException),
        (FakeResponse(503), TokenServiceUnavailableError),
        (FakeResponse(429), TokenServiceUnavailableError),
        (FakeResponse(200, ValueError("not json")), TokenServiceUnavailableError),
        (FakeResponse(200, {"token_type": "Bearer"}), TokenServiceUnavailableError),
        (requests.ConnectionError("down"), TokenServiceUnavailableError),
    ],
)
def test_broker_answers_map_to_bridge_errors(broker, answer, error):
    broker.token = answer
    with pytest.raises(error):
        ops_broker.access_token("google", CALLER)


def test_the_keycloak_retriever_uses_the_broker_for_its_providers(broker, monkeypatch):
    def keycloak(*_):
        raise AssertionError("Keycloak must not be asked for a broker provider")

    monkeypatch.setattr(KeyCloakTokenRetriever, "_get_stored_provider_token", keycloak)
    result = KeyCloakTokenRetriever("google").retrieve_token(CALLER)
    assert result["access_token"] == "provider-token"
    # Not connected at the broker stays a provider problem, not a sign-out.
    broker.token = FakeResponse(404)
    with pytest.raises(ProviderNotConnectedError):
        KeyCloakTokenRetriever("google").retrieve_token(CALLER)
    broker.token = FakeResponse(503)
    with pytest.raises(TokenServiceUnavailableError):
        KeyCloakTokenRetriever("google").retrieve_token(CALLER)


def test_other_aliases_keep_the_keycloak_path(broker, monkeypatch):
    monkeypatch.setattr(
        KeyCloakTokenRetriever,
        "_get_stored_provider_token",
        lambda self, token: {"access_token": "keycloak-token", "expires_in": 3600},
    )
    monkeypatch.setattr(
        KeyCloakTokenRetriever, "_token_needs_refresh", lambda *_: False
    )
    result = KeyCloakTokenRetriever("notion").retrieve_token(CALLER)
    assert result["access_token"] == "keycloak-token"
    assert broker.posts == []
