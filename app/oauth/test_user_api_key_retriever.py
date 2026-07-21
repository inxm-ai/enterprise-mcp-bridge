import jwt
import pytest

from app.oauth import token_exchange
from app.oauth.token_exchange import (
    TokenRetrieverFactory,
    UserApiKeyTokenRetriever,
    UserLoggedOutException,
)


def _user_token(email="alice@inxm.ai"):
    return jwt.encode({"email": email}, "secret", algorithm="HS256")


class DummyResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


@pytest.fixture
def retriever(monkeypatch):
    r = UserApiKeyTokenRetriever()
    # unit tests bypass JWKS verification; verified_keycloak_claims has its
    # own dedicated tests below
    monkeypatch.setattr(
        r,
        "_verified_claims",
        lambda token: jwt.decode(
            token, options={"verify_signature": False, "verify_aud": False}
        ),
    )
    r.auth_tokens_url = "http://auth-tokens.orchestrator.svc:8000"
    r.connection_id = "github-mcp"
    r.internal_secret = "s3cret"
    r.service_id = "enterprise-mcp-bridge"
    return r


def test_factory_returns_user_api_key_retriever(monkeypatch):
    monkeypatch.setattr(token_exchange, "AUTH_PROVIDER", "user-api-key")
    assert isinstance(TokenRetrieverFactory().get(), UserApiKeyTokenRetriever)


def test_retrieve_token_success(monkeypatch, retriever):
    captured = {}

    def fake_get(url, headers=None, timeout=None, verify=None):
        captured["url"] = url
        captured["headers"] = headers
        return DummyResponse(
            200, {"success": True, "connection": "github-mcp", "key": "remote-key-1"}
        )

    monkeypatch.setattr(token_exchange.requests, "get", fake_get)

    result = retriever.retrieve_token(_user_token())

    assert result["success"] is True
    assert result["access_token"] == "remote-key-1"
    assert "alice@inxm.ai/github-mcp" in captured["url"]
    assert captured["headers"]["X-Internal-Secret"] == "s3cret"
    assert captured["headers"]["X-Service-ID"] == "enterprise-mcp-bridge"


def test_retrieve_token_missing_key_raises_logged_out(monkeypatch, retriever):
    monkeypatch.setattr(
        token_exchange.requests, "get", lambda *a, **k: DummyResponse(404)
    )
    with pytest.raises(UserLoggedOutException) as excinfo:
        retriever.retrieve_token(_user_token())
    assert "github-mcp" in str(excinfo.value)


def test_retrieve_token_without_email_claim_raises(retriever):
    token = jwt.encode({"sub": "user-1"}, "secret", algorithm="HS256")
    with pytest.raises(UserLoggedOutException):
        retriever.retrieve_token(token)


def test_retrieve_token_misconfigured_returns_error(retriever):
    retriever.connection_id = ""
    result = retriever.retrieve_token(_user_token())
    assert result == {"success": False, "error": "user_api_key_misconfigured"}


def test_retrieve_token_upstream_error_returns_error(monkeypatch, retriever):
    monkeypatch.setattr(
        token_exchange.requests, "get", lambda *a, **k: DummyResponse(500)
    )
    result = retriever.retrieve_token(_user_token())
    assert result == {"success": False, "error": "connection_key_lookup_failed"}


def test_verified_claims_rejects_unverifiable_token(monkeypatch):
    # no JWKS reachable / self-signed token -> hard failure, never trusted
    from app.oauth.token_exchange import verified_keycloak_claims

    class FailingJwks:
        def get_signing_key_from_jwt(self, token):
            raise Exception("no jwks")

    monkeypatch.setattr(token_exchange, "_get_jwks_client", lambda: FailingJwks())
    with pytest.raises(UserLoggedOutException):
        verified_keycloak_claims(_user_token())


def test_verified_claims_rejects_foreign_realm(monkeypatch):
    from app.oauth.token_exchange import verified_keycloak_claims

    class OkJwks:
        def get_signing_key_from_jwt(self, token):
            class K:
                key = "irrelevant"

            return K()

    monkeypatch.setattr(token_exchange, "_get_jwks_client", lambda: OkJwks())
    monkeypatch.setattr(
        token_exchange.jwt,
        "decode",
        lambda *a, **k: {"iss": "https://evil.example/realms/other", "email": "x@y"},
    )
    with pytest.raises(UserLoggedOutException):
        verified_keycloak_claims("token")


def test_retrieve_token_requires_internal_secret(retriever):
    retriever.internal_secret = ""
    result = retriever.retrieve_token(_user_token())
    assert result == {"success": False, "error": "user_api_key_misconfigured"}


class _OkJwks:
    def get_signing_key_from_jwt(self, token):
        class K:
            key = "irrelevant"

        return K()


def _mock_verified(monkeypatch, claims):
    monkeypatch.setattr(token_exchange, "_get_jwks_client", lambda: _OkJwks())
    monkeypatch.setattr(token_exchange.jwt, "decode", lambda *a, **k: claims)
    monkeypatch.setattr(token_exchange, "KEYCLOAK_ISSUER", "https://auth/realms/inxm")


def test_verified_claims_requires_client_allowlist(monkeypatch):
    from app.oauth.token_exchange import verified_keycloak_claims

    _mock_verified(monkeypatch, {"iss": "https://auth/realms/inxm", "azp": "web"})
    monkeypatch.setattr(token_exchange, "USER_API_KEY_ALLOWED_CLIENTS", [])
    with pytest.raises(UserLoggedOutException):
        verified_keycloak_claims("token")


def test_verified_claims_rejects_non_allowlisted_client(monkeypatch):
    from app.oauth.token_exchange import verified_keycloak_claims

    _mock_verified(
        monkeypatch,
        {"iss": "https://auth/realms/inxm", "azp": "desktop", "aud": "account"},
    )
    monkeypatch.setattr(
        token_exchange, "USER_API_KEY_ALLOWED_CLIENTS", ["orchestrator-client"]
    )
    with pytest.raises(UserLoggedOutException):
        verified_keycloak_claims("token")


def test_verified_claims_accepts_allowlisted_azp(monkeypatch):
    from app.oauth.token_exchange import verified_keycloak_claims

    _mock_verified(
        monkeypatch,
        {
            "iss": "https://auth/realms/inxm",
            "azp": "orchestrator-client",
            "email": "alice@inxm.ai",
        },
    )
    monkeypatch.setattr(
        token_exchange, "USER_API_KEY_ALLOWED_CLIENTS", ["orchestrator-client"]
    )
    assert verified_keycloak_claims("token")["email"] == "alice@inxm.ai"
