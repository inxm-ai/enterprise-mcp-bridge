"""An expired linked-provider token answers 401 with a typed connection code.

The person is signed in; only their link to the provider (e.g. Entra for the
azure-management server) needs renewing, so the platform must ask them to
reconnect that provider rather than to sign in again.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.oauth.connection_policy import ConnectionAttestationMiddleware
from app.oauth.token_exchange import ProviderTokenExpiredError, UserLoggedOutException
from app.routes import PROVIDER_TOKEN_EXPIRED_CODE, _unauthorized


def _client(error: Exception) -> TestClient:
    app = FastAPI()

    @app.get("/call")
    def call():
        try:
            raise error
        except UserLoggedOutException as e:
            raise _unauthorized(e)

    app.add_middleware(ConnectionAttestationMiddleware)
    return TestClient(app)


def test_an_expired_provider_link_is_a_401_that_asks_to_reconnect():
    response = _client(ProviderTokenExpiredError("azure-management")).get("/call")

    assert response.status_code == 401
    assert response.headers["x-inxm-connection-code"] == PROVIDER_TOKEN_EXPIRED_CODE
    assert "Reconnect azure-management" in response.json()["detail"]


def test_a_signed_out_caller_is_a_plain_401():
    response = _client(UserLoggedOutException("User is logged out")).get("/call")

    assert response.status_code == 401
    assert "x-inxm-connection-code" not in response.headers


def test_a_never_linked_provider_is_a_typed_connect_your_account_401():
    from app.oauth.token_exchange import ProviderNotConnectedError

    response = _client(ProviderNotConnectedError("microsoft")).get("/call")

    assert response.status_code == 401
    assert response.headers["x-inxm-connection-code"] == "upstream_auth_required"
    detail = response.json()["detail"]
    assert detail["error"] == "upstream_auth_required"
    assert "microsoft requires you to connect your account" in detail["message"]
