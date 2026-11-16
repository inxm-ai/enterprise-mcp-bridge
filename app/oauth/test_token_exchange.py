from datetime import datetime
import pytest
import jwt
from unittest.mock import patch, MagicMock
import app.oauth.token_exchange as token_exchange
from app.oauth.token_exchange import KeyCloakTokenRetriever, UserLoggedOutException


@pytest.fixture
def retriever(monkeypatch):
    # Patch module-level constants used by KeyCloakTokenRetriever (sourced from app.vars)
    monkeypatch.setattr(
        token_exchange, "AUTH_BASE_URL", "https://test.keycloak", raising=False
    )
    monkeypatch.setattr(token_exchange, "KEYCLOAK_REALM", "testrealm", raising=False)
    monkeypatch.setattr(
        token_exchange, "KEYCLOAK_PROVIDER_ALIAS", "testprovider", raising=False
    )
    return KeyCloakTokenRetriever()


def make_token(exp=None):
    payload = {"exp": exp} if exp else {}
    return jwt.encode(payload, "secret", algorithm="HS256")


def test_retrieve_token_success(monkeypatch, retriever):
    token_data = {
        "access_token": make_token(exp=(datetime.now().timestamp() + 120)),
        "token_type": "Bearer",
        "expires_in": 120,
    }
    monkeypatch.setattr(
        retriever, "_get_stored_provider_token", lambda t: token_data.copy()
    )
    monkeypatch.setattr(retriever, "_token_needs_refresh", lambda td: False)
    result = retriever.retrieve_token("dummy")
    assert result["success"] is True
    assert result["access_token"] == token_data["access_token"]
    assert result["token_type"] == "Bearer"
    assert result["expires_in"] == 120


def test_retrieve_token_refresh(monkeypatch, retriever):
    token_data = {
        "access_token": make_token(exp=(datetime.now().timestamp() - 10)),
        "refresh_token": "refresh",
    }
    monkeypatch.setattr(
        retriever, "_get_stored_provider_token", lambda t: token_data.copy()
    )
    monkeypatch.setattr(retriever, "_token_needs_refresh", lambda td: True)
    monkeypatch.setattr(
        retriever,
        "_refresh_provider_token",
        lambda td, kt: {
            "access_token": "newtoken",
            "token_type": "Bearer",
            "expires_in": 3600,
        },
    )
    result = retriever.retrieve_token("dummy")
    assert result["success"] is True
    assert result["access_token"] == "newtoken"
    assert result["token_type"] == "Bearer"
    assert result["expires_in"] == 3600


def test_retrieve_token_failure(monkeypatch, retriever):
    monkeypatch.setattr(retriever, "_get_stored_provider_token", lambda t: None)
    result = retriever.retrieve_token("dummy")
    assert result["success"] is False
    assert "Failed to retrieve" in result["error"]


def test_retrieve_token_exception(monkeypatch, retriever):
    def raise_exc(token):
        raise Exception("fail")

    monkeypatch.setattr(retriever, "_get_stored_provider_token", raise_exc)
    with pytest.raises(UserLoggedOutException) as excinfo:
        retriever.retrieve_token("dummy")
    assert (
        str(excinfo.value)
        == "Token retrieval failed, user probably logged out. Please log in again."
    )


def test_token_needs_refresh_expired(monkeypatch, retriever):
    token = make_token(exp=(datetime.now().timestamp() - 10))
    assert retriever._token_needs_refresh({"access_token": token}) is True


def test_token_needs_refresh_valid(monkeypatch, retriever):
    token = make_token(exp=(datetime.now().timestamp() + 3600))
    assert retriever._token_needs_refresh({"access_token": token}) is False


def test_token_needs_refresh_no_exp(monkeypatch, retriever):
    token = make_token()
    assert retriever._token_needs_refresh({"access_token": token}) is False


def test_token_needs_refresh_invalid_token(monkeypatch, retriever):
    assert retriever._token_needs_refresh({"access_token": "notajwt"}) is False


def test_token_needs_refresh_no_exp_with_refresh_token(monkeypatch, retriever):
    token = make_token()
    assert (
        retriever._token_needs_refresh(
            {"access_token": token, "refresh_token": "refresh"}
        )
        is True
    )


def test_refresh_provider_token_success(monkeypatch, retriever):
    token_data = {"refresh_token": "refresh"}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "newtoken",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    with patch("requests.post", return_value=mock_response):
        result = retriever._refresh_provider_token(token_data.copy(), "keycloak_token")
        assert result["access_token"] == "newtoken"
        assert result["token_type"] == "Bearer"
        assert result["expires_in"] == 3600


def test_refresh_provider_token_failure(monkeypatch, retriever):
    token_data = {"refresh_token": "refresh"}
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "fail"
    with patch("requests.post", return_value=mock_response):
        with pytest.raises(UserLoggedOutException):
            retriever._refresh_provider_token(token_data.copy(), "keycloak_token")


def test_refresh_provider_token_no_refresh(monkeypatch, retriever):
    with pytest.raises(Exception):
        retriever._refresh_provider_token({})


def test_get_stored_provider_token_success(monkeypatch, retriever):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "token"}
    with patch("requests.get", return_value=mock_response):
        result = retriever._get_stored_provider_token("tok")
        assert result["access_token"] == "token"


def test_get_stored_provider_token_unauthorized(monkeypatch, retriever):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "unauthorized"
    with patch("requests.get", return_value=mock_response):
        with pytest.raises(UserLoggedOutException):
            retriever._get_stored_provider_token("tok")


def test_get_stored_provider_token_other_error(monkeypatch, retriever):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "fail"
    with patch("requests.get", return_value=mock_response):
        with pytest.raises(Exception):
            retriever._get_stored_provider_token("tok")


def test_force_token_refresh_refresh(monkeypatch, retriever):
    monkeypatch.setattr(
        retriever, "_get_stored_provider_token", lambda t: {"refresh_token": "refresh"}
    )
    monkeypatch.setattr(
        retriever,
        "_refresh_provider_token",
        lambda td, kt: {"access_token": "newtoken"},
    )
    result = retriever.force_token_refresh("tok")
    assert result["success"] is True
    assert result["access_token"] == "newtoken"


def test_force_token_refresh_broker(monkeypatch, retriever):
    monkeypatch.setattr(retriever, "_get_stored_provider_token", lambda t: {})
    monkeypatch.setattr(
        retriever, "_force_broker_refresh", lambda t: {"access_token": "broker_token"}
    )
    result = retriever.force_token_refresh("tok")
    assert result["success"] is True
    assert result["access_token"] == "broker_token"


def test_force_token_refresh_exception(monkeypatch, retriever):
    monkeypatch.setattr(
        retriever,
        "_get_stored_provider_token",
        lambda t: (_ for _ in ()).throw(Exception("fail")),
    )
    result = retriever.force_token_refresh("tok")
    assert result["success"] is False
    assert "Force refresh failed" in result["error"]


def test_force_broker_refresh_success(monkeypatch, retriever):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "broker_token"}
    with patch("requests.get", return_value=mock_response):
        result = retriever._force_broker_refresh("tok")
        assert result["access_token"] == "broker_token"


def test_force_broker_refresh_failure(monkeypatch, retriever):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "fail"
    with patch("requests.get", return_value=mock_response):
        with pytest.raises(Exception):
            retriever._force_broker_refresh("tok")


def test_extract_keycloak_token_bearer(retriever):
    headers = {"X-Auth-Request-Access-Token": "Bearer sometoken"}
    assert retriever._extract_keycloak_token(headers) == "sometoken"


def test_extract_keycloak_token_plain(retriever):
    headers = {"X-Auth-Request-Access-Token": "sometoken"}
    assert retriever._extract_keycloak_token(headers) == "sometoken"


def test_extract_keycloak_token_none(retriever):
    headers = {}
    assert retriever._extract_keycloak_token(headers) is None


def test_retrieve_token_passthrough_when_no_provider(monkeypatch):
    # No KEYCLOAK_PROVIDER_ALIAS set -> patch module constant
    monkeypatch.setattr(token_exchange, "KEYCLOAK_PROVIDER_ALIAS", "", raising=False)
    monkeypatch.setattr(
        token_exchange, "AUTH_BASE_URL", "https://test.keycloak", raising=False
    )
    monkeypatch.setattr(token_exchange, "KEYCLOAK_REALM", "testrealm", raising=False)
    r = KeyCloakTokenRetriever()
    result = r.retrieve_token("kc_token")
    assert result["success"] is True
    assert result["access_token"] == "kc_token"
    assert result["token_type"] == "Bearer"


def test_force_token_refresh_passthrough_when_no_provider(monkeypatch):
    # No KEYCLOAK_PROVIDER_ALIAS set -> patch module constant
    monkeypatch.setattr(token_exchange, "KEYCLOAK_PROVIDER_ALIAS", "", raising=False)
    monkeypatch.setattr(
        token_exchange, "AUTH_BASE_URL", "https://test.keycloak", raising=False
    )
    monkeypatch.setattr(token_exchange, "KEYCLOAK_REALM", "testrealm", raising=False)
    r = KeyCloakTokenRetriever()
    result = r.force_token_refresh("kc_token")
    assert result["success"] is True
    assert result["access_token"] == "kc_token"


def test_force_token_refresh_never_logs_token_values(monkeypatch, retriever, caplog):
    tokens = {
        "access_token": "access-canary",
        "refresh_token": "refresh-canary",
        "expires_in": 300,
    }
    monkeypatch.setattr(
        retriever, "_get_stored_provider_token", lambda t: {"refresh_token": "old"}
    )
    monkeypatch.setattr(retriever, "_refresh_provider_token", lambda td, kt: tokens)
    monkeypatch.setattr(retriever, "logger", token_exchange.logging.getLogger("t"))
    with caplog.at_level("DEBUG"):
        assert retriever.force_token_refresh("tok")["success"] is True
        monkeypatch.setattr(retriever, "_get_stored_provider_token", lambda t: {})
        monkeypatch.setattr(retriever, "_force_broker_refresh", lambda t: tokens)
        assert retriever.force_token_refresh("tok")["success"] is True
    rendered = " ".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert "Forced refresh successful" in rendered
    assert "access-canary" not in rendered
    assert "refresh-canary" not in rendered


# --- linked external providers: a failed bridge refresh ----------------------

from app.oauth.token_exchange import ProviderTokenExpiredError  # noqa: E402


def _tokens(seconds_left, name="stored"):
    return {
        "access_token": jwt.encode(
            {"exp": int(datetime.now().timestamp() + seconds_left), "n": name},
            "secret",
            algorithm="HS256",
        ),
        "refresh_token": "entra-refresh",
    }


def _refused(token_data, keycloak_token):
    raise UserLoggedOutException("User is logged out or unauthorized")


def test_a_still_valid_token_is_used_when_the_refresh_fails(monkeypatch, retriever):
    # The bridge refreshes a minute early; Keycloak renews only after expiry.
    stored = _tokens(40)
    reads = []
    monkeypatch.setattr(
        retriever,
        "_get_stored_provider_token",
        lambda t: reads.append(t) or stored,
    )
    monkeypatch.setattr(retriever, "_refresh_provider_token", _refused)

    result = retriever.retrieve_token("callers-keycloak-token")

    assert result["access_token"] == stored["access_token"]
    assert len(reads) == 1


def test_an_expired_token_is_read_again_and_keycloaks_renewal_used(
    monkeypatch, retriever
):
    expired, renewed = _tokens(-20, "old"), _tokens(3600, "new")
    answers = iter([expired, renewed])
    monkeypatch.setattr(
        retriever, "_get_stored_provider_token", lambda t: next(answers)
    )
    monkeypatch.setattr(retriever, "_refresh_provider_token", _refused)

    result = retriever.retrieve_token("callers-keycloak-token")

    assert result["access_token"] == renewed["access_token"]


def test_an_expired_token_that_cannot_be_renewed_names_the_provider(
    monkeypatch, retriever
):
    expired = _tokens(-20)
    monkeypatch.setattr(retriever, "_get_stored_provider_token", lambda t: expired)
    monkeypatch.setattr(retriever, "_refresh_provider_token", _refused)

    with pytest.raises(ProviderTokenExpiredError) as raised:
        retriever.retrieve_token("callers-keycloak-token")
    assert raised.value.provider_alias == "testprovider"
    assert "Reconnect testprovider" in raised.value.message


def test_an_unlinked_reread_is_also_an_expired_link(monkeypatch, retriever):
    answers = iter([_tokens(-20), None])
    monkeypatch.setattr(
        retriever, "_get_stored_provider_token", lambda t: next(answers)
    )
    monkeypatch.setattr(retriever, "_refresh_provider_token", _refused)

    with pytest.raises(ProviderTokenExpiredError):
        retriever.retrieve_token("callers-keycloak-token")
