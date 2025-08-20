from datetime import datetime, timedelta
import pytest
import jwt
import types
from unittest.mock import patch, MagicMock
from app.oauth.token_exchange import KeyCloakTokenRetriever


@pytest.fixture
def retriever(monkeypatch):
    monkeypatch.setenv("AUTH_BASE_URL", "https://test.keycloak")
    monkeypatch.setenv("KEYCLOAK_REALM", "testrealm")
    monkeypatch.setenv("KEYCLOAK_PROVIDER_ALIAS", "testprovider")
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
        lambda td: {
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
    result = retriever.retrieve_token("dummy")
    assert result["success"] is False
    assert "Token retrieval failed" in result["error"]


def test_token_needs_refresh_expired(monkeypatch, retriever):
    token = make_token(exp=(datetime.now().timestamp() - 10))
    assert retriever._token_needs_refresh({"access_token": token}) is True


def test_token_needs_refresh_valid(monkeypatch, retriever):
    token = make_token(exp=(datetime.now().timestamp() + 3600))
    assert retriever._token_needs_refresh({"access_token": token}) is False


def test_token_needs_refresh_no_exp(monkeypatch, retriever):
    token = make_token()
    assert retriever._token_needs_refresh({"access_token": token}) is True


def test_token_needs_refresh_invalid_token(monkeypatch, retriever):
    assert retriever._token_needs_refresh({"access_token": "notajwt"}) is True


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
        result = retriever._refresh_provider_token(token_data.copy())
        assert result["access_token"] == "newtoken"
        assert result["token_type"] == "Bearer"
        assert result["expires_in"] == 3600


def test_refresh_provider_token_failure(monkeypatch, retriever):
    token_data = {"refresh_token": "refresh"}
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "fail"
    with patch("requests.post", return_value=mock_response):
        with pytest.raises(Exception):
            retriever._refresh_provider_token(token_data.copy())


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
        with pytest.raises(Exception):
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
        retriever, "_refresh_provider_token", lambda td: {"access_token": "newtoken"}
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
