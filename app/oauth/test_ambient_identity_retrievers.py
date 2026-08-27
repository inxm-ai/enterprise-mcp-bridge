import pytest
import requests

from app.oauth import token_exchange
from app.oauth.token_exchange import (
    AwsWebIdentityTokenRetriever,
    AzureManagedIdentityTokenRetriever,
    GcpMetadataTokenRetriever,
    TokenRetrieverFactory,
    UserLoggedOutException,
)


class DummyResponse:
    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.mark.parametrize(
    "provider,cls",
    [
        ("gcp-metadata", GcpMetadataTokenRetriever),
        ("azure-metadata", AzureManagedIdentityTokenRetriever),
        ("aws-metadata", AwsWebIdentityTokenRetriever),
    ],
)
def test_factory_returns_ambient_identity_retriever(monkeypatch, provider, cls):
    monkeypatch.setattr(token_exchange, "AUTH_PROVIDER", provider)
    assert isinstance(TokenRetrieverFactory().get(), cls)


def test_gcp_retrieve_token_success(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyResponse(200, text="  gcp-identity-jwt  \n")

    monkeypatch.setattr(token_exchange.requests, "get", fake_get)
    monkeypatch.setattr(token_exchange, "GCP_METADATA_IDENTITY_AUDIENCE", "")
    monkeypatch.setattr(token_exchange, "MCP_REMOTE_SERVER", "https://remote.example")

    result = GcpMetadataTokenRetriever().retrieve_token("")

    assert result == {
        "success": True,
        "access_token": "gcp-identity-jwt",
        "token_type": "Bearer",
    }
    assert captured["headers"] == {"Metadata-Flavor": "Google"}
    assert captured["params"] == {"audience": "https://remote.example"}
    assert "computeMetadata/v1/instance/service-accounts/default/identity" in captured["url"]


def test_gcp_retrieve_token_explicit_audience_overrides_remote_server(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["params"] = params
        return DummyResponse(200, text="jwt")

    monkeypatch.setattr(token_exchange.requests, "get", fake_get)
    monkeypatch.setattr(
        token_exchange, "GCP_METADATA_IDENTITY_AUDIENCE", "https://custom-audience"
    )
    monkeypatch.setattr(token_exchange, "MCP_REMOTE_SERVER", "https://remote.example")

    GcpMetadataTokenRetriever().retrieve_token("")

    assert captured["params"] == {"audience": "https://custom-audience"}


def test_gcp_retrieve_token_unreachable_fails_closed(monkeypatch):
    def fake_get(*a, **k):
        raise requests.ConnectionError("no route to metadata server")

    monkeypatch.setattr(token_exchange.requests, "get", fake_get)

    with pytest.raises(UserLoggedOutException) as excinfo:
        GcpMetadataTokenRetriever().retrieve_token("")
    assert "GCP metadata server unreachable" in str(excinfo.value)


def test_gcp_retrieve_token_non_2xx_fails_closed(monkeypatch):
    monkeypatch.setattr(
        token_exchange.requests, "get", lambda *a, **k: DummyResponse(404, text="")
    )
    with pytest.raises(UserLoggedOutException):
        GcpMetadataTokenRetriever().retrieve_token("")


def test_azure_retrieve_token_success(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        return DummyResponse(200, json_data={"access_token": "azure-access-token"})

    monkeypatch.setattr(token_exchange.requests, "get", fake_get)
    monkeypatch.setattr(token_exchange, "AZURE_METADATA_IDENTITY_RESOURCE", "")
    monkeypatch.setattr(token_exchange, "AZURE_METADATA_CLIENT_ID", "")
    monkeypatch.setattr(token_exchange, "MCP_REMOTE_SERVER", "https://remote.example")

    result = AzureManagedIdentityTokenRetriever().retrieve_token("")

    assert result == {
        "success": True,
        "access_token": "azure-access-token",
        "token_type": "Bearer",
    }
    assert captured["headers"] == {"Metadata": "true"}
    assert captured["params"] == {
        "api-version": "2018-02-01",
        "resource": "https://remote.example",
    }
    assert "/metadata/identity/oauth2/token" in captured["url"]


def test_azure_retrieve_token_includes_client_id_when_set(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["params"] = params
        return DummyResponse(200, json_data={"access_token": "tok"})

    monkeypatch.setattr(token_exchange.requests, "get", fake_get)
    monkeypatch.setattr(token_exchange, "AZURE_METADATA_IDENTITY_RESOURCE", "res")
    monkeypatch.setattr(token_exchange, "AZURE_METADATA_CLIENT_ID", "user-assigned-id")

    AzureManagedIdentityTokenRetriever().retrieve_token("")

    assert captured["params"]["client_id"] == "user-assigned-id"


def test_azure_retrieve_token_unreachable_fails_closed(monkeypatch):
    def fake_get(*a, **k):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(token_exchange.requests, "get", fake_get)

    with pytest.raises(UserLoggedOutException) as excinfo:
        AzureManagedIdentityTokenRetriever().retrieve_token("")
    assert "Azure Instance Metadata Service unreachable" in str(excinfo.value)


def test_aws_retrieve_token_success(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("aws-oidc-jwt\n")
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(token_file))

    result = AwsWebIdentityTokenRetriever().retrieve_token("")

    assert result == {
        "success": True,
        "access_token": "aws-oidc-jwt",
        "token_type": "Bearer",
    }


def test_aws_retrieve_token_env_var_unset_fails_closed(monkeypatch):
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)

    with pytest.raises(UserLoggedOutException) as excinfo:
        AwsWebIdentityTokenRetriever().retrieve_token("")
    assert "AWS_WEB_IDENTITY_TOKEN_FILE" in str(excinfo.value)


def test_aws_retrieve_token_missing_file_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "AWS_WEB_IDENTITY_TOKEN_FILE", str(tmp_path / "does-not-exist")
    )

    with pytest.raises(UserLoggedOutException):
        AwsWebIdentityTokenRetriever().retrieve_token("")
