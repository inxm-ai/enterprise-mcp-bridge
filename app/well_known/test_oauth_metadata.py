"""Tests for OAuth discovery well-known endpoints."""

import pytest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.well_known.oauth_metadata import router


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Protected Resource Metadata (RFC 9728)
# ---------------------------------------------------------------------------


class TestProtectedResourceMetadata:
    def test_returns_404_when_not_configured(self, client):
        with patch("app.well_known.oauth_metadata._derive_issuer", return_value=None):
            resp = client.get("/.well-known/oauth-protected-resource")
            assert resp.status_code == 404

    def test_returns_metadata_with_keycloak_config(self, client):
        with patch(
            "app.well_known.oauth_metadata._derive_issuer",
            return_value="https://id.example.com/realms/test",
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_RESOURCE_URL",
            "https://bridge.example.com",
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_SCOPES",
            ["openid", "profile"],
        ):
            resp = client.get("/.well-known/oauth-protected-resource")
            assert resp.status_code == 200
            data = resp.json()
            assert data["resource"] == "https://bridge.example.com"
            assert "https://id.example.com/realms/test" in data["authorization_servers"]
            assert data["scopes_supported"] == ["openid", "profile"]
            assert data["bearer_methods_supported"] == ["header"]

    def test_derives_resource_url_from_request_when_not_configured(self, client):
        with patch(
            "app.well_known.oauth_metadata._derive_issuer",
            return_value="https://id.example.com/realms/test",
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_RESOURCE_URL",
            None,
        ), patch(
            "app.well_known.oauth_metadata.MCP_BASE_PATH",
            "/api/mcp",
        ):
            resp = client.get("/.well-known/oauth-protected-resource")
            assert resp.status_code == 200
            data = resp.json()
            # TestClient uses http://testserver
            assert data["resource"].endswith("/api/mcp")

    def test_includes_resource_name_when_client_id_set(self, client):
        with patch(
            "app.well_known.oauth_metadata._derive_issuer",
            return_value="https://id.example.com/realms/test",
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_RESOURCE_URL",
            "https://bridge.example.com",
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_CLIENT_ID",
            "my-bridge-client",
        ):
            resp = client.get("/.well-known/oauth-protected-resource")
            data = resp.json()
            assert data["resource_name"] == "my-bridge-client"

    def test_cache_control_header(self, client):
        with patch(
            "app.well_known.oauth_metadata._derive_issuer",
            return_value="https://id.example.com/realms/test",
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_RESOURCE_URL",
            "https://bridge.example.com",
        ):
            resp = client.get("/.well-known/oauth-protected-resource")
            assert "max-age=3600" in resp.headers.get("cache-control", "")


# ---------------------------------------------------------------------------
# OAuth Authorization Server Metadata (RFC 8414)
# ---------------------------------------------------------------------------


class TestOAuthAuthorizationServerMetadata:
    def test_returns_404_when_not_configured(self, client):
        with patch("app.well_known.oauth_metadata._derive_issuer", return_value=None):
            resp = client.get("/.well-known/oauth-authorization-server")
            assert resp.status_code == 404

    def test_returns_keycloak_metadata(self, client):
        issuer = "https://id.example.com/realms/test"
        with patch(
            "app.well_known.oauth_metadata._derive_issuer",
            return_value=issuer,
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_REGISTRATION_ENDPOINT",
            None,
        ):
            resp = client.get("/.well-known/oauth-authorization-server")
            assert resp.status_code == 200
            data = resp.json()
            assert data["issuer"] == issuer
            assert (
                data["authorization_endpoint"]
                == f"{issuer}/protocol/openid-connect/auth"
            )
            assert data["token_endpoint"] == f"{issuer}/protocol/openid-connect/token"
            assert "code" in data["response_types_supported"]
            assert "S256" in data["code_challenge_methods_supported"]
            # Default Keycloak registration endpoint
            assert (
                "clients-registrations/openid-connect" in data["registration_endpoint"]
            )

    def test_custom_registration_endpoint(self, client):
        issuer = "https://id.example.com/realms/test"
        custom_reg = "https://id.example.com/custom-register"
        with patch(
            "app.well_known.oauth_metadata._derive_issuer",
            return_value=issuer,
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_REGISTRATION_ENDPOINT",
            custom_reg,
        ):
            resp = client.get("/.well-known/oauth-authorization-server")
            data = resp.json()
            assert data["registration_endpoint"] == custom_reg

    def test_includes_scopes(self, client):
        issuer = "https://id.example.com/realms/test"
        with patch(
            "app.well_known.oauth_metadata._derive_issuer",
            return_value=issuer,
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_SCOPES",
            ["openid", "offline_access"],
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_REGISTRATION_ENDPOINT",
            None,
        ):
            resp = client.get("/.well-known/oauth-authorization-server")
            data = resp.json()
            assert data["scopes_supported"] == ["openid", "offline_access"]

    def test_grant_types_include_refresh(self, client):
        issuer = "https://id.example.com/realms/test"
        with patch(
            "app.well_known.oauth_metadata._derive_issuer",
            return_value=issuer,
        ), patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_REGISTRATION_ENDPOINT",
            None,
        ):
            resp = client.get("/.well-known/oauth-authorization-server")
            data = resp.json()
            assert "authorization_code" in data["grant_types_supported"]
            assert "refresh_token" in data["grant_types_supported"]


# ---------------------------------------------------------------------------
# _derive_issuer
# ---------------------------------------------------------------------------


class TestDeriveIssuer:
    def test_explicit_issuer_takes_precedence(self):
        with patch(
            "app.well_known.oauth_metadata.MCP_OAUTH_ISSUER",
            "https://custom-issuer.example.com",
        ):
            from app.well_known.oauth_metadata import _derive_issuer

            assert _derive_issuer() == "https://custom-issuer.example.com"

    def test_derives_from_keycloak_vars(self):
        with patch("app.well_known.oauth_metadata.MCP_OAUTH_ISSUER", None), patch(
            "app.well_known.oauth_metadata.AUTH_BASE_URL",
            "https://keycloak.example.com",
        ), patch("app.well_known.oauth_metadata.KEYCLOAK_REALM", "myrealm"):
            from app.well_known.oauth_metadata import _derive_issuer

            assert _derive_issuer() == "https://keycloak.example.com/realms/myrealm"

    def test_returns_none_when_nothing_configured(self):
        with patch("app.well_known.oauth_metadata.MCP_OAUTH_ISSUER", None), patch(
            "app.well_known.oauth_metadata.AUTH_BASE_URL", ""
        ):
            from app.well_known.oauth_metadata import _derive_issuer

            assert _derive_issuer() is None
