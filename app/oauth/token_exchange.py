from datetime import datetime, timedelta
import logging
from typing import Dict, Any
from json import JSONDecodeError
from urllib.parse import parse_qsl

from app.vars import (
    AUTH_ALLOW_UNSAFE_CERT,
    AUTH_BASE_URL,
    AUTH_PROVIDER,
    AUTH_TOKENS_INTERNAL_URL,
    INTERNAL_API_SECRET,
    KEYCLOAK_PROVIDER_ALIAS,
    KEYCLOAK_PROVIDER_REFRESH_MODE,
    KEYCLOAK_REALM,
    LOG_TOKEN_VALUES,
    MCP_CONNECTION_ID,
    SERVICE_NAME,
)
import jwt
from jwt import DecodeError, InvalidTokenError

import requests

from app.utils import mask_token, token_fingerprint

logger = logging.getLogger("uvicorn.error")


class TokenRetriever:
    def retrieve_token(self, token: str) -> Dict[str, Any]:
        pass


class TokenRetrieverFactory:
    def get(self) -> TokenRetriever:
        """
        Factory method to retrieve the appropriate token retriever based on environment variables.
        """
        provider: str = AUTH_PROVIDER
        if provider == "keycloak":
            return KeyCloakTokenRetriever()
        elif provider == "user-api-key":
            return UserApiKeyTokenRetriever()
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class UserLoggedOutException(Exception):
    """Exception raised when the user is logged out."""

    def __init__(self, message: str = "User is logged out or unauthorized"):
        self.message = message
        super().__init__(message)


_jwks_client = None


def _get_jwks_client():
    """Cached PyJWKClient for the configured Keycloak realm."""
    global _jwks_client
    if _jwks_client is None:
        import ssl

        jwks_url = (
            f"{AUTH_BASE_URL}/realms/{KEYCLOAK_REALM}"
            "/protocol/openid-connect/certs"
        )
        ssl_context = None
        if AUTH_ALLOW_UNSAFE_CERT:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        _jwks_client = jwt.PyJWKClient(
            jwks_url, cache_keys=True, ssl_context=ssl_context
        )
    return _jwks_client


def verified_keycloak_claims(token: str) -> Dict[str, Any]:
    """Verify signature and expiry against Keycloak's JWKS and return claims.

    Raises UserLoggedOutException on any validation failure. Audience is
    not enforced (Keycloak access tokens commonly carry aud=account and
    vary per client); the signature check already pins the issuer's keys,
    and the realm is additionally asserted via the iss claim suffix.
    """
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            options={"verify_aud": False},
        )
    except Exception as exc:
        raise UserLoggedOutException(
            f"Access token failed verification: {exc}"
        ) from exc
    issuer = claims.get("iss", "")
    if not issuer.endswith(f"/realms/{KEYCLOAK_REALM}"):
        raise UserLoggedOutException(
            f"Access token issued by unexpected realm: {issuer}"
        )
    return claims


class UserApiKeyTokenRetriever(TokenRetriever):
    """Per-user API keys for a connection (AUTH_PROVIDER=user-api-key).

    The requesting user's Keycloak token identifies them (email claim);
    their stored key for MCP_CONNECTION_ID is fetched from app-auth-tokens'
    internal API and used as the bearer credential toward the remote MCP.
    """

    def __init__(self):
        self.auth_tokens_url = AUTH_TOKENS_INTERNAL_URL.rstrip("/")
        self.connection_id = MCP_CONNECTION_ID
        self.internal_secret = INTERNAL_API_SECRET
        # X-Service-ID toward app-auth-tokens: the bridge's own identity,
        # e.g. "mcp-notes-server" (matched by the mcp-* allow-list there).
        self.service_id = SERVICE_NAME
        self.allow_unsafe_cert = AUTH_ALLOW_UNSAFE_CERT
        self.logger = logger

    def _verified_claims(self, keycloak_token: str) -> Dict[str, Any]:
        return verified_keycloak_claims(keycloak_token)

    def _resolve_email(self, keycloak_token: str) -> str:
        # A raw credential is released based on this identity — the token
        # signature MUST be verified, unlike the broker path where Keycloak
        # itself re-validates the token server-side.
        claims = self._verified_claims(keycloak_token)
        email = (
            claims.get("email")
            or claims.get("preferred_username")
            or claims.get("upn")
        )
        if not email:
            raise UserLoggedOutException(
                "The user's access token carries no email claim"
            )
        return email

    def retrieve_token(self, keycloak_token: str) -> Dict[str, Any]:
        if not self.auth_tokens_url or not self.connection_id:
            self.logger.error(
                "[UserApiKey] AUTH_TOKENS_INTERNAL_URL and MCP_CONNECTION_ID "
                "must be set for AUTH_PROVIDER=user-api-key"
            )
            return {"success": False, "error": "user_api_key_misconfigured"}

        email = self._resolve_email(keycloak_token)
        url = (
            f"{self.auth_tokens_url}/api/internal/connection-key/"
            f"{email}/{self.connection_id}"
        )
        headers = {"X-Service-ID": self.service_id}
        if self.internal_secret:
            headers["X-Internal-Secret"] = self.internal_secret

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=10,
                verify=not self.allow_unsafe_cert,
            )
        except requests.RequestException as exc:
            self.logger.error("[UserApiKey] Connection key lookup failed: %s", exc)
            return {"success": False, "error": "connection_key_lookup_failed"}

        if response.status_code == 404:
            raise UserLoggedOutException(
                f"No API key configured for connection '{self.connection_id}'. "
                "Add one in your profile under Connections."
            )
        if response.status_code != 200:
            self.logger.error(
                "[UserApiKey] Connection key lookup returned HTTP %s",
                response.status_code,
            )
            return {"success": False, "error": "connection_key_lookup_failed"}

        try:
            payload = response.json()
        except JSONDecodeError:
            return {"success": False, "error": "connection_key_lookup_failed"}
        key = payload.get("key")
        if not key:
            raise UserLoggedOutException(
                f"No API key configured for connection '{self.connection_id}'. "
                "Add one in your profile under Connections."
            )
        self.logger.info(
            mask_token(
                f"[UserApiKey] Using per-user key for connection {self.connection_id}",
                key,
            )
        )
        return {"success": True, "access_token": key, "token_type": "Bearer"}


class KeyCloakTokenRetriever(TokenRetriever):
    def __init__(self):
        self.keycloak_base_url = AUTH_BASE_URL
        self.realm = KEYCLOAK_REALM
        self.provider_alias = KEYCLOAK_PROVIDER_ALIAS
        self.allow_unsafe_cert = AUTH_ALLOW_UNSAFE_CERT
        self.logger = logger

    def retrieve_token(self, keycloak_token: str) -> Dict[str, Any]:
        """
        Retrieve provider API token using Keycloak stored tokens
        """
        try:
            # If no provider alias configured, pass through the original Keycloak token
            if not self.provider_alias or self.provider_alias.strip() == "":
                self.logger.info(
                    "No KEYCLOAK_PROVIDER_ALIAS configured; passing through Keycloak token"
                )
                return {
                    "success": True,
                    "access_token": keycloak_token,
                    "token_type": "Bearer",
                }
            provider_tokens = self._get_stored_provider_token(keycloak_token)
            if not provider_tokens:
                self.logger.info(
                    f"Failed to retrieve {self.provider_alias} token from Keycloak"
                )
                return {
                    "success": False,
                    "error": f"Failed to retrieve {self.provider_alias} token from Keycloak",
                }
            self.logger.info(
                "[Keycloak] Stored provider access token: %s",
                token_fingerprint(provider_tokens.get("access_token")),
            )
            if LOG_TOKEN_VALUES:
                self.logger.info(
                    "[Keycloak] Stored provider access token (raw): %s",
                    provider_tokens.get("access_token"),
                )
            # Check if token needs refresh
            if self._token_needs_refresh(provider_tokens):
                provider_tokens = self._refresh_provider_token(
                    provider_tokens, keycloak_token
                )
            self.logger.info(
                f"Successfully retrieved {self.provider_alias} token from Keycloak"
            )
            return {
                "success": True,
                "access_token": provider_tokens.get("access_token"),
                "token_type": provider_tokens.get("token_type", "Bearer"),
                "expires_in": provider_tokens.get("expires_in"),
            }
        except Exception as e:
            self.logger.error(f"Token retrieval failed: {str(e)}")
            raise UserLoggedOutException(
                "Token retrieval failed, user probably logged out. Please log in again."
            )

    def _extract_keycloak_token(self, headers: Dict[str, str]) -> str:
        """Extract Keycloak token from request headers"""
        # Try different header formats
        auth_header = headers.get("X-Auth-Request-Access-Token")

        if auth_header:
            if auth_header.startswith("Bearer "):
                return auth_header.split(" ")[1]
            return auth_header

        return None

    def _get_stored_provider_token(self, keycloak_token: str) -> Dict[str, Any]:
        """
        Retrieve stored provider token from Keycloak broker endpoint
        """
        url = f"{self.keycloak_base_url}/realms/{self.realm}/broker/{self.provider_alias}/token"
        headers = {
            "Authorization": f"Bearer {keycloak_token}",
            "Accept": "application/json",
        }
        response = requests.get(url, headers=headers, verify=not self.allow_unsafe_cert)
        self.logger.info(
            f"Requesting {self.provider_alias} token from Keycloak: {response.status_code}"
        )
        if response.status_code == 200:
            try:
                return response.json()
            except JSONDecodeError:
                text = response.text or ""
                parsed_token = dict(parse_qsl(text)) if text else {}
                if parsed_token:
                    self.logger.info(
                        f"Received form-encoded token response for {self.provider_alias}; converting to JSON."
                    )
                    # Normalize keys to align with JSON expectation
                    normalized = {
                        "access_token": parsed_token.get("access_token"),
                        "refresh_token": parsed_token.get("refresh_token"),
                        "token_type": parsed_token.get("token_type", "Bearer"),
                        "expires_in": parsed_token.get("expires_in"),
                        "scope": parsed_token.get("scope"),
                    }
                    return {k: v for k, v in normalized.items() if v is not None}
                self.logger.error(
                    mask_token(
                        f"Unable to parse token response for {self.provider_alias}: {text}",
                        keycloak_token,
                    )
                )
                raise
        elif response.status_code == 401:
            self.logger.warning(
                "Unauthorized access to Keycloak broker endpoint - User may not have broker.read-token role"
            )
            raise UserLoggedOutException("User is logged out or unauthorized")
        else:
            self.logger.error(
                mask_token(
                    f"Failed to retrieve token from {url}: {response.status_code} - {response.text}",
                    keycloak_token,
                )
            )
            raise Exception(
                mask_token(
                    f"Failed to retrieve token: {response.status_code} - {response.text}",
                    keycloak_token,
                )
            )

    def _token_needs_refresh(self, token_data: Dict[str, Any]) -> bool:
        """Check if the access token needs to be refreshed"""
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")

        if not access_token:
            return True

        try:
            payload = jwt.decode(
                access_token,
                options={"verify_signature": False, "verify_exp": False},
            )
        except (InvalidTokenError, DecodeError):
            self.logger.debug(
                "Access token appears opaque and no refresh token is available; assuming still valid."
            )
            return False
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error(f"Failed to decode access token: {str(exc)}")
            return bool(refresh_token)

        exp_timestamp = payload.get("exp")
        if exp_timestamp is None:
            return bool(refresh_token)

        try:
            exp_time = datetime.fromtimestamp(int(exp_timestamp))
        except (TypeError, ValueError):
            self.logger.debug(
                "Access token exp claim is not an int (%s); falling back to refresh token presence.",
                str(exp_timestamp),
            )
            return bool(refresh_token)

        return exp_time <= datetime.now() + timedelta(seconds=60)

    def _refresh_provider_token(
        self, token_data: Dict[str, Any], keycloak_token: str
    ) -> Dict[str, Any]:
        """Refresh provider token using appropriate strategy based on provider type"""
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            self.logger.info(
                f"No refresh token available for {self.provider_alias}; re-fetching from broker"
            )
            return self._force_broker_refresh(keycloak_token)

        refresh_mode = KEYCLOAK_PROVIDER_REFRESH_MODE

        # For external OAuth2 identity providers (mode='broker'), refresh via broker endpoint
        # For OIDC providers registered as clients (mode='oidc'), use Keycloak's token endpoint
        if refresh_mode == "broker":
            self.logger.info(
                f"Refreshing {self.provider_alias} token via broker endpoint (mode=broker)"
            )
            return self._force_broker_refresh(keycloak_token)
        else:
            # Default OIDC mode: use Keycloak's token endpoint with provider alias as client_id
            self.logger.info(
                f"Refreshing {self.provider_alias} token via OIDC endpoint (mode=oidc)"
            )
            url = f"{self.keycloak_base_url}/realms/{self.realm}/protocol/openid-connect/token"
            payload = {
                "grant_type": "refresh_token",
                "client_id": self.provider_alias,
                "refresh_token": refresh_token,
            }
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            response = requests.post(
                url, data=payload, headers=headers, verify=not self.allow_unsafe_cert
            )
            self.logger.info(
                f"Refreshing {self.provider_alias} token: {response.status_code}"
            )
            if response.status_code == 200:
                refreshed_tokens = response.json()
                token_data.update(refreshed_tokens)
                return token_data
            else:
                if response.status_code == 401:
                    raise UserLoggedOutException("User is logged out or unauthorized")
                self.logger.error(
                    mask_token(
                        f"Failed to refresh token: {response.status_code} - {response.text}",
                        refresh_token,
                    )
                )
                raise UserLoggedOutException("Failed to refresh token")

    def force_token_refresh(self, keycloak_token: str) -> Dict[str, Any]:
        """Force a token refresh by re-requesting from Keycloak broker"""
        try:
            # When no provider alias is configured, just return the incoming token
            if not self.provider_alias:
                self.logger.info(
                    "No KEYCLOAK_PROVIDER_ALIAS configured; returning original Keycloak token on force refresh"
                )
                return {"success": True, "access_token": keycloak_token}
            current_tokens = self._get_stored_provider_token(keycloak_token)
            if current_tokens and current_tokens.get("refresh_token"):
                new_token = self._refresh_provider_token(current_tokens, keycloak_token)
                self.logger.info(
                    f"[REFRESH]Forced refresh successful using refresh token: {str(new_token)}"
                )
                return {
                    "success": True,
                    "access_token": new_token.get("access_token"),
                }
            else:
                new_token = self._force_broker_refresh(keycloak_token)
                self.logger.info(
                    f"[BROKER]Forced refresh successful using broker: {str(new_token)}"
                )
                return {
                    "success": True,
                    "access_token": new_token.get("access_token"),
                }
        except Exception as e:
            self.logger.error(f"Force refresh failed: {str(e)}")
            return {"success": False, "error": "Force refresh failed"}

    def _force_broker_refresh(self, keycloak_token: str) -> Dict[str, Any]:
        """Force a fresh token retrieval from the broker"""
        url = f"{self.keycloak_base_url}/realms/{self.realm}/broker/{self.provider_alias}/token"
        headers = {
            "Authorization": f"Bearer {keycloak_token}",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        }
        response = requests.get(url, headers=headers, verify=not self.allow_unsafe_cert)
        self.logger.info(
            f"Force refreshing {self.provider_alias} token from Keycloak: {response.status_code}"
        )
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(
                mask_token(
                    f"Failed to force broker refresh: {response.status_code} - {response.text}",
                    keycloak_token,
                )
            )
            raise Exception("Failed to force broker refresh")
