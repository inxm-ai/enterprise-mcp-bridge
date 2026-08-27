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
    AZURE_METADATA_CLIENT_ID,
    AZURE_METADATA_IDENTITY_RESOURCE,
    AZURE_METADATA_SERVER_URL,
    AZURE_METADATA_TOKEN_TIMEOUT_SECONDS,
    GCP_METADATA_IDENTITY_AUDIENCE,
    GCP_METADATA_SERVER_URL,
    GCP_METADATA_TOKEN_TIMEOUT_SECONDS,
    INTERNAL_API_SECRET,
    KEYCLOAK_PROVIDER_ALIAS,
    KEYCLOAK_PROVIDER_REFRESH_MODE,
    KEYCLOAK_REALM,
    KEYCLOAK_ISSUER,
    LOG_TOKEN_VALUES,
    MCP_CONNECTION_ID,
    MCP_REMOTE_SERVER,
    SERVICE_NAME,
    USER_API_KEY_ALLOWED_CLIENTS,
)
import os
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
        elif provider == "gcp-metadata":
            return GcpMetadataTokenRetriever()
        elif provider == "azure-metadata":
            return AzureManagedIdentityTokenRetriever()
        elif provider == "aws-metadata":
            return AwsWebIdentityTokenRetriever()
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
            f"{AUTH_BASE_URL}/realms/{KEYCLOAK_REALM}" "/protocol/openid-connect/certs"
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


def verified_caller_claims(token: str) -> Dict[str, Any]:
    """Verify a Keycloak token and return its claims, or fail closed.

    Enforces: signature against the realm JWKS, a present and valid exp,
    the exact configured issuer (KEYCLOAK_ISSUER, defaulting to
    {AUTH_BASE_URL}/realms/{realm}). Fails closed with
    UserLoggedOutException on any violation.
    The bridge accepts direct Bearer tokens (desktop clients bypass the
    ingress proxy), so any authorization decision based on token claims
    must go through this function, never an unverified decode.
    """
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            options={"verify_aud": False, "require": ["exp", "iss"]},
        )
    except Exception as exc:
        raise UserLoggedOutException(
            f"Access token failed verification: {exc}"
        ) from exc

    expected_issuer = KEYCLOAK_ISSUER or f"{AUTH_BASE_URL}/realms/{KEYCLOAK_REALM}"
    if claims.get("iss") != expected_issuer:
        raise UserLoggedOutException(
            f"Access token issued by unexpected issuer: {claims.get('iss')}"
        )

    return claims


def _ensure_allowlisted_client(
    claims: Dict[str, Any], *, allowed_clients: list, allowlist_name: str
) -> None:
    if not allowed_clients:
        raise UserLoggedOutException(
            f"{allowlist_name} is not configured; refusing to trust tokens "
            "from unspecified clients"
        )
    azp = claims.get("azp")
    aud = claims.get("aud")
    audiences = aud if isinstance(aud, list) else [aud] if aud else []
    token_clients = [c for c in [azp, *audiences] if c]
    if not any(c in allowed_clients for c in token_clients):
        raise UserLoggedOutException(
            f"Access token client(s) {token_clients} not in the allowlist"
        )


def verified_keycloak_claims(token: str) -> Dict[str, Any]:
    """Verify a Keycloak token before it may release a stored credential."""
    claims = verified_caller_claims(token)
    _ensure_allowlisted_client(
        claims,
        allowed_clients=USER_API_KEY_ALLOWED_CLIENTS,
        allowlist_name="USER_API_KEY_ALLOWED_CLIENTS",
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
            claims.get("email") or claims.get("preferred_username") or claims.get("upn")
        )
        if not email:
            raise UserLoggedOutException(
                "The user's access token carries no email claim"
            )
        return email

    def retrieve_token(self, keycloak_token: str) -> Dict[str, Any]:
        if (
            not self.auth_tokens_url
            or not self.connection_id
            or not self.internal_secret
        ):
            # INTERNAL_API_SECRET is mandatory: without it the only thing
            # sent to the credential store is the spoofable X-Service-ID,
            # which is identity metadata, not authentication.
            self.logger.error(
                "[UserApiKey] AUTH_TOKENS_INTERNAL_URL, MCP_CONNECTION_ID "
                "and INTERNAL_API_SECRET must all be set for "
                "AUTH_PROVIDER=user-api-key"
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


class AmbientIdentityTokenRetriever(TokenRetriever):
    """Base for retrievers that self-fetch a fresh, short-lived bearer
    token ambiently from the host platform on every call.

    Unlike KeyCloakTokenRetriever/UserApiKeyTokenRetriever, these ignore the
    caller's token entirely and never persist anything — the platform is
    trusted to vouch for the bridge's own identity. Any failure (timeout,
    connection error, non-2xx, empty body, missing file) fails closed via
    UserLoggedOutException: there is no legitimate fallback to a shared or
    caller-supplied token for an ambient-identity provider.
    """

    unavailable_message = "Ambient identity provider unreachable"
    logger = logger

    def _fetch_token(self) -> str:
        raise NotImplementedError

    def retrieve_token(self, token: str) -> Dict[str, Any]:
        try:
            raw_token = self._fetch_token()
        except UserLoggedOutException:
            raise
        except Exception as exc:
            self.logger.error(
                "[%s] %s: %s", type(self).__name__, self.unavailable_message, exc
            )
            raise UserLoggedOutException(self.unavailable_message) from exc

        if not raw_token:
            raise UserLoggedOutException(self.unavailable_message)

        return {"success": True, "access_token": raw_token, "token_type": "Bearer"}


class GcpMetadataTokenRetriever(AmbientIdentityTokenRetriever):
    """AUTH_PROVIDER=gcp-metadata: GCE/Cloud Run/GKE metadata-server identity token."""

    unavailable_message = (
        "GCP metadata server unreachable — this auth mode requires running "
        "on GCE/Cloud Run/GKE"
    )

    def _fetch_token(self) -> str:
        audience = GCP_METADATA_IDENTITY_AUDIENCE or MCP_REMOTE_SERVER
        url = (
            f"{GCP_METADATA_SERVER_URL.rstrip('/')}/computeMetadata/v1/"
            "instance/service-accounts/default/identity"
        )
        response = requests.get(
            url,
            headers={"Metadata-Flavor": "Google"},
            params={"audience": audience},
            timeout=GCP_METADATA_TOKEN_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.text.strip()


class AzureManagedIdentityTokenRetriever(AmbientIdentityTokenRetriever):
    """AUTH_PROVIDER=azure-metadata: Azure Instance Metadata Service managed-identity token."""

    unavailable_message = (
        "Azure Instance Metadata Service unreachable — this auth mode "
        "requires running on an Azure resource with a managed identity"
    )

    def _fetch_token(self) -> str:
        resource = AZURE_METADATA_IDENTITY_RESOURCE or MCP_REMOTE_SERVER
        url = f"{AZURE_METADATA_SERVER_URL.rstrip('/')}/metadata/identity/oauth2/token"
        params = {"api-version": "2018-02-01", "resource": resource}
        if AZURE_METADATA_CLIENT_ID:
            params["client_id"] = AZURE_METADATA_CLIENT_ID
        response = requests.get(
            url,
            headers={"Metadata": "true"},
            params=params,
            timeout=AZURE_METADATA_TOKEN_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json().get("access_token")


class AwsWebIdentityTokenRetriever(AmbientIdentityTokenRetriever):
    """AUTH_PROVIDER=aws-metadata: EKS IRSA/Pod Identity projected OIDC token.

    AWS's instance metadata service hands out SigV4 signing credentials, not
    a portable bearer JWT for an arbitrary external audience, so there is no
    metadata-server HTTP call to make here. The equivalent ambient,
    short-lived identity token on AWS is the OIDC token Kubernetes projects
    into the pod filesystem for IRSA/Pod Identity, at the path given by the
    standard AWS_WEB_IDENTITY_TOKEN_FILE env var (set automatically by EKS).
    """

    unavailable_message = (
        "AWS_WEB_IDENTITY_TOKEN_FILE unreadable — this auth mode requires "
        "running on EKS with IAM Roles for Service Accounts (IRSA) / Pod Identity"
    )

    def _fetch_token(self) -> str:
        token_file = os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE", "")
        if not token_file:
            raise UserLoggedOutException(self.unavailable_message)
        with open(token_file, "r") as f:
            return f.read().strip()


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
                self.logger.warning(
                    "LOG_TOKEN_VALUES is ignored because credentials must not be logged"
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
                    "Unable to parse token response for %s",
                    self.provider_alias,
                )
                raise
        elif response.status_code == 401:
            self.logger.warning(
                "Unauthorized access to Keycloak broker endpoint - User may not have broker.read-token role"
            )
            raise UserLoggedOutException("User is logged out or unauthorized")
        else:
            self.logger.error(
                "Failed to retrieve provider token from %s: HTTP %s",
                url,
                response.status_code,
            )
            raise Exception(f"Failed to retrieve token: HTTP {response.status_code}")

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
