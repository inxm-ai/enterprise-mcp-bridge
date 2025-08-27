from datetime import datetime, timedelta
import os
import logging
from typing import Dict, Any

import jwt

import requests

from app.utils import mask_token

logger = logging.getLogger("uvicorn.error")


class TokenRetriever:
    def retrieve_token(self, token: str) -> Dict[str, Any]:
        pass


class TokenRetrieverFactory:
    def get(self) -> TokenRetriever:
        """
        Factory method to retrieve the appropriate token retriever based on environment variables.
        """
        provider: str = os.getenv("AUTH_PROVIDER", "keycloak").lower()
        if provider == "keycloak":
            return KeyCloakTokenRetriever()
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class UserLoggedOutException(Exception):
    """Exception raised when the user is logged out."""

    def __init__(self, message: str = "User is logged out or unauthorized"):
        self.message = message
        super().__init__(message)


class KeyCloakTokenRetriever(TokenRetriever):
    def __init__(self):
        self.keycloak_base_url = os.getenv("AUTH_BASE_URL")
        self.realm = os.getenv("KEYCLOAK_REALM", "inxm")
        self.provider_alias = os.getenv("KEYCLOAK_PROVIDER_ALIAS")
        self.allow_unsafe_cert = (
            os.getenv("AUTH_ALLOW_UNSAFE_CERT", "false").lower() == "true"
        )
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
            # Check if token needs refresh
            if self._token_needs_refresh(provider_tokens):
                provider_tokens = self._refresh_provider_token(provider_tokens)
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
            return response.json()
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
        if not access_token:
            return True
        try:
            payload = jwt.decode(
                access_token, options={"verify_signature": False}, algorithms=["HS256"]
            )
            exp_timestamp = payload.get("exp")
            if exp_timestamp:
                exp_time = datetime.fromtimestamp(exp_timestamp)
                return exp_time <= datetime.now() + timedelta(seconds=60)
            else:
                return True
        except Exception as e:
            self.logger.error(f"Failed to decode access token: {str(e)}")
            return True

    def _refresh_provider_token(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh provider token using Keycloak's token refresh endpoint"""
        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            raise Exception("No refresh token available")
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
            # for now we return the user is logged out exception there too
            #  might also be an error that keycloak is not configured for token refresh
            #  so this might be confusing. It should be somewhat clear from the
            #  error message though.
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
                new_token = self._refresh_provider_token(current_tokens)
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
