"""Provider tokens from INXM's central OAuth broker (ops-auth-broker).

Hosted platforms connect people's provider accounts (Google, Microsoft,
GitHub, ...) at the broker, not at a Keycloak identity provider. A server
keeps ``KEYCLOAK_PROVIDER_ALIAS=<name>``; when ``OPS_AUTH_BROKER_URL`` is set
(app-mcp-manager sets it on every bridge of a hosted platform) and the
broker's public catalog lists that name, the token comes from the broker:

    POST {OPS_AUTH_BROKER_URL}/v1/providers/<name>/access-token
    Authorization: Bearer <the caller's own platform token>

The broker refreshes the grant itself; refresh tokens never leave it. 404 is
"not connected", 409 "sign in to the provider again". An unreachable broker
or a server error is 503, never a sign-out.

A provider the broker doesn't hold (a Keycloak identity provider such as
Dropbox) never depends on the broker: while its catalog can't be read, the
alias is tried at Keycloak first, and only an alias Keycloak can't serve
either answers 503 (it may be one of the broker's).
"""

import logging
import threading
import time
from typing import Any, Dict, Optional, Set
from urllib.parse import quote

import requests

logger = logging.getLogger("uvicorn.error")

CATALOG_TTL_SECONDS = 60.0
# How soon to ask again while no catalog was ever read.
RETRY_SECONDS = 10.0
TIMEOUT_SECONDS = 5.0

_lock = threading.Lock()
_catalog: Optional[Set[str]] = None
_fetched_at = 0.0


def broker_url() -> str:
    """``OPS_AUTH_BROKER_URL`` without a trailing slash; empty when off."""
    from app import vars as app_vars

    url = (app_vars.OPS_AUTH_BROKER_URL or "").strip().rstrip("/")
    if url and not url.startswith("https://"):
        logger.error("OPS_AUTH_BROKER_URL must be https; broker providers are off")
        return ""
    return url


def _reset_catalog() -> None:
    """Tests: forget the cached catalog."""
    global _catalog, _fetched_at
    with _lock:
        _catalog = None
        _fetched_at = 0.0


def _names(url: str) -> Optional[Set[str]]:
    """The catalog's provider names, or None when it can't be read."""
    try:
        response = requests.get(f"{url}/v1/catalog", timeout=TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}")
        providers = response.json().get("providers")
        if not isinstance(providers, list):
            raise ValueError("catalog without a providers list")
        return {
            entry["name"]
            for entry in providers
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        }
    except Exception as exc:  # noqa: BLE001 - any failure keeps the last catalog
        logger.warning("ops-auth-broker catalog unavailable: %s", exc)
        return None


def serves(provider_alias: str) -> Optional[bool]:
    """Whether the broker holds ``provider_alias``'s grants.

    The catalog is cached for a minute; when the broker can't answer, the
    last one stays (so a short outage doesn't send the alias to Keycloak).
    While no catalog was ever read, nobody can tell: that is ``None``, and
    the caller tries Keycloak before it says "temporarily unavailable". The
    next call asks the broker again after RETRY_SECONDS.
    """
    global _catalog, _fetched_at
    url = broker_url()
    if not url or not provider_alias:
        return False
    with _lock:
        ttl = CATALOG_TTL_SECONDS if _catalog is not None else RETRY_SECONDS
        if _fetched_at == 0.0 or time.monotonic() - _fetched_at >= ttl:
            names = _names(url)
            if names is not None:
                _catalog = names
            _fetched_at = time.monotonic()
        if _catalog is None:
            return None
        return provider_alias in _catalog


def access_token(provider_alias: str, platform_token: str) -> Dict[str, Any]:
    """The caller's provider access token from the broker."""
    from app.oauth.token_exchange import (
        ProviderNotConnectedError,
        ProviderTokenExpiredError,
        TokenServiceUnavailableError,
        UserLoggedOutException,
    )

    if not platform_token:
        raise UserLoggedOutException("A signed-in caller is required")
    url = f"{broker_url()}/v1/providers/{quote(provider_alias, safe='')}/access-token"
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {platform_token}",
                "Accept": "application/json",
            },
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.error("[OpsAuthBroker] %s token request failed: %s", provider_alias, exc)
        raise TokenServiceUnavailableError("The OAuth broker is unreachable") from exc
    status = response.status_code
    if status == 404:
        raise ProviderNotConnectedError(provider_alias)
    if status == 409:
        raise ProviderTokenExpiredError(provider_alias)
    if status == 401:
        raise UserLoggedOutException("The OAuth broker refused the platform token")
    if status != 200:
        logger.error(
            "[OpsAuthBroker] %s token request: HTTP %s", provider_alias, status
        )
        raise TokenServiceUnavailableError(f"The OAuth broker answered HTTP {status}")
    try:
        body = response.json()
    except ValueError as exc:
        raise TokenServiceUnavailableError("Invalid OAuth broker response") from exc
    token = body.get("access_token") if isinstance(body, dict) else None
    if not isinstance(token, str) or not token:
        raise TokenServiceUnavailableError("The OAuth broker returned no token")
    logger.info("[OpsAuthBroker] Retrieved the %s token", provider_alias)
    return {
        "success": True,
        "access_token": token,
        "token_type": body.get("token_type") or "Bearer",
        "expires_in": body.get("expires_in"),
    }
