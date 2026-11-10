"""Callers from many issuers that share an address pattern (TRUSTED_ISSUERS).

Some deployments serve users of many identity-provider tenants, one issuer
per tenant, and new tenants appear without the bridge being redeployed. A
fixed issuer list doesn't fit, so the bridge trusts patterns instead, such as
``https://*.auth.example.com/realms/main``, where ``*`` stands for exactly
one DNS label. Only whoever controls the domain can create a matching issuer.

Every request must carry a token that passes, in this order:

1. Its issuer fits a pattern. Otherwise 401, without any lookup.
2. Its signing key comes from ``<issuer>/protocol/openid-connect/certs``,
   never from an address the token suggests. Key sets are cached per issuer
   for KEY_SET_LIFESPAN_SECONDS. A token naming a key the cached set lacks
   refetches once (key rotation), at most once per MIN_REFETCH_SECONDS per
   issuer. A key the fresh set still lacks is 401.
3. If the keys can't be fetched, the answer is 503, not 401: the issuer is
   down or slow, so the caller should hear "try again", not "not allowed".
   The failure is remembered for TRUSTED_ISSUER_KEY_FAILURE_SECONDS, so
   tokens naming unreachable issuers can't flood DNS or the issuer with
   lookups, and stale keys are never used.
4. Signature, expiry (with TRUSTED_ISSUER_CLOCK_LEEWAY_SECONDS), audience
   (TRUSTED_ISSUER_AUDIENCE) and requesting client (``azp`` in
   TRUSTED_ISSUER_CLIENTS). The audience must be this bridge alone: a token
   that is also meant for other services would work there too, so it is
   refused even when the requesting client fits. Otherwise 401.

Every refusal is logged with its reason, the issuer, the client and the
audience, so it is clear which tenant or client to fix; never with the
user's identity.
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Union

import jwt
import requests

from app import vars as app_vars
from app.oauth.user_info import CallerNotAuthorizedError

logger = logging.getLogger("uvicorn.error")

KEY_SET_LIFESPAN_SECONDS = 300
MIN_REFETCH_SECONDS = 10
KEY_FETCH_TIMEOUT_SECONDS = 5
MAX_CACHED_ISSUERS = 1024
ALGORITHMS = ["RS256", "ES256"]
REQUIRED_CLAIMS = ["exp", "iss", "aud", "azp"]
KEYS_PATH = "/protocol/openid-connect/certs"
REQUIRED_SCHEME = "https://"
WILDCARD = "*"
# One DNS label: lower-case letters, digits and inner hyphens, at most 63.
DNS_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"

UNAUTHORIZED = 401
UNAVAILABLE = 503
REFUSED_DETAIL = "Access token failed verification"
UNAVAILABLE_DETAIL = "Sign-in is temporarily unavailable"

_monotonic = time.monotonic


class KeyLookupFailure(Enum):
    UNAVAILABLE = "keys could not be fetched"
    UNKNOWN_KEY = "signing key not in the issuer's key set"


@dataclass(frozen=True)
class _CachedKeySet:
    key_set: jwt.PyJWKSet
    fetched_at: float


_key_sets: dict[str, _CachedKeySet] = {}
_failed_at: dict[str, float] = {}


def forget_cached_keys() -> None:
    _key_sets.clear()
    _failed_at.clear()


def cached_issuer_count() -> int:
    return len(_key_sets) + len(_failed_at)


def _fetch_key_set(url: str) -> Optional[Mapping[str, Any]]:
    """The issuer's published key set, or None if it can't be fetched."""
    try:
        response = requests.get(url, timeout=KEY_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
    except (requests.RequestException, ValueError):
        return None
    return body if isinstance(body, dict) else None


def _remember(cache: dict, issuer: str, value: Any) -> None:
    cache.pop(issuer, None)
    while len(cache) >= MAX_CACHED_ISSUERS:
        cache.pop(next(iter(cache)))
    cache[issuer] = value


def _key_with_id(key_set: jwt.PyJWKSet, kid: str) -> Optional[jwt.PyJWK]:
    return next((key for key in key_set.keys if key.key_id == kid), None)


def _fresh_key_set(issuer: str, now: float) -> Optional[jwt.PyJWKSet]:
    published = _fetch_key_set(f"{issuer}{KEYS_PATH}")
    try:
        key_set = jwt.PyJWKSet.from_dict(published) if published else None
    except jwt.PyJWKSetError:
        key_set = None
    if key_set is None:
        _key_sets.pop(issuer, None)
        _remember(_failed_at, issuer, now)
        return None
    _failed_at.pop(issuer, None)
    _remember(_key_sets, issuer, _CachedKeySet(key_set=key_set, fetched_at=now))
    return key_set


def _signing_key(issuer: str, kid: str) -> Union[jwt.PyJWK, KeyLookupFailure]:
    now = _monotonic()
    failed_at = _failed_at.get(issuer)
    if (
        failed_at is not None
        and now - failed_at < app_vars.TRUSTED_ISSUER_KEY_FAILURE_SECONDS
    ):
        return KeyLookupFailure.UNAVAILABLE

    cached = _key_sets.get(issuer)
    if cached is not None and now - cached.fetched_at < KEY_SET_LIFESPAN_SECONDS:
        key = _key_with_id(cached.key_set, kid)
        if key is not None:
            return key
        if now - cached.fetched_at < MIN_REFETCH_SECONDS:
            return KeyLookupFailure.UNKNOWN_KEY

    fresh = _fresh_key_set(issuer, now)
    if fresh is None:
        return KeyLookupFailure.UNAVAILABLE
    key = _key_with_id(fresh, kid)
    return key if key is not None else KeyLookupFailure.UNKNOWN_KEY


def _pattern_regex(pattern: str) -> str:
    return DNS_LABEL.join(re.escape(part) for part in pattern.split(WILDCARD))


def trusted_issuer(issuer: object) -> bool:
    if not isinstance(issuer, str):
        return False
    return any(
        re.fullmatch(_pattern_regex(pattern), issuer)
        for pattern in app_vars.TRUSTED_ISSUERS
    )


def _misconfiguration() -> Optional[str]:
    if not app_vars.TRUSTED_ISSUER_AUDIENCE:
        return "TRUSTED_ISSUER_AUDIENCE is empty"
    if not app_vars.TRUSTED_ISSUER_CLIENTS:
        return "TRUSTED_ISSUER_CLIENTS is empty"
    if not all(p.startswith(REQUIRED_SCHEME) for p in app_vars.TRUSTED_ISSUERS):
        return "every TRUSTED_ISSUERS pattern must start with https://"
    return None


def _refusal(
    status_code: int, reason: str, claims: Mapping[str, Any]
) -> CallerNotAuthorizedError:
    # Claims here may be unverified, so they are logged quoted (%r) and only
    # the ones that name an issuer or client, never `sub`, email or name.
    logger.warning(
        "[TrustedIssuers] %s refused a token (%d): %s; iss=%r azp=%r aud=%r",
        app_vars.SERVICE_NAME,
        status_code,
        reason,
        claims.get("iss"),
        claims.get("azp"),
        claims.get("aud"),
    )
    detail = UNAVAILABLE_DETAIL if status_code == UNAVAILABLE else REFUSED_DETAIL
    return CallerNotAuthorizedError(status_code, detail)


def ensure_trusted_issuer_caller(access_token: Optional[str]) -> None:
    """Fail closed unless the token comes from a trusted issuer pattern.

    A no-op while TRUSTED_ISSUERS is empty. Raises CallerNotAuthorizedError
    (401 or 503) like the bridge's other caller gates.
    """
    if not app_vars.TRUSTED_ISSUERS:
        return
    problem = _misconfiguration()
    if problem:
        logger.error("[TrustedIssuers] %s; refusing every caller", problem)
        raise CallerNotAuthorizedError(UNAVAILABLE, UNAVAILABLE_DETAIL)
    if not access_token:
        raise _refusal(UNAUTHORIZED, "no token", {})

    try:
        unverified = jwt.decode(access_token, options={"verify_signature": False})
        kid = jwt.get_unverified_header(access_token).get("kid")
    except jwt.InvalidTokenError:
        raise _refusal(UNAUTHORIZED, "not a token", {})

    issuer = unverified.get("iss")
    if not trusted_issuer(issuer):
        raise _refusal(UNAUTHORIZED, "issuer fits no pattern", unverified)
    if not isinstance(kid, str):
        raise _refusal(UNAUTHORIZED, "no key id", unverified)

    key = _signing_key(issuer, kid)
    if key is KeyLookupFailure.UNAVAILABLE:
        raise _refusal(UNAVAILABLE, key.value, unverified)
    if isinstance(key, KeyLookupFailure):
        raise _refusal(UNAUTHORIZED, key.value, unverified)

    try:
        claims = jwt.decode(
            access_token,
            key.key,
            algorithms=ALGORITHMS,
            audience=app_vars.TRUSTED_ISSUER_AUDIENCE,
            issuer=issuer,
            leeway=app_vars.TRUSTED_ISSUER_CLOCK_LEEWAY_SECONDS,
            options={"require": REQUIRED_CLAIMS},
        )
    except jwt.InvalidTokenError as exc:
        raise _refusal(UNAUTHORIZED, type(exc).__name__, unverified)

    if claims.get("azp") not in app_vars.TRUSTED_ISSUER_CLIENTS:
        raise _refusal(UNAUTHORIZED, "requesting client not allowed", claims)
    if not _audience_is_only(claims.get("aud"), app_vars.TRUSTED_ISSUER_AUDIENCE):
        raise _refusal(UNAUTHORIZED, "token is also meant for others", claims)


def _audience_is_only(aud: object, audience: str) -> bool:
    return aud == audience or aud == [audience]
