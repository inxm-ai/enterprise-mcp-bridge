"""Central connection policy enforcement and attestation (app-mcp-manager).

When ``CONNECTION_POLICY_URL`` is set, every transport asks the manager's
``POST /v2/connections/enforce`` as the caller before any downstream contact,
after the existing ``BRIDGE_REQUIRED_GROUPS`` gate (both must allow). A denial
is final (403), a hidden or unknown server is 404, and an authority outage is
a retryable 503: the bridge never falls back to "allowed".

The decision is attested on the HTTP response so the manager's connection
probe can tell that this bridge enforced policy for this caller:

- ``X-Inxm-Connection-Policy-Version`` / ``X-Inxm-Connector-Revision``: the
  authority's decision this request was checked against.
- ``X-Inxm-Caller-Subject``: the verified caller, which must match the
  decision's subject.
- ``X-Inxm-Credential-Kind``: what this bridge actually sends upstream, from
  its own configuration (``oauth`` = the caller's broker grant, ``api_key`` =
  the caller's own key, ``service_key`` = a shared credential, ``public`` =
  none).

Unset, the bridge behaves as before and sends no attestation, so the manager
reports ``enforcement_unavailable`` instead of a verified connection.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from app.oauth.user_info import CallerNotAuthorizedError

logger = logging.getLogger("uvicorn.error")

ATTESTATION_HEADERS = (
    "x-inxm-connection-policy-version",
    "x-inxm-connector-revision",
    "x-inxm-caller-subject",
    "x-inxm-credential-kind",
)
_TIMEOUT_SECONDS = 5.0
_MAX_BODY_BYTES = 16_384
# Allowed decisions may be reused for at most 30 seconds (the agreed maximum
# revocation propagation). Denials and failures are never cached.
_CACHE_SECONDS = 30.0
_CACHE_MAX_ENTRIES = 2048
_cache: dict[tuple[str, str, str], tuple[float, "Attestation"]] = {}


def _cache_key(url: str, access_token: str, server: str) -> tuple[str, str, str]:
    return (url, server, hashlib.sha256(access_token.encode()).hexdigest())


def _cached(key) -> Optional["Attestation"]:
    entry = _cache.get(key)
    if entry is None:
        return None
    if entry[0] <= time.monotonic():
        _cache.pop(key, None)
        return None
    return entry[1]


def _remember(key, attestation: "Attestation") -> None:
    now = time.monotonic()
    if len(_cache) >= _CACHE_MAX_ENTRIES:
        for stale in [k for k, (expiry, _) in _cache.items() if expiry <= now]:
            _cache.pop(stale, None)
        while len(_cache) >= _CACHE_MAX_ENTRIES:
            _cache.pop(next(iter(_cache)))
    _cache[key] = (now + _CACHE_SECONDS, attestation)


@dataclass(frozen=True)
class Attestation:
    policy_version: int
    connector_revision: str
    subject: str
    credential_kind: str

    def headers(self) -> list[tuple[bytes, bytes]]:
        return [
            (b"x-inxm-connection-policy-version", str(self.policy_version).encode()),
            (b"x-inxm-connector-revision", self.connector_revision.encode()),
            (b"x-inxm-caller-subject", self.subject.encode()),
            (b"x-inxm-credential-kind", self.credential_kind.encode()),
        ]


class _Holder:
    __slots__ = ("attestation",)

    def __init__(self) -> None:
        self.attestation: Optional[Attestation] = None


_current: contextvars.ContextVar[Optional[_Holder]] = contextvars.ContextVar(
    "inxm_connection_attestation", default=None
)


def policy_url() -> str:
    # One authority per bridge process, also in multi-server mode.
    return os.getenv("CONNECTION_POLICY_URL", "").strip()


def server_id() -> str:
    from app import vars as app_vars
    from app.multi_server import current_server_id

    # Single-server mode: the bridge's own service name, with its default.
    return (current_server_id() or app_vars.SERVICE_NAME or "").strip()


def credential_kind() -> str:
    from app import vars as app_vars
    from app.multi_server import (
        current_auth_provider,
        current_keycloak_provider_alias,
        current_setting,
    )

    if current_keycloak_provider_alias(app_vars.KEYCLOAK_PROVIDER_ALIAS):
        return "oauth"
    if current_auth_provider(app_vars.AUTH_PROVIDER) == "user-api-key":
        return "api_key"
    # A configured shared bearer (MCP_REMOTE_BEARER_TOKEN, or the per-server
    # anonymous one) is a service credential, never a personal connection.
    anon = current_setting("MCP_REMOTE_ANON_BEARER_TOKEN")
    if anon is None:
        anon = os.getenv("MCP_REMOTE_ANON_BEARER_TOKEN", "")
    shared = anon or os.getenv("MCP_REMOTE_BEARER_TOKEN", "")
    return "service_key" if shared else "public"


def _verified_subject(access_token: str) -> str:
    from app.oauth.token_exchange import UserLoggedOutException, verified_caller_claims

    try:
        subject = verified_caller_claims(access_token).get("sub")
    except UserLoggedOutException:
        raise CallerNotAuthorizedError(401, "Access token failed verification")
    if not isinstance(subject, str) or not subject:
        raise CallerNotAuthorizedError(401, "Access token has no subject")
    return subject


async def _decide(url: str, access_token: str, server: str) -> httpx.Response:
    async with httpx.AsyncClient(
        timeout=_TIMEOUT_SECONDS, follow_redirects=False
    ) as client:
        return await client.post(
            url,
            json={"server": server},
            headers={"Authorization": f"Bearer {access_token}"},
        )


async def enforce_connection_policy(
    access_token: Optional[str], *, listing: bool = False
) -> Optional[Attestation]:
    """Raise CallerNotAuthorizedError unless the central authority allows this caller.

    Returns None (no attestation) when no authority is configured, for a tool
    listing (``listing``) without a verifiable caller, and for a caller that
    cannot be verified on a connector that uses no personal credential
    (``public`` or ``service_key``): there is no person to attribute the call
    to, so there is nothing the authority could decide, and such a call has
    always been served (the platform gateway and the manager proxy are its
    boundary, and ``BRIDGE_REQUIRED_GROUPS`` above still applies). A connector
    that acts as a person (``oauth``, ``api_key``) always needs a verified one.
    """
    url = policy_url()
    if not url:
        return None
    server = server_id()
    if not server:
        raise CallerNotAuthorizedError(503, "Connection policy is unavailable")
    # Calls nobody can be named for are served unattested where no personal
    # credential is involved.
    unattributed_ok = listing or credential_kind() in ("public", "service_key")
    if not access_token:
        if unattributed_ok:
            return None
        raise CallerNotAuthorizedError(401, "Authentication required")
    try:
        subject = _verified_subject(access_token)
    except CallerNotAuthorizedError:
        # Tool metadata for an anonymous reader (the manager's tools crawl), or
        # a bearer that is not a platform token (a shared service key): served
        # as before, never attested.
        if unattributed_ok:
            return None
        raise
    key = _cache_key(url, access_token, server)
    attestation = _cached(key)
    if attestation is not None and attestation.subject == subject:
        attestation = Attestation(
            attestation.policy_version,
            attestation.connector_revision,
            subject,
            credential_kind(),
        )
        _record(attestation)
        return attestation
    try:
        response = await _decide(url, access_token, server)
    except httpx.HTTPError:
        logger.warning("[ConnectionPolicy] Authority unreachable for %s", server)
        raise CallerNotAuthorizedError(503, "Connection policy is unavailable")
    status = response.status_code
    if status == 409:
        raise CallerNotAuthorizedError(403, "Blocked by policy; request access")
    if status == 404:
        raise CallerNotAuthorizedError(404, "Not found")
    if status == 401:
        raise CallerNotAuthorizedError(401, "Access token failed verification")
    if status != 200 or len(response.content) > _MAX_BODY_BYTES:
        raise CallerNotAuthorizedError(503, "Connection policy is unavailable")
    try:
        decision = response.json()
    except ValueError:
        raise CallerNotAuthorizedError(503, "Connection policy is unavailable")
    version = decision.get("policy_version") if isinstance(decision, dict) else None
    revision = (
        decision.get("connector_revision") if isinstance(decision, dict) else None
    )
    if (
        decision.get("allowed") is not True
        or decision.get("server") != server
        or decision.get("subject") != subject
        or not isinstance(version, int)
        or isinstance(version, bool)
        or not isinstance(revision, str)
        or not revision
        or not revision.isascii()
        or not revision.isalnum()
    ):
        raise CallerNotAuthorizedError(503, "Connection policy is unavailable")
    attestation = Attestation(version, revision, subject, credential_kind())
    _remember(key, attestation)
    _record(attestation)
    return attestation


def _record(attestation: Attestation) -> None:
    holder = _current.get()
    if holder is not None:
        holder.attestation = attestation


class ConnectionAttestationMiddleware:
    """Adds the request's policy attestation to its response headers.

    Pure ASGI, so the holder set here is the one the endpoint (or a mounted
    SSE/MCP app) writes to. Downstream servers can never supply these headers.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        holder = _Holder()
        token = _current.set(holder)

        async def send_with_attestation(message):
            if message["type"] == "http.response.start":
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower().decode("latin-1") not in ATTESTATION_HEADERS
                ]
                if holder.attestation is not None:
                    headers.extend(holder.attestation.headers())
                message = dict(message, headers=headers)
            await send(message)

        try:
            await self.app(scope, receive, send_with_attestation)
        finally:
            _current.reset(token)
