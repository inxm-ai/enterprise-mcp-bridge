"""Callers from many issuers that share an address pattern (TRUSTED_ISSUERS).

With TRUSTED_ISSUERS set, the bridge accepts only a token whose issuer fits
a pattern (``*`` = one DNS label), signed with that issuer's keys, meant for
this bridge (``aud``) and requested by an allowed client (``azp``). Keys are
fetched only from the issuer's own address. Unset (default), nothing changes.

No test reaches the network: the key fetch is replaced by a fake that serves
key sets from memory and records every address asked for.
"""

import logging
import time
from contextlib import asynccontextmanager

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app import vars as app_vars
from app.oauth import trusted_issuers
from app.oauth.user_info import (
    CallerNotAuthorizedError,
    ensure_caller_in_required_groups,
)
from app.server import app as fastapi_app
from app.session_manager import session_context
from app.test_bridge_required_groups import FakeSession

TOKEN_HEADER = "X-Auth-Request-Access-Token"
PATTERNS = [
    "https://*.auth.example.com/realms/main",
    "https://*.login.example.org/realms/main",
]
TENANT_ISSUER = "https://acme.auth.example.com/realms/main"
TENANT_KEYS_URL = f"{TENANT_ISSUER}/protocol/openid-connect/certs"
OTHER_DOMAIN_ISSUER = "https://globex.login.example.org/realms/main"
AUDIENCE = "mcp-bridge"
ALLOWED_CLIENT = "token-exchange"
OTHER_CLIENT = "web-app"
KEY_ID = "key-1"
ROTATED_KEY_ID = "key-2"
TOKEN_LIFETIME_SECONDS = 300
LEEWAY_SECONDS = 60
FAILURE_SECONDS = 30
USER_ID = "user-42"


def _private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


SIGNING_KEY = _private_key()
ROTATED_KEY = _private_key()
OTHER_KEY = _private_key()


def _public_jwk(private_key, kid):
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    return {**jwk, "kid": kid, "use": "sig", "alg": "RS256"}


def _token(
    *,
    issuer=TENANT_ISSUER,
    audience=AUDIENCE,
    azp=ALLOWED_CLIENT,
    key=SIGNING_KEY,
    kid=KEY_ID,
    expires_in=TOKEN_LIFETIME_SECONDS,
):
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "azp": azp,
        "sub": USER_ID,
        "iat": now,
        "exp": now + expires_in,
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


class FakeKeyServer:
    """Serves key sets by address; ``None`` means the address can't be reached."""

    def __init__(self):
        self.key_sets = {}
        self.requested = []

    def publish(self, issuer, *jwks):
        self.key_sets[f"{issuer}/protocol/openid-connect/certs"] = {"keys": list(jwks)}

    def take_down(self, issuer):
        self.key_sets[f"{issuer}/protocol/openid-connect/certs"] = None

    def fetch(self, url):
        self.requested.append(url)
        return self.key_sets.get(url)


class FakeClock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def key_server(monkeypatch):
    server = FakeKeyServer()
    server.publish(TENANT_ISSUER, _public_jwk(SIGNING_KEY, KEY_ID))
    server.publish(OTHER_DOMAIN_ISSUER, _public_jwk(SIGNING_KEY, KEY_ID))
    monkeypatch.setattr(trusted_issuers, "_fetch_key_set", server.fetch)
    return server


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(trusted_issuers, "_monotonic", fake)
    return fake


@pytest.fixture
def trusted_issuers_on(monkeypatch, key_server, clock):
    monkeypatch.setattr(app_vars, "TRUSTED_ISSUERS", PATTERNS)
    monkeypatch.setattr(app_vars, "TRUSTED_ISSUER_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(app_vars, "TRUSTED_ISSUER_CLIENTS", [ALLOWED_CLIENT])
    monkeypatch.setattr(app_vars, "TRUSTED_ISSUER_CLOCK_LEEWAY_SECONDS", LEEWAY_SECONDS)
    monkeypatch.setattr(app_vars, "TRUSTED_ISSUER_KEY_FAILURE_SECONDS", FAILURE_SECONDS)
    monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", [])
    monkeypatch.setattr(app_vars, "BRIDGE_ALLOWED_USERS", [])
    trusted_issuers.forget_cached_keys()
    yield
    trusted_issuers.forget_cached_keys()


def _status(token):
    with pytest.raises(CallerNotAuthorizedError) as excinfo:
        ensure_caller_in_required_groups(token)
    return excinfo.value.status_code


class TestOffByDefault:
    def test_no_patterns_keeps_todays_behaviour(self, monkeypatch, key_server):
        monkeypatch.setattr(app_vars, "TRUSTED_ISSUERS", [])
        monkeypatch.setattr(app_vars, "BRIDGE_REQUIRED_GROUPS", [])
        monkeypatch.setattr(app_vars, "BRIDGE_ALLOWED_USERS", [])
        ensure_caller_in_required_groups(None)
        assert key_server.requested == []


class TestAccepts:
    def test_token_from_a_matching_issuer_passes(self, trusted_issuers_on, key_server):
        ensure_caller_in_required_groups(_token())
        assert key_server.requested == [TENANT_KEYS_URL]

    def test_every_pattern_is_trusted(self, trusted_issuers_on):
        ensure_caller_in_required_groups(_token(issuer=OTHER_DOMAIN_ISSUER))

    def test_audience_list_containing_ours_passes(self, trusted_issuers_on):
        ensure_caller_in_required_groups(_token(audience=[AUDIENCE, "other"]))

    def test_expiry_within_the_leeway_passes(self, trusted_issuers_on):
        ensure_caller_in_required_groups(_token(expires_in=-(LEEWAY_SECONDS - 10)))

    def test_keys_are_cached_per_issuer(self, trusted_issuers_on, key_server):
        ensure_caller_in_required_groups(_token())
        ensure_caller_in_required_groups(_token())
        assert key_server.requested == [TENANT_KEYS_URL]


class TestRefusesWith401:
    def test_missing_token(self, trusted_issuers_on):
        assert _status(None) == 401

    def test_not_a_token(self, trusted_issuers_on, key_server):
        assert _status("not-a-jwt") == 401
        assert key_server.requested == []

    @pytest.mark.parametrize(
        "issuer",
        [
            "https://acme.auth.example.net/realms/main",
            "https://a.b.auth.example.com/realms/main",
            "https://auth.example.com/realms/main",
            "http://acme.auth.example.com/realms/main",
            "https://acme.auth.example.com/realms/other",
            "https://acme.auth.example.com.evil.example/realms/main",
            "https://acme.auth.example.com/realms/main/extra",
            "https://acme.auth.example.com:8443/realms/main",
            "https://-acme.auth.example.com/realms/main",
            "https://ACME.auth.example.com/realms/main",
            "https://acme.authXexample.com/realms/main",
        ],
    )
    def test_issuer_fitting_no_pattern_is_refused_without_fetching(
        self, trusted_issuers_on, key_server, issuer
    ):
        assert _status(_token(issuer=issuer)) == 401
        assert key_server.requested == []

    def test_wrong_audience(self, trusted_issuers_on):
        assert _status(_token(audience=OTHER_CLIENT)) == 401

    def test_token_requested_by_another_client(self, trusted_issuers_on):
        assert _status(_token(azp=OTHER_CLIENT)) == 401

    def test_signature_from_another_key(self, trusted_issuers_on):
        assert _status(_token(key=OTHER_KEY)) == 401

    def test_expired_beyond_the_leeway(self, trusted_issuers_on):
        assert _status(_token(expires_in=-(LEEWAY_SECONDS + 10))) == 401

    def test_unknown_key_after_a_fresh_fetch(self, trusted_issuers_on):
        assert _status(_token(kid="no-such-key")) == 401


class TestKeyRotation:
    def test_new_key_is_fetched_once_before_deciding(
        self, trusted_issuers_on, key_server, clock
    ):
        ensure_caller_in_required_groups(_token())
        key_server.publish(
            TENANT_ISSUER,
            _public_jwk(SIGNING_KEY, KEY_ID),
            _public_jwk(ROTATED_KEY, ROTATED_KEY_ID),
        )
        clock.now += trusted_issuers.MIN_REFETCH_SECONDS + 1
        ensure_caller_in_required_groups(_token(key=ROTATED_KEY, kid=ROTATED_KEY_ID))
        assert key_server.requested == [TENANT_KEYS_URL, TENANT_KEYS_URL]

    def test_unknown_keys_cannot_force_a_fetch_per_request(
        self, trusted_issuers_on, key_server
    ):
        ensure_caller_in_required_groups(_token())
        for _ in range(5):
            assert _status(_token(kid="random")) == 401
        assert key_server.requested == [TENANT_KEYS_URL]

    def test_removed_key_stops_working_once_the_cache_expires(
        self, trusted_issuers_on, key_server, clock
    ):
        ensure_caller_in_required_groups(_token())
        key_server.publish(TENANT_ISSUER, _public_jwk(ROTATED_KEY, ROTATED_KEY_ID))
        clock.now += trusted_issuers.KEY_SET_LIFESPAN_SECONDS + 1
        assert _status(_token()) == 401


class TestIssuerTroubleIs503:
    def test_unreachable_keys_are_503_not_401(self, trusted_issuers_on, key_server):
        key_server.take_down(TENANT_ISSUER)
        assert _status(_token()) == 503

    def test_failure_is_remembered(self, trusted_issuers_on, key_server, clock):
        key_server.take_down(TENANT_ISSUER)
        assert _status(_token()) == 503
        assert _status(_token()) == 503
        assert key_server.requested == [TENANT_KEYS_URL]
        clock.now += FAILURE_SECONDS + 1
        key_server.publish(TENANT_ISSUER, _public_jwk(SIGNING_KEY, KEY_ID))
        ensure_caller_in_required_groups(_token())
        assert key_server.requested == [TENANT_KEYS_URL, TENANT_KEYS_URL]

    def test_stale_keys_are_not_used_when_the_refresh_fails(
        self, trusted_issuers_on, key_server, clock
    ):
        ensure_caller_in_required_groups(_token())
        key_server.take_down(TENANT_ISSUER)
        clock.now += trusted_issuers.KEY_SET_LIFESPAN_SECONDS + 1
        assert _status(_token()) == 503


class TestMisconfigurationRefusesEveryone:
    @pytest.mark.parametrize(
        "setting, value",
        [
            ("TRUSTED_ISSUER_AUDIENCE", ""),
            ("TRUSTED_ISSUER_CLIENTS", []),
            ("TRUSTED_ISSUERS", ["http://*.auth.example.com/realms/main"]),
        ],
    )
    def test_incomplete_or_unsafe_settings(
        self, trusted_issuers_on, monkeypatch, key_server, setting, value
    ):
        monkeypatch.setattr(app_vars, setting, value)
        assert _status(_token()) == 503
        assert key_server.requested == []


class TestCacheIsBounded:
    def test_many_issuers_do_not_grow_the_cache_without_limit(
        self, trusted_issuers_on, key_server, monkeypatch
    ):
        monkeypatch.setattr(trusted_issuers, "MAX_CACHED_ISSUERS", 2)
        for label in ("one", "two", "three"):
            issuer = f"https://{label}.auth.example.com/realms/main"
            key_server.take_down(issuer)
            assert _status(_token(issuer=issuer)) == 503
        assert trusted_issuers.cached_issuer_count() <= 2


class TestRefusalLog:
    def test_refusal_names_issuer_client_and_audience_but_not_the_user(
        self, trusted_issuers_on, caplog
    ):
        with caplog.at_level(logging.WARNING):
            _status(_token(azp=OTHER_CLIENT))
        logged = caplog.text
        assert TENANT_ISSUER in logged
        assert OTHER_CLIENT in logged
        assert AUDIENCE in logged
        assert USER_ID not in logged


@pytest.fixture
def fake_downstream(monkeypatch):
    fake = FakeSession()

    @asynccontextmanager
    async def fake_mcp_session(**_kwargs):
        yield fake

    monkeypatch.setattr(session_context, "mcp_session", fake_mcp_session)
    return fake


class TestRoutes:
    def test_anonymous_tool_call_is_refused_before_downstream(
        self, trusted_issuers_on, fake_downstream
    ):
        response = TestClient(fastapi_app).post("/tools/test_tool", json={})
        assert response.status_code == 401
        assert fake_downstream.calls == 0

    def test_listing_with_another_clients_token_is_refused(
        self, trusted_issuers_on, fake_downstream
    ):
        response = TestClient(fastapi_app).get(
            "/tools", headers={TOKEN_HEADER: _token(azp=OTHER_CLIENT)}
        )
        assert response.status_code == 401
        assert fake_downstream.calls == 0

    def test_trusted_token_reaches_the_tool(self, trusted_issuers_on, fake_downstream):
        response = TestClient(fastapi_app).post(
            "/tools/test_tool", json={}, headers={TOKEN_HEADER: _token()}
        )
        assert response.status_code == 200
        assert fake_downstream.calls > 0
