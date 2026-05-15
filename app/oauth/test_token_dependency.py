"""
Unit tests for app.oauth.token_dependency.get_access_token.

Scenarios
---------
* TOKEN_SOURCE=cookie, cookie present          → returns cookie value
* TOKEN_SOURCE=cookie, cookie absent           → falls back to TOKEN_NAME header
* TOKEN_SOURCE=cookie, cookie absent, no hdr  → returns None
* TOKEN_SOURCE=header (default)               → returns header value, ignores cookie
"""

import pytest
from starlette.requests import Request

import app.oauth.token_dependency as td_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(cookie_header: str = "") -> Request:
    """Build a minimal Starlette Request with an optional Cookie header."""
    headers = []
    if cookie_header:
        headers.append((b"cookie", cookie_header.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "query_string": b"",
    }
    return Request(scope)


# ---------------------------------------------------------------------------
# Tests – TOKEN_SOURCE=cookie
# ---------------------------------------------------------------------------


class TestCookieSource:
    @pytest.mark.asyncio
    async def test_cookie_present_returns_cookie(self, monkeypatch):
        monkeypatch.setattr(td_module, "TOKEN_SOURCE", "cookie")
        monkeypatch.setattr(td_module, "TOKEN_COOKIE_NAME", "my_cookie")

        request = _make_request("my_cookie=cookie-tok")
        token = await td_module.get_access_token(request, header_token="hdr-tok")

        assert token == "cookie-tok"

    @pytest.mark.asyncio
    async def test_cookie_absent_falls_back_to_header(self, monkeypatch, caplog):
        """Core new behaviour: missing cookie → use injected header instead of None."""
        monkeypatch.setattr(td_module, "TOKEN_SOURCE", "cookie")
        monkeypatch.setattr(td_module, "TOKEN_COOKIE_NAME", "my_cookie")
        monkeypatch.setattr(td_module, "TOKEN_NAME", "x-test-token")
        caplog.set_level("DEBUG", logger="uvicorn.error")

        request = _make_request()  # no cookie at all
        token = await td_module.get_access_token(request, header_token="hdr-tok")

        assert token == "hdr-tok"
        assert (
            "Cookie my_cookie not found; falling back to x-test-token header"
            in caplog.text
        )

    @pytest.mark.asyncio
    async def test_cookie_absent_no_header_returns_none(self, monkeypatch, caplog):
        monkeypatch.setattr(td_module, "TOKEN_SOURCE", "cookie")
        monkeypatch.setattr(td_module, "TOKEN_COOKIE_NAME", "my_cookie")
        monkeypatch.setattr(td_module, "TOKEN_NAME", "x-test-token")
        caplog.set_level("WARNING", logger="uvicorn.error")

        request = _make_request()  # no cookie, no header
        token = await td_module.get_access_token(request, header_token=None)

        assert token is None
        assert (
            "Cookie my_cookie not found and x-test-token header is also missing"
            in caplog.text
        )

    @pytest.mark.asyncio
    async def test_wrong_cookie_name_is_not_used(self, monkeypatch):
        """A different cookie present should not satisfy TOKEN_COOKIE_NAME."""
        monkeypatch.setattr(td_module, "TOKEN_SOURCE", "cookie")
        monkeypatch.setattr(td_module, "TOKEN_COOKIE_NAME", "my_cookie")

        request = _make_request("other_cookie=other-tok")
        token = await td_module.get_access_token(request, header_token="hdr-tok")

        # my_cookie absent → falls back to header
        assert token == "hdr-tok"


# ---------------------------------------------------------------------------
# Tests – TOKEN_SOURCE=header (default)
# ---------------------------------------------------------------------------


class TestHeaderSource:
    @pytest.mark.asyncio
    async def test_header_source_returns_header(self, monkeypatch):
        monkeypatch.setattr(td_module, "TOKEN_SOURCE", "header")

        request = _make_request(
            "_oauth2_proxy=cookie-tok"
        )  # cookie present but ignored
        token = await td_module.get_access_token(request, header_token="hdr-tok")

        assert token == "hdr-tok"

    @pytest.mark.asyncio
    async def test_header_source_no_header_returns_none(self, monkeypatch):
        monkeypatch.setattr(td_module, "TOKEN_SOURCE", "header")

        request = _make_request()
        token = await td_module.get_access_token(request, header_token=None)

        assert token is None
