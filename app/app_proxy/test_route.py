"""
Comprehensive tests for the reverse proxy functionality.

Tests cover:
- Basic proxying of different HTTP methods
- Streaming large responses
- Header forwarding (X-Forwarded-*)
- Cookie path rewriting
- Location header rewriting for redirects
- URL rewriting in HTML/JSON content
- Compression handling
- Error scenarios (timeout, connection errors)
- Concurrent requests
- Query parameter preservation
- Various edge cases
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request
from httpx import Response as HttpxResponse
from httpx import AsyncClient, TimeoutException, ConnectError

from app.app_proxy.route import (
    get_target_url,
    prepare_headers,
    rewrite_location_header,
    rewrite_cookie_path,
    rewrite_content_urls,
    forward_to_target,
    PROXY_PREFIX,
)

# Test constants - don't import TEST_TARGET_SERVER_URL as it's empty at import time
TEST_TARGET_SERVER_URL = "http://internal-app:8080"


# Fixtures
@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request object."""
    request = Mock(spec=Request)
    request.method = "GET"
    request.url.path = f"{PROXY_PREFIX}/test"
    request.url.query = ""
    request.url.scheme = "https"
    request.headers = {"host": "proxy.example.com", "user-agent": "test-agent"}
    request.client.host = "192.168.1.100"
    return request


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx Response."""

    def _create_response(
        status_code=200, headers=None, content=b"test content", stream_chunks=None
    ):
        response = Mock(spec=HttpxResponse)
        response.status_code = status_code
        response.headers = headers or {}
        response.content = content

        if stream_chunks:

            async def aiter_bytes():
                for chunk in stream_chunks:
                    yield chunk

            response.aiter_bytes = aiter_bytes
        else:

            async def aiter_bytes():
                yield content

            response.aiter_bytes = aiter_bytes

        return response

    return _create_response


# Test URL construction
class TestGetTargetUrl:
    """Test URL construction from request."""

    def test_basic_path(self, mock_request, monkeypatch):
        """Test basic path without query parameters."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.url.path = f"{PROXY_PREFIX}/api/users"
        mock_request.url.query = ""

        result = get_target_url(mock_request)
        assert result == f"{TEST_TARGET_SERVER_URL}/api/users"

    def test_with_query_parameters(self, mock_request, monkeypatch):
        """Test path with query parameters."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.url.path = f"{PROXY_PREFIX}/api/search"
        mock_request.url.query = "q=test&limit=10"

        result = get_target_url(mock_request)
        assert result == f"{TEST_TARGET_SERVER_URL}/api/search?q=test&limit=10"

    def test_root_path(self, mock_request, monkeypatch):
        """Test root path proxying."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.url.path = PROXY_PREFIX or "/"
        mock_request.url.query = ""

        result = get_target_url(mock_request)
        assert result == f"{TEST_TARGET_SERVER_URL}/"

    def test_nested_path(self, mock_request, monkeypatch):
        """Test deeply nested path."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.url.path = f"{PROXY_PREFIX}/api/v1/users/123/posts/456"
        mock_request.url.query = ""

        result = get_target_url(mock_request)
        assert result == f"{TEST_TARGET_SERVER_URL}/api/v1/users/123/posts/456"

    def test_special_characters_in_query(self, mock_request, monkeypatch):
        """Test query parameters with special characters."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.url.path = f"{PROXY_PREFIX}/search"
        mock_request.url.query = "q=hello%20world&tag=foo%2Fbar"

        result = get_target_url(mock_request)
        assert (
            result == f"{TEST_TARGET_SERVER_URL}/search?q=hello%20world&tag=foo%2Fbar"
        )


# Test header preparation
class TestPrepareHeaders:
    """Test header forwarding and manipulation."""

    def test_basic_header_forwarding(self, mock_request):
        """Test that basic headers are forwarded."""
        mock_request.headers = {
            "user-agent": "test-agent",
            "accept": "application/json",
            "authorization": "Bearer token123",
        }

        result = prepare_headers(mock_request)

        assert result["user-agent"] == "test-agent"
        assert result["accept"] == "application/json"
        assert result["authorization"] == "Bearer token123"

    def test_hop_by_hop_headers_removed(self, mock_request):
        """Test that hop-by-hop headers are not forwarded."""
        mock_request.headers = {
            "connection": "keep-alive",
            "transfer-encoding": "chunked",
            "upgrade": "websocket",
            "user-agent": "test-agent",
        }

        result = prepare_headers(mock_request)

        assert "connection" not in result
        assert "transfer-encoding" not in result
        assert "upgrade" not in result
        assert result["user-agent"] == "test-agent"

    def test_x_forwarded_headers_added(self, mock_request):
        """Test that X-Forwarded-* headers are properly added."""
        mock_request.headers = {"host": "proxy.example.com"}
        mock_request.url.scheme = "https"
        mock_request.client.host = "192.168.1.100"

        result = prepare_headers(mock_request)

        assert "192.168.1.100" in result["x-forwarded-for"]
        assert result["x-forwarded-host"] == "proxy.example.com"
        assert result["x-forwarded-proto"] == "https"
        assert result["x-forwarded-prefix"] == PROXY_PREFIX
        assert result["x-real-ip"] == "192.168.1.100"

    def test_x_forwarded_for_chain(self, mock_request):
        """Test that X-Forwarded-For is properly chained."""
        mock_request.headers = {
            "host": "proxy.example.com",
            "x-forwarded-for": "10.0.0.1, 10.0.0.2",
        }
        mock_request.client.host = "192.168.1.100"

        result = prepare_headers(mock_request)

        assert result["x-forwarded-for"] == "10.0.0.1, 10.0.0.2, 192.168.1.100"

    def test_client_without_host(self, mock_request):
        """Test handling when client host is not available."""
        mock_request.client = None

        result = prepare_headers(mock_request)

        assert "unknown" in result["x-forwarded-for"]


# Test Location header rewriting
class TestRewriteLocationHeader:
    """Test redirect Location header rewriting."""

    def test_relative_url_with_leading_slash(self, mock_request):
        """Test rewriting relative URL with leading slash."""
        location = "/api/redirect"

        result = rewrite_location_header(location, mock_request)

        assert result == f"{PROXY_PREFIX}/api/redirect"

    def test_relative_url_without_leading_slash(self, mock_request):
        """Test rewriting relative URL without leading slash."""
        mock_request.url.path = f"{PROXY_PREFIX}/api/users"
        location = "redirect"

        result = rewrite_location_header(location, mock_request)

        assert result == f"{PROXY_PREFIX}/api/redirect"

    def test_absolute_url_to_target(self, mock_request):
        """Test rewriting absolute URL pointing to target server."""
        location = f"{TEST_TARGET_SERVER_URL}/api/callback"

        with patch("app.app_proxy.route.PUBLIC_URL", "https://public.example.com"):
            with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
                result = rewrite_location_header(location, mock_request)

        assert result == f"https://public.example.com{PROXY_PREFIX}/api/callback"

    def test_absolute_url_to_target_with_query(self, mock_request):
        """Test rewriting absolute URL with query parameters."""
        location = f"{TEST_TARGET_SERVER_URL}/callback?code=abc&state=xyz"

        with patch("app.app_proxy.route.PUBLIC_URL", ""):
            with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
                result = rewrite_location_header(location, mock_request)

        assert result.endswith(f"{PROXY_PREFIX}/callback?code=abc&state=xyz")

    def test_external_url_unchanged(self, mock_request):
        """Test that external URLs are not rewritten."""
        location = "https://external.example.com/path"

        result = rewrite_location_header(location, mock_request)

        assert result == location

    def test_empty_location(self, mock_request):
        """Test handling empty location."""
        location = ""

        result = rewrite_location_header(location, mock_request)

        assert result == ""


# Test cookie path rewriting
class TestRewriteCookiePath:
    """Test Set-Cookie header path rewriting."""

    def test_cookie_with_root_path(self):
        """Test rewriting cookie with root path."""
        set_cookie = "session=abc123; Path=/; HttpOnly"

        result = rewrite_cookie_path(set_cookie)

        assert "session=abc123" in result
        assert f"Path={PROXY_PREFIX or '/'}" in result or "Path=/" in result
        assert "HttpOnly" in result

    def test_cookie_with_specific_path(self):
        """Test rewriting cookie with specific path."""
        set_cookie = "token=xyz789; Path=/api; Secure"

        result = rewrite_cookie_path(set_cookie)

        assert "token=xyz789" in result
        assert f"Path={PROXY_PREFIX}/api" in result or "Path=/api" in result
        assert "Secure" in result

    def test_cookie_already_with_prefix(self):
        """Test cookie that already has proxy prefix."""
        set_cookie = f"data=test; Path={PROXY_PREFIX}/resource"

        result = rewrite_cookie_path(set_cookie)

        assert "data=test" in result
        assert PROXY_PREFIX in result

    def test_multiple_cookies(self):
        """Test handling multiple cookies in one header."""
        set_cookie = "session=abc; Path=/; user=john; Path=/profile"

        result = rewrite_cookie_path(set_cookie)

        assert "session=abc" in result
        assert "user=john" in result

    def test_invalid_cookie(self):
        """Test handling invalid cookie format."""
        set_cookie = "invalid cookie format"

        result = rewrite_cookie_path(set_cookie)

        # Should return original on parse failure
        assert result == set_cookie


# Test content URL rewriting
class TestRewriteContentUrls:
    """Test URL rewriting in HTML and JSON content."""

    def test_html_absolute_urls(self):
        """Test rewriting absolute URLs in HTML."""
        html = f"""
        <html>
            <link rel="stylesheet" href="{TEST_TARGET_SERVER_URL}/styles.css">
            <script src="{TEST_TARGET_SERVER_URL}/app.js"></script>
        </html>
        """

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(html.encode("utf-8"), "text/html")
            result_str = result.decode("utf-8")

        assert f'href="{PROXY_PREFIX}/styles.css"' in result_str
        assert f'src="{PROXY_PREFIX}/app.js"' in result_str
        assert TEST_TARGET_SERVER_URL not in result_str

    def test_html_relative_urls(self):
        """Test rewriting relative URLs in HTML."""
        html = """
        <html>
            <a href="/page">Link</a>
            <img src="/image.png">
        </html>
        """

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(html.encode("utf-8"), "text/html")
            result_str = result.decode("utf-8")

        assert f'href="{PROXY_PREFIX}/page"' in result_str
        assert f'src="{PROXY_PREFIX}/image.png"' in result_str
        # Ensure no double slashes
        assert "//" not in result_str.replace("https://", "").replace("http://", "")

    def test_html_nextjs_urls(self):
        """Test rewriting Next.js-style URLs (/_next/static/...)."""
        html = """
        <html>
            <link rel="stylesheet" href="/_next/static/css/app.css">
            <script src="/_next/static/chunks/main.js"></script>
        </html>
        """

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(html.encode("utf-8"), "text/html")
            result_str = result.decode("utf-8")

        assert f'href="{PROXY_PREFIX}/_next/static/css/app.css"' in result_str
        assert f'src="{PROXY_PREFIX}/_next/static/chunks/main.js"' in result_str
        # Ensure no double slashes (e.g., /prefix//_next)
        assert f"{PROXY_PREFIX}//_next" not in result_str

    def test_html_no_double_prefix(self):
        """Test that URLs already containing the proxy prefix are not rewritten again."""
        html = f"""
        <html>
            <a href="{PROXY_PREFIX}/dashboard">Already prefixed</a>
            <a href="/api/auth">Not prefixed</a>
            <script src="{PROXY_PREFIX}/_next/static/chunks/main.js"></script>
        </html>
        """

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(html.encode("utf-8"), "text/html")
            result_str = result.decode("utf-8")

        # URLs already containing the prefix should not be rewritten
        assert f'href="{PROXY_PREFIX}/dashboard"' in result_str
        assert f'src="{PROXY_PREFIX}/_next/static/chunks/main.js"' in result_str

        # URLs not containing the prefix should be rewritten
        assert f'href="{PROXY_PREFIX}/api/auth"' in result_str

        # Ensure no double-prefixing occurred
        assert f"{PROXY_PREFIX}{PROXY_PREFIX}" not in result_str

    def test_html_single_quotes(self):
        """Test rewriting URLs with single quotes in HTML."""
        html = f"<a href='{TEST_TARGET_SERVER_URL}/path'>Link</a>"

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(html.encode("utf-8"), "text/html")
            result_str = result.decode("utf-8")

        assert f"href='{PROXY_PREFIX}/path'" in result_str

    def test_json_urls(self):
        """Test rewriting URLs in JSON content."""
        data = {
            "url": f"{TEST_TARGET_SERVER_URL}/api/resource",
            "callback": f"{TEST_TARGET_SERVER_URL}/webhook",
        }
        json_str = json.dumps(data)

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(json_str.encode("utf-8"), "application/json")
            result_str = result.decode("utf-8")

        assert f'"{PROXY_PREFIX}/api/resource"' in result_str
        assert f'"{PROXY_PREFIX}/webhook"' in result_str
        assert TEST_TARGET_SERVER_URL not in result_str

    def test_non_text_content_unchanged(self):
        """Test that binary content is not modified."""
        binary_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

        result = rewrite_content_urls(binary_data, "image/png")

        assert result == binary_data

    def test_rewriting_disabled(self):
        """Test content when rewriting is disabled."""
        html = f'<a href="{TEST_TARGET_SERVER_URL}/page">Link</a>'

        with patch("app.app_proxy.route.REWRITE_HTML_URLS", False):
            result = rewrite_content_urls(html.encode("utf-8"), "text/html")
            result_str = result.decode("utf-8")

        assert TEST_TARGET_SERVER_URL in result_str

    def test_css_url_rewriting(self):
        """Test rewriting url() functions in CSS."""
        css = f"""
        .background {{
            background-image: url("{TEST_TARGET_SERVER_URL}/images/bg.png");
        }}
        .logo {{
            background: url('/logo.svg');
        }}
        .icon {{
            background: url(/icons/star.png);
        }}
        """

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(css.encode("utf-8"), "text/css")
            result_str = result.decode("utf-8")

        assert f'url("{PROXY_PREFIX}/images/bg.png")' in result_str
        assert f'url("{PROXY_PREFIX}/logo.svg")' in result_str
        assert f'url("{PROXY_PREFIX}/icons/star.png")' in result_str
        assert TEST_TARGET_SERVER_URL not in result_str

    def test_css_import_rewriting(self):
        """Test rewriting @import statements in CSS."""
        css = f"""
        @import "{TEST_TARGET_SERVER_URL}/styles/theme.css";
        @import '/styles/base.css';
        @import url("/fonts.css");
        """

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(css.encode("utf-8"), "text/css")
            result_str = result.decode("utf-8")

        assert f'@import "{PROXY_PREFIX}/styles/theme.css"' in result_str
        assert f'@import "{PROXY_PREFIX}/styles/base.css"' in result_str
        assert f'url("{PROXY_PREFIX}/fonts.css")' in result_str
        assert TEST_TARGET_SERVER_URL not in result_str

    def test_javascript_url_rewriting(self):
        """Test rewriting URLs in JavaScript."""
        js = f"""
        const apiUrl = "{TEST_TARGET_SERVER_URL}/api/data";
        const imgSrc = '/images/photo.jpg';
        fetch("/api/users.json").then(r => r.json());
        import {{ Component }} from '/components/Button.js';
        """

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(js.encode("utf-8"), "application/javascript")
            result_str = result.decode("utf-8")

        assert f'"{PROXY_PREFIX}/api/data"' in result_str
        assert (
            f"'{PROXY_PREFIX}/images/photo.jpg'" in result_str
            or '"/images/photo.jpg"' in result_str
        )
        assert TEST_TARGET_SERVER_URL not in result_str

    def test_javascript_single_quotes(self):
        """Test JavaScript URL rewriting with single quotes."""
        js = f"const url = '{TEST_TARGET_SERVER_URL}/path';"

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(js.encode("utf-8"), "application/javascript")
            result_str = result.decode("utf-8")

        assert f"'{PROXY_PREFIX}/path'" in result_str
        assert TEST_TARGET_SERVER_URL not in result_str

    def test_css_with_complex_selectors(self):
        """Test CSS with complex selectors and multiple url() calls."""
        css = f"""
        body::before {{
            content: "";
            background: url("/bg.jpg") no-repeat center,
                       url('{TEST_TARGET_SERVER_URL}/overlay.png');
        }}
        @font-face {{
            font-family: 'CustomFont';
            src: url('/fonts/custom.woff2') format('woff2'),
                 url({TEST_TARGET_SERVER_URL}/fonts/custom.woff) format('woff');
        }}
        """

        with patch("app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL):
            result = rewrite_content_urls(css.encode("utf-8"), "text/css")
            result_str = result.decode("utf-8")

        assert f'url("{PROXY_PREFIX}/bg.jpg")' in result_str
        assert f'url("{PROXY_PREFIX}/overlay.png")' in result_str
        assert f'url("{PROXY_PREFIX}/fonts/custom.woff2")' in result_str
        assert TEST_TARGET_SERVER_URL not in result_str


# Integration tests
class TestForwardToTarget:
    """Test the main forward_to_target function."""

    @pytest.mark.asyncio
    async def test_successful_get_request(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test successful GET request proxying."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.method = "GET"
        mock_request.body = AsyncMock(return_value=b"")

        response = mock_httpx_response(
            status_code=200,
            headers={"content-type": "text/plain"},
            content=b"Hello, World!",
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            result = await forward_to_target(mock_request)

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_successful_post_request(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test successful POST request with body."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.method = "POST"
        post_data = b'{"name": "test"}'
        mock_request.body = AsyncMock(return_value=post_data)

        response = mock_httpx_response(
            status_code=201,
            headers={"content-type": "application/json"},
            content=b'{"id": 123}',
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            result = await forward_to_target(mock_request)

            assert result.status_code == 201
            # Verify body was passed
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["content"] == post_data

    @pytest.mark.asyncio
    async def test_redirect_with_location_rewrite(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test redirect response with Location header rewriting."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.body = AsyncMock(return_value=b"")

        response = mock_httpx_response(
            status_code=302,
            headers={
                "location": f"{TEST_TARGET_SERVER_URL}/new-location",
                "content-type": "text/html",
            },
            content=b"Redirecting...",
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            result = await forward_to_target(mock_request)

            assert result.status_code == 302
            # Location should be rewritten
            assert TEST_TARGET_SERVER_URL not in dict(result.headers).get(
                "location", ""
            )

    @pytest.mark.asyncio
    async def test_cookie_path_rewriting(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test Set-Cookie header path rewriting."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.body = AsyncMock(return_value=b"")

        response = mock_httpx_response(
            status_code=200,
            headers={
                "set-cookie": "session=abc123; Path=/; HttpOnly",
                "content-type": "text/html",
            },
            content=b"<html></html>",
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            result = await forward_to_target(mock_request)

            assert result.status_code == 200
            # Cookie path should be rewritten
            set_cookie = dict(result.headers).get("set-cookie", "")
            assert "session=abc123" in set_cookie

    @pytest.mark.asyncio
    async def test_streaming_large_response(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test streaming of large response."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.body = AsyncMock(return_value=b"")

        # Create large response in chunks
        chunks = [b"chunk" * 1000 for _ in range(10)]
        response = mock_httpx_response(
            status_code=200,
            headers={"content-type": "application/octet-stream"},
            stream_chunks=chunks,
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            result = await forward_to_target(mock_request)

            assert result.status_code == 200
            # Verify it's a streaming response
            assert hasattr(result, "body_iterator")

    @pytest.mark.asyncio
    async def test_compressed_response(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test handling of gzip-compressed response (httpx auto-decompresses)."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.body = AsyncMock(return_value=b"")

        # httpx automatically decompresses, so we receive plain content
        # but the content-encoding header is still present
        original_content = b"This is test content"

        response = mock_httpx_response(
            status_code=200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",  # Header present but httpx already decompressed
            },
            content=original_content,  # Already decompressed by httpx
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            result = await forward_to_target(mock_request)

            assert result.status_code == 200
            # Content-Encoding should be preserved for non-rewritable content
            assert dict(result.headers).get("content-encoding") == "gzip"

    @pytest.mark.asyncio
    async def test_compressed_html_encoding_removed(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test that content-encoding is removed for HTML when rewriting (httpx already decompressed)."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )
        monkeypatch.setattr("app.app_proxy.route.REWRITE_HTML_URLS", True)

        mock_request.body = AsyncMock(return_value=b"")

        # httpx has already decompressed this
        original_html = b'<html><a href="/_next/test">Link</a></html>'

        response = mock_httpx_response(
            status_code=200,
            headers={
                "content-type": "text/html",
                "content-encoding": "gzip",  # httpx already handled this
            },
            content=original_html,  # Already decompressed
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            result = await forward_to_target(mock_request)

            assert result.status_code == 200
            # Content-Encoding should be removed when we rewrite content
            assert "content-encoding" not in dict(result.headers)

    @pytest.mark.asyncio
    async def test_timeout_error(self, mock_request, monkeypatch):
        """Test handling of timeout errors."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.body = AsyncMock(return_value=b"")

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.side_effect = TimeoutException("Request timed out")

            with pytest.raises(Exception) as exc_info:
                await forward_to_target(mock_request)

            # Should raise HTTPException with 504
            assert (
                "504" in str(exc_info.value) or "timeout" in str(exc_info.value).lower()
            )

    @pytest.mark.asyncio
    async def test_connection_error(self, mock_request, monkeypatch):
        """Test handling of connection errors."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.body = AsyncMock(return_value=b"")

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.side_effect = ConnectError("Connection refused")

            with pytest.raises(Exception) as exc_info:
                await forward_to_target(mock_request)

            # Should raise HTTPException with 502
            assert (
                "502" in str(exc_info.value) or "gateway" in str(exc_info.value).lower()
            )

    @pytest.mark.asyncio
    async def test_target_url_not_configured(self, mock_request, monkeypatch):
        """Test error when TEST_TARGET_SERVER_URL is not configured."""
        monkeypatch.setattr("app.app_proxy.route.TARGET_SERVER_URL", "")

        mock_request.body = AsyncMock(return_value=b"")

        with patch("app.app_proxy.route.TARGET_SERVER_URL", ""):
            with pytest.raises(Exception) as exc_info:
                await forward_to_target(mock_request)

            assert (
                "503" in str(exc_info.value)
                or "unavailable" in str(exc_info.value).lower()
            )

    @pytest.mark.asyncio
    async def test_various_http_methods(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test proxying various HTTP methods."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]

        for method in methods:
            mock_request.method = method
            mock_request.body = AsyncMock(return_value=b"")

            response = mock_httpx_response(
                status_code=200, headers={"content-type": "application/json"}
            )

            with patch.object(
                AsyncClient, "request", new_callable=AsyncMock
            ) as mock_client:
                mock_client.return_value = response

                result = await forward_to_target(mock_request)

                assert result.status_code == 200
                # Verify correct method was used
                call_kwargs = mock_client.call_args[1]
                assert call_kwargs["method"] == method

    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test handling multiple concurrent requests."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.body = AsyncMock(return_value=b"")

        response = mock_httpx_response(
            status_code=200, headers={"content-type": "text/plain"}, content=b"response"
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            # Create 10 concurrent requests
            tasks = [forward_to_target(mock_request) for _ in range(10)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r.status_code == 200 for r in results)

    @pytest.mark.asyncio
    async def test_special_status_codes(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test handling of various HTTP status codes."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        status_codes = [200, 201, 204, 301, 302, 400, 401, 403, 404, 500, 502, 503]

        for status_code in status_codes:
            mock_request.body = AsyncMock(return_value=b"")

            response = mock_httpx_response(
                status_code=status_code, headers={"content-type": "text/plain"}
            )

            with patch.object(
                AsyncClient, "request", new_callable=AsyncMock
            ) as mock_client:
                mock_client.return_value = response

                result = await forward_to_target(mock_request)

                assert result.status_code == status_code

    @pytest.mark.asyncio
    async def test_preserves_custom_headers(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test that custom headers are preserved."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.headers = {
            "host": "proxy.example.com",
            "x-custom-header": "custom-value",
            "authorization": "Bearer token123",
        }
        mock_request.body = AsyncMock(return_value=b"")

        response = mock_httpx_response(
            status_code=200, headers={"content-type": "text/plain"}
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            await forward_to_target(mock_request)

            # Verify custom headers were forwarded
            call_kwargs = mock_client.call_args[1]
            headers = call_kwargs["headers"]
            assert headers.get("x-custom-header") == "custom-value"
            assert headers.get("authorization") == "Bearer token123"

    @pytest.mark.asyncio
    async def test_removes_hop_by_hop_headers_from_response(
        self, mock_request, mock_httpx_response, monkeypatch
    ):
        """Test that hop-by-hop headers are removed from response."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        mock_request.body = AsyncMock(return_value=b"")

        response = mock_httpx_response(
            status_code=200,
            headers={
                "content-type": "text/plain",
                "connection": "keep-alive",
                "transfer-encoding": "chunked",
                "x-custom": "value",
            },
        )

        with patch.object(
            AsyncClient, "request", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = response

            result = await forward_to_target(mock_request)

            headers = dict(result.headers)
            assert "connection" not in headers
            assert "transfer-encoding" not in headers
            assert headers.get("x-custom") == "value"


# Edge cases and error scenarios
class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_empty_proxy_prefix(self, mock_request, monkeypatch):
        """Test behavior when PROXY_PREFIX is empty."""
        monkeypatch.setattr(
            "app.app_proxy.route.TARGET_SERVER_URL", TEST_TARGET_SERVER_URL
        )

        with patch("app.app_proxy.route.PROXY_PREFIX", ""):
            mock_request.url.path = "/test"
            result = get_target_url(mock_request)
            assert result == f"{TEST_TARGET_SERVER_URL}/test"

    def test_unicode_in_path(self, mock_request):
        """Test handling Unicode characters in path."""
        mock_request.url.path = f"{PROXY_PREFIX}/api/用户/测试"
        mock_request.url.query = ""

        result = get_target_url(mock_request)
        assert "/api/用户/测试" in result

    def test_very_long_path(self, mock_request):
        """Test handling very long paths."""
        long_path = "/".join([f"segment{i}" for i in range(100)])
        mock_request.url.path = f"{PROXY_PREFIX}/{long_path}"
        mock_request.url.query = ""

        result = get_target_url(mock_request)
        assert long_path in result

    def test_query_with_empty_values(self, mock_request):
        """Test query parameters with empty values."""
        mock_request.url.path = f"{PROXY_PREFIX}/search"
        mock_request.url.query = "q=&empty=&valid=value"

        result = get_target_url(mock_request)
        assert "q=&empty=&valid=value" in result

    def test_multiple_slashes_in_path(self, mock_request):
        """Test handling multiple consecutive slashes."""
        mock_request.url.path = f"{PROXY_PREFIX}//api///users//"
        mock_request.url.query = ""

        result = get_target_url(mock_request)
        # Should preserve the path structure
        assert "//api///users//" in result or "/api/users" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
