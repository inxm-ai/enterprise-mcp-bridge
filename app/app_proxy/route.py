import logging
import os
import re
from typing import Optional, Dict, AsyncIterator
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from http.cookies import SimpleCookie

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from opentelemetry import trace

from app.vars import MCP_BASE_PATH

# Initialize components
router = APIRouter(prefix="/app")
tracer = trace.get_tracer(__name__)
logger = logging.getLogger("uvicorn.error")

# Configuration from environment
PROXY_PREFIX = os.environ.get("PROXY_PREFIX", MCP_BASE_PATH + "/app")
TARGET_SERVER_URL = os.environ.get("TARGET_SERVER_URL", "").rstrip("/")
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")  # Public-facing URL for rewrites
PROXY_TIMEOUT = int(os.environ.get("PROXY_TIMEOUT", "300"))  # 5 minutes default
REWRITE_HTML_URLS = os.environ.get("REWRITE_HTML_URLS", "true").lower() == "true"
REWRITE_JSON_URLS = os.environ.get("REWRITE_JSON_URLS", "true").lower() == "true"
REWRITE_CSS_URLS = os.environ.get("REWRITE_CSS_URLS", "true").lower() == "true"
REWRITE_JS_URLS = os.environ.get("REWRITE_JS_URLS", "true").lower() == "true"

# Hop-by-hop headers that should NOT be forwarded (RFC 2616)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

# Headers that identify the original request for the backend
FORWARDED_HEADERS = {
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-forwarded-prefix",
    "x-real-ip",
    "x-scheme",
}


def get_target_url(request: Request) -> str:
    """Construct the target URL from the request path."""
    # Remove the proxy prefix from the path
    path = request.url.path
    if path.startswith(PROXY_PREFIX):
        path = path[len(PROXY_PREFIX):]
    
    # Ensure path starts with /
    if not path.startswith("/"):
        path = "/" + path
    
    # Preserve query parameters
    query_string = str(request.url.query)
    if query_string:
        path = f"{path}?{query_string}"
    
    return urljoin(TARGET_SERVER_URL + "/", path.lstrip("/"))


def prepare_headers(request: Request) -> Dict[str, str]:
    """
    Prepare headers for forwarding to target server.
    Removes hop-by-hop headers and adds proxy headers.
    """
    headers = {}
    
    # Copy headers, excluding hop-by-hop headers
    for name, value in request.headers.items():
        name_lower = name.lower()
        if name_lower not in HOP_BY_HOP_HEADERS:
            headers[name] = value
    
    # Add/update forwarded headers
    # X-Forwarded-For: append client IP
    client_ip = request.client.host if request.client else "unknown"
    existing_xff = headers.get("x-forwarded-for", "")
    headers["x-forwarded-for"] = f"{existing_xff}, {client_ip}".strip(", ")
    
    # X-Forwarded-Host: original host
    headers["x-forwarded-host"] = request.headers.get("host", "")
    
    # X-Forwarded-Proto: original scheme
    headers["x-forwarded-proto"] = request.url.scheme
    
    # X-Forwarded-Prefix: proxy prefix for apps that need to know their base path
    headers["x-forwarded-prefix"] = PROXY_PREFIX
    
    # X-Real-IP: client IP (for single client identification)
    headers["x-real-ip"] = client_ip
    
    return headers


def rewrite_location_header(location: str, request: Request) -> str:
    """
    Rewrite Location header from target server to proxy URL.
    Handles both absolute and relative URLs.
    """
    if not location:
        return location
    
    # Parse the location URL
    parsed = urlparse(location)
    
    # If it's a relative URL, make it absolute with proxy prefix
    if not parsed.scheme and not parsed.netloc:
        # Relative URL - prepend proxy prefix
        if location.startswith("/"):
            return f"{PROXY_PREFIX}{location}"
        else:
            # Relative to current path
            current_path = request.url.path
            if current_path.startswith(PROXY_PREFIX):
                current_path = current_path[len(PROXY_PREFIX):]
            base_path = "/".join(current_path.split("/")[:-1])
            return f"{PROXY_PREFIX}{base_path}/{location}"
    
    # If it's an absolute URL pointing to target server, rewrite to proxy
    if parsed.netloc == urlparse(TARGET_SERVER_URL).netloc:
        # Replace target server URL with proxy URL
        path = parsed.path
        query = f"?{parsed.query}" if parsed.query else ""
        fragment = f"#{parsed.fragment}" if parsed.fragment else ""
        
        # Use PUBLIC_URL if set, otherwise construct from request
        if PUBLIC_URL:
            base = PUBLIC_URL
        else:
            base = f"{request.url.scheme}://{request.headers.get('host', request.client.host)}"
        
        return f"{base}{PROXY_PREFIX}{path}{query}{fragment}"
    
    # Otherwise, return as-is (external redirect)
    return location


def rewrite_cookie_path(set_cookie: str) -> str:
    """
    Rewrite the Path attribute in Set-Cookie header to use proxy prefix.
    """
    cookie = SimpleCookie()
    try:
        cookie.load(set_cookie)
    except Exception as e:
        logger.warning(f"Failed to parse cookie: {set_cookie}, error: {e}")
        return set_cookie
    
    # Rewrite each cookie's path
    for morsel in cookie.values():
        path = morsel.get("path", "/")
        # If path is /, set it to proxy prefix
        if path == "/":
            morsel["path"] = PROXY_PREFIX or "/"
        # If path doesn't start with proxy prefix, prepend it
        elif not path.startswith(PROXY_PREFIX):
            morsel["path"] = f"{PROXY_PREFIX}{path}"
    
    # Return the rewritten Set-Cookie header
    result = []
    for morsel in cookie.values():
        result.append(morsel.OutputString())
    return "; ".join(result) if result else set_cookie


def rewrite_content_urls(content: bytes, content_type: str) -> bytes:
    """
    Rewrite URLs in HTML/CSS/JS/JSON content to use proxy prefix.
    This is best-effort and may not catch all cases.
    """
    # Only rewrite if configured and content type is appropriate
    if not content:
        return content
    
    try:
        text = content.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        return content
    
    target_pattern = re.escape(TARGET_SERVER_URL)
    proxy_prefix_pattern = re.escape(PROXY_PREFIX) if PROXY_PREFIX else ""
    
    # HTML URL rewriting
    if REWRITE_HTML_URLS and "text/html" in content_type:
        # Rewrite href and src attributes pointing to target
        # Replace absolute URLs
        text = re.sub(
            f'(href|src)="({target_pattern})(/[^"]*)"',
            f'\\1="{PROXY_PREFIX}\\3"',
            text
        )
        # Replace absolute URLs with single quotes
        text = re.sub(
            f"(href|src)='({target_pattern})(/[^']*)'",
            f"\\1='{PROXY_PREFIX}\\3'",
            text
        )
        # Rewrite root-relative URLs (href="/path") but NOT if already prefixed
        if proxy_prefix_pattern:
            text = re.sub(
                f'(href|src)="((?!{proxy_prefix_pattern})/)([^"]*)"',
                f'\\1="{PROXY_PREFIX}/\\3"',
                text
            )
            text = re.sub(
                f"(href|src)='((?!{proxy_prefix_pattern})/)([^']*)'",
                f"\\1='{PROXY_PREFIX}/\\3'",
                text
            )
        else:
            text = re.sub(
                r'(href|src)="(/[^"]*)"',
                f'\\1="{PROXY_PREFIX}\\2"',
                text
            )
            text = re.sub(
                r"(href|src)='(/[^']*)'",
                f"\\1='{PROXY_PREFIX}\\2'",
                text
            )
    
    # CSS URL rewriting
    if REWRITE_CSS_URLS and ("text/css" in content_type or "stylesheet" in content_type):
        # Rewrite url() functions with absolute URLs
        text = re.sub(
            f'url\\(["\']?({target_pattern})(/[^)"\'\\ ]+)["\']?\\)',
            f'url("{PROXY_PREFIX}\\2")',
            text
        )
        # Rewrite url() functions with root-relative URLs (but not if already prefixed)
        if proxy_prefix_pattern:
            text = re.sub(
                f'url\\(["\']?((?!{proxy_prefix_pattern})/)([^)"\'\\ ]+)["\']?\\)',
                f'url("{PROXY_PREFIX}/\\2")',
                text
            )
        else:
            text = re.sub(
                r'url\(["\']?(/[^)"\'\\ ]+)["\']?\)',
                f'url("{PROXY_PREFIX}\\1")',
                text
            )
        # Rewrite @import statements with absolute URLs
        text = re.sub(
            f'@import\\s+["\']({target_pattern})(/[^"\']+)["\']',
            f'@import "{PROXY_PREFIX}\\2"',
            text
        )
        # Rewrite @import statements with root-relative URLs (but not if already prefixed)
        if proxy_prefix_pattern:
            text = re.sub(
                f'@import\\s+["\']?((?!{proxy_prefix_pattern})/)([^"\']+)["\']?',
                f'@import "{PROXY_PREFIX}/\\2"',
                text
            )
        else:
            text = re.sub(
                r'@import\s+["\'](/[^"\']+)["\']',
                f'@import "{PROXY_PREFIX}\\1"',
                text
            )
    
    # JavaScript URL rewriting
    if REWRITE_JS_URLS and ("javascript" in content_type or "application/json" in content_type):
        # Rewrite absolute URLs in strings (both single and double quotes)
        text = re.sub(
            f'(["\'])({target_pattern})(/[^"\']+)(["\'])',
            f'\\1{PROXY_PREFIX}\\3\\4',
            text
        )
        # Rewrite root-relative URLs in strings that look like file paths
        # Only rewrite if it looks like a path (starts with /, contains . or /)
        # and is not already prefixed
        if proxy_prefix_pattern:
            # Match paths but not already prefixed ones
            text = re.sub(
                f'(["\'])((?!{proxy_prefix_pattern})/)([a-zA-Z0-9_.-]+/)*([a-zA-Z0-9_.-]*\\.[a-zA-Z0-9]+)(["\'])',
                f'\\1{PROXY_PREFIX}/\\3\\4\\5',
                text
            )
        else:
            text = re.sub(
                r'(["\'](/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]*\.[a-zA-Z0-9]+)["\'])',
                lambda m: f'"{PROXY_PREFIX}{m.group(2)}"' if m.group(0)[0] == '"' else f"'{PROXY_PREFIX}{m.group(2)}'",
                text
            )
    
    # JSON URL rewriting (for API responses)
    if REWRITE_JSON_URLS and "application/json" in content_type:
        # Replace URLs in JSON strings with absolute target URLs
        text = re.sub(
            f'"{target_pattern}(/[^"]*)"',
            f'"{PROXY_PREFIX}\\1"',
            text
        )
        # Replace root-relative URLs that look like API paths (but not if already prefixed)
        if proxy_prefix_pattern:
            text = re.sub(
                f'"((?!{proxy_prefix_pattern})/)((?:api|v\\d+|_next|static)/[^"]+)"',
                f'"{PROXY_PREFIX}/\\2"',
                text
            )
        else:
            text = re.sub(
                r'"(/(?:api|v\d+|_next|static)/[^"]+)"',
                f'"{PROXY_PREFIX}\\1"',
                text
            )
    
    return text.encode("utf-8")


async def stream_response(response: httpx.Response, content_type: str) -> AsyncIterator[bytes]:
    """
    Stream response from target server, optionally rewriting content.
    """
    # For text content that needs rewriting, we need to buffer
    should_rewrite = (
        (REWRITE_HTML_URLS and "text/html" in content_type) or
        (REWRITE_JSON_URLS and "application/json" in content_type) or
        (REWRITE_CSS_URLS and ("text/css" in content_type or "stylesheet" in content_type)) or
        (REWRITE_JS_URLS and "javascript" in content_type)
    )
    
    if should_rewrite:
        # Buffer entire response for rewriting
        content = b""
        async for chunk in response.aiter_bytes():
            content += chunk
        
        # Rewrite and yield
        rewritten = rewrite_content_urls(content, content_type)
        yield rewritten
    else:
        # Stream without modification
        async for chunk in response.aiter_bytes():
            yield chunk


async def forward_to_target(request: Request) -> Response:
    """
    Forward incoming requests to the target webserver.
    This function acts as a comprehensive reverse proxy with:
    - Proper header forwarding (X-Forwarded-*)
    - Cookie path rewriting
    - Location header rewriting for redirects
    - Optional URL rewriting in HTML/JSON content
    - Streaming support for large responses
    - Compression handling
    """
    if not TARGET_SERVER_URL:
        raise HTTPException(
            status_code=503,
            detail="TARGET_SERVER_URL is not configured. Proxy is unavailable."
        )
    
    with tracer.start_as_current_span("proxy_request") as span:
        target_url = get_target_url(request)
        span.set_attribute("proxy.target_url", target_url)
        span.set_attribute("proxy.method", request.method)
        
        logger.debug(f"Proxying {request.method} {request.url.path} -> {target_url}")
        
        # Prepare headers
        headers = prepare_headers(request)
        
        # Read request body
        body = await request.body()
        
        try:
            # Create HTTP client with timeout
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(PROXY_TIMEOUT),
                follow_redirects=False  # Handle redirects manually for rewriting
            ) as client:
                # Forward request to target
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body,
                )
                
                span.set_attribute("proxy.status_code", response.status_code)
                
                # Prepare response headers
                response_headers = {}
                for name, value in response.headers.items():
                    name_lower = name.lower()
                    
                    # Skip hop-by-hop headers
                    if name_lower in HOP_BY_HOP_HEADERS:
                        continue
                    
                    # Rewrite Location header for redirects
                    if name_lower == "location":
                        value = rewrite_location_header(value, request)
                        span.set_attribute("proxy.rewritten_location", value)
                    
                    # Rewrite Set-Cookie path
                    elif name_lower == "set-cookie":
                        value = rewrite_cookie_path(value)
                    
                    response_headers[name] = value
                
                # Get content type for potential rewriting
                content_type = response.headers.get("content-type", "")
                
                # Return streaming response
                return StreamingResponse(
                    stream_response(response, content_type),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=content_type or None
                )
        
        except httpx.TimeoutException as e:
            logger.error(f"Proxy timeout for {target_url}: {e}")
            span.set_attribute("proxy.error", "timeout")
            raise HTTPException(status_code=504, detail="Gateway timeout")
        
        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to target {target_url}: {e}")
            span.set_attribute("proxy.error", "connection_failed")
            raise HTTPException(status_code=502, detail="Bad gateway - cannot connect to target")
        
        except Exception as e:
            logger.error(f"Proxy error for {target_url}: {e}", exc_info=True)
            span.set_attribute("proxy.error", str(e))
            raise HTTPException(status_code=502, detail=f"Bad gateway: {str(e)}")


# Register catch-all route for proxying
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_all(request: Request, path: str):
    """Catch-all route that proxies all requests to the target server."""
    return await forward_to_target(request)
       