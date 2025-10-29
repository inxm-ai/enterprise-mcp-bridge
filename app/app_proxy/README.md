# Generic Web Application Reverse Proxy

A comprehensive, production-ready reverse proxy implementation designed to work with almost any web application. This proxy handles HTTP/HTTPS requests, performs intelligent URL rewriting, manages cookies, headers, and supports streaming responses.

## Features

### Core Functionality
- ✅ **Full HTTP Method Support**: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
- ✅ **Streaming Responses**: Memory-efficient streaming for large files and responses
- ✅ **Compression Support**: Properly handles gzip, deflate, and other compression schemes
- ✅ **Error Handling**: Robust error handling with appropriate HTTP status codes

### Header Management
- ✅ **X-Forwarded-* Headers**: Automatically adds proxy headers for backend awareness
  - `X-Forwarded-For`: Client IP chain
  - `X-Forwarded-Host`: Original host
  - `X-Forwarded-Proto`: Original protocol (http/https)
  - `X-Forwarded-Prefix`: Proxy path prefix
  - `X-Real-IP`: Direct client IP
- ✅ **Hop-by-Hop Headers**: Properly filters headers that shouldn't be forwarded
- ✅ **Custom Headers**: Preserves all custom headers from client

### URL Rewriting
- ✅ **HTML**: Rewrites `href` and `src` attributes in HTML content
- ✅ **CSS**: Rewrites `url()` functions and `@import` statements
- ✅ **JavaScript**: Rewrites URL strings in JS code
- ✅ **JSON**: Rewrites URLs in API responses

### Cookie & Redirect Handling
- ✅ **Cookie Path Rewriting**: Automatically adjusts cookie paths for proxy prefix
- ✅ **Location Header Rewriting**: Rewrites redirect locations to proxy URLs
- ✅ **Set-Cookie Attributes**: Preserves all cookie attributes while fixing paths

### Security & Performance
- ✅ **Configurable Timeouts**: Prevent hung connections
- ✅ **Connection Pooling**: Efficient HTTP client with connection reuse
- ✅ **HTTPS Termination**: Supports HTTPS on both sides
- ✅ **OpenTelemetry Tracing**: Built-in observability

## Environment Variables

### Required Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `TARGET_SERVER_URL` | The backend server to proxy to | `http://localhost:3000` |

### Optional Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_PREFIX` | `${MCP_BASE_PATH}/app` | URL prefix for the proxy endpoint |
| `PUBLIC_URL` | (auto-detected) | Public-facing URL for redirect rewriting |
| `PROXY_TIMEOUT` | `300` | Request timeout in seconds |
| `REWRITE_HTML_URLS` | `true` | Enable URL rewriting in HTML content |
| `REWRITE_JSON_URLS` | `true` | Enable URL rewriting in JSON responses |
| `REWRITE_CSS_URLS` | `true` | Enable URL rewriting in CSS files |
| `REWRITE_JS_URLS` | `true` | Enable URL rewriting in JavaScript files |

## Usage Examples

### Basic Setup

```bash
# Set the target server
export TARGET_SERVER_URL="http://internal-app:8080"

# Optional: Set a custom prefix
export PROXY_PREFIX="/apps/myapp"

# Optional: Set public URL for proper redirects
export PUBLIC_URL="https://public.example.com"

# Start the server
python -m uvicorn app.server:app
```

### Accessing the Proxied Application

If your proxy is configured with `PROXY_PREFIX=/apps/myapp`:

- Original app URL: `http://internal-app:8080/dashboard`
- Proxied URL: `https://your-proxy.com/apps/myapp/dashboard`

### Advanced Configuration

#### Disable URL Rewriting for Specific Content Types

```bash
# Disable CSS rewriting if the app uses relative paths correctly
export REWRITE_CSS_URLS="false"

# Disable JS rewriting for apps that handle paths dynamically
export REWRITE_JS_URLS="false"
```

#### Configure for High Traffic

```bash
# Increase timeout for long-running requests
export PROXY_TIMEOUT="600"

# Use connection pooling (handled automatically by httpx)
```

## How It Works

### Request Flow

1. **Client Request** → Proxy receives request at `/apps/myapp/resource`
2. **Header Preparation** → Adds X-Forwarded-* headers, removes hop-by-hop headers
3. **URL Translation** → Converts `/apps/myapp/resource` → `http://target/resource`
4. **Forward Request** → Sends request to target server with prepared headers
5. **Response Processing** → Rewrites Location, Set-Cookie, and content URLs
6. **Stream Response** → Returns processed response to client

### URL Rewriting Logic

The proxy intelligently rewrites URLs in various content types:

#### HTML Content
```html
<!-- Before -->
<a href="http://internal-app:8080/page">Link</a>
<img src="/image.png">

<!-- After (with PROXY_PREFIX=/apps/myapp) -->
<a href="/apps/myapp/page">Link</a>
<img src="/apps/myapp/image.png">
```

#### CSS Content
```css
/* Before */
.bg { background: url('/images/bg.png'); }
@import "/styles/theme.css";

/* After */
.bg { background: url('/apps/myapp/images/bg.png'); }
@import "/apps/myapp/styles/theme.css";
```

#### JavaScript Content
```javascript
// Before
const apiUrl = "/api/data";
fetch("/api/users.json");

// After
const apiUrl = "/apps/myapp/api/data";
fetch("/apps/myapp/api/users.json");
```

#### JSON Responses
```json
// Before
{
  "next": "/api/page/2",
  "resource": "http://internal-app:8080/resource/123"
}

// After
{
  "next": "/apps/myapp/api/page/2",
  "resource": "/apps/myapp/resource/123"
}
```

### Cookie Handling

Cookies from the target server are automatically adjusted:

```http
# Response from target server
Set-Cookie: session=abc123; Path=/; HttpOnly; Secure

# Rewritten by proxy (with PROXY_PREFIX=/apps/myapp)
Set-Cookie: session=abc123; Path=/apps/myapp; HttpOnly; Secure
```

This ensures cookies are sent back on subsequent requests through the proxy.

### Redirect Handling

Redirects are automatically rewritten to proxy URLs:

```http
# Response from target server
Location: http://internal-app:8080/login

# Rewritten by proxy
Location: https://public.example.com/apps/myapp/login
```

## Common Use Cases

### 1. Proxying a Web Application Behind Authentication

```bash
export TARGET_SERVER_URL="http://grafana:3000"
export PROXY_PREFIX="/monitoring/grafana"
export PUBLIC_URL="https://portal.example.com"
```

Users access Grafana at: `https://portal.example.com/monitoring/grafana`

### 2. Hosting Multiple Applications

```bash
# App 1
export PROXY_PREFIX="/apps/dashboard"
export TARGET_SERVER_URL="http://dashboard:8080"

# App 2 (separate instance)
export PROXY_PREFIX="/apps/analytics"
export TARGET_SERVER_URL="http://analytics:3000"
```

### 3. Development/Staging Environment

```bash
# Proxy to local development server
export TARGET_SERVER_URL="http://localhost:3000"
export PROXY_PREFIX="/dev"
export REWRITE_HTML_URLS="true"
export REWRITE_JS_URLS="true"
```

### 4. HTTPS Termination

The proxy handles HTTPS termination automatically:

```
[Client HTTPS] → [Proxy HTTPS] → [HTTP] → [Target App HTTP]
```

The target app receives proper `X-Forwarded-Proto: https` header.

## Troubleshooting

### Issue: Application Assets Not Loading

**Symptom**: CSS, JavaScript, or images return 404 errors

**Solution**: Check if URL rewriting is enabled:
```bash
export REWRITE_HTML_URLS="true"
export REWRITE_CSS_URLS="true"
export REWRITE_JS_URLS="true"
```

### Issue: Redirects Point to Internal URLs

**Symptom**: Redirects take you to `http://internal-app:8080` instead of proxy URL

**Solution**: Set `PUBLIC_URL`:
```bash
export PUBLIC_URL="https://your-public-domain.com"
```

### Issue: Cookies Not Persisting

**Symptom**: User session is lost after navigation

**Solution**: Verify cookie path rewriting is working. Check that:
1. `PROXY_PREFIX` is set correctly
2. The target app is setting cookies
3. Cookie domain doesn't conflict

### Issue: WebSocket Connection Fails

**Current Limitation**: This proxy currently doesn't support WebSocket upgrades.

**Workaround**: Use a dedicated WebSocket proxy or configure a separate route.

### Issue: Large File Upload Timeout

**Solution**: Increase timeout:
```bash
export PROXY_TIMEOUT="3600"  # 1 hour for large uploads
```

### Issue: Application Performs Client-Side URL Construction

**Symptom**: JavaScript builds URLs that bypass the proxy

**Solution**: 
1. Check if app respects `X-Forwarded-Prefix` header
2. Configure the app's base URL setting
3. Adjust `REWRITE_JS_URLS` to be more aggressive (may require custom logic)

## Performance Considerations

### Streaming vs Buffering

The proxy uses intelligent buffering:
- **Text content** (HTML, CSS, JS, JSON): Buffered for URL rewriting
- **Binary content** (images, videos, PDFs): Streamed directly for memory efficiency
- **Large responses**: Streamed when rewriting is disabled

### Memory Usage

- Minimal memory footprint for streaming responses
- Buffering only occurs for content that needs URL rewriting
- Connection pooling reduces overhead

### Latency

- Adds ~1-5ms for header processing
- URL rewriting adds ~10-50ms depending on content size
- Streaming eliminates waiting for full response

## Security Considerations

### Headers Removed for Security

The following hop-by-hop headers are never forwarded:
- `Connection`
- `Keep-Alive`
- `Proxy-Authenticate`
- `Proxy-Authorization`
- `TE`
- `Trailers`
- `Transfer-Encoding`
- `Upgrade`

### Headers Added for Backend Awareness

The proxy adds these headers so backends know about the original request:
- `X-Forwarded-For`: Track client IP through proxy chain
- `X-Forwarded-Host`: Original hostname
- `X-Forwarded-Proto`: Original protocol (important for redirect generation)
- `X-Forwarded-Prefix`: Proxy path prefix (for base URL construction)

### Recommendations

1. **Always use HTTPS** in production
2. **Validate TARGET_SERVER_URL** is internal/trusted
3. **Set appropriate timeouts** to prevent resource exhaustion
4. **Monitor proxy logs** for unusual patterns
5. **Consider rate limiting** at the proxy level
6. **Use authentication** before the proxy when exposing internal apps

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest app/app_proxy/test_route.py -v

# Run specific test categories
pytest app/app_proxy/test_route.py::TestGetTargetUrl -v
pytest app/app_proxy/test_route.py::TestRewriteContentUrls -v

# Run with coverage
pytest app/app_proxy/test_route.py --cov=app.app_proxy --cov-report=html
```

## API Reference

### Routes

#### `/{path:path}` - Catch-all Proxy Route

Proxies all HTTP methods to the target server.

**Methods**: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS

**Parameters**:
- `path`: Any path after the proxy prefix

**Example**:
```
GET /apps/myapp/api/users → GET http://target/api/users
POST /apps/myapp/api/users → POST http://target/api/users
```

## Monitoring & Observability

The proxy includes OpenTelemetry tracing with these spans:

- `proxy_request`: Overall request processing
  - Attributes:
    - `proxy.target_url`: Final target URL
    - `proxy.method`: HTTP method
    - `proxy.status_code`: Response status
    - `proxy.rewritten_location`: Rewritten redirect location (if applicable)
    - `proxy.error`: Error type (timeout, connection_failed, etc.)

View traces in your OpenTelemetry-compatible backend (Jaeger, Zipkin, etc.).

## Contributing

When adding new features to the proxy:

1. Update the URL rewriting logic in `rewrite_content_urls()`
2. Add corresponding tests in `test_route.py`
3. Update this documentation
4. Ensure backward compatibility

## License

[Include your license information here]

## Support

For issues, questions, or contributions, please [contact information or issue tracker link].
