# Use Remote MCP Servers

Connect the Enterprise MCP Bridge to hosted MCP servers instead of running them locally.

## Overview

Remote mode allows the bridge to:
- Connect to MCP servers hosted elsewhere
- Forward requests over HTTPS
- Centralize MCP server management
- Reduce local resource usage

## Basic Remote Configuration

### Environment Variables

```bash
# Remote MCP server URL
export MCP_REMOTE_SERVER="https://mcp.example.com"

# Optional: Bearer token for authentication
export MCP_REMOTE_BEARER_TOKEN="your-service-token"

# Optional: OAuth scopes
export MCP_REMOTE_SCOPE="offline_access api.read"
```

### Start the Bridge

```bash
# Do NOT set MCP_SERVER_COMMAND when using remote mode
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## Authentication Methods

### 1. Bearer Token (Simplest)

Use a static service token:

```bash
export MCP_REMOTE_SERVER="https://mcp.example.com"
export MCP_REMOTE_BEARER_TOKEN="static-service-token-123"
```

All requests will include:
```
Authorization: Bearer static-service-token-123
```

### 2. Token Exchange (Recommended)

Exchange user tokens for service tokens:

```bash
export MCP_REMOTE_SERVER="https://mcp.example.com"
export MCP_REMOTE_SERVER_OAUTH_ISSUER="https://auth.example.com"
export MCP_REMOTE_CLIENT_ID="bridge-client"
export MCP_REMOTE_CLIENT_SECRET="client-secret"
export MCP_REMOTE_SCOPE="offline_access api.read"
```

The bridge will:
1. Receive user OAuth token
2. Exchange it for a service token
3. Use service token for remote MCP calls

### 3. Anonymous Access

For public endpoints:

```bash
export MCP_REMOTE_SERVER="https://mcp.example.com"
export MCP_REMOTE_ANON_BEARER_TOKEN="read-only-token"
```

Used for:
- Health checks
- Public tool listings
- Anonymous tool calls

### 4. Token Forwarding

Forward the user's token directly:

```bash
export MCP_REMOTE_SERVER="https://mcp.example.com"
# No bearer token set - user's token is forwarded
```

## Header Forwarding

### Static Headers

Set headers from environment variables:

```bash
# Headers starting with MCP_REMOTE_HEADER_ are forwarded
export MCP_REMOTE_HEADER_X_API_KEY="secret-api-key"
export MCP_REMOTE_HEADER_X_CUSTOM_VALUE="custom-value"
```

Sent as:
```
X-API-KEY: secret-api-key
X-Custom-Value: custom-value
```

### Dynamic Headers

Forward specific request headers:

```bash
export MCP_REMOTE_SERVER_FORWARD_HEADERS="X-Request-ID,X-Correlation-ID,User-Agent"
```

The bridge forwards these headers from incoming requests to the remote server.

## Complete Example

### Docker Compose

```yaml
version: '3.8'

services:
  mcp-bridge:
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    environment:
      # Remote server configuration
      - MCP_REMOTE_SERVER=https://mcp-github.example.com
      
      # OAuth token exchange
      - MCP_REMOTE_SERVER_OAUTH_ISSUER=https://keycloak.example.com/realms/mcp
      - MCP_REMOTE_CLIENT_ID=bridge-client
      - MCP_REMOTE_CLIENT_SECRET=${CLIENT_SECRET}
      - MCP_REMOTE_REDIRECT_URI=https://bridge.example.com/oauth/callback
      - MCP_REMOTE_SCOPE=offline_access api.read
      
      # Static headers
      - MCP_REMOTE_HEADER_X_API_VERSION=v1
      
      # Forward request headers
      - MCP_REMOTE_SERVER_FORWARD_HEADERS=X-Request-ID,User-Agent
      
      # Anonymous access token
      - MCP_REMOTE_ANON_BEARER_TOKEN=${ANON_TOKEN}
    ports:
      - "8000:8000"
```

## GitHub Remote Example

Connect to a hosted GitHub MCP server:

```bash
# Configure remote GitHub MCP server
export MCP_REMOTE_SERVER="https://github-mcp.example.com"

# OAuth configuration for token exchange
export MCP_REMOTE_SERVER_OAUTH_ISSUER="https://keycloak.example.com/realms/mcp"
export MCP_REMOTE_CLIENT_ID="github-bridge-client"
export MCP_REMOTE_CLIENT_SECRET="your-client-secret"
export MCP_REMOTE_REDIRECT_URI="https://your-bridge.com/oauth/callback"
export MCP_REMOTE_SCOPE="offline_access"

# Start bridge
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Usage:

```bash
# User authenticates
curl http://localhost:8000/oauth/login

# Create GitHub issue (token automatically exchanged)
curl -X POST http://localhost:8000/tools/create_issue \
  -H "Authorization: Bearer USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "myorg",
    "repo": "myrepo",
    "title": "New issue",
    "body": "Issue description"
  }'
```

## Atlassian Remote Example

Connect to hosted Jira/Confluence MCP:

```bash
# Remote Atlassian MCP server
export MCP_REMOTE_SERVER="https://atlassian-mcp.example.com"

# OAuth configuration
export MCP_REMOTE_SERVER_OAUTH_ISSUER="https://keycloak.example.com/realms/mcp"
export MCP_REMOTE_CLIENT_ID="atlassian-bridge"
export MCP_REMOTE_CLIENT_SECRET="client-secret"
export MCP_REMOTE_SCOPE="offline_access jira:read jira:write"

# Start bridge
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## Multiple Remote Servers

To connect to multiple remote servers, run multiple bridge instances:

```yaml
# docker-compose.yml
version: '3.8'

services:
  github-bridge:
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    environment:
      - MCP_REMOTE_SERVER=https://github-mcp.example.com
      - MCP_BASE_PATH=/api/github
      # ... other config
    ports:
      - "8001:8000"

  atlassian-bridge:
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    environment:
      - MCP_REMOTE_SERVER=https://atlassian-mcp.example.com
      - MCP_BASE_PATH=/api/atlassian
      # ... other config
    ports:
      - "8002:8000"
```

Use an API gateway to route:
- `/api/github/*` → `github-bridge:8000`
- `/api/atlassian/*` → `atlassian-bridge:8000`

## Testing Remote Connection

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. List Tools

```bash
curl http://localhost:8000/tools
```

Should return tools from the remote server.

### 3. Test Tool Call

```bash
curl -X POST http://localhost:8000/tools/some_tool \
  -H "Content-Type: application/json" \
  -d '{"param": "value"}'
```

### 4. Verify Headers

Enable debug logging:

```bash
export LOG_LEVEL=debug
uvicorn app.server:app --reload
```

Check logs for forwarded headers and authentication.

## Performance Considerations

### Latency

Remote calls add network latency. Consider:

- **Caching:** Enable schema caching
- **Connection Pooling:** HTTP/2 for multiplexing
- **Timeout Configuration:** Adjust for slow remote servers

```bash
# Configure timeouts
export MCP_REMOTE_TIMEOUT=30  # seconds
export MCP_REMOTE_CONNECT_TIMEOUT=10  # seconds
```

### Connection Limits

```bash
# Configure connection pool
export MCP_REMOTE_MAX_CONNECTIONS=100
export MCP_REMOTE_MAX_KEEPALIVE_CONNECTIONS=20
```

## Security Considerations

### 1. Use HTTPS

Always use HTTPS for remote servers:

```bash
# Good
export MCP_REMOTE_SERVER="https://mcp.example.com"

# Bad - never in production
export MCP_REMOTE_SERVER="http://mcp.example.com"
```

### 2. Verify TLS Certificates

```bash
# Verify certificates (default)
export MCP_REMOTE_VERIFY_TLS=true

# Custom CA bundle
export MCP_REMOTE_CA_BUNDLE=/path/to/ca-bundle.crt
```

### 3. Secure Token Storage

Don't log or expose tokens:

```bash
# Use secret management
export MCP_REMOTE_BEARER_TOKEN_FILE=/run/secrets/remote-token
```

### 4. Limit Token Scope

Request minimal scopes:

```bash
export MCP_REMOTE_SCOPE="api.read"  # Read-only
```

## Troubleshooting

### Connection Refused

**Error:** `Connection refused to https://mcp.example.com`

**Solutions:**
- Verify remote server is running
- Check firewall rules
- Verify DNS resolution
- Test with `curl https://mcp.example.com/health`

### Authentication Failed

**Error:** `401 Unauthorized`

**Solutions:**
- Verify `MCP_REMOTE_BEARER_TOKEN` is correct
- Check token hasn't expired
- Verify OAuth configuration
- Test token with curl:
  ```bash
  curl https://mcp.example.com/tools \
    -H "Authorization: Bearer YOUR_TOKEN"
  ```

### Token Exchange Failed

**Error:** `Failed to exchange token`

**Solutions:**
- Verify OAuth issuer URL
- Check client ID and secret
- Verify redirect URI matches
- Check Keycloak broker configuration

### Timeout Errors

**Error:** `Request timeout`

**Solutions:**
- Increase timeout:
  ```bash
  export MCP_REMOTE_TIMEOUT=60
  ```
- Check remote server performance
- Verify network connectivity

## Monitoring Remote Connections

### Metrics

The bridge exposes metrics for remote calls:

```
mcp_remote_requests_total
mcp_remote_request_duration_seconds
mcp_remote_errors_total
```

### Logging

Enable request/response logging:

```bash
export LOG_LEVEL=debug
export MCP_REMOTE_LOG_REQUESTS=true
export MCP_REMOTE_LOG_RESPONSES=true
```

### Tracing

Enable distributed tracing:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:4318"
export OTEL_SERVICE_NAME="mcp-bridge-remote"
```

## Summary

You now know how to:

✅ Connect to remote MCP servers  
✅ Configure authentication methods  
✅ Forward headers and tokens  
✅ Handle multiple remote servers  
✅ Troubleshoot connection issues  

## Next Steps

- [Configure OAuth](configure-oauth.md)
- [Deploy to Production](deploy-production.md)
- [Monitor Your Deployment](monitoring.md)

## Resources

- [Remote MCP Example](../reference/examples.md#remote-mcp)
- [Configuration Reference](../reference/configuration.md)
- [Token Exchange Flow](../explanation/token-exchange.md)
