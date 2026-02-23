# Configuration Reference

Complete reference for all configuration options in Enterprise MCP Bridge.

## Environment Variables

### Core Configuration

#### MCP_SERVER_COMMAND

Command to launch the local MCP server.

- **Type:** String
- **Default:** `python mcp/server.py`
- **Required:** No (unless not using remote mode)
- **Example:** `npx -y @modelcontextprotocol/server-memory`

Supports placeholders:
- `{data_path}` - Data directory path
- `{user_id}` - Current user ID
- `{group_id}` - Current user's primary group ID

```bash
MCP_SERVER_COMMAND="python server.py --data {data_path}/{user_id}"
```

#### MCP_BASE_PATH

Base path for all API endpoints.

- **Type:** String
- **Default:** `/` (root)
- **Required:** No
- **Example:** `/api/v1/mcp`

Useful for:
- API versioning
- Ingress routing
- Multi-tenant deployments

```bash
MCP_BASE_PATH="/api/v1/mcp"
# API available at: http://host:port/api/v1/mcp/docs
```

### LLM Configuration

#### TGI_CONVERSATION_MODE

Controls which OpenAI-compatible endpoint style the bridge uses for LLM calls.

- **Type:** String
- **Values:** `chat/completions`, `/chat/completions`, `chat`, `chat_completions`, `responses`, `/responses`
- **Default:** `chat/completions`
- **Required:** No

Notes:
- Invalid values fall back to `chat/completions` and emit a warning.
- For Codex-style models that reject chat-completions, set this to `responses`.

```bash
# Default behavior (chat-completions)
TGI_CONVERSATION_MODE="chat/completions"

# Codex / responses-style models
TGI_CONVERSATION_MODE="responses"
```

#### ENV

Environment mode.

- **Type:** String
- **Values:** `dev`, `prod`
- **Default:** `prod`
- **Required:** No

In `dev` mode:
- Auto-installs dependencies from `requirements.txt` or `pyproject.toml`
- More verbose logging
- Debug features enabled

```bash
ENV=dev
```

### Remote MCP Server

#### MCP_REMOTE_SERVER

URL of the remote MCP server.

- **Type:** String (URL)
- **Default:** None
- **Required:** Only for remote mode
- **Example:** `https://mcp.example.com`

When set, local `MCP_SERVER_COMMAND` is ignored.

```bash
MCP_REMOTE_SERVER="https://mcp-github.example.com"
```

#### MCP_REMOTE_BEARER_TOKEN

Static bearer token for remote server authentication.

- **Type:** String
- **Default:** None
- **Required:** No
- **Security:** Sensitive - use secret management

```bash
MCP_REMOTE_BEARER_TOKEN="service-token-abc123"
```

#### MCP_REMOTE_ANON_BEARER_TOKEN

Bearer token for anonymous/unauthenticated requests.

- **Type:** String
- **Default:** None
- **Required:** No

Used for:
- Health checks
- Public tool listings
- Guest access

```bash
MCP_REMOTE_ANON_BEARER_TOKEN="readonly-token-xyz789"
```

#### MCP_REMOTE_SCOPE

OAuth scopes for remote server.

- **Type:** String (space-separated)
- **Default:** `offline_access`
- **Required:** No

```bash
MCP_REMOTE_SCOPE="offline_access api.read api.write"
```

#### MCP_REMOTE_SERVER_FORWARD_HEADERS

Request headers to forward to remote server.

- **Type:** String (comma-separated)
- **Default:** None
- **Required:** No

```bash
MCP_REMOTE_SERVER_FORWARD_HEADERS="X-Request-ID,X-Correlation-ID,User-Agent"
```

#### MCP_REMOTE_HEADER_*

Static headers to send to remote server.

- **Type:** String
- **Default:** None
- **Required:** No
- **Pattern:** `MCP_REMOTE_HEADER_<HEADER_NAME>`

```bash
MCP_REMOTE_HEADER_X_API_KEY="secret-key-123"
MCP_REMOTE_HEADER_X_CLIENT_VERSION="1.0.0"
```

Sent as HTTP headers:
- `X-API-KEY: secret-key-123`
- `X-Client-Version: 1.0.0`

### OAuth Configuration

#### OAUTH_ISSUER_URL

OAuth2 issuer URL (e.g., Keycloak realm).

- **Type:** String (URL)
- **Default:** None
- **Required:** For OAuth authentication

```bash
OAUTH_ISSUER_URL="https://keycloak.example.com/realms/mcp"
```

#### OAUTH_CLIENT_ID

OAuth2 client ID.

- **Type:** String
- **Default:** None
- **Required:** For OAuth authentication

```bash
OAUTH_CLIENT_ID="mcp-bridge-client"
```

#### OAUTH_CLIENT_SECRET

OAuth2 client secret.

- **Type:** String
- **Default:** None
- **Required:** For OAuth authentication
- **Security:** Sensitive - use secret management

```bash
OAUTH_CLIENT_SECRET="client-secret-here"
```

#### OAUTH_REDIRECT_URI

OAuth2 redirect URI for callbacks.

- **Type:** String (URL)
- **Default:** `http://localhost:8000/oauth/callback`
- **Required:** No

Must match exactly in OAuth provider configuration.

```bash
OAUTH_REDIRECT_URI="https://bridge.example.com/oauth/callback"
```

#### OAUTH_SCOPE

OAuth2 scopes to request.

- **Type:** String (space-separated)
- **Default:** `openid profile email`
- **Required:** No

```bash
OAUTH_SCOPE="openid profile email offline_access"
```

#### OAUTH_ENABLE_TOKEN_EXCHANGE

Enable OAuth2 token exchange.

- **Type:** Boolean
- **Values:** `true`, `false`
- **Default:** `false`
- **Required:** No

```bash
OAUTH_ENABLE_TOKEN_EXCHANGE=true
```

### Session Management

#### SESSION_MANAGER_TYPE

Type of session manager to use.

- **Type:** String
- **Values:** `memory`, `redis`
- **Default:** `memory`
- **Required:** No

```bash
SESSION_MANAGER_TYPE=redis
```

#### REDIS_URL

Redis connection URL (when using Redis session manager).

- **Type:** String (URL)
- **Default:** `redis://localhost:6379/0`
- **Required:** Only if `SESSION_MANAGER_TYPE=redis`

```bash
REDIS_URL="redis://redis:6379/0"
REDIS_URL="rediss://redis:6379/0"  # TLS
```

#### REDIS_PASSWORD

Redis password.

- **Type:** String
- **Default:** None
- **Required:** No
- **Security:** Sensitive

```bash
REDIS_PASSWORD="redis-secret-password"
```

#### SESSION_TIMEOUT_SECONDS

Session inactivity timeout in seconds.

- **Type:** Integer
- **Default:** `1800` (30 minutes)
- **Required:** No

```bash
SESSION_TIMEOUT_SECONDS=3600  # 1 hour
```

#### SESSION_CLEANUP_INTERVAL_SECONDS

How often to clean up expired sessions.

- **Type:** Integer
- **Default:** `300` (5 minutes)
- **Required:** No

```bash
SESSION_CLEANUP_INTERVAL_SECONDS=600  # 10 minutes
```

### Logging and Monitoring

#### LOG_LEVEL

Logging level.

- **Type:** String
- **Values:** `debug`, `info`, `warning`, `error`, `critical`
- **Default:** `info`
- **Required:** No

```bash
LOG_LEVEL=debug
```

#### LOG_FORMAT

Log output format.

- **Type:** String
- **Values:** `json`, `text`
- **Default:** `text`
- **Required:** No

```bash
LOG_FORMAT=json
```

#### OTEL_EXPORTER_OTLP_ENDPOINT

OpenTelemetry collector endpoint.

- **Type:** String (URL)
- **Default:** None
- **Required:** No

```bash
OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:4318"
```

#### OTEL_SERVICE_NAME

Service name for telemetry.

- **Type:** String
- **Default:** `enterprise-mcp-bridge`
- **Required:** No

```bash
OTEL_SERVICE_NAME="mcp-bridge-prod"
```

#### PROMETHEUS_ENABLED

Enable Prometheus metrics.

- **Type:** Boolean
- **Values:** `true`, `false`
- **Default:** `true`
- **Required:** No

```bash
PROMETHEUS_ENABLED=true
```

### Performance

#### WORKERS

Number of worker processes (for Gunicorn).

- **Type:** Integer
- **Default:** `1`
- **Required:** No
- **Recommendation:** `2 * CPU_CORES + 1`

```bash
WORKERS=4
```

#### WORKER_CLASS

Worker class to use.

- **Type:** String
- **Default:** `uvicorn.workers.UvicornWorker`
- **Required:** No

```bash
WORKER_CLASS="uvicorn.workers.UvicornWorker"
```

#### MAX_REQUESTS

Maximum requests per worker before restart.

- **Type:** Integer
- **Default:** `0` (disabled)
- **Required:** No

Helps prevent memory leaks:

```bash
MAX_REQUESTS=10000
MAX_REQUESTS_JITTER=1000
```

#### TIMEOUT

Worker timeout in seconds.

- **Type:** Integer
- **Default:** `30`
- **Required:** No

```bash
TIMEOUT=120  # 2 minutes
```

### Security

#### CORS_ORIGINS

Allowed CORS origins.

- **Type:** String (comma-separated)
- **Default:** `*` (allow all)
- **Required:** No

```bash
CORS_ORIGINS="https://app.example.com,https://admin.example.com"
```

#### CORS_ALLOW_CREDENTIALS

Allow credentials in CORS requests.

- **Type:** Boolean
- **Values:** `true`, `false`
- **Default:** `true`
- **Required:** No

```bash
CORS_ALLOW_CREDENTIALS=true
```

#### ALLOWED_HOSTS

Allowed host headers.

- **Type:** String (comma-separated)
- **Default:** `*` (allow all)
- **Required:** No

```bash
ALLOWED_HOSTS="mcp.example.com,bridge.example.com"
```

### Advanced

#### MCP_ENV_*

Environment variables to pass to MCP server.

- **Type:** String
- **Default:** None
- **Required:** No
- **Pattern:** `MCP_ENV_<VAR_NAME>`

```bash
MCP_ENV_API_KEY="secret-api-key"
MCP_ENV_DATABASE_URL="postgresql://..."
```

Becomes available to MCP server as:
- `API_KEY=secret-api-key`
- `DATABASE_URL=postgresql://...`

#### ENABLE_SCHEMA_CACHE

Enable tool schema caching.

- **Type:** Boolean
- **Values:** `true`, `false`
- **Default:** `true`
- **Required:** No

```bash
ENABLE_SCHEMA_CACHE=true
```

#### SCHEMA_CACHE_TTL

Schema cache time-to-live in seconds.

- **Type:** Integer
- **Default:** `3600` (1 hour)
- **Required:** No

```bash
SCHEMA_CACHE_TTL=7200  # 2 hours
```

## Configuration Files

### Loading from .env File

Create `.env` file:

```bash
# .env
MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-memory
MCP_BASE_PATH=/api/mcp
SESSION_MANAGER_TYPE=redis
REDIS_URL=redis://redis:6379
LOG_LEVEL=info
```

The bridge automatically loads `.env` files.

## Configuration by Deployment Type

### Development

```bash
ENV=dev
MCP_SERVER_COMMAND="python mcp/server.py"
LOG_LEVEL=debug
SESSION_MANAGER_TYPE=memory
```

### Testing

```bash
ENV=dev
MCP_SERVER_COMMAND="python mcp/server.py"
LOG_LEVEL=info
SESSION_MANAGER_TYPE=memory
```

### Staging

```bash
ENV=prod
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/memory.json"
MCP_BASE_PATH="/api/v1/mcp"
LOG_LEVEL=info
LOG_FORMAT=json
SESSION_MANAGER_TYPE=redis
REDIS_URL="redis://redis:6379"
OAUTH_ISSUER_URL="https://staging-auth.example.com/realms/mcp"
WORKERS=2
```

### Production

```bash
ENV=prod
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/{group_id}/memory.json"
MCP_BASE_PATH="/api/v1/mcp"
LOG_LEVEL=info
LOG_FORMAT=json
SESSION_MANAGER_TYPE=redis
REDIS_URL="rediss://redis:6379"
REDIS_PASSWORD="${REDIS_PASSWORD}"
OAUTH_ISSUER_URL="https://auth.example.com/realms/mcp"
OAUTH_CLIENT_ID="mcp-bridge"
OAUTH_CLIENT_SECRET="${OAUTH_CLIENT_SECRET}"
OAUTH_ENABLE_TOKEN_EXCHANGE=true
CORS_ORIGINS="https://app.example.com"
WORKERS=8
PROMETHEUS_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:4318"
```

## Priority Order

Configuration is loaded in this order (later overrides earlier):

1. Default values
2. Configuration file (`.env`, `config.yaml`)
3. Environment variables
4. Command-line arguments (if applicable)

## Validation

The bridge validates configuration on startup:

- Required variables must be set
- URLs must be valid
- Integers must be in valid ranges
- Enum values must match allowed values

Invalid configuration causes startup failure with clear error messages.

## Summary

This reference covers all configuration options for the Enterprise MCP Bridge. Use it to:

✅ Configure for different environments  
✅ Tune performance settings  
✅ Set up security features  
✅ Enable monitoring and logging  

## Next Steps

- [Environment Variables Guide](environment-variables.md)
- [Deploy to Production](../how-to/deploy-production.md)
- [Examples](examples.md)
