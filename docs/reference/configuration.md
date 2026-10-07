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

#### MCP_RUN_ON_START

Command to run before the bridge starts serving traffic.

- **Type:** String
- **Default:** Not set
- **Required:** No
- **Example:** `npm install -g @modelcontextprotocol/server-memory`

By default, this command runs in a clean environment with only `HOME` and `PATH`
set. Use it for startup work that should not inherit runtime secrets or
deployment-specific environment variables.

```bash
MCP_RUN_ON_START="npm install -g @modelcontextprotocol/server-memory"
```

#### MCP_RUN_ON_START_INHERIT_ENVIRONMENT

Command to run before the bridge starts serving traffic while inheriting the
current process environment.

- **Type:** String
- **Default:** Not set
- **Required:** No
- **Example:** `alembic upgrade head`

Use this instead of `MCP_RUN_ON_START` when the startup command needs access to
environment variables provided by the runtime. This can be useful for database
migrations on MCP startup, where the migration command needs variables such as
`POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`.

`MCP_RUN_ON_START` and `MCP_RUN_ON_START_INHERIT_ENVIRONMENT` are mutually
exclusive. Set only one startup hook.

```bash
MCP_RUN_ON_START_INHERIT_ENVIRONMENT="alembic upgrade head"
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

### Multi-server mode

#### MCP_SERVERS

Optional JSON array defining multiple MCP backends hosted by one bridge process.
When unset, the existing single-server variables such as `MCP_SERVER_COMMAND`,
`MCP_REMOTE_SERVER`, `MCP_BASE_PATH`, `INCLUDE_TOOLS`, and `EXCLUDE_TOOLS`
keep their current behavior.

Each entry requires:

- `id`: unique server identifier
- `base_path`: existing public path to preserve
- exactly one of `command` (local stdio) or `url` (remote MCP)

Optional entry fields:

- `env`: environment variables added to a local stdio server
- `include_tools`: glob patterns for allowed tools
- `exclude_tools`: glob patterns for hidden/blocked tools
- `sessionless`: override the global `MCP_SESSIONLESS` value for this server
- `auth_provider` (string): override the global `AUTH_PROVIDER` for requests
  to this server (e.g. `keycloak`, `user-api-key`, `gcp-metadata`). Accepts the
  same values as `AUTH_PROVIDER`; an empty string means "not overridden".
- `keycloak_provider_alias` (string): override the global
  `KEYCLOAK_PROVIDER_ALIAS`, i.e. the Keycloak identity provider whose stored
  token is exchanged for the caller's token. An empty string means "no alias":
  the Keycloak token is passed through, as with an empty global.
- `effect_tools` (array of strings): override the global `EFFECT_TOOLS` glob
  list used for dry runs (`X-Inxm-Dry-Run: true`) and generated-UI sampling.
  `[]` means no effect tools for this server. The entry `"auto"` enables
  [automatic effect classification](#automatic-effect-classification) and can
  be combined with explicit globs.
- `forward_access_token` (boolean, default `true`): whether the caller's access
  token may reach this server. When `false`, the bridge never sends it
  upstream: no token exchange (including `user-api-key` lookups), no fallback
  to the incoming token, no `oauth_token` tool argument, and no credential
  headers (`Authorization`, `Cookie`, `TOKEN_NAME`/`X-Auth-Request-Access-Token`,
  `X-Forwarded-Access-Token`, or any header containing the caller's token) via
  `MCP_REMOTE_SERVER_FORWARD_HEADERS` or `MCP_MAP_HEADER_TO_INPUT`, even if
  allowed globally. Only explicitly configured credentials are sent
  (`MCP_REMOTE_BEARER_TOKEN`, `MCP_REMOTE_ANON_BEARER_TOKEN`,
  `MCP_REMOTE_HEADER_*`, ambient cloud identity); with none configured there is
  no `Authorization` header. The bridge still authenticates its own callers as
  usual.

> **Security:** an orchestrator should set `"forward_access_token": false` for
> every remote that does not use a `keycloak_provider_alias` (e.g. a public
> third-party MCP such as DeepWiki). Otherwise the default sends the caller's
> own platform token to that third party.

Fields that are absent fall back to the global environment variables, and
unknown fields are ignored.

Example:

```bash
MCP_SERVERS='[
  {
    "id": "files",
    "base_path": "/api/mcp/files",
    "command": "python /servers/files.py",
    "env": {"FILES_ROOT": "/data"},
    "include_tools": ["read_*"],
    "sessionless": true
  },
  {
    "id": "remote",
    "base_path": "/api/mcp/remote",
    "url": "https://mcp.example.com/mcp",
    "exclude_tools": ["admin_*"]
  }
]'
```

#### MCP_ISOLATE_CHILDREN

Run local stdio servers as their own unprivileged users (default `false`;
an `MCP_SERVERS` entry's `isolate` overrides it per server). Each server gets
a stable uid derived from its id, a private 0700 `HOME`/`TMPDIR` under
`MCP_CHILD_ROOT` (default `/var/lib/mcp-children`, also the npm and uv
caches), no capabilities, `no_new_privs`, and only its own environment
(`PATH`, locale/CA/proxy variables, its `env`, its token variable). Siblings
then cannot read each other's `/proc/<pid>/environ` (where `OAUTH_ENV`
tokens live) or ptrace each other. The tools cache moves to the root-only
`MCP_BRIDGE_PRIVATE_DIR` (default `/var/lib/mcp-bridge`).

Requires the bridge to run as root with `setpriv` (util-linux) and the
`SETUID`, `SETGID`, `CHOWN` and `KILL` capabilities; isolation that cannot be
applied fails the request rather than running the child unisolated.

#### MCP_CHILD_MEMORY_RESERVE_MB / MCP_CHILD_ADMISSION_TIMEOUT / MCP_ADMISSION_DIR

Every request to a local stdio server starts its own child process, so a
bridge serving many local servers can be asked to start dozens at once (a
client listing every server's tools) and exceed its memory limit. With
`MCP_CHILD_MEMORY_RESERVE_MB` set (default `0`, off), a child only starts
while the container's cgroup memory leaves that many MiB for it and for every
child still initializing. Otherwise the request waits for room for up to
`MCP_CHILD_ADMISSION_TIMEOUT` seconds (default `30`) and then fails with
`503`, `Retry-After: 5` and `X-MCP-Admission: refused`. That header means the
server never ran, so clients may retry even a tool call. Memory in use is the
working set (droppable page cache excluded); one child may always run; without
a cgroup memory limit nothing waits. Set the reserve to roughly the largest
local server's memory.

Children are counted across all worker processes of the container through
lock files in `MCP_ADMISSION_DIR` (default `/tmp/mcp-admission`); a crashed
worker's children stop counting. Streaming tool calls (`/tools/{name}/stream`)
have already answered `200` when a refusal happens, so their `error` event
carries `details: {"status": 503, "retry_after": "5", "admission": "refused"}`.
`/session/start` returns only once the session's server is up, so a refused
start is answered with the `503` itself.

Session state, SSE transport state, tool filtering, and tool-cache files are
isolated by server. Overlapping base paths are supported; the most specific
matching base path selects the server context. In sessionless mode,
`/session/start` remains available
for client compatibility but does not create a persistent downstream session.

Central remote host proxying several remote MCP servers, each exchanging the
caller's Keycloak token through its own identity provider and classifying tool
effects automatically:

```bash
MCP_SERVERS='[
  {
    "id": "mcp-cloudflare-server",
    "base_path": "/api/mcp-cloudflare-server",
    "url": "https://mcp.cloudflare.com/mcp",
    "sessionless": true,
    "auth_provider": "keycloak",
    "keycloak_provider_alias": "cloudflare",
    "effect_tools": ["auto"]
  },
  {
    "id": "mcp-notion-server",
    "base_path": "/api/mcp-notion-server",
    "url": "https://mcp.notion.com/mcp",
    "sessionless": true,
    "auth_provider": "keycloak",
    "keycloak_provider_alias": "notion",
    "effect_tools": ["auto"]
  },
  {
    "id": "mcp-deepwiki-server",
    "base_path": "/api/mcp-deepwiki-server",
    "url": "https://mcp.deepwiki.com/mcp",
    "sessionless": true,
    "forward_access_token": false,
    "effect_tools": ["auto"]
  }
]'
```

#### Automatic effect classification

When the effective effect-tool list contains `"auto"` (a server's
`effect_tools`, or the global `EFFECT_TOOLS=auto`), every tool is an effect
tool unless it is confidently read-only:

1. MCP tool annotations win: `readOnlyHint: true` means read-only;
   `readOnlyHint: false` or `destructiveHint: true` means effect.
2. Without a usable annotation, a tool is read-only only if its name — compared
   case-insensitively, after stripping a namespace prefix up to the last `.` or
   `/` — starts with a read verb followed by `_`, `-`, the end of the name, or a
   camelCase boundary (`getUser`). Read verbs: `get`, `list`, `search`, `read`,
   `fetch`, `find`, `describe`, `query`, `lookup`, `show`, `view`, `count`,
   `check`, `validate`, `preview`, `explain`, `summarize`, `whoami`.
3. Everything else is an effect, so a misclassified write tool never runs for
   real during a dry run.

A tool matching an explicit glob is always an effect. Tool definitions are only
listed for classification when a dry run is actually requested.

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

#### MCP_REMOTE_AUTH_HEADER_NAME

Header name used to send the auth token (exchanged provider token, user API key, or bearer fallback) to the remote server.

- **Type:** String
- **Default:** `Authorization`
- **Required:** No

Useful when a gateway expects the credential in a custom header, e.g. an ESB in front of Azure APIM using `esb-subscription-key` instead of `ocp-apim-subscription-key`:

```bash
MCP_REMOTE_AUTH_HEADER_NAME="esb-subscription-key"
```

#### MCP_REMOTE_AUTH_HEADER_VALUE_TEMPLATE

Format of the auth header value. Supports the placeholders `{token}` and `{token_type}`.

- **Type:** String
- **Default:** None (sends `{token_type} {token}`, e.g. `Bearer <token>`)
- **Required:** No

Subscription keys are usually sent raw, without a `Bearer` prefix:

```bash
MCP_REMOTE_AUTH_HEADER_VALUE_TEMPLATE="{token}"
```

Combined example — send each user's stored API key as an ESB subscription key
(see the full provider documentation below):

```bash
AUTH_PROVIDER="user-api-key"
MCP_REMOTE_AUTH_HEADER_NAME="esb-subscription-key"
MCP_REMOTE_AUTH_HEADER_VALUE_TEMPLATE="{token}"
```

#### AUTH_PROVIDER=user-api-key (INXM-specific integration)

> **Note:** this provider integrates with INXM's `app-auth-tokens` service
> (currently a private repository). Open-source operators can implement the
> credential-store contract below themselves, or treat this provider as
> INXM-platform-specific.

Each authenticated user stores their own API key for this connection in the
credential store; the bridge fetches the *requesting user's* key per request
and sends it to the remote MCP (via `Authorization` or the custom header
configured above). Retrieval failures fail closed — the bridge never falls
back to `MCP_REMOTE_BEARER_TOKEN` or the incoming Keycloak token in this
mode, so a store outage cannot silently replace a user identity with a
shared one.

Required configuration (the provider refuses to operate if any is missing):

| Variable | Purpose |
| --- | --- |
| `AUTH_BASE_URL` | Keycloak base URL (JWKS verification of the caller's token) |
| `KEYCLOAK_REALM` | Realm whose JWKS/issuer are trusted (default `inxm`) |
| `USER_API_KEY_ALLOWED_CLIENTS` | Comma-separated client ids (matched against `azp`/`aud`) whose tokens may release credentials; empty = reject all |
| `AUTH_TOKENS_INTERNAL_URL` | Base URL of the credential store (in-cluster) |
| `MCP_CONNECTION_ID` | Connection id users stored their key under |
| `INTERNAL_API_SECRET` | Shared secret authenticating the bridge to the credential store — mandatory, `X-Service-ID` alone is spoofable metadata |
| `SERVICE_NAME` | Sent as `X-Service-ID`; the store enforces that only the connection's owning service may fetch its keys |

Optional: `KEYCLOAK_ISSUER` overrides the expected exact `iss` claim
(default `{AUTH_BASE_URL}/realms/{KEYCLOAK_REALM}`).

Incoming-token requirements: RS256/ES256 signature valid against the realm
JWKS, `exp` present and unexpired, `iss` exactly the configured issuer,
`azp` (or an `aud` entry) in `USER_API_KEY_ALLOWED_CLIENTS`, and an
`email`/`preferred_username`/`upn` claim identifying the user.

Credential-store contract (what an alternative implementation must serve):

```
GET {AUTH_TOKENS_INTERNAL_URL}/api/internal/connection-key/{email}/{connection}
Headers: X-Service-ID: {SERVICE_NAME}, X-Internal-Secret: {INTERNAL_API_SECRET}

200 {"success": true, "connection": "...", "key": "<raw key>"}
404 no key stored           -> surfaced to the user as "add a key in your profile"
401/403 auth failure        -> retrieval error (fail closed)
```

#### AUTH_PROVIDER=gcp-metadata / azure-metadata / aws-metadata (ambient cloud identity)

These three providers self-fetch a fresh, short-lived bearer token from the
platform the bridge is running on, on every remote-MCP connection. There is
no caller-token dependency and no persistence — the platform vouches for the
*bridge's own* identity, not the end user's. This makes them the right fit
for unattended, service-to-service auth toward a Cloud Run/Azure/EKS-hosted
remote MCP server, in contrast to `keycloak`/`user-api-key`, which forward or
exchange the *requesting user's* identity.

All three fail closed: if the platform call fails (unreachable, timeout,
non-2xx, missing file), the bridge raises rather than falling back to
`MCP_REMOTE_BEARER_TOKEN` or the incoming access token — a metadata-service
outage must never silently swap in a shared or wrong credential.

**`gcp-metadata`** — fetches a GCE/Cloud Run/GKE service-account identity
token from the local metadata server. Only reachable when the bridge itself
runs on that infrastructure.

| Variable | Purpose |
| --- | --- |
| `GCP_METADATA_SERVER_URL` | Metadata server base URL (default `http://metadata.google.internal`; override to point at a stub server in tests) |
| `GCP_METADATA_IDENTITY_AUDIENCE` | Audience claim for the identity token. Empty (default) falls back to `MCP_REMOTE_SERVER`, matching Cloud Run's own `--audiences=<service-url>` convention — override if the remote server validates against origin only |
| `GCP_METADATA_TOKEN_TIMEOUT_SECONDS` | Timeout for the metadata-server call (default `3`) |

```bash
AUTH_PROVIDER="gcp-metadata"
MCP_REMOTE_SERVER="https://my-mcp-service-abc123.a.run.app"
```

```
GET {GCP_METADATA_SERVER_URL}/computeMetadata/v1/instance/service-accounts/default/identity?audience=<audience>
Headers: Metadata-Flavor: Google

200 <raw JWT text body>   -> used directly as the bearer token
non-2xx / timeout         -> retrieval error (fail closed)
```

**`azure-metadata`** — fetches an OAuth2 access token for a resource from
Azure's Instance Metadata Service (IMDS), using the resource's managed
identity. Only reachable when the bridge runs on an Azure resource
(VM, Container App, AKS, etc.) with a managed identity assigned.

| Variable | Purpose |
| --- | --- |
| `AZURE_METADATA_SERVER_URL` | IMDS base URL (default `http://169.254.169.254`; override to point at a stub server in tests) |
| `AZURE_METADATA_IDENTITY_RESOURCE` | Resource/audience for the token. Empty (default) falls back to `MCP_REMOTE_SERVER` |
| `AZURE_METADATA_CLIENT_ID` | Client ID of a user-assigned managed identity. Empty (default) uses the resource's system-assigned identity |
| `AZURE_METADATA_TOKEN_TIMEOUT_SECONDS` | Timeout for the IMDS call (default `3`) |

```bash
AUTH_PROVIDER="azure-metadata"
MCP_REMOTE_SERVER="https://my-mcp-app.azurewebsites.net"
```

```
GET {AZURE_METADATA_SERVER_URL}/metadata/identity/oauth2/token?api-version=2018-02-01&resource=<resource>[&client_id=<AZURE_METADATA_CLIENT_ID>]
Headers: Metadata: true

200 {"access_token": "...", ...}   -> access_token field used as the bearer token
non-2xx / timeout                  -> retrieval error (fail closed)
```

**`aws-metadata`** — reads the IRSA/EKS Pod Identity OIDC token from the
file path in the standard `AWS_WEB_IDENTITY_TOKEN_FILE` env var, which EKS
sets automatically for pods with a service-account IAM role (no
bridge-specific config needed). **This is a file read, not an HTTP metadata
call** — unlike GCP/Azure, AWS's instance metadata service hands out SigV4
signing credentials for the AWS API, not a portable bearer JWT for an
arbitrary external audience, so there is no metadata-server request that
fits this use case on AWS. The IRSA-projected token (a JWT the Kubernetes
kubelet rotates automatically, hourly by default) is the closest ambient,
short-lived equivalent.

```bash
AUTH_PROVIDER="aws-metadata"
# AWS_WEB_IDENTITY_TOKEN_FILE is set automatically by EKS; no other config needed.
```

```
read({AWS_WEB_IDENTITY_TOKEN_FILE})

file readable, non-empty -> contents (trimmed) used directly as the bearer token
unset / missing / empty  -> retrieval error (fail closed)
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

### UI Generation

#### GENERATED_UI_EXPLORE_TOOLS

Enable automatic tool exploration before code generation. When enabled, the
generator probes discovery/listing tools to find additional domain tools that
are not exposed at the top level.

- **Type:** Boolean (`true`/`false`)
- **Default:** `false`
- **Required:** No

```bash
GENERATED_UI_EXPLORE_TOOLS=true
```

#### GENERATED_UI_EXPLORE_TOOLS_MAX_CALLS

Maximum number of MCP tool calls the exploration step may make. A higher
budget allows deeper discovery across more servers.

- **Type:** Integer
- **Default:** `5`
- **Required:** No

```bash
GENERATED_UI_EXPLORE_TOOLS_MAX_CALLS=10
```

#### GENERATED_UI_GATEWAY_LIST_SERVERS

Tool name for the gateway role that lists available sub-servers.

- **Type:** String
- **Default:** `get_servers`
- **Required:** No

```bash
GENERATED_UI_GATEWAY_LIST_SERVERS=list_servers
```

#### GENERATED_UI_GATEWAY_LIST_TOOLS

Tool name for the gateway role that lists tools on a specific server.
This is one of the two **required** roles for gateway detection.

- **Type:** String
- **Default:** `get_tools`
- **Required:** No

```bash
GENERATED_UI_GATEWAY_LIST_TOOLS=list_all_tools
```

#### GENERATED_UI_GATEWAY_GET_TOOL

Tool name for the gateway role that fetches a full tool definition
(inputSchema, outputSchema). Optional — when absent, the generator uses
whatever summary the list-tools call returns.

- **Type:** String
- **Default:** `get_tool`
- **Required:** No

```bash
GENERATED_UI_GATEWAY_GET_TOOL=describe_tool
```

#### GENERATED_UI_GATEWAY_CALL_TOOL

Tool name for the gateway role that invokes a tool on a sub-server.
This is one of the two **required** roles for gateway detection.

- **Type:** String
- **Default:** `call_tool`
- **Required:** No

```bash
GENERATED_UI_GATEWAY_CALL_TOOL=execute_tool
```

#### GENERATED_UI_GATEWAY_ROLE_ARGS

Optional JSON mapping for gateway discovery call arguments. This lets you
inject additional parameters (for example `prompt`) when calling gateway
roles during tool exploration.

- **Type:** JSON object
- **Default:** `{}`
- **Required:** No

Supported roles:
- `list_servers`
- `list_tools`
- `get_tool`

Supported placeholders (exact-match values only):
- `${prompt}`
- `${server_id}`
- `${tool_name}`

Rules:
- Default call args are built first, then template keys override them.
- If a template value is exactly a placeholder and that value is missing in
  context, the key is omitted from the final call args.
- Unknown roles are ignored with a warning.

```bash
GENERATED_UI_GATEWAY_ROLE_ARGS='{"list_tools":{"prompt":"${prompt}"}}'
```

```bash
GENERATED_UI_GATEWAY_ROLE_ARGS='{"get_tool":{"sid":"${server_id}","name":"${tool_name}"}}'
```

#### GENERATED_UI_GATEWAY_PROMPT_ARG_MAX_CHARS

Maximum length for `${prompt}` when substituted into
`GENERATED_UI_GATEWAY_ROLE_ARGS`.

- **Type:** Integer
- **Default:** `800`
- **Required:** No

```bash
GENERATED_UI_GATEWAY_PROMPT_ARG_MAX_CHARS=1200
```

#### GENERATED_UI_GATEWAY_SERVER_ID_FIELDS

Comma-separated field paths used to derive `server_id` from tool summaries
returned by gateway `list_tools` calls when no explicit `server_id` is present.

- **Type:** String (comma-separated paths)
- **Default:** `server_id,server,meta.server_id,meta.server,mcp_server_id,meta.mcp_server_id,url,meta.url`
- **Required:** No

Paths may be nested via dot notation (for example `meta.mcp_server_id`).
If a matched value looks like a URL/path, the server ID is extracted with
`GENERATED_UI_GATEWAY_SERVER_ID_URL_REGEX`.

```bash
GENERATED_UI_GATEWAY_SERVER_ID_FIELDS=meta.mcp_server_id,url
```

#### GENERATED_UI_GATEWAY_SERVER_ID_URL_REGEX

Regex used to extract `server_id` from URL-like values found via
`GENERATED_UI_GATEWAY_SERVER_ID_FIELDS`.

- **Type:** String (regular expression)
- **Default:** `/api/(?P<server_id>[^/]+)/tools/[^/?#]+`
- **Required:** No

Use either:
- named capture group `server_id`, or
- first capture group.

```bash
GENERATED_UI_GATEWAY_SERVER_ID_URL_REGEX=/api/(?P<server_id>[^/]+)/tools/[^/?#]+
```

## Workflow Storage

#### WORKFLOW_DB_BACKEND

Select the storage backend for persisted workflow execution state.

- **Type:** String
- **Values:** `sqlite`, `postgres`
- **Default:** `sqlite`
- **Required:** No

```bash
WORKFLOW_DB_BACKEND=sqlite
```

#### WORKFLOW_DB_PATH

SQLite database path for workflow execution state. This is used only when
`WORKFLOW_DB_BACKEND=sqlite`.

- **Type:** String
- **Default:** `<workflows base path>/workflow_state.db`
- **Required:** No

```bash
WORKFLOW_DB_PATH=/data/workflows/workflow_state.db
```

#### WORKFLOW_DATABASE_URL

Postgres DSN for workflow execution state. This is required when
`WORKFLOW_DB_BACKEND=postgres`.

- **Type:** String
- **Default:** None
- **Required:** Only with `WORKFLOW_DB_BACKEND=postgres`

```bash
WORKFLOW_DB_BACKEND=postgres
WORKFLOW_DATABASE_URL=postgresql://workflow_user:secret@postgres:5432/workflows
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

## Hub memory scopes

`X-INXM-Memory-Scope` selects a `c/<canonical conversation UUID>` graph asserted
by the agent hub, or a `p/<form-URL-encoded Keycloak group>` project graph with a
matching `?group=`. Both require the caller's bearer credential and
`X-Internal-Secret` matching the nonempty `INTERNAL_API_SECRET`. The hub is
responsible for checking chat membership on each call; projects additionally
pass the bridge's existing Keycloak group access check.

The override applies only to `MEMORY_TENANT`, only for a local child configured
with `MCP_ENV_MEMORY_TENANT={data_path}`, and only on sessionless REST calls.
Invalid scopes, groups or secrets fail closed before starting a child. They
never resolve to the caller's personal graph. Requests without the header retain
their existing user/group template behavior. The scope header does not support
remote MCP backends or persistent sessions.

The hub uses `POST /memory/tools/{tool}` for scoped calls. This endpoint requires
a scope assertion; older bridges do not implement it and therefore fail closed
instead of ignoring the header and reaching a personal graph. Existing
`POST /tools/{tool}` calls may also present a validated scope header.

In multi-server mode, `INTERNAL_API_SECRET` can be a per-server bridge setting
(including `settings_from`) so the hosted memory definition can reference a
Secret without exposing it to the child. Otherwise the global env is used.
