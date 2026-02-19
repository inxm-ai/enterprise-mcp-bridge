# Enterprise MCP Bridge

FastAPI-based wrapper that exposes any Model Context Protocol (MCP) server over plain HTTP/JSON.

## Why this project?

Most existing MCP examples are designed for local development or simple demos and fall short for real-world applications. They are typically:

* Single-user CLI processes driven by a local client.
* Ephemeral, with state being lost as soon as the process ends.
* Lacking multi-tenancy, with no built-in orchestration for concurrent users or sessions.
* Hard to monitor/control in an Enterprise environment.
* No simple way for integrating MCP components into REST-based microservice architectures.
* Missing a consistent security model for handling delegated permissions (e.g., OAuth) for downstream resources.

This project directly addresses these gaps. It's designed for enterprise production use, with a focus on security, scalability, and ease of integration. For small private deployments it's probably not the right fit.

## Key Capabilities

### Robust Session & User Management
* **Centralized hosting of the MCP server:** The server can be hosted in a centralized manner, allowing multiple clients to connect and interact with it concurrently, while your IT manages the underlying infrastructure, permissions and monitors it.
* **Multi-User & Multi-Session:** A single server instance can securely manage multiple, isolated user contexts, namespaced by OAuth tokens. Each user can run multiple concurrent sessions (`/session/start`), enabling parallel tool execution and long-running conversational state.
* **Stateful & Stateless Modes:** Supports both **stateless** ("fire-and-forget") tool calls for low-latency tasks and **stateful** sessions for complex, multi-step interactions.
* **Lifecycle & Resource Hygiene:** Provides explicit endpoints to start and close sessions, coupled with automatic cleanup of idle sessions via inactivity pings. This prevents resource leaks and supports autoscaling.
* **Pluggable for Scalability:** Features a swappable session manager (`SessionManagerBase`). While it defaults to a simple in-memory store, it can be replaced with a distributed backend (like Redis or a database) for horizontal scaling.

### Integrated Security & Authentication
* **Built-in OAuth2 Token Exchange:** Natively handles the OAuth2 token exchange flow (e.g., with Keycloak as a broker) to securely acquire tokens for downstream resource providers.
* **Automatic Token Injection:** If a tool's input schema includes an `oauth_token` argument, the server automatically and securely injects the correct token. This simplifies client-side logic and prevents tokens from being exposed.
* **Automated Token Refresh:** Manages token refresh logic transparently, allowing for long-lived, secure sessions without requiring clients to handle re-authentication.
* **Group-Based Data Access:** Dynamically resolves data sources based on OAuth token group membership. Users can access group-specific or user-specific data based on their authenticated identity and group memberships, eliminating the need to expose sensitive file paths or database identifiers.

### Developer Experience & API Design
* **REST-first Interface:** All tool discovery and invocation happens over standard HTTP/JSON endpoints, ensuring maximum compatibility with any client, platform, or automation tool (`curl`, browsers, etc.).
* **Automatic Tool Endpoints:** Each tool exposed by the MCP server is automatically mapped to a canonical REST endpoint (e.g., `/tools/{tool_name}`), making the API discoverable and easy to integrate with.
* **AI-Generated Web Applications:** Generate complete, production-ready web applications and dashboards directly from natural language prompts. Uses LLM-powered generation to create reactive interfaces built on pfusch (progressive enhancement) that integrate seamlessly with your MCP tools. Supports user/group scoping, versioned updates, and maintains full generation history.
* **Structured Error Handling:** Maps MCP-level errors to standard HTTP status codes (`400`, `404`, `500`), providing a predictable and developer-friendly integration experience.
* **Auto-Generated API Docs:** Because it's built on FastAPI, it automatically generates interactive OpenAPI (Swagger) and ReDoc documentation, making the API easy to explore and test.
* **Containerized Deployment:** Supports running the MCP server as a containerized application (e.g., Docker), simplifying deployment and scaling.
* **Observability:** Structured logging and monitoring capabilities are built-in, allowing for easy tracking of requests, errors, and performance metrics.

### Flexible Deployment & Integration
* **Protocol Agnostic:** Works with any MCP server that communicates over `stdio`, regardless of the language it's written in (Python, Node.js, Go, etc.).
* **Configurable Routing & Base Path:** Deployment is simplified through environment variables like `MCP_SERVER_COMMAND` (to define the tool process) and `MCP_BASE_PATH` (for clean ingress integration and API versioning).


---


## Quick Start

```bash
git clone https://github.com/inxm-ai/enterprise-mcp-bridge.git
cd enterprise-mcp-bridge
pip install app
uvicorn app.server:app --reload
```

Open: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Using MCPs

### Local development workflow
1. Clone or copy your MCP server into the `mcp/` folder so it ships with the bridge.
2. When no override is supplied, the bridge runs `mcp/server.py` bundled with the repo.
3. Set `ENV=dev` when using the published Docker image to auto-install dependencies declared in `requirements.txt` or `pyproject.toml` within your MCP directory.

**Example**
```bash
# clone the repo you want to run
git clone https://github.com/modelcontextprotocol/servers.git mcp
# run it in docker
docker run -p 8000:8000 -e ENV=dev -v $PWD/mcp/src/fetch:/mcp -it ghcr.io/inxm-ai/enterprise-mcp-bridge:latest python -m mcp_server_fetch
```

For production, create a dedicated Docker image from `ghcr.io/inxm-ai/enterprise-mcp-bridge` that bundles your MCP server and its dependencies.

If you have specific requirements (e.g. different python version, additional system packages), use the `Dockerfile` in this repo as a starting point.

### Running custom commands with `MCP_SERVER_COMMAND`
Set `MCP_SERVER_COMMAND` to the exact command you want the bridge to launch. The value is shell-split, so quoting works as expected, and it can include placeholders such as `{data_path}`, `{user_id}`, and `{group_id}` that are resolved per request.

```bash
export MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory"
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Precedence:
1. `MCP_SERVER_COMMAND` if present.
2. Command-line arguments passed after `--` when starting `uvicorn`.
3. The default `mcp/server.py`.

### Connecting to remote MCP servers
Set `MCP_REMOTE_SERVER` to switch the bridge into remote mode and forward every request to a hosted MCP endpoint over HTTPS. Leave `MCP_SERVER_COMMAND` unset to avoid conflicting strategies.

```bash
export MCP_REMOTE_SERVER="https://mcp.example.com"
export MCP_REMOTE_SCOPE="offline_access api.read"
export MCP_REMOTE_REDIRECT_URI="https://bridge.example.com/oauth/callback"
export MCP_REMOTE_CLIENT_ID="bridge-client"
export MCP_REMOTE_CLIENT_SECRET="change-me"
# Optional shortcut token:
# export MCP_REMOTE_BEARER_TOKEN="service-token-123"
# Optional read-only token for anonymous flows (agent card, health checks):
# export MCP_REMOTE_ANON_BEARER_TOKEN="read-only-service-token"

# Set any additional MCP env vars here:
# export MCP_ENV_API_KEY="supersecret"

# Set any additional MCP Header vars here (static headers from environment):
# export MCP_REMOTE_HEADER_X_CUSTOM="custom-value"

# Configure which incoming request headers to forward to the remote MCP server:
# export MCP_REMOTE_SERVER_FORWARD_HEADERS="X-Request-ID,X-Correlation-ID,User-Agent"

uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Authentication hierarchy:
1. Exchange the incoming OAuth token using `TokenRetrieverFactory` (if configured).
2. Use `MCP_REMOTE_BEARER_TOKEN` when it is provided.
3. Anonymous requests first use `MCP_REMOTE_ANON_BEARER_TOKEN` (if set).
4. Fall back to forwarding the caller's token.

#### Header Forwarding
When connecting to a remote MCP server, you can configure the bridge to forward specific headers from incoming requests:

- **`MCP_REMOTE_HEADER_*`**: Set static headers from environment variables (e.g., `MCP_REMOTE_HEADER_X_API_KEY="secret123"`)
- **`MCP_REMOTE_SERVER_FORWARD_HEADERS`**: Comma-separated list of incoming request header names to forward (e.g., `"X-Request-ID,X-Correlation-ID,User-Agent"`)

Header forwarding is case-insensitive and only forwards headers that are explicitly configured. This is useful for:
- Propagating request IDs and correlation IDs for distributed tracing
- Forwarding user-agent information for analytics
- Passing through custom application headers

Example:
```bash
export MCP_REMOTE_SERVER_FORWARD_HEADERS="X-Request-ID,X-Correlation-ID"
```

Authentication hierarchy:
1. Exchange the incoming OAuth token using `TokenRetrieverFactory` (if configured).
2. Use `MCP_REMOTE_BEARER_TOKEN` when it is provided.
3. Anonymous requests first use `MCP_REMOTE_ANON_BEARER_TOKEN` (if set).
4. Fall back to forwarding the caller’s token.

## Deploying to Production

### Container image
Run the published image and mount your MCP server or data as volumes.

```bash
docker run -it -p 8000:8000 \
  -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/memory.json" \
  -v $(pwd)/data:/data \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

Set `ENV=dev` to auto-install dependencies found in mounted MCP directories.

### Kubernetes/Helm
Embed the bridge into your platform by defining the command and volumes in your pod spec.

```yaml
containers:
  - name: enterprise-mcp-bridge
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    env:
      - name: MCP_SERVER_COMMAND
        value: "npx -y @modelcontextprotocol/server-memory /data/memory.json"
      - name: MCP_BASE_PATH
        value: "/api/mcp"
    volumeMounts:
      - name: data
        mountPath: /data
volumes:
  - name: data
    persistentVolumeClaim:
      claimName: mcp-data
```

### Reference deployments
- `example/minimal-example` – tiny starter with the default memory server.
- `example/memory-group-access` – demonstrates group-aware data routing.
- `example/token-exchange-m365` – complete token-exchange stack with monitoring and tracing.
- `example/remote-mcp-github` – Use the remote github mcp server with integrated token exchange.
- `example/remote-mcp-atlassian` – Atlassian MCP workflow demo (Jira + Confluence).

### Security checklist
- Terminate TLS and place an auth proxy in front of the bridge (expects `_oauth2_proxy` or the `X-Auth-Request-Access-Token` header).
- Scope Keycloak broker roles minimally (requires `broker.read-token` only).
- Rotate sensitive values such as `MCP_SERVER_COMMAND` secrets and OAuth credentials outside the container image.
- Monitor request volume and errors using your preferred logging/metrics stack.

## Full Configuration Guidelines

### Environment variables
| Variable                  | Purpose                                                   | Default                |
| ------------------------- | --------------------------------------------------------- | ---------------------- |
| `MCP_SERVER_COMMAND`      | Full command to launch MCP server (supports placeholders) | (unset)                |
| `MCP_BASE_PATH`           | Prefix all routes (e.g., `/api/mcp`)                      | ""                     |
| `OAUTH_ENV`               | Name of env var injected into MCP subprocess              | (unset)                |
| `AUTH_PROVIDER`           | Token exchange provider selector (`keycloak`)             | keycloak               |
| `AUTH_BASE_URL`           | Keycloak base URL for broker/token endpoints              | (required)             |
| `KEYCLOAK_REALM`          | Keycloak realm                                            | inxm                   |
| `KEYCLOAK_PROVIDER_ALIAS` | External IdP alias used in broker path                    | (required)             |
| `MCP_SESSION_MANAGER`     | Implementation name (e.g., future Redis)                  | InMemorySessionManager |
| `SESSION_FIELD_NAME`      | Name of the header/cookie containing the session ID       | x-inxm-mcp-session     |
| `TOKEN_NAME`              | Name of the header containing the OAuth token             | X-Auth-Request-Access-Token |
| `TOKEN_COOKIE_NAME`       | Name of the cookie containing the OAuth token             | _oauth2_proxy          |
| `TOKEN_SOURCE`            | Source of the token (`header` or `cookie`)                | header                 |
| `INCLUDE_TOOLS`           | Comma-separated list of tool name patterns to include     | ""                     |
| `EXCLUDE_TOOLS`           | Comma-separated list of tool name patterns to exclude     | ""                     |
| `EFFECT_TOOLS`            | Comma-separated list of tool name patterns that modify data or have side effects | ""          |
| `LLM_MAX_PAYLOAD_BYTES`   | Maximum serialized payload size sent to the backing LLM; large histories are compacted or truncated to stay below this limit | 120000                |
| `MCP_REMOTE_SERVER`       | HTTPS endpoint of a remote MCP server (enables remote mode) | ""                   |
| `MCP_REMOTE_SCOPE`        | OAuth scope to request when exchanging tokens              | ""                     |
| `MCP_REMOTE_REDIRECT_URI` | Redirect URI used during dynamic OAuth client registration | `https://localhost/unused-callback` |
| `MCP_REMOTE_CLIENT_ID`    | Pre-registered OAuth client id for the remote server       | ""                     |
| `MCP_REMOTE_CLIENT_SECRET`| Client secret for the remote OAuth client (if required)    | ""                     |
| `MCP_REMOTE_BEARER_TOKEN` | Static bearer token to send if OAuth negotiation is skipped | ""                    |
| `MCP_REMOTE_ANON_BEARER_TOKEN` | Static bearer token used for anonymous remote calls (e.g., agent card generation) | "" |
| `MCP_REMOTE_HEADER_*`     | Any additional static headers to send to the remote MCP server (e.g., `MCP_REMOTE_HEADER_X_API_KEY`) |                        |
| `MCP_REMOTE_SERVER_FORWARD_HEADERS` | Comma-separated list of incoming request header names to forward to remote MCP server | "" |
| `MCP_MAP_HEADER_TO_INPUT` | Comma-separated list of mappings from tool input property to incoming HTTP header name, in the form `input=Header-Name` (e.g. `userId=x-auth-user-id,email=x-auth-user-email`). Mapped input properties are removed from tool input schemas and will be automatically filled from headers at call time. | "" |
| `SYSTEM_DEFINED_PROMPTS`  | JSON array of built-in prompts available to all users     | "[]"                   |
| `TOOL_OUTPUT_SCHEMAS`     | JSON object mapping tool names to JSON Schemas for their output. Ensures tools advertise structured output capability and parsed JSON strings into `structuredContent`. If a value is a string, it is treated as a file path to load the schema from. | "{}" |
| `GENERATED_WEB_PATH`      | Base directory for storing AI-generated web applications  | ""                     |
| `MCP_ENV_*`               | Forwarded to the MCP server process                       |                        |
| `MCP_*_DATA_ACCESS_TEMPLATE` | Template for specific data resources. See [User and Group Management](#user-and-group-management) for details. | `{*}/{placeholder}` |
| `MCP_OAUTH_RESOURCE_URL`  | Public URL of this bridge (for RFC 9728 protected resource metadata). Auto-derived from request if unset. | (auto) |
| `MCP_OAUTH_ISSUER`        | OAuth issuer / authorization server URL. Auto-derived from `AUTH_BASE_URL`+`KEYCLOAK_REALM` when unset. | (auto) |
| `MCP_OAUTH_CLIENT_ID`     | OAuth client ID that MCP clients should use               | (unset)                |
| `MCP_OAUTH_SCOPES`        | Space-separated scopes the bridge accepts                 | `openid profile email` |
| `MCP_OAUTH_REGISTRATION_ENDPOINT` | Dynamic client registration URL (defaults to Keycloak's) | (auto)          |

### Data access templates
| Template                           | Default value          | Description                                     |
| ---------------------------------- | ---------------------- | ----------------------------------------------- |
| `MCP_GROUP_DATA_ACCESS_TEMPLATE`   | `g/{group_id}`         | Group-specific access path template           |
| `MCP_USER_DATA_ACCESS_TEMPLATE`    | `u/{user_id}`          | User-specific access path template            |
| `MCP_SHARED_DATA_ACCESS_TEMPLATE`  | `shared/{resource_id}` | Shared access path template                   |

### System-defined prompts
Provide curated prompts to every user by setting `SYSTEM_DEFINED_PROMPTS` to a JSON array. Each prompt includes a `name`, `title`, `description`, optional `arguments`, and a `template`.

```json
[
  {
    "name": "greeting",
    "title": "Hello You",
    "description": "Get a personalized greeting.",
    "arguments": [{ "name": "name" }],
    "template": {
      "role": "system",
      "content": "The user is called {name}. For any greeting, use their name.",
      "file": "prompts/greeting.md" // alternative to `content`
    }
  }
]
```

Prompts are merged with prompts returned by the MCP server and are available through `/prompts`.

#### Template files note

Templates may provide content directly via `template.content` or point to a file using `template.file`.

- If `template.file` is a relative path (for example `prompts/greeting.md`) it is resolved relative to the `app/` directory and opened.
- If `template.file` is an absolute path (for example `/tmp/prompt.md`) it will be opened directly as provided. This is intentional: system-defined prompts are controlled by the environment owner via `SYSTEM_DEFINED_PROMPTS` and may reference external files.

Be aware that allowing absolute paths means the runtime will attempt to read any file the process user can access.

### Tool Output Schemas

You can define output schemas for tools that do not natively provide them (or provide them as unstructured text) via the `TOOL_OUTPUT_SCHEMAS` environment variable. This variable accepts a JSON object where keys are tool names and values are JSON Schema objects.

If the value is a string, it is interpreted as a file path (absolute or relative to the working directory) to read the schema from.

```json
{
  "my-complex-tool": {
    "type": "object",
    "properties": {
      "result": { "type": "string" }
    }
  },
  "my-other-tool": "/path/to/schema.json"
}
```

When a schema is defined for a tool:
1. The schema is advertised in the tool definition under `outputSchema`.
2. When the tool is executed, if the output is a JSON-encoded string, the bridge will attempt to parse it and populate the `structuredContent` field in the result.

## API Docs
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc rendering: `http://localhost:8000/redoc`
- Key operations:
  - `GET /tools` – list available tools.
  - `GET /tools/{tool_name}` – retrieve a tool's details.
  - `POST /tools/{tool_name}` – invoke a tool (stateless unless a session header is provided).
  - `POST /tools/{tool_name}/stream` – invoke a tool with SSE progress streaming (see [Progress Streaming](#progress-streaming-sse)).
  - `GET /prompts` – list available prompts (including system-defined).
  - `GET /resource` – list available resources.
  - `GET /resource/{resource_id}` – access resources. If the resource is html/text, it is rendered directly.
  - `POST /session/start` and `POST /session/close` – manage stateful sessions.
  - `POST /app/_generated/{scope}` – generate a new web application from a natural language prompt.
  - `GET /app/_generated/{scope}/{ui_id}/{name}` – retrieve a generated application (as metadata, page, or snippet).
  - `POST /app/_generated/{scope}/{ui_id}/{name}` – update an existing generated application.
  - `GET /.well-known/agent.json` – discover agent metadata when agent mode is enabled.
  - `GET /sse` – MCP-native SSE endpoint (full MCP protocol over SSE transport). See [Native MCP SSE Endpoint](#native-mcp-sse-endpoint).
  - `GET /.well-known/oauth-protected-resource` – RFC 9728 resource metadata for MCP client auth discovery.
  - `GET /.well-known/oauth-authorization-server` – RFC 8414 authorization server metadata.

## Native MCP SSE Endpoint

The bridge exposes a **fully MCP-compliant SSE transport** at `/sse`. Any standard MCP client (Claude Code, VS Code Copilot, Cursor, `mcp-client-cli`, etc.) can connect directly using the MCP SSE protocol — no custom HTTP/JSON wrapper needed.

All requests flow through the same auth pipeline as the REST API: OAuth2 token exchange, tool filtering (`INCLUDE_TOOLS`/`EXCLUDE_TOOLS`), header-to-input mapping, and group-based data access.

### How it works

```
┌──────────────┐       SSE           ┌──────────────────────┐       stdio / HTTP       ┌─────────────┐
│  MCP Client  │ ◄──── GET /sse ──► │  Enterprise Bridge   │ ◄─────────────────────► │  MCP Server  │
│ (Claude Code)│  POST /sse/messages │  (SSE proxy server)  │   (token exchanged)      │  (downstream)│
└──────────────┘                     └──────────────────────┘                          └─────────────┘
```

1. Client opens `GET /sse` → receives an SSE stream with a `session_id`-bearing message endpoint.
2. Client sends JSON-RPC messages to `POST /sse/messages?session_id=...`.
3. The bridge opens a fresh downstream MCP session per SSE connection (with full OAuth token exchange).
4. All MCP operations (`tools/list`, `tools/call`, `prompts/list`, `resources/read`, etc.) are proxied transparently.

### Connecting Claude Code

1. **Set the OAuth discovery variables** so Claude Code can authenticate automatically:

```bash
# Keycloak-based setup (most common)
export AUTH_BASE_URL="https://keycloak.example.com"
export KEYCLOAK_REALM="myrealm"

# Optional: explicit overrides
# export MCP_OAUTH_ISSUER="https://keycloak.example.com/realms/myrealm"
# export MCP_OAUTH_CLIENT_ID="enterprise-mcp-bridge"
# export MCP_OAUTH_SCOPES="openid profile email offline_access"

uvicorn app.server:app --host 0.0.0.0 --port 8000
```

2. **Add the server to Claude Code's config** (`~/.claude/settings.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "enterprise-bridge": {
      "url": "https://bridge.example.com/sse"
    }
  }
}
```

Claude Code will:
- Connect to `GET /sse` and get a 401 (if auth is configured)
- Discover `/.well-known/oauth-protected-resource` → finds the authorization server
- Discover `/.well-known/oauth-authorization-server` → gets token/auth endpoints
- Perform the OAuth authorization code flow with PKCE
- Reconnect to `/sse` with a valid bearer token
- Use all MCP tools/prompts/resources through the standard protocol

3. **Without OAuth** (e.g., local development behind a VPN):

If you don't need auth, just point the client at the bare URL. No OAuth variables needed:

```json
{
  "mcpServers": {
    "enterprise-bridge": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### Connecting VS Code (GitHub Copilot)

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "enterprise-bridge": {
      "type": "sse",
      "url": "https://bridge.example.com/sse"
    }
  }
}
```

### Connecting Cursor

Add to Cursor's MCP settings (Settings → MCP Servers → Add):

```json
{
  "mcpServers": {
    "enterprise-bridge": {
      "url": "https://bridge.example.com/sse"
    }
  }
}
```

### Group scoping

Pass a `group` query parameter to scope the session to a specific data group:

```json
{
  "mcpServers": {
    "enterprise-bridge": {
      "url": "https://bridge.example.com/sse?group=my-team"
    }
  }
}
```

### Using with `MCP_BASE_PATH`

When the bridge is deployed behind a reverse proxy with a path prefix, both the SSE endpoint and the message channel respect `MCP_BASE_PATH`:

```bash
export MCP_BASE_PATH="/api/mcp"
```

Clients should then connect to:
```
https://bridge.example.com/api/mcp/sse
```

### Keycloak Integration

The bridge integrates with Keycloak in two distinct ways depending on how it is
deployed. Choose the section that matches your setup.

---

#### Scenario A: Embedded in a Kubernetes application (recommended)

This is the standard production setup. The bridge runs as an in-cluster service
behind an application that already handles user authentication via Keycloak. A
proxy layer (e.g. [oauth2-proxy](https://oauth2-proxy.github.io/oauth2-proxy/),
an OIDC-aware ingress) sits in front of the bridge and injects the authenticated
user's token into every request as a header. The bridge reads that header — no
OAuth client or PKCE flow needed inside the bridge itself.

```
User's browser / app frontend
        │  already authenticated via Keycloak
        ▼
 Ingress / oauth2-proxy
        │  validates session, injects X-Auth-Request-Access-Token header
        ▼
 enterprise-mcp-bridge  (reads header, optionally exchanges for downstream token)
        ▼
 Downstream MCP server
```

**Keycloak setup:** none required beyond what your application already has.
The bridge does not initiate any OAuth flows.

**Bridge environment variables:**

```bash
# ── How the user token arrives at the bridge ─────────────────────────
export TOKEN_SOURCE="header"                        # or "cookie"
export TOKEN_NAME="X-Auth-Request-Access-Token"     # injected by oauth2-proxy
# export TOKEN_COOKIE_NAME="_oauth2_proxy"          # if using cookie mode

# ── (Optional) Exchange user token for a downstream service token ─────
# If MCP tools need to call third-party APIs on behalf of the user,
# configure Keycloak broker token exchange:
export AUTH_BASE_URL="https://keycloak.example.com"
export KEYCLOAK_REALM="myrealm"
export KEYCLOAK_PROVIDER_ALIAS="github"             # IDP alias in Keycloak
export KEYCLOAK_PROVIDER_REFRESH_MODE="broker"      # or "oidc"

# ── MCP server to proxy ──────────────────────────────────────────────
export MCP_SERVER_COMMAND="python mcp/server.py"
# or: export MCP_REMOTE_SERVER="https://remote-mcp.example.com/"
```

When `KEYCLOAK_PROVIDER_ALIAS` is set, the bridge calls Keycloak's broker
endpoint (`/realms/{realm}/broker/{alias}/token`) on each tool call that
requires an `oauth_token`, exchanging the user's Keycloak JWT for a stored
downstream provider token (GitHub, Microsoft 365, etc.).

To enable token storage in Keycloak for a given Identity Provider:

```
Keycloak Admin → Identity Providers → <your provider>
  → Enable "Store tokens"

  Mappers → Add mapper:
    Name:  Store Token
    Type:  Hardcoded Attribute
```

---

#### Scenario B: Direct external MCP client access

Use this when MCP clients (Claude Code, Cursor, VS Code Copilot) connect
**directly** to the bridge from outside the cluster — i.e. the bridge is the
public-facing endpoint and there is no proxy handling auth upstream of it.

In this case the MCP client must obtain a Keycloak token itself using the
OAuth 2.0 Authorization Code + PKCE flow. The bridge's `/.well-known/` endpoints
advertise where to do that.

```
MCP Client (Claude Code, Cursor, …)
   │ 1. GET /.well-known/oauth-protected-resource  → discovers Keycloak
   │ 2. PKCE Authorization Code flow with Keycloak → gets access token
   │ 3. GET /sse  (Authorization: Bearer <token>)
   ▼
 Ingress / oauth2-proxy          ← must accept bearer tokens directly
   │ validates JWT, injects X-Auth-Request-Access-Token
   ▼
 enterprise-mcp-bridge
```

> **Important:** The ingress / oauth2-proxy in front of the bridge must be
> configured to accept Bearer tokens (not just session cookies), otherwise it
> will redirect the MCP client to the Keycloak login page, breaking the SSE
> connection. With oauth2-proxy, set `--skip-jwt-bearer-tokens=true` and
> `--oidc-issuer-url=https://keycloak.example.com/realms/myrealm`.

##### 1. Create (or reuse) a Keycloak realm

```
Keycloak Admin → Realm Settings → Create Realm
  Name: myrealm
```

##### 2. Create a public OAuth client

MCP clients run on the user's machine and cannot keep a client secret, so the
client must be **public** (no `client_secret`). PKCE provides the security instead.

```
Keycloak Admin → Clients → Create Client

  Client ID:      enterprise-mcp-bridge
  Client type:    OpenID Connect
  Authentication: OFF            ← makes it a "public" client

  Valid Redirect URIs:
    http://localhost/*            ← for local development / CLI clients
    http://127.0.0.1/*           ← alternative localhost
    https://bridge.example.com/* ← production callback (if used)

  Web Origins:
    +                            ← or list your allowed origins

  Advanced → Proof Key for Code Exchange:
    S256                         ← required by MCP clients
```

> **Tip:** Enable the *clients-registrations* endpoint on the realm if you want
> MCP clients to register themselves dynamically. Otherwise all clients share the
> same pre-configured `client_id`.

##### 3. Configure scopes

| Scope | Purpose |
|---|---|
| `openid` | Standard OIDC identity |
| `profile` | User's display name |
| `email` | User's email (used for group access resolution) |
| `offline_access` | *(optional)* Enables refresh tokens for long-lived sessions |

In Keycloak: **Clients → enterprise-mcp-bridge → Client Scopes → Add client scope**.

##### 4. Bridge environment variables

```bash
# ── Keycloak connection (drives /.well-known/* discovery) ────────────
export AUTH_BASE_URL="https://keycloak.example.com"
export KEYCLOAK_REALM="myrealm"

# ── OAuth discovery metadata ─────────────────────────────────────────
# AUTH_BASE_URL + KEYCLOAK_REALM are usually enough; override if needed:
# export MCP_OAUTH_ISSUER="https://keycloak.example.com/realms/myrealm"
export MCP_OAUTH_CLIENT_ID="enterprise-mcp-bridge"
export MCP_OAUTH_SCOPES="openid profile email"

# ── How the injected token arrives at the bridge ─────────────────────
export TOKEN_SOURCE="header"
export TOKEN_NAME="X-Auth-Request-Access-Token"

# ── (Optional) Downstream token exchange — same as Scenario A ────────
# export KEYCLOAK_PROVIDER_ALIAS="github"
# export KEYCLOAK_PROVIDER_REFRESH_MODE="broker"

# ── MCP server to proxy ──────────────────────────────────────────────
export MCP_SERVER_COMMAND="python mcp/server.py"

uvicorn app.server:app --host 0.0.0.0 --port 8000
```

##### 5. Verify the discovery endpoints

```bash
# Protected resource metadata — points clients at your Keycloak realm
curl -s http://localhost:8000/.well-known/oauth-protected-resource | jq .
# {
#   "resource": "http://localhost:8000",
#   "authorization_servers": ["https://keycloak.example.com/realms/myrealm"],
#   "scopes_supported": ["openid", "profile", "email"],
#   "bearer_methods_supported": ["header"]
# }

# Authorization server metadata — tells clients where to get tokens
curl -s http://localhost:8000/.well-known/oauth-authorization-server | jq .
# {
#   "issuer": "https://keycloak.example.com/realms/myrealm",
#   "authorization_endpoint": "…/protocol/openid-connect/auth",
#   "token_endpoint": "…/protocol/openid-connect/token",
#   "code_challenge_methods_supported": ["S256"],
#   …
# }
```

##### 6. Connect your MCP client

```json
{
  "mcpServers": {
    "enterprise-bridge": {
      "url": "https://bridge.example.com/sse"
    }
  }
}
```

On first connection the client will:

1. `GET /sse` → 401 Unauthorized
2. `GET /.well-known/oauth-protected-resource` → discovers Keycloak as the auth server
3. `GET /.well-known/oauth-authorization-server` → gets the `/auth` and `/token` URLs
4. Opens a browser for the user to log in to Keycloak
5. Exchanges the authorization code (with PKCE) for tokens
6. Reconnects to `GET /sse` with `Authorization: Bearer <token>`
7. Full MCP session is active — tools, prompts, resources all work

##### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Client gets 404 on `/.well-known/*` | `AUTH_BASE_URL` or `KEYCLOAK_REALM` not set | Set both or set `MCP_OAUTH_ISSUER` explicitly |
| Ingress redirects client to login page | oauth2-proxy not accepting bearer tokens | Add `--skip-jwt-bearer-tokens=true` to oauth2-proxy |
| Client gets "invalid redirect URI" | Keycloak client missing the redirect URI the MCP client uses | Add `http://localhost/*` and `http://127.0.0.1/*` to Valid Redirect URIs |
| `PKCE challenge method not supported` | Keycloak client not configured for S256 | Enable PKCE S256 in client Advanced settings |
| Token exchange returns empty | Provider alias mismatch or "Store tokens" not enabled | Verify `KEYCLOAK_PROVIDER_ALIAS` matches and token storage is on |
| "User is logged out" errors | Keycloak token expired and refresh failed | Check `offline_access` scope; verify `KEYCLOAK_PROVIDER_REFRESH_MODE` |

## Progress Streaming (SSE)

The bridge supports real-time progress streaming for long-running tool executions via Server-Sent Events (SSE). This is useful for tools that report progress updates during execution.

### When to use

Use the streaming endpoint when:
- The tool execution takes a significant amount of time
- The tool reports progress updates (using `ctx.report_progress()` in FastMCP)
- You want to provide real-time feedback to users during execution

### Endpoint

`POST /tools/{tool_name}/stream`

This endpoint accepts the same parameters as the regular `POST /tools/{tool_name}` endpoint but returns a stream of SSE events instead of waiting for the complete result.

### Event Types

The stream emits the following event types:

| Event Type | Description | Fields |
|------------|-------------|--------|
| `progress` | Progress update from the tool | `progress` (float), `total` (float, optional), `message` (string, optional) |
| `result` | Final tool execution result | `data` (object with `content`, `isError`, `structuredContent`) |
| `error` | Error during execution | `data.message` (string), `data.details` (optional) |

### Example Usage

**Server-side tool (FastMCP)**
```python
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("my-server")

@mcp.tool()
async def long_running_task(ctx: Context, items: list[str]):
    """Process items with progress reporting"""
    total = len(items)
    results = []
    
    for i, item in enumerate(items):
        # Report progress to the client
        await ctx.report_progress(
            progress=(i + 1) * 100 / total,
            total=100.0,
            message=f"Processing {item}..."
        )
        
        # Do actual work
        result = await process_item(item)
        results.append(result)
    
    return {"processed": results}
```

**Client-side (curl)**
```bash
curl -N -X POST "http://localhost:8000/tools/long_running_task/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"items": ["a", "b", "c"]}'
```

**Client-side (JavaScript)**
```javascript
// Using fetch with streaming
async function callToolWithProgress(toolName, args, onProgress) {
  const response = await fetch(`/tools/${toolName}/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(args)
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop(); // Keep incomplete event in buffer
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const event = JSON.parse(line.slice(6));
        
        if (event.type === 'progress') {
          onProgress(event.progress, event.total, event.message);
        } else if (event.type === 'result') {
          return event.data;
        } else if (event.type === 'error') {
          throw new Error(event.data.message);
        }
      }
    }
  }
}

// Usage
const result = await callToolWithProgress(
  'long_running_task',
  { items: ['a', 'b', 'c'] },
  (progress, total, message) => {
    console.log(`Progress: ${progress}/${total} - ${message}`);
    // Update progress bar, etc.
  }
);
```

**Client-side (Python)**
```python
import httpx
import json

async def call_tool_with_progress(tool_name: str, args: dict):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            'POST',
            f'http://localhost:8000/tools/{tool_name}/stream',
            json=args,
            headers={'Accept': 'text/event-stream'}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    event = json.loads(line[6:])
                    
                    if event['type'] == 'progress':
                        print(f"Progress: {event['progress']}% - {event.get('message', '')}")
                    elif event['type'] == 'result':
                        return event['data']
                    elif event['type'] == 'error':
                        raise Exception(event['data']['message'])

# Usage
import asyncio
result = asyncio.run(call_tool_with_progress('long_running_task', {'items': ['a', 'b', 'c']}))
```

### Response Format

Each SSE event is a JSON object prefixed with `data: ` and followed by two newlines:

```
data: {"type": "progress", "progress": 33.33, "total": 100, "message": "Processing item 1..."}

data: {"type": "progress", "progress": 66.67, "total": 100, "message": "Processing item 2..."}

data: {"type": "progress", "progress": 100, "total": 100, "message": "Processing item 3..."}

data: {"type": "result", "data": {"isError": false, "content": [...], "structuredContent": {"processed": [...]}}}

```

### Headers

The streaming endpoint sets appropriate headers for SSE:
- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `Connection: keep-alive`
- `X-Accel-Buffering: no` (disables nginx buffering)

### Limitations

- **Sessionful mode**: Progress streaming works best with sessionless tool calls. For sessionful sessions, the progress callbacks cannot currently be forwarded through the session task architecture, so the endpoint falls back to a regular tool call without progress updates.
- **Log messages**: MCP log notifications (`ctx.log()`) are handled at the session level and are not currently streamed through the SSE endpoint. They appear in server logs instead.

## OAuth Token Exchange & Extension
1. Set `OAUTH_ENV` to the environment variable that should receive the downstream provider token.
2. Supply an OAuth token via cookie (`_oauth2_proxy`) or the `X-Auth-Request-Access-Token` header.
3. `TokenRetrieverFactory` exchanges the token (e.g., Keycloak → Microsoft) and injects it into the MCP subprocess.

```bash
export OAUTH_ENV=MS_TOKEN
uvicorn app.server:app --reload
```

### Adding a custom provider
Implement your own retriever in `oauth/token_exchange.py` and register it with `TokenRetrieverFactory`.

```python
class MyCustomTokenRetriever:
    def retrieve_token(self, input_token: str) -> dict:
        return {"access_token": exchange_token_somewhere(input_token)}

TokenRetrieverFactory.register("my-provider", MyCustomTokenRetriever)
```

Select the retriever via configuration or environment variables and the bridge will handle injection automatically.

## Session Management & Extension

### Core behaviors
- Stateless tool calls (`POST /tools/{tool}`) spin up a short-lived MCP connection.
- Stateful sessions (`/session/start`) reuse the MCP process and share context until closed.
- Session identifiers are namespaced by OAuth identity to prevent collisions.
- Tokens flagged as `oauth_token` in tool schemas are injected automatically.
- Token exchange output is exposed to the MCP process via `OAUTH_ENV`.

### Stateless vs stateful requests
| Mode      | How                                            | When to use                     |
|-----------|------------------------------------------------|---------------------------------|
| Stateless | `POST /tools/{tool}` (no session header)       | One-off execution, low latency  |
| Stateful  | `POST /session/start` then include session id  | Conversations or long workflows |

### Common operations
```bash
# Start a session and store cookies locally
curl -X POST http://localhost:8000/session/start -c cookies.txt

# Invoke a tool within the session
curl -b cookies.txt -X POST http://localhost:8000/tools/add \
     -H 'Content-Type: application/json' \
     -d '{"a":2,"b":3}'

# Close the session when it is no longer needed
curl -b cookies.txt -X POST http://localhost:8000/session/close
```

### Extending the session manager
1. Subclass `SessionManagerBase` to back sessions with Redis, SQL, or another store.
2. Export the class from `session_manager/session_manager.py`.
3. Set `MCP_SESSION_MANAGER=<YourClassName>`.

## Testing and Development

### Dry-run (mock) mode for side-effect tools

Some tools perform modifying operations or otherwise have side effects (for example: creating, deleting or updating data). To safely preview or test those tools without executing their side effects, the bridge supports a dry-run (mock) mode that synthesizes realistic responses using the configured LLM.

How it works:

- Mark which tools are allowed to be dry-run by listing their names (comma-separated) in the `EFFECT_TOOLS` environment variable. Example `export EFFECT_TOOLS="add,create_memory,delete"`

- Trigger a dry-run by including the header `X-Inxm-Dry-Run: true` (case-insensitive) on a `POST /tools/{tool_name}` request. The dry-run path is taken only when the requested tool name is present in `EFFECT_TOOLS`; otherwise the tool executes normally.

- The server validates the supplied arguments against the tool's `inputSchema` (if present). Invalid inputs return an error comparable to normal tool validation.

- The bridge then asks the configured LLM to generate a mock response. The dry-run implementation:
  - Requires an LLM endpoint to be configured (the code checks `TGI_URL` / LLM client settings). Calls will fail if no LLM URL is configured.
  - If the tool declares an `outputSchema` the bridge requests JSON that conforms to that schema and attempts to parse the aggregated streamed output into structured JSON. When parsing succeeds the structured result is returned in the MCP-style `structuredContent` field. If parsing fails an error result is returned.
  - If no `outputSchema` is declared, the aggregated LLM text is returned as a single entry in the `content` array (object with a `text` field).

- Prompting: dry-run uses the same prompt service as the agent flow. You can provide a prompt named `dryrun_<tool_name>` to control a specific tool's mock output, or `dryrun_default` as a fallback. System-defined prompts (`SYSTEM_DEFINED_PROMPTS`) are merged with MCP-provided prompts and exposed via `GET /prompts`.

Example

```sh
export EFFECT_TOOLS='add'
export TGI_URL='https://api.example.com/v1'
export TGI_TOKEN="your_api_token"
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
curl -X POST http://localhost:8000/tools/add \
  -H 'Content-Type: application/json' \
  -H 'X-Inxm-Dry-Run: true' \
  -d '{"a":2,"b":3}'
```

Possible response (tool with an output schema):

```sh
{
  "isError": false,
  "content": [],
  "structuredContent": {"result": 5}
}
```

Notes:

- Dry-run is intentionally conservative: it only runs for tools explicitly listed in `EFFECT_TOOLS` to avoid accidentally mocking read-only tools.
- You can extend the dry-run functionality by tailored prompts - for example, by creating a prompt named `dryrun_<tool_name>` to control the mock output for a specific tool. It will choose the prompt in the following order:
  1. `dryrun_<tool_name>`
  2. `dryrun_default`
  3. fallback built-in prompt that asks for a reasonable mock response.
- Because dry-run uses an LLM, it can incur costs and introduce latency; use it for testing, previews or UI previews rather than high-volume production traffic.

#### Header → Input mapping (MCP_MAP_HEADER_TO_INPUT)

You can configure the bridge to map HTTP request headers directly into tool input arguments. This is useful for passing authenticated user identifiers, emails, or other per-request metadata into tools without exposing them as required fields in the public tool schema.

Format

Set `MCP_MAP_HEADER_TO_INPUT` to a comma-separated list of `input=Header-Name` entries. Example:

```bash
export MCP_MAP_HEADER_TO_INPUT="userId=x-auth-user-id,email=x-auth-user-email"
```

Behavior

- When tools are listed (`GET /tools`), any input properties that appear on the left-hand side of a mapping (e.g. `userId`) are removed from the returned `inputSchema`. This prevents those fields from being shown as required inputs to callers.
- When a tool is invoked (`POST /tools/{tool}`), if the tool declares an input property that matches a mapping and that argument is not supplied in the request body, the bridge will fill the argument from the incoming HTTP header value (case-insensitive header name matching). Existing explicit arguments provided in the request body are not overwritten.
- Mapping values are used as-is (string). If a header is missing the argument remains unset and normal validation applies.

Notes

- Mapping matches input properties exactly (no nested path support yet). If you need nested or more advanced mappings, I can extend this behavior.
- Mapped headers are looked up case-insensitively.
- Because mapped properties are removed from the schema, ensure that callers understand which inputs are supplied by the infrastructure vs. the client.


## User and Group Management

### How it works
1. Users authenticate via OAuth and receive a token containing identifiers and groups.
2. Requested group access is validated against token claims.
3. Resource paths are resolved using the templates described above.
4. Sanitized identifiers prevent traversal or injection attacks.

### Usage patterns

**Sessionless group access**
```bash
curl -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -X POST "http://localhost:8000/tools/search?group=finance" \
     -d '{"query": "budget reports"}'
```

**Session-based group access**
```bash
curl -H "Authorization: Bearer $TOKEN" \
     -X POST "http://localhost:8000/session/start?group=marketing" \
     -c cookies.txt

curl -b cookies.txt \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8000/tools/create_memory \
     -d '{"content": "Campaign ideas for Q4"}'
```

### Security features
- Group membership is derived from token claims such as `groups`, `realm_access.roles`, and `resource_access`.
- Unauthorized group access returns HTTP 403; invalid tokens return HTTP 401.
- All identifiers are sanitized and length-limited before being used.
- Data paths can separate user (`u/`), group (`g/`), and shared (`shared/`) resources.

### Directory layout example
```
/data/
├── u/
│   ├── user123.json
│   └── alice_smith.json
├── g/
│   ├── finance.json
│   ├── marketing.json
│   └── engineering.json
└── shared/
    └── knowledge-base.json
```

## Deploying as Agent

### Enable agent endpoints
Set the following environment variables to expose OpenAI-compatible agent endpoints:

- `TGI_URL` – Base URL of your OpenAI-compatible endpoint.
- `TGI_TOKEN` – Optional API token if the provider requires authentication.
- `TGI_MODEL_NAME` – Default model name used for completions.

### Endpoints
- `POST /tgi/v1/chat/completions` – Streaming or non-streaming chat completions with automatic MCP tool invocation.
- `POST /tgi/v1/a2a` – A2A JSON-RPC wrapper around the same capability for clients that speak the A2A protocol.
- `DELETE /tgi/v1/workflows/<execution_id>` – Cancel a background workflow execution.
- `GET /tgi/v1/workflows` – List workflows for the current user, ordered by `created_at` (desc). Supports paging via `limit` (1-100, default 20), `before` (ISO-8601 UTC timestamp), and `before_id` (execution id tiebreaker when multiple rows share the same timestamp). Response includes `created_at`, `last_change`, `status`, `awaiting_feedback`, and `current_agent`.
- `GET /.well-known/agent.json` – Agent metadata describing available tools.

#### Workflow listing

list workflows with paging (newest first), then page using the last item’s `created_at` + `execution_id`.

```bash
# First page
curl -X GET "http://localhost:8000/tgi/v1/workflows?limit=2" \
  -H "X-Auth-Request-Access-Token: $TOKEN"

# Next page (use last item's created_at + execution_id)
curl -X GET "http://localhost:8000/tgi/v1/workflows?limit=2&before=2024-01-15T10:30:00.123000Z&before_id=exec-123" \
  -H "X-Auth-Request-Access-Token: $TOKEN"
```

Example response:

```json
{
  "workflows": [
    {
      "execution_id": "exec-456",
      "workflow_id": "newsletter_flow",
      "status": "awaiting_feedback",
      "awaiting_feedback": true,
      "current_agent": "review_copy",
      "created_at": "2024-01-15T10:45:00.123000Z",
      "last_change": "2024-01-15T10:50:12.456000Z"
    },
    {
      "execution_id": "exec-123",
      "workflow_id": "plan_run",
      "status": "completed",
      "awaiting_feedback": false,
      "current_agent": "finalize",
      "created_at": "2024-01-15T10:30:00.123000Z",
      "last_change": "2024-01-15T10:35:10.789000Z"
    }
  ]
}
```

### System prompts
`SYSTEM_DEFINED_PROMPTS` can include prompts with `role: "system"` content. When no system message is supplied in the request, the bridge automatically prepends the configured system prompt, ensuring consistent agent behavior.

### Workflow-aware routing
Workflow-aware routing lets the bridge chain multiple agents with dependency ordering, reroute rules, and stateful resumes. It runs entirely on the bridge side, and is only for simple use cases: only MCP tools available to the current session are exposed to agents, reroutes cannot re-run agents that already completed, and cycles/loops in dependencies or reroutes will halt with an error instead of spinning forever.

It is not a full-featured orchestration engine, but it is useful for coordinating multi-step tasks without building a separate service.

You can let the bridge orchestrate simple multi-agent workflows. Configure:
- `WORKFLOWS_PATH`: Directory containing workflow JSON files (one per file).
- `WORKFLOW_DB_PATH` (optional): SQLite file for execution state (defaults to `<WORKFLOWS_PATH>/workflow_state.db`).
- Existing agent settings (`TGI_URL`, `TGI_TOKEN`, `TGI_MODEL_NAME`) still apply for LLM calls.

Each workflow file must include `flow_id`, `root_intent`, and an ordered `agents` list, for example:
```json
{
  "flow_id": "what_can_i_do_today",
  "root_intent": "SUGGEST_TODAY_ACTIVITIES",
  "agents": [
    {
      "agent": "get_location",
      "description": "Ask or infer the user's current city or coordinates.",
      "tools": ["get_location"]
    },
    {
      "agent": "get_weather",
      "description": "Fetch the weather forecast for a given location.",
      "pass_through": true,
      "depends_on": ["get_location"],
      "tools": ["get_weather", "get_forecast_for_location"]
    },
    {
      "agent": "get_outdoor",
      "description": "Suggest outdoor activities.",
      "depends_on": ["get_weather"],
      "when": "context.weather_ok is True",
      "reroute": {
        "on": ["NO_GOOD_OUTDOOR_OPTIONS"],
        "to": "get_restaurants"
      }
    },
    {
      "agent": "get_restaurants",
      "description": "Indoor ideas if outdoor is not suitable.",
      "depends_on": ["get_weather"]
    }
  ]
}
```
Field reference:
- `flow_id` (string): Unique id used by `use_workflow`.
- `root_intent` (string): Intent label; used for auto-selection when `use_workflow: true`.
- `loop` (bool, optional, default false): If true, the workflow stays open and reruns its agents for each new user message, preserving history so you can continue a normal chat across turns.
- `agents` (array, ordered): Steps executed in order when their dependencies are met.
  - `agent` (string): Name of the agent/prompt to run (also used to look up a custom prompt by name).
  - `description` (string): Default prompt text if no custom prompt exists for this agent.
  - `pass_through` (bool | string, default false): Controls response visibility. If `true`, the agent's streamed content is shown to the user. If a string, it acts as a response guideline instruction added to the agent's system prompt (e.g., `"Return only the searches you are performing"`).
  - `context` (bool | array[string] | `"user_prompt"`, default true): Controls how much workflow context is sent to the agent. `true` sends the full context (current behavior), `false` sends no context, `"user_prompt"` sends only the original user prompt captured when the workflow started, and an array limits context to specific references (e.g., `["detect.intent", "plan.steps"]`) using the same notation as arg mappings.
  - `depends_on` (array[string]): Agent names that must complete before this agent runs.
  - `when` (string, optional): Python-style expression evaluated against `context`; if falsy, the agent is skipped and marked with `reason: condition_not_met`.
  - `reroute` (object, optional): `{ "on": ["CODE1", ...], "to": "agent_name" }`. If the agent emits `<reroute>CODE1</reroute>`, the router jumps to the `to` agent next.
    - Tool-result reroutes are supported via `on` entries like `tool:tool_name:success` or `tool:tool_name:error`. These trigger reroutes immediately after tool results are processed, even if the LLM does not emit a `<reroute>` tag.
    - You can also hand off to another workflow by emitting `<reroute start_with='{"args": {...}}'>workflows[target_flow]</reroute>`. The engine completes the current workflow, resets state for `target_flow` within the same execution id (optionally applying the provided `start_with` payload), and continues streaming from the new workflow. Looping reroutes between workflows are detected and aborted.
    - If a reroute target is missing, has unmet dependencies, or was already completed earlier in the run, the engine stops and emits an error chunk explaining the reason (e.g., missing dependencies, already completed).
  - `tools` (array, optional): Limit available tools for this agent. Can be:
    - Array of strings: Simple tool names (e.g., `["get_weather", "search"]`)
    - Array of objects: Advanced tool configurations with settings and argument mappings (see below)
    - Empty array `[]`: Disable all tools for this agent
    - Omitted: All MCP tools are available
  - `returns` (array[string], optional): Field names to extract from tool results and store in context for use by subsequent agents.
  - `on_tool_error` (string, optional): Agent name to reroute to when a tool call fails. This provides automatic error handling when the LLM doesn't emit an explicit `<reroute>` tag. Useful for gracefully handling API failures or validation errors.

You can visualize your workflows using the [Workflow Visualizer](bin/viz-workflow.py):

```bash
python bin/viz-workflow.py ./your-workflow.json -o ./your-workflow.png
```

This will require `graphviz` to be installed (`pip install graphviz` & `apt install graphviz`).

##### Automatic tool error handling

When a tool call fails (returns an error response like `400 Bad Request`, `Error:`, etc.), the workflow engine can automatically reroute to a failure-handling agent. This is useful when you want guaranteed error handling without relying on the LLM to detect and emit reroute tags.

Configure `on_tool_error` on any agent that uses tools which may fail:

```json
{
  "agent": "save_plan",
  "description": "Save the plan to the database",
  "tools": ["save_plan"],
  "on_tool_error": "summarize_creation_failure",
  "depends_on": ["create_plan"]
},
{
  "agent": "summarize_creation_failure",
  "description": "Explain to the user that saving failed and suggest next steps",
  "tools": []
}
```

Behavior:
- When `save_plan` tool returns an error, the workflow automatically reroutes to `summarize_creation_failure`
- If the LLM emits an explicit `<reroute>` tag, that takes precedence over `on_tool_error`
- If a `reroute.on` tool trigger matches (e.g., `tool:save_plan:error`), it takes precedence over `on_tool_error`
- Error detection recognizes common patterns: HTTP status codes (4xx, 5xx), "Error:", "Failed:", exception messages, etc.
- The error details are stored in the agent's context and available to the failure handler

##### Tool-result reroutes

You can also reroute directly on tool outcomes without requiring the LLM to emit `<reroute>`:

```json
{
  "agent": "get_plan",
  "description": "Fetch the plan",
  "tools": ["get_plan"],
  "reroute": [
    { "on": ["tool:get_plan:success"], "to": "analyse_plan" },
    { "on": ["tool:get_plan:error"], "to": "get_plan_failed" }
  ]
}
```

When a tool-result reroute matches, the engine skips the post-tool LLM turn, so the tool payload isn't sent back to the model.

##### Advanced tool configuration

Tools can be configured with settings and argument mappings:

```json
{
  "agent": "create_plan",
  "description": "Create a detailed plan",
  "tools": [
    "simple_tool",
    {
      "plan": {
        "settings": { "streaming": true },
        "args": { "selected-tools": "select_tools.selected_tools" }
      }
    }
  ],
  "depends_on": ["select_tools"]
}
```

Tool configuration options:
- `settings.streaming` (bool): If `true`, the tool is streaming to receive progress updates.
- `args` (object): Maps tool input arguments to values from previous agent contexts. Format: `"input_arg": "agent_name.field_name"`. Mapped arguments are automatically injected and removed from the tool schema presented to the LLM.

##### Using `returns` to capture tool outputs

The `returns` field allows you to capture specific fields from tool results for use in subsequent agents. This allows you to pass complex/plentiful information without bloating the context. It supports multiple formats for flexible data extraction:

**Simple field capture (from any tool):**
```json
{
  "agent": "select_tools",
  "tools": ["select_tools"],
  "returns": ["selected_tools"]
}
```

**Nested path capture:**
```json
{
  "agent": "create_plan",
  "tools": ["plan"],
  "returns": ["payload.result"]
}
```
This captures the nested value at `payload.result` from the tool response and stores it at the same path in the agent's context.

**Tool-specific capture (when agent calls multiple tools):**
```json
{
  "agent": "create_plan",
  "tools": ["select_tools", "plan"],
  "returns": [
    {"field": "selected_tools", "from": "select_tools"},
    {"field": "payload.result", "from": "plan", "as": "plan"}
  ]
}
```
- `field`: The path to capture from the tool result
- `from`: Only capture from this specific tool (ignores results from other tools)
- `as` (optional): Store the value under this alias instead of the original field path

Notes:
- If the same agent emits multiple tool results for the same `returns` target (e.g., calling `resolve_mcp_properties_defaults` several times with `{ "as": "properties" }`), the value is overwritten each time and only the last call is kept in context.

**Complete example with tool-specific captures:**
```json
{
  "agent": "select_tools",
  "description": "Select appropriate tools for the task",
  "tools": ["select_tools"],
  "returns": ["selected_tools"],
  "pass_through": true
},
{
  "agent": "create_plan",
  "description": "Create a detailed plan using selected tools",
  "tools": [
    {"plan": {"args": {"selected_tools": "select_tools.selected_tools"}}}
  ],
  "returns": [{"field": "payload.result", "from": "plan", "as": "plan"}],
  "depends_on": ["select_tools"]
},
{
  "agent": "confirm_plan",
  "description": "Present the plan and ask for approval. On the first run, always emit <user_feedback_needed> with two choices: approve (save) or retry (workflows[plan_create]). On resume, interpret free-text feedback: emit <reroute>APPROVE</reroute> to proceed or <reroute>RETRY</reroute> to regenerate.",
  "context": ["create_plan.plan", "feedback", "user_query"],
  "pass_through": "Summarize the plan in 3-6 bullets and ask if it looks good or should be reworked.",
  "reroute": [
    { "on": ["APPROVE"], "to": "save_plan" },
    { "on": ["RETRY"], "to": "workflows[plan_create]" }
  ],
  "depends_on": ["create_plan"]
},
{
  "agent": "save_plan",
  "description": "Save the created plan",
  "tools": [
    {"save_plan": {"args": {"plan": "create_plan.plan"}}}
  ],
  "depends_on": ["confirm_plan"]
}
```

In this example:
1. `select_tools` agent runs and captures `selected_tools` from its tool result
2. `create_plan` agent receives `selected_tools` via arg mapping, runs the `plan` tool, and captures `payload.result` specifically from the `plan` tool, storing it as `plan`
3. `confirm_plan` presents the plan, asks for approval, and either reroutes to `save_plan` or restarts with `workflows[plan_create]`
4. `save_plan` agent receives the plan via `create_plan.plan` arg mapping

Note:
- Workflow reroutes must use the plural form `workflows[plan_create]`. This is required by the engine’s reroute parser.
- If the user replies in free text instead of clicking a button, the `confirm_plan` prompt handles it by emitting `<reroute>APPROVE</reroute>` or `<reroute>RETRY</reroute>` based on the response.

Prompt usage:
- For each agent the router looks for a prompt named exactly like the `agent` value via the MCP prompt service. If found, that prompt content is used; otherwise it falls back to the agent’s `description`.
- Prompts receive: the original user request, the workflow goal (`root_intent`), and the accumulated `context` JSON from prior agents.
- Streaming is used for agent runs; `pass_through: true` agents stream their content to the user while still being stored in context.
- A `routing_agent` prompt can be provided in the MCP prompt store; if absent, a built-in default is used. The routing agent checks overall intent fit, evaluates `when` conditions, and chooses reroutes.
- Routing prompt lookup: the routing agent prompt is resolved by the workflow’s `flow_id` as the prompt name. If found, it is appended with the default routing instructions (tags for `<run>`, `<reroute>`, `<next_agent>`) so custom text always inherits the control contract.

Rerouting and feedback:
- Agents can emit special tags in their output:
  - `<reroute>REASON</reroute>`: If the current agent has a matching `reroute.on`, execution jumps to that target agent. Otherwise the reason is stored on the agent context.
  - `<user_feedback_needed>...optional text...</user_feedback_needed>`: The workflow pauses, persists state, and streams a “User feedback needed” event. When the user calls the same `workflow_execution_id` again with a new message, the engine resumes from that point.
- Agents can emit `<return name="key.path">value</return>` to store lightweight values directly on their agent context (supports dotted paths like `meta.author`).
- Reroute entries support `"with": ["field1", ...]` to copy those agent fields into the shared workflow context when the reroute triggers (useful for shared values like `plan_id` that downstream agents can reference without knowing which agent produced them).
- To override a reroute inside a prompt, use `<no_reroute>` (stored but not forced; user can still choose to proceed by omitting reroute tags).
- If the routing agent judges that the user request doesn’t match `root_intent`, it returns only `<reroute>reason</reroute>` and stops. For `when` conditions, the routing agent evaluates the condition against the current `context` (LLM-driven) and skips the agent if it returns `<run>false</run>`. When a reroute reason isn’t mapped, the routing agent can suggest the next agent via `<next_agent>name</next_agent>`. Users can force continuing without reroute by including `<no_reroute>` in their request.

Runtime behavior:
- `use_workflow: "<flow_id>"` forces that workflow; `use_workflow: true` auto-selects by matching `root_intent` to the user request.
- Optional `workflow_execution_id` resumes a prior execution (state is persisted in SQLite). Without it, a new execution id is generated.
- Optional `return_full_state: true` replays the stored workflow event history when resuming; omit or set to `false` to stream only newly generated events.
- Optional background execution: include the header `x-inxm-workflow-background: true` on streaming requests to keep the workflow running even if the client disconnects. Reconnect with the same `workflow_execution_id` to receive the full history and then continue streaming live updates. Use `DELETE /tgi/v1/workflows/<execution_id>` to cancel a background run.
- Workflow executions are bound to the user token that started them; resuming with a different token returns an access error.
- The routing agent streams status events; agents marked `pass_through: true` stream their content to the user.
- `persist_inner_thinking` (boolean, optional) controls how much agent output is stored in workflow context for later turns. Defaults to `false`, which keeps only pass-through content and explicit returns, trimming internal reasoning to avoid oversized LLM payloads. Set to `true` if you need full agent transcripts preserved across resumes.
- `start_with` (object, optional) lets you prefill workflow context and optionally jump directly to a specific agent. Shape: `{"args": {"plan_id": "abc"}, "agent": "get_plan"}`. `args` are copied into the workflow context (top-level keys), and `agent` forces the first runnable agent; its declared dependencies are marked completed with reason `start_with_prefill` so it can run immediately.
- Agent-level `context` controls how much prior workflow state each agent sees: set to `false` to omit context entirely, or provide a list of references (e.g., `["plan.steps", "detect.intent"]`) to send only selected values using the same notation as arg mappings.

#### Example: 3-step workflow with feedback + client response
Below is a short workflow that finds matching plans, asks the user to pick one (or create a new plan), then runs the selected plan.

```json
{
  "flow_id": "plan_run",
  "root_intent": "RUN_PLAN",
  "agents": [
    {
      "agent": "find_plan",
      "description": "Find matching plans. Emit plan list or reroute to create.",
      "tools": ["search_plan"],
      "reroute": [
        {
          "on": ["FOUND_MULTIPLE_MATCHES"],
          "ask": {
            "question": "Present matching plans and ask the user to select one or create a new plan.",
            "expected_responses": [
              {
                "select_plan": { "to": "select_run_mode", "each": "plans", "with": ["plan_id"] },
                "create_new": { "to": "workflows[plan_create]" }
              }
            ]
          }
        }
      ]
    },
    {
      "agent": "select_run_mode",
      "description": "Choose how to run the selected plan.",
      "depends_on": ["find_plan"]
    },
    {
      "agent": "run_plan",
      "description": "Run the plan now.",
      "depends_on": ["select_run_mode"]
    }
  ]
}
```

**What the server streams when a selection is needed:**
```xml
<user_feedback_needed>{"question":"Which plan should I run?","expected_responses":[{"id":"select_plan","to":"select_run_mode","each":"plans","with":["plan_id"],"options":[{"key":"plan-1","value":"Plan One - ..."},{"key":"plan-2","value":"Plan Two - ..."}]},{"id":"create_new","to":"workflows[plan_create]"}]}</user_feedback_needed>
```

**Client response (user selects plan-2):**
```xml
<user_feedback>select_run_mode("plan-2")</user_feedback>
```

Notes:
- The client should return the `to` target plus the selected `key`.
- If a choice needs to propagate context, it must be explicitly declared in `with`. The engine does not infer it.

#### Example: start or resume a workflow
```bash
curl -X POST http://localhost:8000/tgi/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What can I do today in SF?"}],
    "use_workflow": "what_can_i_do_today",
    "workflow_execution_id": "my-session-123"
  }'
```

#### Example: prefill and start at a specific agent
```bash
curl -X POST http://localhost:8000/tgi/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Run the plan"}],
    "use_workflow": "plan_run",
    "start_with": {
      "args": { "plan_id": "plan-123" },
      "agent": "get_plan"
    }
  }'
```

#### Typical Problems

- **Reroute to missing agent**: Ensure the target agent exists and has all dependencies met.
- **Empty Tools vs No Tools**: An empty `tools: []` disables all tools for that agent, while omitting `tools` allows all MCP tools.
- **Context Overload**: Large contexts can exceed LLM input limits; use agent-level `context` settings to limit data sent.

### Example call
```bash
curl -X POST http://localhost:8000/tgi/v1/chat/completions \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Agent mode can generate significant LLM traffic; monitor costs and apply tool filters (`INCLUDE_TOOLS`/`EXCLUDE_TOOLS`) as needed.

## Using resources

The `/resources` endpoint provides a way to list and retrieve details about resources managed by the MCP server. This endpoint supports both session-based and sessionless access, making it flexible for various use cases.

### Listing Resources

To list all available resources, send a `GET` request to `/resources`.

### Retrieving Resource Details

To retrieve details about a specific resource, send a `GET` request to `/resources/{resource_name}`.

Resources can have different MIME types, such as `text/plain`, `text/html`, or `application/json`. The server will return the appropriate content type based on the resource's MIME type, and render it accordingly. Thus, it allows you to directly serve HTML content or plain text as needed.

## AI-Generated Web Applications

The bridge includes a powerful feature for generating complete web applications and dashboards directly from natural language prompts. This enables rapid prototyping and deployment of custom interfaces on top of your MCP tools.

### Overview

The generated web application feature:
- **LLM-powered generation**: Uses your configured LLM to translate natural language requirements into working HTML applications
- **Progressive enhancement**: Built on [pfusch](https://matthiaskainer.github.io/pfusch/), a lightweight library for creating reactive web components without build steps
- **MCP tool integration**: Automatically connects to your MCP tools with proper authentication and data handling
- **User and group isolation**: Each application is scoped to specific users or groups with proper access control
- **Versioned updates**: Maintains full history of changes with the ability to iterate and refine applications
- **Streaming generation**: Real-time Server-Sent Events (SSE) for responsive UI updates during generation
- **Zero build process**: Generated applications work directly in the browser with no compilation or bundling required

### Configuration

To enable AI-generated applications, configure these environment variables:

```bash
# Required: Path where generated applications are stored
export GENERATED_WEB_PATH="/data/generated-apps"

# Required: LLM endpoint (already required for agent mode)
export TGI_URL="https://api.openai.com/v1"
export TGI_TOKEN="your-api-token"
export TGI_MODEL_NAME="gpt-4o"
```

### Creating Applications

#### Create a new application

```bash
curl -X POST "http://localhost:8000/app/_generated/group=engineering" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "absence-dashboard",
    "name": "absence-tracker",
    "prompt": "Create a dashboard showing employee absence data with filters for date range and absence type",
    "tools": ["list_absence_types", "get_absences"]
  }'
```

The response is streamed as Server-Sent Events (SSE):
- `data:` events contain incremental generation progress
- `event: done` signals completion with the full application record
- `event: error` indicates generation failures

#### Scope Patterns
Applications can be scoped to users or groups:
- **Group scope**: `group=engineering` - accessible by all members of the engineering group
- **User scope**: `user=alice_smith` - private to a specific user

#### Request Parameters
- `id`: Unique identifier for the application (alphanumeric with dashes/underscores)
- `name`: Human-readable name (also used in file paths)
- `prompt`: Natural language description of what the application should do
- `tools` (optional): Array of MCP tool names to include; if omitted, relevant tools are auto-selected based on the prompt

### Retrieving Applications

#### Get application metadata and HTML

```bash
# Get full record with metadata (default)
curl "http://localhost:8000/app/_generated/group=engineering/absence-dashboard/absence-tracker?as=card" \
  -H "Authorization: Bearer $TOKEN"

# Get rendered HTML page
curl "http://localhost:8000/app/_generated/group=engineering/absence-dashboard/absence-tracker?as=page" \
  -H "Authorization: Bearer $TOKEN"

# Get HTML snippet for embedding
curl "http://localhost:8000/app/_generated/group=engineering/absence-dashboard/absence-tracker?as=snippet" \
  -H "Authorization: Bearer $TOKEN"
```

Response formats:
- `as=card` (default): Returns complete metadata including generation history, tools used, and current HTML
- `as=page`: Returns the full HTML document ready to display
- `as=snippet`: Returns just the HTML body content for embedding in another page

### Updating Applications

Iterate on existing applications by posting update prompts:

```bash
curl -X POST "http://localhost:8000/app/_generated/group=engineering/absence-dashboard/absence-tracker" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Add a summary card showing total absences by type",
    "tools": ["list_absence_types", "get_absences", "calculate_totals"]
  }'
```

Updates:
- Preserve the original requirements and generation history
- Use context from previous versions to maintain consistency
- Allow incremental refinement without starting from scratch
- Return the updated application with new timestamp and history entry

### Customizing Design Systems

Applications can be styled according to your design system. Set a global design prompt:

```bash
export SYSTEM_DEFINED_PROMPTS='[
  {
    "name": "design-system",
    "description": "Corporate design system guidelines",
    "template": "Use our corporate color palette: primary #0066cc, secondary #00cc66. All cards should have rounded corners and subtle shadows. Use system fonts."
  }
]'
```

The design prompt is automatically incorporated during generation to ensure visual consistency across all applications.

### Access Control

Applications respect OAuth-based access control:
- **User-scoped apps** can only be created and accessed by the owning user
- **Group-scoped apps** require the actor to be a member of the target group
- All requests must include a valid access token
- Group membership is derived from token claims (`groups`, `realm_access.roles`, `resource_access`)

### Storage Structure

Generated applications are stored in a hierarchical structure:

```
${GENERATED_WEB_PATH}/
├── user/
│   └── alice_smith/
│       └── my-dashboard/
│           └── personal-view/
│               └── ui.json
└── group/
    └── engineering/
        └── absence-dashboard/
            └── absence-tracker/
                └── ui.json
```

Each `ui.json` contains:
- `metadata`: Creation timestamps, owner info, generation history
- `current`: The latest HTML and metadata including component lists
- Version history with prompts and tool selections for each iteration

### Technical Implementation

Generated applications use [pfusch](https://github.com/matthiaskainer/pfusch), a progressive enhancement library that:
- Works without build tools or compilation
- Provides reactive state management through ES6 Proxies
- Uses Web Components (custom elements) for encapsulation
- Supports shadow DOM with automatic style injection via `data-pfusch` attributes
- Enables event-driven communication between components

Example generated component:

```javascript
pfusch('data-viewer', { items: [], loading: false }, (state, trigger, helpers) => [
  script(async function() {
    state.loading = true;
    const response = await fetch('/app/tools/list_items', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    const mcpResponse = await response.json();
    if (!mcpResponse.isError) {
      state.items = mcpResponse.structuredContent.result;
    }
    state.loading = false;
  }),
  html.ul(...state.items.map(item => html.li(item.name)))
])
```

### Best Practices

1. **Be specific in prompts**: Include details about layout, interactions, and data requirements
2. **List relevant tools**: Pre-selecting tools improves generation speed and relevance
3. **Iterate incrementally**: Make small updates rather than regenerating from scratch
4. **Use descriptive IDs**: Application IDs should clearly indicate their purpose
5. **Leverage design prompts**: Define reusable design systems for visual consistency
6. **Test with real tokens**: Ensure your OAuth configuration provides proper group claims

### Example: Complete Workflow

```bash
# 1. Create initial dashboard
curl -X POST "http://localhost:8000/app/_generated/group=finance/budget-tracker/overview" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "budget-tracker",
    "name": "overview",
    "prompt": "Create a budget overview dashboard with monthly spending charts and category breakdowns. Include filters for date range.",
    "tools": ["get_transactions", "list_categories"]
  }'

# 2. View the generated application
curl "http://localhost:8000/app/_generated/group=finance/budget-tracker/overview?as=page" \
  -H "Authorization: Bearer $TOKEN" > dashboard.html

# 3. Refine with additional features
curl -X POST "http://localhost:8000/app/_generated/group=finance/budget-tracker/overview" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Add an export button to download transactions as CSV"
  }'

# 4. Get metadata including history
curl "http://localhost:8000/app/_generated/group=finance/budget-tracker/overview?as=card" \
  -H "Authorization: Bearer $TOKEN"
```

## Generic Web Application Proxy

The bridge includes a production-ready reverse proxy at `/app/*` that can forward requests to any web application. This enables you to:

- Run a web application on top of your MCP, and to for instance host dashboards, notebooks, or other UIs behind the same authentication layer as your MCP server.
- Provide a unified entry point for multiple services
- Handle HTTPS termination and header forwarding transparently
- Automatically rewrite URLs in HTML, CSS, JavaScript, and JSON responses

### Configuration

Set the following environment variables to enable the proxy:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TARGET_SERVER_URL` | Yes | - | Backend server URL to proxy to (e.g., `http://internal-app:8080`) |
| `PROXY_PREFIX` | No | `${MCP_BASE_PATH}/app` | URL prefix for proxy endpoints |
| `PUBLIC_URL` | No | (auto-detected) | Public-facing URL for redirect rewriting |
| `PROXY_TIMEOUT` | No | `300` | Request timeout in seconds |
| `REWRITE_HTML_URLS` | No | `true` | Enable URL rewriting in HTML content |
| `REWRITE_JSON_URLS` | No | `true` | Enable URL rewriting in JSON responses |
| `REWRITE_CSS_URLS` | No | `true` | Enable URL rewriting in CSS files |
| `REWRITE_JS_URLS` | No | `true` | Enable URL rewriting in JavaScript files |

### Features

**Header Management**
- Automatically adds `X-Forwarded-*` headers (For, Host, Proto, Prefix)
- Removes hop-by-hop headers that shouldn't be forwarded
- Preserves all custom headers from clients

**URL Rewriting**
- Rewrites absolute and relative URLs in HTML (`href`, `src` attributes)
- Rewrites `url()` functions and `@import` statements in CSS
- Rewrites URL strings in JavaScript code
- Rewrites URLs in JSON API responses
- Prevents double-prefixing when URLs are already rewritten

**Cookie & Redirect Handling**
- Automatically adjusts cookie paths to match the proxy prefix
- Rewrites `Location` headers in redirects to point to the proxy
- Preserves all cookie attributes (HttpOnly, Secure, etc.)

**Performance & Security**
- Streams large responses for memory efficiency
- Buffers only text content that needs URL rewriting
- Handles compression (gzip, deflate) transparently
- Proper error handling with appropriate HTTP status codes

### Usage Examples

**Basic proxying**
```bash
export TARGET_SERVER_URL="http://internal-dashboard:3000"
export PROXY_PREFIX="/apps/dashboard"
export PUBLIC_URL="https://portal.example.com"
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Users can then access the dashboard at: `https://portal.example.com/apps/dashboard`

**Multiple applications**
Run multiple bridge instances with different prefixes to host multiple apps:

```yaml
# Application 1 - Grafana
- name: grafana-proxy
  env:
    - name: TARGET_SERVER_URL
      value: "http://grafana:3000"
    - name: PROXY_PREFIX
      value: "/apps/grafana"

# Application 2 - Jupyter
- name: jupyter-proxy
  env:
    - name: TARGET_SERVER_URL
      value: "http://jupyter:8888"
    - name: PROXY_PREFIX
      value: "/apps/jupyter"
```

**Disable URL rewriting for specific content types**
```bash
# If your app handles paths correctly via X-Forwarded-Prefix
export REWRITE_JS_URLS="false"
export REWRITE_CSS_URLS="false"
```

### How It Works

1. **Request arrives** at `/apps/myapp/resource`
2. **Headers prepared**: Adds X-Forwarded-* headers, removes hop-by-hop headers
3. **URL translated**: `/apps/myapp/resource` → `http://target/resource`
4. **Request forwarded** to target server
5. **Response processed**: Rewrites Location, Set-Cookie, and content URLs as needed
6. **Response streamed** back to client

### Common Use Cases

**Hosting applications behind authentication**
```bash
export TARGET_SERVER_URL="http://grafana:3000"
export PROXY_PREFIX="/monitoring/grafana"
# Users authenticate to the bridge, then access Grafana seamlessly
```

**Development/staging environments**
```bash
export TARGET_SERVER_URL="http://localhost:3000"
export PROXY_PREFIX="/dev"
# Access local dev server through the proxy for testing
```

**HTTPS termination**
The proxy handles HTTPS termination automatically:
```
[Client HTTPS] → [Bridge HTTPS] → [HTTP] → [Target App HTTP]
```

The target app receives `X-Forwarded-Proto: https` to generate correct URLs.

### Troubleshooting

**Assets returning 404**
- Ensure URL rewriting is enabled for the content type
- Check that `PROXY_PREFIX` matches your ingress configuration

**Redirects pointing to internal URLs**
- Set `PUBLIC_URL` to your public-facing domain
- Verify the target app isn't hardcoding absolute URLs
