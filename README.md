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

# Set any additional MCP Header vars here:
# export MCP_REMOTE_HEADER_X_CUSTOM="custom-value"

uvicorn app.server:app --host 0.0.0.0 --port 8000
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
| `TOKEN_NAME`              | Name of the cookie containing the OAuth token             | _oauth2_proxy          |
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
| `MCP_REMOTE_HEADER_*`     | Any additional headers to send to the remote MCP server   |                        |
| `SYSTEM_DEFINED_PROMPTS`  | JSON array of built-in prompts available to all users     | "[]"                   |
| `MCP_ENV_*`               | Forwarded to the MCP server process                       |                        |
| `MCP_*_DATA_ACCESS_TEMPLATE` | Template for specific data resources. See [User and Group Management](#user-and-group-management) for details. | `{*}/{placeholder}` |

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
    "template": "Hello, {name}!"
  }
]
```

Prompts are merged with prompts returned by the MCP server and are available through `/prompts`.

## API Docs
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc rendering: `http://localhost:8000/redoc`
- Key operations:
  - `GET /tools` – list available tools.
  - `GET /tools/{tool_name}` – retrieve a tool's details.
  - `POST /tools/{tool_name}` – invoke a tool (stateless unless a session header is provided).
  - `GET /prompts` – list available prompts (including system-defined).
  - `GET /resource` – list available resources.
  - `GET /resource/{resource_id}` – access resources. If the resource is html/text, it is rendered directly.
  - `POST /session/start` and `POST /session/close` – manage stateful sessions.
  - `GET /.well-known/agent.json` – discover agent metadata when agent mode is enabled.

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
export TGI_API_TOKEN="your_api_token"
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

- `TGI_API_URL` – Base URL of your OpenAI-compatible endpoint.
- `TGI_API_TOKEN` – Optional API token if the provider requires authentication.
- `TGI_MODEL_NAME` – Default model name used for completions.

### Endpoints
- `POST /tgi/v1/chat/completions` – Streaming or non-streaming chat completions with automatic MCP tool invocation.
- `POST /tgi/v1/a2a` – A2A JSON-RPC wrapper around the same capability for clients that speak the A2A protocol.
- `GET /.well-known/agent.json` – Agent metadata describing available tools.

### System prompts
`SYSTEM_DEFINED_PROMPTS` can include prompts with `role: "system"` content. When no system message is supplied in the request, the bridge automatically prepends the configured system prompt, ensuring consistent agent behavior.

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
