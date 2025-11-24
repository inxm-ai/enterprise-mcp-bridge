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
| `GENERATED_WEB_PATH`      | Base directory for storing AI-generated web applications  | ""                     |
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
  - `POST /app/_generated/{scope}` – generate a new web application from a natural language prompt.
  - `GET /app/_generated/{scope}/{ui_id}/{name}` – retrieve a generated application (as metadata, page, or snippet).
  - `POST /app/_generated/{scope}/{ui_id}/{name}` – update an existing generated application.
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
export TGI_API_TOKEN="your-api-token"
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

