# MCP REST Server

FastAPI-based wrapper that exposes any Model Context Protocol (MCP) server over plain HTTP/JSON.

## Why this project? (Key Capabilities & Differences)

Most MCP examples today are:
- Single-user CLI processes (stdio pipes) driven by an LLM client
- One-session-per-process (state lost when process ends)
- Lack built-in multi-session & multi-user orchestration
- Assume direct access instead of sandboxed / remote / browser environments
- Provide no integrated OAuth token exchange for downstream resource access

This server bridges those gaps:

| Capability | What it means here | Why it matters |
|------------|--------------------|----------------|
| REST-first | All MCP tool discovery & invocation via standard HTTP endpoints | Works with any platform (browser, mobile, gateway, curl, automation) |
| Multi-user | Session keys are namespaced by (optional) user OAuth token | A single deployment can safely serve many authenticated users |
| Multi-session per user | Start/close multiple concurrent sessions (`/session/start`) | Parallel tool chains, isolation, long-running conversational state |
| Stateless default calls | You can invoke tools without starting a session | Low-latency fire-and-forget usage / health checks |
| Session lifecycle mgmt | Explicit start/close endpoints & inactivity timeout pings | Prevent resource leaks & enable autoscaling-friendly behavior |
| Pluggable session manager | Environment-switchable (default in-memory) | Future: Redis / DB backed horizontal scale |
| OAuth token exchange built-in | Keycloak→downstream provider token retrieval & injection | Securely pass resource provider tokens into MCP tools without exposing them to clients |
| Automatic token arg injection | If a tool declares `oauth_token` in its input schema it is auto-satisfied | Cleaner client payloads; security gating |
| Command routing & precedence | `MCP_SERVER_COMMAND` / CLI `--` / default Python server | Flexible deployment (Docker, K8s, local dev) |
| Runtime base path | `MCP_BASE_PATH` environment variable | Clean ingress integration / versioned APIs |
| Tool URL mapping | Each tool gets a canonical REST endpoint | Easier discovery & linking |
| Structured error handling | HTTP 400 / 404 / 500 mapping from MCP responses | Predictable integration DX |
| Token refresh logic | Keycloak broker + refresh token reuse | Long-lived sessions without manual refresh |

And one more thing:
- Future extension hook: swap `SessionManagerBase` for distributed stores.
- Ping-based inactivity detection (auto-closes idle sessions >60s inactivity inside task loop) — helps resource hygiene.
- Minimal surface area: only a few focused endpoints; easy to audit.
- Works with any stdio MCP server (Python, Node, etc.) launched as a subprocess.

---

## Core Concepts

1. Ad-hoc (stateless) tool calls: Just POST to `/tools/{tool_name}` with arguments. A temporary MCP connection is created and torn down.
2. Managed sessions: Start with `/session/start`, receive a session ID cookie (`x-inxm-mcp-session`). Subsequent tool calls reuse a persistent MCP process context (supports stateful tools).
3. Multi-user separation: If an OAuth token is present (cookie `_oauth2_proxy`) it is appended internally to the session identifier to avoid collisions and enforce per-user isolation.
4. Automatic OAuth token propagation: When a tool declares an `oauth_token` property in its JSON schema, the server injects a validated token value; clients need not send it explicitly.
5. Token exchange: Incoming Keycloak token can be exchanged for a provider (e.g., Microsoft) token using `TokenRetrieverFactory` and exported into the MCP subprocess environment variable defined by `OAUTH_ENV`.

---

## Quick Start

```bash
git clone https://github.com/inxm-ai/mcp-rest-server.git
cd mcp-rest-server
pip install app
uvicorn app.server:app --reload
```

Open: http://localhost:8000/docs

---

## Session vs Stateless Calls

| Mode | How | When to use |
|------|-----|-------------|
| Stateless | `POST /tools/{tool}` (no session header) | Simple one-off tool execution |
| Stateful | `POST /session/start` then include header/cookie | Conversations, multi-step pipelines |

### Start a Session

```bash
curl -X POST http://localhost:8000/session/start -c cookies.txt
```

The response sets `x-inxm-mcp-session` cookie.

### Use the Session

```bash
curl -b cookies.txt -X POST http://localhost:8000/tools/add -H 'Content-Type: application/json' -d '{"a":2,"b":3}'
```

### Close the Session

```bash
curl -b cookies.txt -X POST http://localhost:8000/session/close
```

If you prefer headers:

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/session/start | jq -r '."x-inxm-mcp-session"')
curl -H "x-inxm-mcp-session: $SESSION_ID" -X POST http://localhost:8000/tools/add -H 'Content-Type: application/json' -d '{"a":1,"b":4}'
```

---

## Multi-User Handling

If your ingress / auth proxy (e.g., oauth2-proxy) sets a user token cookie `_oauth2_proxy`, the server internally namespaces sessions by that token. Two users can both have a logical session ID `abc123` but their tokens create distinct internal keys (`abc123:<token-hash>`). You do not need to manage this manually.

---

## OAuth Token Exchange & Injection

1. Set `OAUTH_ENV=MS_TOKEN` (example) when starting the REST server.
2. Provide a Keycloak user token (via cookie `_oauth2_proxy` or CLI argument when launching underlying MCP server if applicable).
3. The `TokenRetrieverFactory` chooses the retriever (currently Keycloak) and exchanges the Keycloak token for the provider token.
4. The resulting provider token is exported into the MCP subprocess environment (`MS_TOKEN`).
5. If a tool input schema contains `oauth_token`, the call decorator injects the raw user Keycloak token (or accessible token) unless customized.

Example (local dev with environment variable):

```bash
export OAUTH_ENV=MS_TOKEN
uvicorn app.server:app --reload
```

### Adding a Custom Provider
Implement a new retriever in `oauth/token_exchange.py` and register it in `TokenRetrieverFactory.get()`.

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MCP_SERVER_COMMAND` | Full command to launch MCP server (overrides all) | (unset) |
| `MCP_BASE_PATH` | Prefix all routes (e.g., `/api/mcp`) | "" |
| `OAUTH_ENV` | Name of env var injected into MCP subprocess with provider token | (unset) |
| `AUTH_PROVIDER` | Token exchange provider selector (`keycloak`) | keycloak |
| `AUTH_BASE_URL` | Keycloak base URL for broker/token endpoints | (required for exchange) |
| `KEYCLOAK_REALM` | Keycloak realm | inxm |
| `KEYCLOAK_PROVIDER_ALIAS` | External IdP alias used in broker path | (required) |
| `MCP_SESSION_MANAGER` | Implementation name (e.g., future Redis) | InMemorySessionManager |

---

## Command Precedence & Launching MCP Server

Launch order logic:
1. If `MCP_SERVER_COMMAND` is set → split & exec.
2. Else run the default demo `../mcp/server.py`.

Docker override example:

```bash
docker run -it -p 8000:8000 -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/memory.json" inxm-ai/mcp-rest-server
```

---

## Base Path / Ingress Integration

Set `MCP_BASE_PATH=/api/mcp` to serve endpoints like `/api/mcp/tools`. Helpful when sitting behind a shared gateway or versioned path.

---

## Tool Discovery & Invocation

List tools:

```bash
curl http://localhost:8000/tools
```

Each listed tool returns a JSON object including a direct URL. Invoke by POSTing JSON arguments.

Unknown tools → HTTP 404. Validation errors → HTTP 400. Internal tool errors → HTTP 500.

---

## Inactivity & Resource Hygiene

Each persistent session schedules a lightweight ping every 10s. If no non-ping request is seen for >60s, the underlying MCP stdio session is closed, returning `session_closed` internally. Clients should recreate sessions as needed for very long idle periods.

---

## Extending Session Management

`session_manager()` factory reads `MCP_SESSION_MANAGER`. To add a backend:

1. Implement a subclass of `SessionManagerBase`.
2. Expose it in `session_manager/session_manager.py` globals.
3. Set `MCP_SESSION_MANAGER=<YourClassName>`.

Future direction: Redis-backed manager for horizontal scaling.

---

## Kubernetes / Helm Notes

Same as before, plus optionally set `MCP_BASE_PATH` for ingress-friendly routing. Mount volumes for `/data` or `/config` as needed. Supply `MCP_SERVER_COMMAND` for non-Python servers.

---

## Security Considerations

- Do not expose this service publicly without an auth proxy placing user tokens in `_oauth2_proxy` cookie.
- Ensure TLS termination at ingress.
- Carefully scope Keycloak broker roles (needs `broker.read-token`).
- Consider rotating `MCP_SERVER_COMMAND` secrets outside image (ConfigMap / Secret).

---

## Roadmap Ideas

- Celery / Redis / Postgres session manager
- Metrics endpoint (Prometheus) for session counts & tool latency
- Rate limiting / quota per user
- Pluggable auth providers in `TokenRetrieverFactory`
- WebSocket streaming for long-running tools

---

## Original Usage (Reference)

Below retains original quick examples for continuity.

---

## Running your own MCP App

### 1. Start the REST API Server

By default, the app will use the Demo MCP server at `../mcp/server.py`.

```bash
uvicorn app.server:app --reload
```

- The API will be available at: `http://localhost:8000`
- The OpenAPI docs are at: `http://localhost:8000/docs`

and will serve the default mcp app in the /mcp folder.

You can call the mcp functions over rest by sending requests to the API endpoints defined in the FastAPI app, or by using your favorite client.

```sh
❯ curl -X 'POST' \
  'http://127.0.0.1:8000/tools/add' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "a": 2,
  "b": 1
}'
{"isError":false,"content":[{"text":"3","structuredContent":null}],"structuredContent":{"result":3}}
```


#### Custom MCP Server (in mcp folder)

If you are developing or testing with a custom MCP server, you can easily mount it in the `mcp` folder and run the REST server with docker ie like this:

```bash
docker run -it -e ENV=dev -v $(pwd)/mcp:/mcp -p 8000:8000 inxm-ai/mcp-rest-server python /mcp/subfoldered_mcp_server/server.py
```

Setting the `ENV` variable to `dev` will check if there is a requirements.txt or pyproject.toml file in the mounted directory and install the dependencies in the docker container.

#### Custom MCP Server (Parameter Forwarding & Environment Variable)

You can start the REST server with a custom MCP server command in two ways:

**1. Using the command forwarding (with Docker):**

```bash
docker build -t inxm-ai/mcp-rest-server .
docker run -it -p 8000:8000 inxm-ai/mcp-rest-server npx -y @modelcontextprotocol/server-memory /data/memory.json
```

**2. Using the `MCP_SERVER_COMMAND` environment variable (recommended for Docker/Kubernetes):**

Set the environment variable to the full command you want to run as the MCP server. This takes precedence over any arguments passed via `--`.

```bash
docker run -it -p 8000:8000 -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/memory.json" inxm-ai/mcp-rest-server
```

Or in Kubernetes YAML:

```yaml
containers:
  - name: mcp-rest-server
    image: inxm-ai/mcp-rest-server:latest
    env:
      - name: MCP_SERVER_COMMAND
        value: "npx -y @modelcontextprotocol/server-memory /data/memory.json"
    volumeMounts:
      - name: data
        mountPath: /data
volumes:
  - name: data
    persistentVolumeClaim:
      claimName: my-mcp-memory-pvc
```

**Precedence:**
- If `MCP_SERVER_COMMAND` is set, it is used (split shell-style, so quoting works as expected).
- Otherwise, if `--` is present in the command line, those arguments are used.
- Otherwise, the default Python MCP server is started.

---

#### Kubernetes/Helm

- The image exposes `/data` and `/config` as volumes for persistent storage and configuration.
- You can mount volumes and override the in your deployment YAML or Helm chart.
- You can use the `MCP_SERVER_COMMAND` environment variable to specify the MCP server command.
- And you can use the `MCP_BASE_PATH` environment variable to set a custom base path for the API to match your ingress configuration.
- Example (Kubernetes):
  ```yaml
  containers:
    - name: mcp-rest-server
      image: inxm-ai/mcp-rest-server:latest
      env:
        - name: MCP_SERVER_COMMAND
          value: "npx -y @modelcontextprotocol/server-memory /data/memory.json"
        - name: MCP_BASE_PATH
          value: "/api/mcp-memory-server"
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: my-mcp-data-pvc
  ```

#### Extending the image

- You can add further MCP servers or dependencies by extending the Dockerfile or mounting additional volumes.
- The entrypoint and CMD are designed to be flexible for most use cases.

## API Usage

### List Tools

```http
GET /tools
```

- Returns a list of available tools from the MCP server.

### Run a Tool

```http
POST /tools/{tool_name}
```

- Body: JSON with tool arguments
- Header: `x-inxm-mcp-session` (optional, for session-based calls)

### Session Management
- **Start a session:** `POST /session/start` → returns a session ID
- **Use the session**: `POST /tools/{tool_name}` with header `x-inxm-mcp-session`
- **Close a session:** `POST /session/close` with header `x-inxm-mcp-session`

## Example: List Tools

```bash
curl http://localhost:8000/tools
```

## Example: Run a Tool

```bash
curl -X POST http://localhost:8000/tools/<tool_name> -H 'Content-Type: application/json' -d '{"arg1": "value"}'
```

## Notes
- The REST server launches the MCP server as a subprocess. You can customize the command and arguments as needed.
- For advanced usage, see the code and comments in `routes.py`.

---

For more details, see the source code and comments in the `template/app` directory.

---

## OAuth Token Integration

### Getting a Microsoft Token from Keycloak

To inject a Microsoft token obtained via your Keycloak provider into the MCP server environment, set the `OAUTH_ENV` environment variable to the desired environment variable name (e.g., `MS_TOKEN`). When starting the REST server, provide your Keycloak-issued OAuth token as a parameter. The server will use the `TokenRetrieverFactory` to exchange the Keycloak token for a Microsoft token and inject it into the MCP subprocess environment.

**Example:**

```bash
export OAUTH_ENV=MS_TOKEN
uvicorn server:app --reload -- <your_keycloak_token>
```

This will set `MS_TOKEN` in the MCP server environment with the retrieved Microsoft token.

### General Approach

- The REST server supports dynamic token retrieval and injection using the `TokenRetrieverFactory`.
- When `OAUTH_ENV` is set, the server expects an OAuth token as input.
- The token retriever exchanges the provided token for the required provider token (e.g., Microsoft).
- The resulting token is injected into the MCP server environment for use by downstream tools.

### Implementing Custom Token Providers

To add your own token provider:
1. Create a new class in `oauth/token_exchange.py` that implements a `retrieve_token` method.
2. Register your provider in the `TokenRetrieverFactory`.
3. The factory will select the appropriate retriever based on configuration or environment.

**Example Skeleton:**

```python
# oauth/token_exchange.py

class MyCustomTokenRetriever:
    def retrieve_token(self, input_token):
        # Exchange input_token for your provider's token
        return {"access_token": "<your_provider_token>"}

class TokenRetrieverFactory:
    def get(self):
        # Return the appropriate retriever based on config/env
        return MyCustomTokenRetriever()
```

**Usage:**
- Set `OAUTH_ENV` and provide the input token when starting the server.
- The custom retriever will handle token exchange and injection.
