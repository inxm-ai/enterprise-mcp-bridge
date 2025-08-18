# MCP REST Server

FastAPI-based wrapper that exposes any Model Context Protocol (MCP) server over plain HTTP/JSON.

## Why this project?

Most existing MCP examples are designed for local development or simple demos and fall short for real-world applications. They are typically:

* Single-user CLI processes driven by a local client.
* Ephemeral, with state being lost as soon as the process ends.
* Lacking multi-tenancy, with no built-in orchestration for concurrent users or sessions.
* Unsuitable for sandboxed environments like browsers or secure remote servers.
* No simple way for integrating MCP components into REST-based microservice architectures.
* Missing a consistent security model for handling delegated permissions (e.g., OAuth) for downstream resources.

This project directly addresses these gaps.

## Key Capabilities

### Robust Session & User Management
* **Multi-User & Multi-Session:** A single server instance can securely manage multiple, isolated user contexts, namespaced by OAuth tokens. Each user can run multiple concurrent sessions (`/session/start`), enabling parallel tool execution and long-running conversational state.
* **Stateful & Stateless Modes:** Supports both **stateless** ("fire-and-forget") tool calls for low-latency tasks and **stateful** sessions for complex, multi-step interactions.
* **Lifecycle & Resource Hygiene:** Provides explicit endpoints to start and close sessions, coupled with automatic cleanup of idle sessions via inactivity pings. This prevents resource leaks and supports autoscaling.
* **Pluggable for Scalability:** Features a swappable session manager (`SessionManagerBase`). While it defaults to a simple in-memory store, it can be replaced with a distributed backend (like Redis or a database) for horizontal scaling.

### Integrated Security & Authentication
* **Built-in OAuth2 Token Exchange:** Natively handles the OAuth2 token exchange flow (e.g., with Keycloak as a broker) to securely acquire tokens for downstream resource providers.
* **Automatic Token Injection:** If a tool's input schema includes an `oauth_token` argument, the server automatically and securely injects the correct token. This simplifies client-side logic and prevents tokens from being exposed.
* **Automated Token Refresh:** Manages token refresh logic transparently, allowing for long-lived, secure sessions without requiring clients to handle re-authentication.

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


## Core Concepts

1. **Ad-hoc (stateless) tool calls**: POST to `/tools/{tool_name}` with arguments. A temporary MCP connection is created and torn down.
2. **Managed sessions**: Start with `/session/start`, receive a session ID cookie (`x-inxm-mcp-session`). Subsequent tool calls reuse a persistent MCP process context.
3. **Multi-user separation**: If an OAuth token is present (cookie `_oauth2_proxy`) it is appended internally to the session identifier to avoid collisions and enforce per-user isolation.
4. **Automatic OAuth token propagation**: When a tool declares an `oauth_token` property in its JSON schema, the server injects a validated token value; clients need not send it explicitly.
5. **Token exchange**: Incoming Keycloak token can be exchanged for a provider (e.g., Microsoft) token using `TokenRetrieverFactory` and exported into the MCP subprocess environment variable defined by `OAUTH_ENV`.

---

## Quick Start

```bash
git clone https://github.com/inxm-ai/mcp-rest-server.git
cd mcp-rest-server
pip install app
uvicorn app.server:app --reload
```

Open: [http://localhost:8000/docs](http://localhost:8000/docs)

## Session vs Stateless Calls

| Mode      | How                                      | When to use                     |
|-----------|-----------------------------------------|----------------------------------|
| Stateless | `POST /tools/{tool}` (no session header)| Simple one-off tool execution   |
| Stateful  | `POST /session/start` then include header/cookie | Conversations, multi-step pipelines |

### Session Management

- **Start a Session**:
  ```bash
  curl -X POST http://localhost:8000/session/start -c cookies.txt
  ```
  The response sets `x-inxm-mcp-session` cookie.

- **Use the Session**:
  ```bash
  curl -b cookies.txt -X POST http://localhost:8000/tools/add -H 'Content-Type: application/json' -d '{"a":2,"b":3}'
  ```

- **Close the Session**:
  ```bash
  curl -b cookies.txt -X POST http://localhost:8000/session/close
  ```

## OAuth Token Exchange & Injection

1. Set `OAUTH_ENV=MS_TOKEN` when starting the REST server.
2. Provide a Keycloak user token (via cookie `_oauth2_proxy` or CLI argument).
3. The `TokenRetrieverFactory` exchanges the Keycloak token for the provider token.
4. The resulting provider token is exported into the MCP subprocess environment (`MS_TOKEN`).
5. Tools with `oauth_token` in their schema automatically receive the token.

**Example**:
```bash
export OAUTH_ENV=MS_TOKEN
uvicorn app.server:app --reload
```

### Adding a Custom Provider

1. Implement a new retriever in `oauth/token_exchange.py`.
2. Register it in `TokenRetrieverFactory.get()`.

## Environment Variables

| Variable               | Purpose                                              | Default       |
|------------------------|------------------------------------------------------|---------------|
| `MCP_SERVER_COMMAND`   | Full command to launch MCP server                   | (unset)       |
| `MCP_BASE_PATH`        | Prefix all routes (e.g., `/api/mcp`)                | ""            |
| `OAUTH_ENV`            | Name of env var injected into MCP subprocess        | (unset)       |
| `AUTH_PROVIDER`        | Token exchange provider selector (`keycloak`)       | keycloak      |
| `AUTH_BASE_URL`        | Keycloak base URL for broker/token endpoints        | (required)    |
| `KEYCLOAK_REALM`       | Keycloak realm                                      | inxm          |
| `KEYCLOAK_PROVIDER_ALIAS` | External IdP alias used in broker path            | (required)    |
| `MCP_SESSION_MANAGER`  | Implementation name (e.g., future Redis)            | InMemorySessionManager |

## Extending Session Management

To add a backend:

1. Implement a subclass of `SessionManagerBase`.
2. Expose it in `session_manager/session_manager.py` globals.
3. Set `MCP_SESSION_MANAGER=<YourClassName>`.

## Security Considerations

- Do not expose this service publicly without an auth proxy placing user tokens in `_oauth2_proxy` cookie.
- Ensure TLS termination at ingress.
- Carefully scope Keycloak broker roles (needs `broker.read-token`).
- Consider rotating `MCP_SERVER_COMMAND` secrets outside image (ConfigMap / Secret).

## Roadmap

- Celery/Redis/Postgres session manager
- Other Auth providers
- Metrics endpoint (Prometheus) for session counts & tool latency
- Rate limiting / quota per user
- Pluggable auth providers in `TokenRetrieverFactory`
- WebSocket streaming for long-running tools

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
