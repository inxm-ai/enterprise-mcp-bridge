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


## Core Concepts

1. **Ad-hoc (stateless) tool calls**: POST to `/tools/{tool_name}` with arguments. A temporary MCP connection is created and torn down.
2. **Managed sessions**: Start with `/session/start`, receive a session ID cookie (`x-inxm-mcp-session`). Subsequent tool calls reuse a persistent MCP process context.
3. **Multi-user separation**: If an OAuth token is present (default cookie `_oauth2_proxy`) it is appended internally to the session identifier to avoid collisions and enforce per-user isolation.
4. **Automatic OAuth token propagation**: When a tool declares an `oauth_token` property in its JSON schema, the server injects a validated token value; clients need not send it explicitly.
5. **Token exchange**: Incoming Keycloak token can be exchanged for a provider (e.g., Microsoft) token using `TokenRetrieverFactory` and exported into the MCP subprocess environment variable defined by `OAUTH_ENV`.

---

## Quick Start

```bash
git clone https://github.com/inxm-ai/enterprise-mcp-bridge.git
cd enterprise-mcp-bridge
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

- **Start a Group-Specific Session**:
  ```bash
  curl -X POST "http://localhost:8000/session/start?group=team-alpha" -c cookies.txt
  ```
  Creates a session with access to group-specific data (user must be member of the group).

- **Use the Session**:
  ```bash
  curl -b cookies.txt -X POST http://localhost:8000/tools/add -H 'Content-Type: application/json' -d '{"a":2,"b":3}'
  ```

- **Close the Session**:
  ```bash
  curl -b cookies.txt -X POST http://localhost:8000/session/close
  ```

## Group-Based Data Access

The Enterprise MCP Bridge supports secure, group-based data access through OAuth token validation. This feature allows different users to access different data sources based on their group memberships, without exposing file paths or risking path traversal attacks.

### How It Works

1. **User Authentication**: Users authenticate via OAuth (e.g., Keycloak) and receive a token containing group memberships.
2. **Group Validation**: When requesting group-specific access, the server validates the user's membership in the requested group.
3. **Dynamic Path Resolution**: Based on group membership, the server resolves secure data paths:
   - User-specific: `/data/u/{sanitized_user_id}.json`
   - Group-specific: `/data/g/{sanitized_group_name}.json`
4. **Template Processing**: The `MCP_SERVER_COMMAND` template is processed with the resolved data path.

### Security Features

- **No Path Exposure**: Clients never specify file paths directly
- **Group Membership Validation**: Access is granted only to authorized group members
- **Path Sanitization**: All path components are sanitized to prevent traversal attacks
- **OAuth-Based Authorization**: Leverages existing identity and access management systems

### Usage Examples

**Sessionless Group Access:**
```bash
# Access group-specific data (user must be member of 'finance' group)
curl -H "Authorization: Bearer $TOKEN" \
     -X POST "http://localhost:8000/tools/search?group=finance" \
     -H 'Content-Type: application/json' \
     -d '{"query": "budget reports"}'
```

**Session-Based Group Access:**
```bash
# Start session with group access
curl -H "Authorization: Bearer $TOKEN" \
     -X POST "http://localhost:8000/session/start?group=marketing" \
     -c cookies.txt

# Use the session (automatically uses group-specific data)
curl -b cookies.txt \
     -X POST http://localhost:8000/tools/create_memory \
     -H 'Content-Type: application/json' \
     -d '{"content": "Campaign ideas for Q4"}'
```

### Configuration

Configure your MCP command template to use dynamic data paths:

```bash
# Memory server with group/user-specific data
export MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory {data_path}"

# Custom server with additional parameters
export MCP_SERVER_COMMAND="python custom_mcp.py --data-file {data_path} --user {user_id}"
```

### Data Directory Structure

```
/data/
├── u/           # User-specific data
│   ├── user123.json
│   ├── alice_smith.json
│   └── ...
├── g/           # Group-specific data
│   ├── finance.json
│   ├── marketing.json
│   ├── engineering.json
│   └── ...
└── shared/      # Shared resources (future)
    └── ...
```

## OAuth Token Exchange & Injection

1. Set `OAUTH_ENV=MS_TOKEN` when starting the REST server.
2. Provide a Keycloak user token (via cookie `_oauth2_proxy` or CLI argument).
3. The `TokenRetrieverFactory` exchanges the Keycloak token for the provider (e.g., Microsoft) token.
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
| `SYSTEM_DEFINED_PROMPTS`  | <a href="#system-defined-prompts">JSON array of built-in prompts available to all users</a> | "[]"                    |
| `MCP_ENV_*`               | Those will be passed to the MCP server process            |                        |
| `MCP_*_DATA_ACCESS_TEMPLATE` | Template for specific data resources. See [Data Resource Templates](#data-resource-templates) for details. | `{*}/{placeholder}` |


---

## System Defined Prompts

### SYSTEM_DEFINED_PROMPTS

The `SYSTEM_DEFINED_PROMPTS` environment variable allows you to inject a list of built-in prompts that are available to all users, regardless of the underlying MCP server. This is useful for providing default or global prompt templates (such as greetings, onboarding, or FAQ responses) that do not depend on the MCP toolset.

**Format:**

Set `SYSTEM_DEFINED_PROMPTS` to a JSON array of prompt objects, each with the following fields:

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

**Fields:**
- `name`: Unique identifier for the prompt
- `title`: Display name
- `description`: Short explanation of the prompt's purpose
- `arguments`: List of argument definitions (name, type, etc.)
- `template`: String template with placeholders for arguments

These prompts are merged into the `/prompts` endpoint and can be invoked by name using the standard API. See [prompt_helper.py](app/session_manager/prompt_helper.py) for implementation details.

---
### MCP_SERVER_COMMAND Template Placeholders

The `MCP_SERVER_COMMAND` environment variable supports template placeholders that are dynamically resolved based on the user's OAuth token and requested access:

| Placeholder  | Description                                           | Example Resolution     |
|-------------|-------------------------------------------------------|------------------------|
| `{data_path}` | Resolves to user or group-specific data path        | `/data/u/user123.json` or `/data/g/team-alpha.json` |
| `{user_id}`   | User identifier extracted from OAuth token          | `user123` or `john.doe@company.com` |
| `{group_id}`  | Requested group identifier (if group access)        | `team-alpha` or `finance` |

**Examples:**
```bash
# Memory server with dynamic user/group data
export MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory {data_path}"

# Custom server with user-specific configuration
export MCP_SERVER_COMMAND="python /app/custom_server.py --user {user_id} --data {data_path}"

# Database connection with group-based access
export MCP_SERVER_COMMAND="psql-mcp-server --database group_{group_id} --user {user_id}"
```

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
- Rate limiting / quota per user
- Externally extensible auth providers in `TokenRetrieverFactory` and `SessionManagerBase`
- WebSocket streaming/Long Polling for long-running tools
- Easy mcp client support (ie Claude, Cursor, Windsurf, VSCode...)

Find more ideas in our [GitHub issues](https://github.com/inxm-ai/enterprise-mcp-bridge/issues).

---

## Running your own MCP App

### Start the REST API Server

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

### Try the fully blown example

Go to [example/token-exchange-m365](https://github.com/inxm-ai/enterprise-mcp-bridge/tree/main/example/token-exchange-m365) and try out our full example

#### What it Provides

* Keycloak with token-exchange feature and ingress
* Automated Entra (Azure AD) app registration
* Enterprise MCP Bridge launched with `npx -y @softeria/ms-365-mcp-server --org-mode`
* Minimal chat frontend
* Tracing via Jaeger
* Monitoring via Prometheus/Grafana


#### Custom MCP Server (in mcp folder)

If you are developing or testing with a custom MCP server, you can easily mount it in the `mcp` folder and run the REST server with docker ie like this:

```bash
docker run -it -e ENV=dev -v $(pwd)/mcp:/mcp -p 8000:8000 inxm-ai/enterprise-mcp-bridge python /mcp/subfoldered_mcp_server/server.py
```

Setting the `ENV` variable to `dev` will check if there is a requirements.txt or pyproject.toml file in the mounted directory and install the dependencies in the docker container.

#### Custom MCP Server (Parameter Forwarding & Environment Variable)

You can start the REST server with a custom MCP server command in two ways:

**1. Using the command forwarding (with Docker):**

```bash
docker build -t inxm-ai/enterprise-mcp-bridge .
docker run -it -p 8000:8000 inxm-ai/enterprise-mcp-bridge npx -y @modelcontextprotocol/server-memory /data/memory.json
```

**2. Using the `MCP_SERVER_COMMAND` environment variable (recommended for Docker/Kubernetes):**

Set the environment variable to the full command you want to run as the MCP server. This takes precedence over any arguments passed via `--`.

```bash
docker run -it -p 8000:8000 -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/memory.json" inxm-ai/enterprise-mcp-bridge
```

Or in Kubernetes YAML:

```yaml
containers:
  - name: enterprise-mcp-bridge
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
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
    - name: enterprise-mcp-bridge
      image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
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

## Examples

This project includes several examples to help you get started with different use cases. Below is a summary of the examples and their purposes, along with links to their respective README sections for more details:

### 1. **Minimal Example**
A lightweight example to quickly get started with the Enterprise MCP Bridge. It provides:
- A simple script to start the server.
- Minimal dependencies and configuration.

Check the [Minimal Example README](example/minimal-example/README.md) for more details.

### 2. **Memory Group Access**
This example demonstrates how to manage group-based data access using OAuth tokens. It includes:
- Group-specific and user-specific data files.
- Scripts to start and stop the example environment.
- Integration with Keycloak for authentication.

Refer to the [Memory Group Access README](example/memory-group-access/README.md) for setup and usage instructions.

### 3. **Token Exchange with Microsoft 365**
This example showcases the token exchange feature with Microsoft 365. It includes:
- Dockerized setup for Keycloak and the MCP Bridge.
- Integration with Microsoft Azure AD for token exchange.
- Monitoring and tracing with Prometheus, Grafana, and Jaeger.

See the [Token Exchange README](example/token-exchange-m365/README.md) for a comprehensive guide.

Each example is designed to highlight specific features of the Enterprise MCP Bridge, making it easier to understand and integrate into your workflows.

---

## Group-Based Data Access

The Enterprise MCP Bridge supports dynamic data source resolution based on OAuth token group membership. This allows secure, group-based access to data without exposing sensitive file paths, database connection strings, or table names to clients.

### Overview

When using group-based data access, the server:

1. **Extracts user and group information** from OAuth tokens (JWT payload)
2. **Validates group membership** against requested group access
3. **Dynamically resolves data sources** using configurable templates
4. **Prevents unauthorized access** through automatic permission checking

### Configuration

#### MCP_SERVER_COMMAND Template

Use placeholders in your `MCP_SERVER_COMMAND` or `MCP_ENV_` environment variable:

```bash
# For file-based MCP servers (e.g., memory server)
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/{data_path}.json"
MCP_ENV_MEMORY_FILE_PATH="/data/{data_path}.json"

# For database-based MCP servers
MCP_SERVER_COMMAND="python db-mcp-server.py --table {data_path}"

# Multiple placeholders supported
MCP_SERVER_COMMAND="my-mcp-server --user {user_id} --group {group_id} --resource {data_path}"
```

#### Supported Placeholders

- `{data_path}`: Resolves to group-specific (`g/groupname`) or user-specific (`u/userid`) resource identifier
- `{user_id}`: User identifier from OAuth token
- `{group_id}`: Requested group identifier (if accessing group data)

### Usage

#### Session-Based Group Access

Start a session with group-specific data access:

```bash
# Start session for 'finance' group data
curl -X POST "http://localhost:8000/session/start?group=finance" \
  -H "X-Auth-Request-Access-Token: <your-oauth-token>"

# Use session for tool calls
curl -X POST "http://localhost:8000/tools/read_memory" \
  -H "x-inxm-mcp-session: <session-id>" \
  -H "Content-Type: application/json" \
  -d '{"query": "recent transactions"}'
```

#### Sessionless Group Access

Make direct tool calls with group specification:

```bash
# Access finance group data without session
curl -X POST "http://localhost:8000/tools/read_memory?group=finance" \
  -H "X-Auth-Request-Access-Token: <your-oauth-token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "budget reports"}'
```

#### User-Specific Data Access

Access user-specific data (default behavior):

```bash
# Access user's personal data
curl -X POST "http://localhost:8000/tools/read_memory" \
  -H "X-Auth-Request-Access-Token: <your-oauth-token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "my notes"}'
```

### Security Model

#### Group Membership Validation

The server extracts groups from multiple JWT token claims:
- `groups`: Direct group membership
- `realm_access.roles`: Keycloak realm roles
- `resource_access`: Keycloak client roles
- `roles`: Generic roles claim

#### Permission Checking

- Users can only access groups they are members of
- Unauthorized group access returns HTTP 403 Forbidden
- Invalid tokens return HTTP 401 Unauthorized
- Resource identifiers are automatically sanitized

#### Resource Identifier Sanitization

All resource identifiers are sanitized to prevent security issues:
- Dangerous characters are removed or replaced
- Length is limited to reasonable bounds
- Path traversal attempts are neutralized

### Example: Memory Server with Groups

See the complete example in `example/memory-group-access/`:

```yaml
# docker-compose.yml
services:
  app-mcp-rest:
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    environment:
      MCP_SERVER_COMMAND: npx -y @modelcontextprotocol/server-memory /data/{data_path}.json
      AUTH_PROVIDER: keycloak
      AUTH_BASE_URL: https://auth.example.com
      KEYCLOAK_REALM: company
    volumes:
      - ./data:/data
```

Directory structure:
```
data/
├── g/                    # Group-specific data
│   ├── finance.json      # Finance team data
│   ├── marketing.json    # Marketing team data
│   └── hr.json          # HR team data
└── u/                    # User-specific data
    ├── alice.json        # Alice's personal data
    └── bob.json         # Bob's personal data
```

### Token Claims Example

The server extracts user information from JWT tokens like this:

```json
{
  "sub": "alice",
  "email": "alice@company.com",
  "groups": ["finance", "employees"],
  "realm_access": {
    "roles": ["user", "finance-analyst"]
  },
  "resource_access": {
    "frontend-client": {
      "roles": ["admin"]
    }
  }
}
```

This user can access:
- Their personal data: `u/alice`
- Finance group data: `g/finance` 
- Employee group data: `g/employees`
- Admin data through client role: `g/admin`

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

### Data Resource Templates

The `DataAccessManager` class in the `app.oauth.user_info` module provides configurable templates for resolving data resources based on user or group access. These templates are defined as environment variables:

| Template Name                     | Default Value               | Description                                      |
|-----------------------------------|-----------------------------|--------------------------------------------------|
| `MCP_GROUP_DATA_ACCESS_TEMPLATE` | `g/{group_id}`             | Template for group-specific data resources      |
| `MCP_USER_DATA_ACCESS_TEMPLATE`  | `u/{user_id}`              | Template for user-specific data resources       |
| `MCP_SHARED_DATA_ACCESS_TEMPLATE`| `shared/{resource_id}`     | Template for shared data resources              |

These templates are dynamically resolved based on the user's OAuth token and requested access. For example, a group-specific data resource might resolve to `g/finance` for the `finance` group.
