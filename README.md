# MCP REST Server

This project provides a FastAPI-based REST server that acts as a bridge to an existing Model Context Protocol (MCP) server. It allows you to list and call tools exposed by the MCP server via HTTP endpoints, and manage sessions for tool execution.

## Prerequisites
- Python 3.10+
- An existing MCP server (e.g., `./mcp/server.py`)
- (Optional) Docker, if you want to run the app in a container

## Installation
1. **Clone the repository** (if not already done):

   ```bash
   git clone [inxm-ai/mcp-rest-server](https://github.com/inxm-ai/mcp-rest-server.git)
   cd mcp-rest-server/app
   ```
2. **Install dependencies**:

   ```bash
   pip install .
   ```

---

## Running your own MCP App

### 1. Start the REST API Server

By default, the app will use the Demo MCP server at `../mcp/server.py`.

```bash
uvicorn server:app --reload
```

- The API will be available at: `http://localhost:8000`
- The OpenAPI docs are at: `http://localhost:8000/docs`

and will serve the default mcp app in the /mcp folder.

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
