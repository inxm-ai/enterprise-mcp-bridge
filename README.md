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

## Install from GitHub

You can install the package directly from GitHub (replace `<user>` and `<repo>` with your repository details):

```bash
pip install 'git+https://github.com/inxm-ai/mcp-rest-server.git@main#subdirectory=app'
```

## Build a pip-installable package

To build a wheel and source distribution:

```bash
cd app
python -m pip install build
python -m build
```

The generated `.whl` and `.tar.gz` files will be in the `dist/` directory. You can install them with:

```bash
pip install dist/mcp_rest_server-*.whl
```

## Running the App

### 1. Start the REST API Server

By default, the app will use the Demo MCP server at `../mcp/server.py`.

```bash
uvicorn server:app --reload
```

- The API will be available at: `http://localhost:8000`
- The OpenAPI docs are at: `http://localhost:8000/docs`


#### Custom MCP Server (Parameter Forwarding & Environment Variable)

You can start the REST server with a custom MCP server command in two ways:

**1. Using the `--` separator (Python or Docker):**

```bash
uvicorn server:app --reload -- <command> <arg1> <arg2> ...
```

For example, to use a custom Python MCP server:

```bash
uvicorn server:app --reload -- python3 /path/to/your/mcp_server.py
```

Or with Docker:

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
- You can mount volumes and override the MCP server command in your deployment YAML or Helm chart.
- Example (Kubernetes):
  ```yaml
  containers:
    - name: mcp-rest-server
      image: <your-repo>/mcp-rest-server:latest
      args: ["--", "npx", "@modelcontextprotocol/server-sqlite", "/data/database.db"]
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
