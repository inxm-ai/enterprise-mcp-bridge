# Quick Start

Get up and running with Enterprise MCP Bridge in under 5 minutes.

## Docker Quick Start

The fastest way to get started is using Docker:

```bash
docker run -it -p 8000:8000 ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

Access the API at: **http://localhost:8000/docs**

## Using a Specific MCP Server

### Memory Server (Default)

```bash
docker run -it -p 8000:8000 \
  -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory" \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

### Filesystem Server

```bash
docker run -it -p 8000:8000 \
  -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-filesystem /workspace" \
  -v $(pwd):/workspace \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

### GitHub Server

```bash
docker run -it -p 8000:8000 \
  -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-github" \
  -e GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here" \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

## Local Development

### Clone and Install

```bash
git clone https://github.com/inxm-ai/enterprise-mcp-bridge.git
cd enterprise-mcp-bridge
pip install app
```

### Run the Server

```bash
uvicorn app.server:app --reload
```

### Test It

```bash
# List available tools
curl http://localhost:8000/tools

# Start a session
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session"}'
```

## Remote MCP Server

Connect to a hosted MCP server:

```bash
export MCP_REMOTE_SERVER="https://mcp.example.com"
export MCP_REMOTE_BEARER_TOKEN="your-service-token"

uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## Using Custom MCP Servers

### From a Local Directory

```bash
# Clone your MCP server
git clone https://github.com/your-org/your-mcp-server.git mcp

# Run with auto-install
docker run -p 8000:8000 \
  -e ENV=dev \
  -v $PWD/mcp:/mcp \
  -it ghcr.io/inxm-ai/enterprise-mcp-bridge:latest \
  python -m your_mcp_module
```

### With Node.js-based MCP

```bash
export MCP_SERVER_COMMAND="node /path/to/your/server.js"
uvicorn app.server:app --reload
```

### With Python-based MCP

```bash
export MCP_SERVER_COMMAND="python /path/to/your/server.py"
uvicorn app.server:app --reload
```

## Verify Your Setup

Visit these URLs to confirm everything is working:

- **API Documentation:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health
- **List Tools:** http://localhost:8000/tools

## Common Commands

### List All Tools

```bash
curl http://localhost:8000/tools
```

### Get Tool Schema

```bash
curl http://localhost:8000/tools/tool_name/schema
```

### Call a Tool (Stateless)

```bash
curl -X POST http://localhost:8000/tools/tool_name \
  -H "Content-Type: application/json" \
  -d '{"param1": "value1", "param2": "value2"}'
```

### Session Management

```bash
# Start session
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session"}'

# Ping session (keep alive)
curl -X POST http://localhost:8000/session/my-session/ping

# Close session
curl -X POST http://localhost:8000/session/my-session/close
```

## Configuration Examples

### With Base Path

```bash
export MCP_BASE_PATH="/api/v1/mcp"
uvicorn app.server:app --host 0.0.0.0
```

Now API is available at: http://localhost:8000/api/v1/mcp/docs

### With Data Persistence

```bash
docker run -it -p 8000:8000 \
  -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/memory.json" \
  -v $(pwd)/data:/data \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

### Development Mode with Auto-reload

```bash
docker run -it -p 8000:8000 \
  -e ENV=dev \
  -v $(pwd)/mcp:/mcp \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest \
  python -m mcp.server
```

## Next Steps

Now that you have the bridge running:

- **Explore:** Use the Swagger UI at `/docs` to test endpoints
- **Learn:** Read the [Getting Started Tutorial](getting-started.md)
- **Deploy:** Check out [Deploy to Production](../how-to/deploy-production.md)
- **Customize:** Learn about [Configuration Options](../reference/configuration.md)

## Need Help?

- 📖 [Full Documentation](/)
- 🐛 [Report Issues](https://github.com/inxm-ai/enterprise-mcp-bridge/issues)
- 💬 [Discussions](https://github.com/inxm-ai/enterprise-mcp-bridge/discussions)
