# Enterprise MCP Bridge

FastAPI-based wrapper that exposes any Model Context Protocol (MCP) server over plain HTTP/JSON with enterprise-grade capabilities for production use.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/inxm-ai/enterprise-mcp-bridge)

Read more about deploying to Render [here →](#deploy-on-render-weather-mcp-example)

📖 **[Full Documentation](https://inxm-ai.github.io/enterprise-mcp-bridge)**

## Why Enterprise MCP Bridge?

Many open source MCPs are designed for local development or simple demos and fall short for real-world applications. This project addresses those gaps with enterprise production features: multi-user sessions, OAuth integration, centralized management, and scalability.

**[Learn more about the motivation →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/explanation/architecture)**

## Key Features

The Enterprise MCP Bridge automatically extends any MCP server with enterprise-grade capabilities:

- 🔐 **Security & Authentication** - Built-in OAuth2 token exchange, automatic token injection and refresh, group-based data access
- 👥 **Multi-User Sessions** - Concurrent isolated sessions with automatic cleanup and pluggable storage backends
- 🌐 **REST API** - Standard HTTP/JSON endpoints with auto-generated OpenAPI docs
- 🎨 **UI Generation** - AI-powered web application generation from natural language prompts
- 🔄 **Workflow Orchestration** - Multi-agent workflows with human-in-the-loop feedback
- 📡 **SSE Streaming** - Real-time updates for long-running operations

**[See all features →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/home)**

## Quick Start

```bash
git clone https://github.com/inxm-ai/enterprise-mcp-bridge.git
cd enterprise-mcp-bridge
pip install app
uvicorn app.server:app --reload
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API documentation.

**[Complete getting started guide →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/tutorials/getting-started)**

## Documentation

- 📚 **[Tutorials](https://inxm-ai.github.io/enterprise-mcp-bridge/#/tutorials/getting-started)** - Step-by-step guides to get you started
- 🛠️ **[How-To Guides](https://inxm-ai.github.io/enterprise-mcp-bridge/#/how-to/deploy-production)** - Task-focused guides for specific scenarios
- 📖 **[Reference](https://inxm-ai.github.io/enterprise-mcp-bridge/#/reference/api)** - Complete API and configuration reference
- 💡 **[Explanation](https://inxm-ai.github.io/enterprise-mcp-bridge/#/explanation/architecture)** - Deep dives into architecture and concepts

## Common Use Cases

### Running Local MCP Servers

```bash
# Local MCP: Clone the repo you want to run
git clone https://github.com/modelcontextprotocol/servers.git mcp
docker run -p 8000:8000 -e ENV=dev -v $PWD/mcp/src/fetch:/mcp -it ghcr.io/inxm-ai/enterprise-mcp-bridge:latest python -m mcp_server_fetch

# Custom command: Use environment variable
export MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory"
uvicorn app.server:app --host 0.0.0.0 --port 8000

# Remote MCP: Connect to a hosted MCP endpoint
export MCP_REMOTE_SERVER="https://mcp.example.com"
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

**[Complete MCP configuration guide →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/how-to/remote-mcp-servers)**

### Production Deployment

```bash
# Docker
docker run -it -p 8000:8000 \
  -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/memory.json" \
  -v $(pwd)/data:/data \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

**[See deployment guides →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/how-to/deploy-production)** | **[Docker →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/how-to/docker)** | **[Kubernetes →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/how-to/kubernetes)**

### Deploy on Render (Weather MCP Example)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/inxm-ai/enterprise-mcp-bridge)

This deploys `ghcr.io/inxm-ai/enterprise-mcp-bridge:latest` with the weather MCP server.

Environment variables used by this setup:

- `TGI_URL` - LLM API base URL (`https://api.openai.com/v1`)
- `TGI_TOKEN` - API token for the LLM provider (**set this as a secret in Render**)
- `DEFAULT_MODEL` - Default model name (for example `gpt-5.2-codex`)

- `MCP_GIT_CLONE` - MCP repo cloned into the container at startup - the mcp that will be running
- `MCP_SERVER_COMMAND` - Command used to start the MCP server (`python -m mcp_weather_server`) you cloned earlier

- `TGI_CONVERSATION_MODE` - LLM API mode (`responses`)
- `MCP_TEST_ACTOR` - Default actor id for testing/demo flows
- `APP_CONVERSATIONAL_UI_ENABLED` - Enables generated conversational UI features
- `GENERATED_WEB_PATH` - Path where generated UI artifacts are stored
- `ENV` - Runtime mode (`dev`)

⚠️ **Token usage warning:** When the service runs, calls to your configured model provider consume tokens and can incur costs. Keep `TGI_TOKEN` secret and monitor usage limits/billing.

### OAuth Configuration

The bridge includes built-in OAuth2 token exchange for secure access to downstream resources.

**[OAuth configuration guide →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/how-to/configure-oauth)**

## Examples

The repository includes several complete examples demonstrating different capabilities:

- **minimal-example** - Simple starter with the default memory server
- **memory-group-access** - Group-based data isolation
- **token-exchange-m365** - Microsoft 365 integration with token exchange
- **remote-mcp-github** - GitHub MCP server with token exchange
- **remote-mcp-atlassian** - Jira and Confluence integration

**[Browse all examples →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/reference/examples)**

## Configuration Reference

Key environment variables:

- `MCP_SERVER_COMMAND` - Command to launch the MCP server
- `MCP_REMOTE_SERVER` - URL for remote MCP server
- `MCP_BASE_PATH` - Base path for API routing
- `KEYCLOAK_URL`, `KEYCLOAK_REALM` - OAuth configuration
- And many more...

**[Complete configuration reference →](https://inxm-ai.github.io/enterprise-mcp-bridge/#/reference/configuration)**

## Contributing

Contributions are welcome! Please see our **[contributing guidelines](https://inxm-ai.github.io/enterprise-mcp-bridge/#/contributing)** for details.

## License

GPL-3.0 - See **[LICENSE](https://inxm-ai.github.io/enterprise-mcp-bridge/#/license)** for details.

## Support

- 📖 **[Documentation](https://inxm-ai.github.io/enterprise-mcp-bridge)**
- 💬 **[GitHub Discussions](https://github.com/inxm-ai/enterprise-mcp-bridge/discussions)**
- 🐛 **[Issue Tracker](https://github.com/inxm-ai/enterprise-mcp-bridge/issues)**
- 🔒 **Security**: Report vulnerabilities to security@inxm.ai
