# Enterprise MCP Bridge

> FastAPI-based wrapper that exposes any Model Context Protocol (MCP) server over plain HTTP/JSON

![Architecture](architecture.png)

## What is Enterprise MCP Bridge?

Enterprise MCP Bridge is a production-ready wrapper that transforms any Model Context Protocol (MCP) server into a scalable REST API. It bridges the gap between local MCP development tools and enterprise production requirements.

## Why This Project?

Most existing MCP examples fall short for real-world applications. They are typically:

* **Single-user** CLI processes driven by a local client
* **Ephemeral**, with state lost when the process ends
* **Lacking multi-tenancy**, with no orchestration for concurrent users
* **Hard to monitor** in enterprise environments
* **Missing REST integration** for microservice architectures
* **Without security models** for handling delegated permissions (e.g., OAuth)

This project directly addresses these gaps with enterprise-grade features for security, scalability, and integration.

## Key Features

### 🚀 Production-Ready Architecture
- **Multi-user & multi-session support** with isolated user contexts
- **Stateful and stateless modes** for flexible deployment
- **Automatic session lifecycle management** with cleanup
- **Horizontal scalability** via pluggable session managers

### 🔒 Enterprise Security
- **Built-in OAuth2 token exchange** with Keycloak integration
- **Automatic token injection** for downstream services
- **Group-based data access control**
- **Token refresh handling** for long-lived sessions

### 🛠️ Developer Experience
- **REST-first API** compatible with any HTTP client
- **Auto-generated OpenAPI documentation**
- **Automatic tool endpoint mapping**
- **Structured error handling** with standard HTTP codes

### 🌐 Flexible Deployment
- **Protocol agnostic** - works with any stdio-based MCP server
- **Docker-ready** with production configurations
- **Kubernetes/Helm** compatible
- **Observable** with built-in logging and metrics

## Quick Links

### 📚 Learning Path
- [Getting Started Tutorial](tutorials/getting-started.md) - Your first MCP bridge deployment
- [Quick Start Guide](tutorials/quick-start.md) - 5-minute setup

### 🔧 How-To Guides
- [Deploy to Production](how-to/deploy-production.md)
- [Configure OAuth](how-to/configure-oauth.md)
- [Use Remote MCP Servers](how-to/remote-mcp-servers.md)

### 📖 Reference
- [API Reference](reference/api.md)
- [Configuration](reference/configuration.md)
- [Environment Variables](reference/environment-variables.md)

### 💡 Understanding
- [Architecture Overview](explanation/architecture.md)
- [Security Model](explanation/security.md)
- [Session Management](explanation/sessions.md)

## Community & Support

- **GitHub:** [inxm-ai/enterprise-mcp-bridge](https://github.com/inxm-ai/enterprise-mcp-bridge)
- **Issues:** [Report a bug or request a feature](https://github.com/inxm-ai/enterprise-mcp-bridge/issues)
- **License:** GPL-3.0

---

Ready to get started? Head to the [Getting Started Tutorial](tutorials/getting-started.md) or explore the documentation using the sidebar.
