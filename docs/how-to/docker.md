# Run in Docker

Quick guide to running Enterprise MCP Bridge with Docker.

## Quick Start

### Using the Published Image

Run with default memory server:

```bash
docker run -it -p 8000:8000 ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

Access the API at http://localhost:8000/docs

### With Custom MCP Server

#### Memory Server with Persistence

```bash
docker run -it -p 8000:8000 \
  -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/memory.json" \
  -v $(pwd)/data:/data \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

#### Filesystem Server

```bash
docker run -it -p 8000:8000 \
  -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-filesystem /workspace" \
  -v $(pwd):/workspace \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

#### GitHub Server

```bash
docker run -it -p 8000:8000 \
  -e MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-github" \
  -e GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here" \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

## Development Mode

Auto-install dependencies from mounted MCP directory:

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

## Docker Compose

### Basic Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mcp-bridge:
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    ports:
      - "8000:8000"
    environment:
      - MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-memory /data/memory.json
    volumes:
      - ./data:/data
```

Run:
```bash
docker-compose up
```

### Production Setup

```yaml
version: '3.8'

services:
  mcp-bridge:
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    container_name: enterprise-mcp-bridge
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-memory /data/memory.json
      - MCP_BASE_PATH=/api/mcp
      - LOG_LEVEL=info
      - SESSION_MANAGER_TYPE=redis
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/data
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

## Custom Dockerfile

For production with bundled MCP server:

```dockerfile
FROM ghcr.io/inxm-ai/enterprise-mcp-bridge:latest

# Copy your MCP server
COPY ./my-mcp-server /mcp/

# Install dependencies
RUN pip install -r /mcp/requirements.txt

# Set the command
ENV MCP_SERVER_COMMAND="python /mcp/server.py"
```

Build and run:
```bash
docker build -t my-mcp-bridge .
docker run -it -p 8000:8000 my-mcp-bridge
```

## Environment Variables

Common configuration:

```bash
# MCP Server
MCP_SERVER_COMMAND="your-command-here"
MCP_BASE_PATH="/api/v1/mcp"

# Session Management
SESSION_MANAGER_TYPE=redis
REDIS_URL=redis://redis:6379

# OAuth
OAUTH_ISSUER_URL=https://keycloak.example.com/realms/mcp
OAUTH_CLIENT_ID=mcp-bridge
OAUTH_CLIENT_SECRET=secret

# Logging
LOG_LEVEL=info
LOG_FORMAT=json
```

## Volume Mounts

### Data Persistence

```bash
-v /host/data:/data
```

### Custom MCP Server

```bash
-v /host/mcp-server:/mcp
```

### Configuration Files

```bash
-v /host/config/.env:/app/.env
```

### Logs

```bash
-v /host/logs:/logs
```

## Networking

### Host Network

```bash
docker run --network host ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

### Custom Network

```bash
# Create network
docker network create mcp-network

# Run with network
docker run --network mcp-network \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

## Resource Limits

```bash
docker run \
  --memory="512m" \
  --cpus="1.0" \
  -p 8000:8000 \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

In docker-compose:

```yaml
services:
  mcp-bridge:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '1.0'
        reservations:
          memory: 256M
          cpus: '0.5'
```

## Health Checks

```bash
docker run \
  --health-cmd="curl -f http://localhost:8000/health || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  -p 8000:8000 \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

## Logs

View logs:
```bash
docker logs mcp-bridge

# Follow logs
docker logs -f mcp-bridge

# Last 100 lines
docker logs --tail 100 mcp-bridge
```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker logs mcp-bridge
```

### Port Already in Use

Use different port:
```bash
docker run -p 8080:8000 ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

### Permission Issues

Check volume permissions:
```bash
docker run -u $(id -u):$(id -g) \
  -v $(pwd)/data:/data \
  ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

## Next Steps

- [Deploy to Kubernetes](kubernetes.md)
- [Deploy to Production](deploy-production.md)
- [Configuration Reference](../reference/configuration.md)
