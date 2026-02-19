# Examples

Complete working examples demonstrating various Enterprise MCP Bridge features and deployment scenarios.

## Available Examples

All examples are located in the `example/` directory of the repository.

### 1. Minimal Example

**Path:** `example/minimal-example/`

The simplest possible deployment using the default memory MCP server.

**Features:**
- Basic MCP Bridge setup
- Default memory server
- Docker-based deployment
- No authentication

**Use Case:** Quick testing and development

**Quick Start:**
```bash
cd example/minimal-example
./start.sh
```

Access at: http://localhost:8000/docs

---

### 2. Memory Group Access

**Path:** `example/memory-group-access/`

**Tutorial:** [Group-Based Access Tutorial](../tutorials/group-based-access.md)

Demonstrates group-based data isolation using OAuth and Keycloak.

**Features:**
- Keycloak authentication
- OAuth2 token exchange
- Group-based memory isolation
- Multiple pre-configured users
- SSL/TLS with self-signed certificates

**Pre-configured Users:**

| Username         | Password       | Groups         |
|------------------|----------------|----------------|
| `admin`          | `admin123`     | administrators |
| `john.engineer`  | `engineer123`  | engineering    |
| `jane.marketing` | `marketing123` | marketing      |
| `bob.sales`      | `sales123`     | sales          |

**Use Case:** Multi-tenant applications with data isolation

**Quick Start:**
```bash
cd example/memory-group-access
./start.sh
```

Access at: https://inxm.local

**Configuration:**
- Each group has separate memory storage: `/data/{group}/memory.json`
- Users can only access their group's data
- Admin can access all groups

---

### 3. Remote MCP - GitHub

**Path:** `example/remote-mcp-github/`

**Tutorial:** [GitHub Integration Tutorial](../tutorials/github-integration.md)

Connect to GitHub's hosted MCP service with OAuth token exchange.

**Features:**
- Remote MCP server connection
- GitHub OAuth integration
- Keycloak as identity broker
- Token exchange for GitHub API
- SSL/TLS with self-signed certificates
- Optional LLM integration

**Use Case:** Integrate with GitHub Copilot MCP tools

**Quick Start:**
```bash
cd example/remote-mcp-github
./start.sh
```

You'll be prompted for:
- GitHub OAuth Client ID
- GitHub OAuth Client Secret
- OAuth Scopes
- (Optional) OpenAI configuration

Access at: https://ghmcp.local

**Architecture:**
```
Browser → OAuth2 Proxy → Keycloak → GitHub OAuth
                ↓
        MCP Bridge → GitHub Remote MCP API
```

---

### 4. Remote MCP - Atlassian

**Path:** `example/remote-mcp-atlassian/`

Connect to Atlassian (Jira/Confluence) MCP services.

**Features:**
- Remote MCP server for Atlassian
- OAuth integration with Atlassian
- Keycloak identity provider
- Jira and Confluence tool access
- SSL/TLS configuration

**Use Case:** Integrate with Atlassian products via MCP

**Quick Start:**
```bash
cd example/remote-mcp-atlassian
./start.sh
```

**Supported Services:**
- Jira
- Confluence
- (Extensible to other Atlassian products)

---

### 5. Token Exchange - Microsoft 365

**Path:** `example/token-exchange-m365/`

Complete enterprise stack with Microsoft 365 integration.

**Features:**
- OAuth2 token exchange
- Microsoft 365 / Entra ID integration
- Keycloak as broker
- Full observability stack (Prometheus, Grafana, Jaeger)
- Production-ready configuration
- SSL/TLS with Let's Encrypt ready

**Use Case:** Enterprise deployment with Microsoft ecosystem

**Components:**
- Enterprise MCP Bridge
- Keycloak
- Prometheus (metrics)
- Grafana (dashboards)
- Jaeger (distributed tracing)
- Redis (session storage)

**Quick Start:**
```bash
cd example/token-exchange-m365
./start.sh
```

**Observability:**
- Metrics: http://localhost:9090 (Prometheus)
- Dashboards: http://localhost:3000 (Grafana)
- Tracing: http://localhost:16686 (Jaeger)

---

## Common Patterns

### Starting Examples

Most examples include a `start.sh` script:

```bash
cd example/<example-name>
./start.sh
```

This typically:
1. Generates SSL certificates (first run)
2. Sets up environment variables
3. Configures services (Keycloak, etc.)
4. Starts Docker Compose stack
5. Updates `/etc/hosts` for local DNS

### Stopping Examples

Use the provided `stop.sh` script:

```bash
./stop.sh
```

This:
1. Stops Docker Compose stack
2. Removes containers
3. Cleans up `/etc/hosts`
4. Removes temporary files
5. (Usually preserves certificates for reuse)

### Certificate Management

Examples with SSL use development certificates:

- **Location:** `../dev-local-certs/`
- **CA Certificate:** Needs to be trusted in your browser
- **Reused:** Across multiple examples
- **Production:** Replace with Let's Encrypt or commercial CA

To trust the CA (macOS):
```bash
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain dev-local-certs/rootCA.crt
```

To trust the CA (Linux):
```bash
sudo cp dev-local-certs/rootCA.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

## Customization

### Modify Environment Variables

Each example creates a `.env` file. Edit it and restart:

```bash
# Edit configuration
vim .env

# Restart stack
docker-compose restart
```

### Add Custom MCP Servers

Replace the MCP server command:

```bash
# In .env or docker-compose.yml
MCP_SERVER_COMMAND="python /path/to/your/server.py"
```

### Change Ports

Edit `docker-compose.yml`:

```yaml
services:
  mcp-bridge:
    ports:
      - "8080:8000"  # Changed from 8000:8000
```

## Troubleshooting

### Common Issues

**Certificate Trust Issues**
- Trust the development CA certificate
- Restart your browser
- Check certificate is in correct keychain/store

**Port Conflicts**
- Check if ports are already in use: `lsof -i :8000`
- Change ports in `docker-compose.yml`

**Permission Denied**
- Some scripts need sudo for `/etc/hosts` modification
- Check file permissions on mounted volumes

**Docker Issues**
- Ensure Docker is running
- Check Docker has sufficient resources
- Try `docker-compose down -v` to clean up

### Getting Help

1. Check example's README.md for specific instructions
2. Review logs: `docker-compose logs -f`
3. Verify environment variables: `docker-compose config`
4. Check our [Troubleshooting Guide](../how-to/deploy-production.md#troubleshooting)

## Example Structure

Typical example directory structure:

```
example/<name>/
├── README.md              # Example-specific documentation
├── start.sh               # Start script
├── stop.sh                # Stop script
├── docker-compose.yml     # Service definitions
├── .env                   # Environment variables (generated)
├── keycloak/              # Keycloak configuration
│   └── realm-export/      # Realm templates
├── data/                  # Persistent data
└── certs/                 # SSL certificates (optional)
```

## Creating Your Own Example

To create a new example based on existing ones:

1. Copy an existing example:
   ```bash
   cp -r example/minimal-example example/my-example
   ```

2. Modify `docker-compose.yml` for your needs

3. Update environment variables in `.env`

4. Customize `start.sh` and `stop.sh` if needed

5. Document in `README.md`

6. Test thoroughly:
   ```bash
   ./start.sh
   # Test functionality
   ./stop.sh
   ```

## Contributing Examples

We welcome new examples! Please:

1. Follow existing example structure
2. Include comprehensive README.md
3. Add start.sh and stop.sh scripts
4. Test on clean system
5. Document all requirements
6. Submit a pull request

See [Contributing Guide](../contributing.md)

## Reference Documentation

- [Deploy to Production](../how-to/deploy-production.md)
- [Configuration Reference](configuration.md)
- [Docker Guide](../how-to/docker.md)
- [Kubernetes Guide](../how-to/kubernetes.md)

## Repository

All examples: https://github.com/inxm-ai/enterprise-mcp-bridge/tree/main/example
