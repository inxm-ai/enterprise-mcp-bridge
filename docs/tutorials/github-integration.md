# GitHub Integration Tutorial

Learn how to integrate the Enterprise MCP Bridge with GitHub's Remote MCP service using OAuth token exchange.

## What You'll Learn

- How to connect to GitHub's Remote MCP API
- How to configure OAuth authentication with GitHub
- How to use GitHub Copilot MCP tools
- How to deploy the complete example

## Overview

This tutorial demonstrates connecting the Enterprise MCP Bridge to GitHub's hosted MCP service (`https://api.githubcopilot.com/mcp/`). Instead of running a local MCP server, the bridge forwards requests to GitHub's remote service.

## Architecture

```
Browser ──TLS──▶ oauth2-proxy ──▶ Keycloak (GitHub IdP)
   │                                     │
   │                        GitHub OAuth  │
   ▼                                     ▼
Enterprise MCP Bridge ──▶ GitHub Remote MCP API
```

## Prerequisites

- Docker and Docker Compose installed
- `openssl` for certificate generation
- `envsubst` (usually provided by `gettext`)
- A GitHub OAuth App or GitHub App with OAuth credentials
- Basic understanding of OAuth flows

## Step 1: Create GitHub OAuth App

1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Click "New OAuth App"
3. Configure:
   - **Application name:** "MCP Bridge Demo"
   - **Homepage URL:** `https://ghmcp.local`
   - **Authorization callback URL:** `https://ghmcp.local/oauth2/callback`
4. Click "Register application"
5. Note your **Client ID**
6. Generate and note your **Client Secret**

### Recommended Scopes

- `read:user` - Read user profile information
- `user:email` - Access user email
- `copilot:chat` - Access GitHub Copilot chat
- Plus any tool-specific scopes you need

## Step 2: Start the Example

Navigate to the example directory:

```bash
cd example/remote-mcp-github
./start.sh
```

The wizard will prompt you for:

1. **GitHub Client ID:** Your OAuth app client ID
2. **GitHub Client Secret:** Your OAuth app client secret
3. **GitHub OAuth Scopes:** Space-separated list (e.g., `read:user user:email copilot:chat`)
4. **OpenAI Configuration:** (Optional) Use bundled dummy LLM or provide your own

The script will:
- Generate local SSL certificates
- Configure Keycloak with GitHub identity provider
- Start the MCP Bridge in remote mode
- Set up OAuth2 Proxy
- Configure local DNS

## Step 3: Access the Application

Open your browser to `https://ghmcp.local`

You'll be redirected through:
1. OAuth2 Proxy
2. Keycloak login
3. GitHub OAuth authorization
4. Back to the application

## Step 4: Explore GitHub MCP Tools

### List Available Tools

```bash
curl https://ghmcp.local/api/mcp/github/tools \
  -H "Cookie: _oauth2_proxy=..."
```

You should see GitHub-specific tools available through the MCP interface.

### Call a GitHub Tool

Example - get repository information:

```bash
curl -X POST https://ghmcp.local/api/mcp/github/tools/get_repository \
  -H "Cookie: _oauth2_proxy=..." \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "inxm-ai",
    "repo": "enterprise-mcp-bridge"
  }'
```

### Start a Session

For stateful interactions:

```bash
# Start session
curl -X POST https://ghmcp.local/api/mcp/github/session/start \
  -H "Cookie: _oauth2_proxy=..." \
  -d '{"session_id": "my-github-session"}'

# Use tools in session
curl -X POST https://ghmcp.local/api/mcp/github/session/my-github-session/tools/list_issues \
  -H "Cookie: _oauth2_proxy=..." \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "inxm-ai",
    "repo": "enterprise-mcp-bridge",
    "state": "open"
  }'

# Close session
curl -X POST https://ghmcp.local/api/mcp/github/session/my-github-session/close \
  -H "Cookie: _oauth2_proxy=..."
```

## Step 5: Understanding the Configuration

### Environment Variables

Key variables set by `start.sh`:

| Variable | Description |
|----------|-------------|
| `GITHUB_CLIENT_ID` | OAuth app client ID |
| `GITHUB_CLIENT_SECRET` | OAuth app client secret |
| `GITHUB_OAUTH_SCOPES` | Space-separated scopes |
| `MCP_REMOTE_HEADER_X_MCP_READONLY` | Set to `true` for read-only access |
| `MCP_REMOTE_HEADER_X_MCP_WORKSPACE_ID` | Optional workspace scoping |
| `MCP_REMOTE_REDIRECT_URI` | OAuth callback URL |

### Custom Headers

Any variable prefixed with `MCP_REMOTE_HEADER_` becomes an HTTP header:

```bash
# Becomes: X-MCP-Readonly: true
MCP_REMOTE_HEADER_X_MCP_READONLY=true

# Becomes: X-MCP-Organization-ID: my-org
MCP_REMOTE_HEADER_X_MCP_ORGANIZATION_ID=my-org
```

### Bridge Configuration

The bridge is configured in remote mode:

```yaml
# docker-compose.yml
services:
  mcp-bridge:
    environment:
      # Remote MCP server
      - MCP_REMOTE_SERVER=https://api.githubcopilot.com/mcp/
      
      # OAuth token exchange
      - MCP_REMOTE_SERVER_OAUTH_ISSUER=${KEYCLOAK_URL}/realms/ghmcp
      - MCP_REMOTE_CLIENT_ID=bridge-client
      - MCP_REMOTE_SCOPE=${GITHUB_OAUTH_SCOPES}
      
      # Custom headers
      - MCP_REMOTE_HEADER_X_MCP_READONLY=${MCP_REMOTE_HEADER_X_MCP_READONLY}
```

## Step 6: Token Exchange Flow

Understanding how tokens flow:

```
1. User clicks login
   ↓
2. Redirected to Keycloak
   ↓
3. Keycloak redirects to GitHub
   ↓
4. User authorizes on GitHub
   ↓
5. GitHub returns to Keycloak with code
   ↓
6. Keycloak exchanges code for GitHub token
   ↓
7. Keycloak stores GitHub token
   ↓
8. User receives Keycloak token
   ↓
9. Bridge receives request with Keycloak token
   ↓
10. Bridge exchanges Keycloak token for GitHub token
    ↓
11. Bridge calls GitHub MCP API with GitHub token
```

## Step 7: Customize for Your Use Case

### Read-Write Access

Enable write operations:

```bash
# Edit .env file
MCP_REMOTE_HEADER_X_MCP_READONLY=false

# Restart
docker-compose restart
```

### Workspace Scoping

Scope to specific GitHub organization/workspace:

```bash
# Edit .env file
MCP_REMOTE_HEADER_X_MCP_WORKSPACE_ID=my-org-id

# Restart
docker-compose restart
```

### Additional Scopes

Request more GitHub permissions:

```bash
# Edit .env file
GITHUB_OAUTH_SCOPES="read:user user:email copilot:chat repo admin:org"

# Re-run start.sh to update Keycloak configuration
./start.sh
```

## Production Deployment

### Security Considerations

1. **Use Valid Certificates**
   - Replace self-signed certs with Let's Encrypt or commercial CA
   - Configure certificate auto-renewal

2. **Secure Secrets**
   ```bash
   # Use secret management
   GITHUB_CLIENT_SECRET_FILE=/run/secrets/github-secret
   ```

3. **Rate Limiting**
   ```bash
   # Configure rate limits
   RATE_LIMIT="100/minute"
   ```

4. **Audit Logging**
   ```bash
   # Enable comprehensive logging
   LOG_LEVEL=info
   LOG_FORMAT=json
   ```

### Scaling

For production scale:

```yaml
# docker-compose.yml
services:
  mcp-bridge:
    deploy:
      replicas: 3
    environment:
      - SESSION_MANAGER_TYPE=redis
      - REDIS_URL=redis://redis:6379
```

## Troubleshooting

### Browser Cannot Reach `https://ghmcp.local`

**Cause:** CA certificate not trusted or DNS not configured

**Solution:**
1. Re-run `./start.sh` to regenerate certificates
2. Trust the CA certificate in your browser
3. Verify `/etc/hosts` has entry for `ghmcp.local`

### GitHub Login Loops

**Cause:** OAuth callback URL mismatch

**Solution:**
1. Verify callback URL is **exactly** `https://ghmcp.local/oauth2/callback`
2. Check scopes are valid
3. Ensure GitHub app is not suspended

### Remote MCP Returns 403

**Cause:** Insufficient permissions or missing headers

**Solution:**
1. Set `MCP_REMOTE_HEADER_X_MCP_READONLY=false` for write access
2. Add required workspace headers
3. Verify GitHub token has necessary scopes

### Tools List is Empty

**Cause:** User lacks access to MCP workspace

**Solution:**
1. Verify GitHub account has Copilot access
2. Check workspace scoping headers
3. Confirm OAuth scopes cover required tools

## Step 8: Using with LLM

The example includes integration with LLM for chat:

### Use Bundled Dummy LLM

Default configuration includes a simple LLM for testing:

```bash
curl -X POST https://ghmcp.local/api/chat \
  -H "Cookie: _oauth2_proxy=..." \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "List issues in inxm-ai/enterprise-mcp-bridge"}
    ]
  }'
```

### Use OpenAI-Compatible Endpoint

Configure during `start.sh` or edit `.env`:

```bash
OAI_BASE_URL=https://api.openai.com/v1
OAI_API_KEY=sk-...
OAI_MODEL=gpt-4
```

## Clean Up

Stop the example:

```bash
cd example/remote-mcp-github
./stop.sh
```

This removes:
- Docker containers
- Local DNS entries
- Rendered Keycloak realm
- `.env` file
- (Certificates are preserved)

**Note:** The script does not delete your GitHub OAuth App. Rotate or delete credentials manually if needed.

## Summary

You now know how to:

✅ Set up GitHub OAuth integration  
✅ Connect to GitHub's Remote MCP API  
✅ Configure token exchange flow  
✅ Use GitHub Copilot MCP tools  
✅ Deploy the complete example  
✅ Customize for production use  

## Next Steps

- [Group-Based Access Tutorial](group-based-access.md)
- [Configure OAuth](../how-to/configure-oauth.md)
- [Use Remote MCP Servers](../how-to/remote-mcp-servers.md)
- [Security Model](../explanation/security.md)

## Reference

- [Example Code](https://github.com/inxm-ai/enterprise-mcp-bridge/tree/main/example/remote-mcp-github)
- [GitHub OAuth Apps Documentation](https://docs.github.com/en/apps/oauth-apps)
- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [MCP Protocol](https://modelcontextprotocol.io)
