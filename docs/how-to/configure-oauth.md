# Configure OAuth

Learn how to set up OAuth2 authentication for the Enterprise MCP Bridge with Keycloak or other identity providers.

## Overview

The Enterprise MCP Bridge supports OAuth2 token exchange for:
- User authentication
- Automatic token injection for downstream services
- Token refresh handling
- Group-based access control

## Prerequisites

- Keycloak (or another OAuth2 provider) instance
- Basic understanding of OAuth2 flows
- Admin access to configure clients

## Keycloak Setup

### Step 1: Create a Realm

1. Log in to Keycloak Admin Console
2. Click "Create Realm"
3. Name it `mcp` (or your preferred name)
4. Click "Create"

### Step 2: Create the Bridge Client

1. Navigate to Clients → Create Client
2. Configure:
   - **Client ID:** `mcp-bridge`
   - **Client Protocol:** `openid-connect`
   - **Client Authentication:** ON
   - **Authorization:** OFF

3. Set Redirect URIs:
   ```
   https://your-bridge-domain.com/oauth/callback
   http://localhost:8000/oauth/callback  # for development
   ```

4. Note the **Client Secret** from the Credentials tab

### Step 3: Configure Token Exchange

Enable token exchange for the bridge client:

1. Go to Service Account Roles tab
2. Assign role: `realm-management` → `view-users`
3. Create a client scope for token exchange:
   - Name: `token-exchange`
   - Protocol: `openid-connect`

### Step 4: Create Resource Clients

For each downstream service (e.g., GitHub, Microsoft 365):

1. Create a new client (e.g., `github-client`)
2. Configure identity provider:
   - Navigate to Identity Providers
   - Add provider (GitHub, Microsoft, etc.)
   - Configure OAuth credentials from the service
   - Enable "Store Token"

## Bridge Configuration

### Environment Variables

```bash
# OAuth Provider
OAUTH_ISSUER_URL="https://keycloak.example.com/realms/mcp"
OAUTH_CLIENT_ID="mcp-bridge"
OAUTH_CLIENT_SECRET="your-client-secret"
OAUTH_REDIRECT_URI="https://your-bridge.com/oauth/callback"

# Token Exchange
OAUTH_ENABLE_TOKEN_EXCHANGE=true
OAUTH_TOKEN_EXCHANGE_SCOPE="offline_access"
```

### Full Example

```bash
# docker-compose.yml
version: '3.8'

services:
  keycloak:
    image: quay.io/keycloak/keycloak:latest
    environment:
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: admin
    ports:
      - "8080:8080"
    command: start-dev

  mcp-bridge:
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    environment:
      - MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-github
      - OAUTH_ISSUER_URL=http://keycloak:8080/realms/mcp
      - OAUTH_CLIENT_ID=mcp-bridge
      - OAUTH_CLIENT_SECRET=your-secret-here
      - OAUTH_REDIRECT_URI=http://localhost:8000/oauth/callback
      - OAUTH_ENABLE_TOKEN_EXCHANGE=true
    ports:
      - "8000:8000"
    depends_on:
      - keycloak
```

## Token Exchange Flow

### 1. User Authentication

```bash
# User initiates OAuth flow
curl -X GET "http://localhost:8000/oauth/login?redirect_url=/dashboard"
```

This redirects to Keycloak for authentication.

### 2. Callback Handling

After successful authentication, user is redirected back with an authorization code:

```
http://localhost:8000/oauth/callback?code=AUTH_CODE&state=STATE
```

The bridge exchanges the code for tokens automatically.

### 3. Accessing Protected Resources

```bash
# Make authenticated request
curl -X GET http://localhost:8000/tools \
  -H "Authorization: Bearer USER_ACCESS_TOKEN"
```

### 4. Automatic Token Injection

When calling tools that require downstream tokens:

```bash
# The bridge automatically exchanges tokens
curl -X POST http://localhost:8000/tools/github_create_issue \
  -H "Authorization: Bearer USER_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "myorg/myrepo",
    "title": "Issue title",
    "body": "Issue description"
  }'
```

The bridge:
1. Receives user token
2. Exchanges it for GitHub token via Keycloak
3. Injects GitHub token into the MCP server call
4. Returns the result

## Advanced Configuration

### Custom Token Exchange

```python
# Configure custom token retriever
from app.oauth.token_retriever import TokenRetrieverFactory

# Register custom retriever
TokenRetrieverFactory.register(
    "custom-service",
    CustomTokenRetriever(
        issuer_url="https://auth.example.com",
        client_id="bridge-client",
        client_secret="secret"
    )
)
```

### Group-Based Access Control

Configure Keycloak groups:

1. Create groups in Keycloak (e.g., `engineering`, `marketing`)
2. Assign users to groups
3. Map groups to token claims

In the bridge:

```bash
# Groups are automatically extracted from token
# Data paths resolve based on group membership
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/{group_id}/memory.json"
```

Users in `engineering` group access `/data/engineering/memory.json`

### Token Refresh

The bridge automatically refreshes tokens:

```python
# Automatic refresh when token expires
# No configuration needed - handled automatically
```

Refresh token settings:

```bash
# Request offline access for refresh tokens
OAUTH_SCOPE="openid profile email offline_access"

# Refresh buffer (refresh N seconds before expiry)
OAUTH_REFRESH_BUFFER_SECONDS=300
```

## Testing OAuth Setup

### Test Authentication Flow

```bash
# 1. Start OAuth flow
curl -v http://localhost:8000/oauth/login

# 2. Follow redirect to Keycloak
# 3. Log in with test user
# 4. Get redirected back with token
```

### Verify Token

```bash
# Get token info
curl http://localhost:8000/oauth/userinfo \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Response:
```json
{
  "sub": "user-id-123",
  "email": "user@example.com",
  "groups": ["engineering", "users"],
  "preferred_username": "john.doe"
}
```

### Test Token Exchange

```bash
# Call a tool requiring downstream token
curl -X POST http://localhost:8000/tools/some_protected_tool \
  -H "Authorization: Bearer YOUR_USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"param": "value"}'

# Check logs to verify token exchange occurred
docker logs enterprise-mcp-bridge | grep "token-exchange"
```

## Security Best Practices

### 1. Use HTTPS in Production

```bash
# Never use HTTP for OAuth in production
OAUTH_ISSUER_URL="https://keycloak.example.com/realms/mcp"
OAUTH_REDIRECT_URI="https://bridge.example.com/oauth/callback"
```

### 2. Secure Client Secrets

```bash
# Use secret management
# Don't hardcode secrets in config files

# Kubernetes
kubectl create secret generic oauth-secrets \
  --from-literal=client-secret='your-secret-here'

# Docker
docker run -e OAUTH_CLIENT_SECRET_FILE=/run/secrets/oauth-secret ...
```

### 3. Validate Tokens

```bash
# Enable token validation
OAUTH_VALIDATE_TOKENS=true

# Verify issuer
OAUTH_VERIFY_ISSUER=true

# Check audience
OAUTH_AUDIENCE="mcp-bridge"
```

### 4. Limit Token Scope

Request only necessary scopes:

```bash
OAUTH_SCOPE="openid profile email"  # Minimal scopes
```

### 5. Token Storage

```bash
# Use secure session storage
SESSION_MANAGER_TYPE="redis"
REDIS_URL="rediss://redis:6379"  # Use TLS
REDIS_PASSWORD="strong-password"
```

## Troubleshooting

### Invalid Redirect URI

**Error:** `invalid_redirect_uri`

**Solution:** Verify redirect URI in Keycloak client matches exactly:
```bash
# Must match exactly (including trailing slashes)
OAUTH_REDIRECT_URI="https://bridge.example.com/oauth/callback"
```

### Token Exchange Failed

**Error:** `Token exchange failed for provider 'github'`

**Solution:**
1. Verify identity provider is configured in Keycloak
2. Check "Store Token" is enabled
3. Verify broker permissions: `realm-management.view-users`

### Invalid Token

**Error:** `Invalid or expired token`

**Solution:**
1. Check token expiration
2. Verify `OAUTH_ISSUER_URL` is correct
3. Ensure clock sync between services

### Insufficient Permissions

**Error:** `403 Forbidden`

**Solution:**
1. Check user has required groups/roles
2. Verify group mappings in token
3. Check Keycloak mapper configuration

## Example: GitHub Integration

Complete example with GitHub token exchange:

```yaml
# docker-compose.yml
version: '3.8'

services:
  keycloak:
    image: quay.io/keycloak/keycloak:latest
    environment:
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: admin
    ports:
      - "8080:8080"
    command: start-dev
    volumes:
      - ./keycloak-config:/opt/keycloak/data/import

  mcp-bridge:
    image: ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
    environment:
      # MCP Server
      - MCP_SERVER_COMMAND=npx -y @modelcontextprotocol/server-github
      
      # OAuth Configuration
      - OAUTH_ISSUER_URL=http://keycloak:8080/realms/mcp
      - OAUTH_CLIENT_ID=mcp-bridge
      - OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
      - OAUTH_REDIRECT_URI=http://localhost:8000/oauth/callback
      
      # Token Exchange
      - OAUTH_ENABLE_TOKEN_EXCHANGE=true
      - OAUTH_TOKEN_EXCHANGE_PROVIDER=github
      - OAUTH_SCOPE=openid profile email offline_access
    ports:
      - "8000:8000"
    depends_on:
      - keycloak
```

Keycloak realm configuration (`keycloak-config/realm-mcp.json`):

```json
{
  "realm": "mcp",
  "enabled": true,
  "clients": [{
    "clientId": "mcp-bridge",
    "enabled": true,
    "protocol": "openid-connect",
    "publicClient": false,
    "redirectUris": ["http://localhost:8000/oauth/callback"],
    "serviceAccountsEnabled": true
  }],
  "identityProviders": [{
    "alias": "github",
    "providerId": "github",
    "enabled": true,
    "storeToken": true,
    "config": {
      "clientId": "YOUR_GITHUB_CLIENT_ID",
      "clientSecret": "YOUR_GITHUB_CLIENT_SECRET"
    }
  }]
}
```

## Summary

You now know how to:

✅ Set up Keycloak for OAuth2 authentication  
✅ Configure the bridge for token exchange  
✅ Implement group-based access control  
✅ Secure your OAuth configuration  
✅ Troubleshoot common issues  

## Next Steps

- [Deploy to Production](deploy-production.md)
- [Handle Token Exchange](token-exchange.md)
- [Security Model](../explanation/security.md)

## Resources

- [Keycloak Documentation](https://www.keycloak.org/documentation)
- [OAuth2 RFC](https://tools.ietf.org/html/rfc6749)
- [Token Exchange Example](../reference/examples.md#token-exchange)
