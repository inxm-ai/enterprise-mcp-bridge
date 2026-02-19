# Group-Based Access Tutorial

Learn how to implement group-based data access control using Enterprise MCP Bridge with the Memory MCP Server.

## What You'll Learn

- How to configure group-based memory access
- How to set up users with different group memberships
- How to test group isolation
- How to deploy the complete example

## Overview

Group-based access control allows you to partition data by user groups, ensuring that users can only access data belonging to their groups. This is essential for multi-tenant applications.

## Architecture

```
User A (group: engineering)
  ↓
Bridge → /data/engineering/memory.json

User B (group: marketing)
  ↓
Bridge → /data/marketing/memory.json
```

Each group has its own isolated memory store.

## Prerequisites

- Docker and Docker Compose installed
- Basic understanding of OAuth and Keycloak
- Completed the [Getting Started](getting-started.md) tutorial

## Example Setup

The project includes a complete working example at `example/memory-group-access/`.

### Pre-configured Users

| Username         | Password       | Groups         | Memory Access              |
|------------------|----------------|----------------|----------------------------|
| `admin`          | `admin123`     | administrators | All group memories + personal |
| `john.engineer`  | `engineer123`  | engineering    | Engineering memories + personal |
| `jane.marketing` | `marketing123` | marketing      | Marketing memories + personal |
| `bob.sales`      | `sales123`     | sales          | Sales memories + personal |

## Step 1: Start the Example

Navigate to the example directory:

```bash
cd example/memory-group-access
./start.sh
```

This will:
1. Generate local SSL certificates
2. Start Keycloak with pre-configured realm
3. Start the MCP Bridge with group-based configuration
4. Start OAuth2 Proxy for authentication
5. Set up local DNS entries

## Step 2: Access the Application

Open your browser to `https://inxm.local`

You'll be redirected to Keycloak for login.

## Step 3: Test Group Isolation

### Login as Engineering User

1. Login with username: `john.engineer`, password: `engineer123`
2. Start a session:
   ```bash
   curl -X POST https://inxm.local/api/mcp/session/start \
     -H "Cookie: _oauth2_proxy=..." \
     -d '{"session_id": "john-session"}'
   ```

3. Create an entity:
   ```bash
   curl -X POST https://inxm.local/api/mcp/session/john-session/tools/create_entities \
     -H "Cookie: _oauth2_proxy=..." \
     -H "Content-Type: application/json" \
     -d '{
       "entities": [{
         "name": "secret_project",
         "entityType": "project",
         "observations": ["Engineering secret project"]
       }]
     }'
   ```

4. Read the graph:
   ```bash
   curl -X POST https://inxm.local/api/mcp/session/john-session/tools/read_graph \
     -H "Cookie: _oauth2_proxy=..." \
     -H "Content-Type: application/json" \
     -d '{}'
   ```

You should see the `secret_project` entity.

### Login as Marketing User

1. Logout and login with username: `jane.marketing`, password: `marketing123`
2. Start a session:
   ```bash
   curl -X POST https://inxm.local/api/mcp/session/start \
     -H "Cookie: _oauth2_proxy=..." \
     -d '{"session_id": "jane-session"}'
   ```

3. Read the graph:
   ```bash
   curl -X POST https://inxm.local/api/mcp/session/jane-session/tools/read_graph \
     -H "Cookie: _oauth2_proxy=..." \
     -H "Content-Type: application/json" \
     -d '{}'
   ```

The graph should be **empty** - Jane cannot see John's engineering data!

## Step 4: Understanding the Configuration

The group-based access is configured using placeholders in the MCP server command:

```bash
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/{group_id}/memory.json"
```

### How It Works

1. **User Authenticates:** User logs in via Keycloak
2. **Token Contains Groups:** JWT token includes group membership
3. **Bridge Extracts Group:** Bridge reads `groups` claim from token
4. **Path Resolution:** `{group_id}` is replaced with user's primary group
5. **Isolated Storage:** Each group has separate memory file

### Group Priority

When a user belongs to multiple groups:
- Primary group (first in list) is used by default
- Can be overridden via configuration
- Admin group typically has special access

## Step 5: Customize for Your Use Case

### Configure Keycloak Groups

1. Open Keycloak admin console: `https://inxm.local/auth`
2. Navigate to your realm → Groups
3. Create new groups as needed
4. Assign users to groups

### Update MCP Server Command

For different data partitioning strategies:

```bash
# User-based isolation
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/{user_id}/memory.json"

# Hierarchical groups
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/{group_id}/{user_id}/memory.json"

# Shared + personal
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/shared.json /data/{user_id}/personal.json"
```

### Environment Variables

Key configuration variables:

```bash
# Enable group-based access
OAUTH_ENABLE_TOKEN_EXCHANGE=true
OAUTH_EXTRACT_GROUPS=true

# Group claim in JWT
OAUTH_GROUPS_CLAIM="groups"

# Default group for users without groups
OAUTH_DEFAULT_GROUP="users"
```

## Step 6: Verify Data Isolation

Check the data directory structure:

```bash
ls -la data/
# Should show:
# data/engineering/memory.json
# data/marketing/memory.json
# data/sales/memory.json
# data/administrators/memory.json
```

Inspect a specific group's data:

```bash
cat data/engineering/memory.json | jq
```

## Production Deployment

For production use:

1. **Persistent Storage:** Use volume mounts or network storage
   ```yaml
   volumes:
     - /mnt/secure/mcp-data:/data
   ```

2. **Backup Strategy:** Regular backups per group
   ```bash
   # Backup script
   for group in engineering marketing sales; do
     tar czf backup-$group-$(date +%Y%m%d).tar.gz data/$group/
   done
   ```

3. **Access Logging:** Log all group access
   ```bash
   LOG_LEVEL=info
   LOG_GROUP_ACCESS=true
   ```

4. **Encryption:** Encrypt data at rest
   ```bash
   # Use encrypted volumes
   # OR encrypt individual memory files
   ```

## Security Considerations

### Group Validation

Always validate group membership server-side:

```python
def validate_group_access(user_groups, required_group):
    if required_group not in user_groups:
        raise PermissionError(f"User not in group: {required_group}")
```

### Token Security

- Tokens contain sensitive group information
- Use HTTPS for all communication
- Short token expiration times
- Regular token rotation

### Audit Logging

Log all group-based access:

```json
{
  "timestamp": "2024-01-15T10:00:00Z",
  "user_id": "john.engineer",
  "groups": ["engineering"],
  "action": "read_graph",
  "group_accessed": "engineering",
  "success": true
}
```

## Troubleshooting

### User Can't Access Any Data

**Cause:** User not assigned to any groups

**Solution:**
1. Check Keycloak group assignment
2. Verify `groups` claim in JWT token
3. Check `OAUTH_DEFAULT_GROUP` is set

### User Can Access Wrong Group's Data

**Cause:** Incorrect group resolution

**Solution:**
1. Verify group priority configuration
2. Check token group claims
3. Review path resolution logic

### Data Not Persisting

**Cause:** Volume mounting issues

**Solution:**
1. Check Docker volume configuration
2. Verify file permissions
3. Ensure data directory exists

## Clean Up

Stop the example:

```bash
cd example/memory-group-access
./stop.sh
```

This removes:
- Docker containers
- Local DNS entries
- Temporary files
- (Certificates are preserved for future use)

## Summary

You now know how to:

✅ Set up group-based access control  
✅ Configure users with different group memberships  
✅ Test group isolation  
✅ Deploy the complete example  
✅ Customize for your use case  
✅ Secure group-based access in production  

## Next Steps

- [GitHub Integration Tutorial](github-integration.md)
- [Configure OAuth](../how-to/configure-oauth.md)
- [Security Model](../explanation/security.md)

## Reference

- [Example Code](https://github.com/inxm-ai/enterprise-mcp-bridge/tree/main/example/memory-group-access)
- [Keycloak Groups Documentation](https://www.keycloak.org/docs/latest/server_admin/#groups)
- [Memory MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)
