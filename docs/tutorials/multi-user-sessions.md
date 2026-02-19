# Multi-User Sessions

Learn how to manage concurrent users and sessions in the Enterprise MCP Bridge.

## What You'll Learn

- How session management works in the bridge
- How to handle multiple concurrent users
- How to implement user isolation
- Best practices for session lifecycle

## Prerequisites

- Completed [Getting Started](getting-started.md) tutorial
- Understanding of REST APIs
- Basic knowledge of OAuth (helpful but not required)

## Understanding Sessions

The Enterprise MCP Bridge supports two modes:

1. **Stateless Mode:** One-off tool calls without maintaining state
2. **Stateful Mode:** Long-running sessions with preserved context

### Why Use Sessions?

Sessions are essential for:
- Maintaining conversational context
- Managing long-running operations
- User-specific state and data
- Resource cleanup and lifecycle management

## Session Architecture

Each session:
- Has a unique session ID
- Is tied to a user (via OAuth token or session ID)
- Maintains its own MCP server process
- Has independent state and memory
- Can be pinged to stay alive
- Is automatically cleaned up when idle

![Session Architecture](../architecture.png)

## Step 1: Start a Session

### Basic Session Creation

```bash
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-alice-session-1"
  }'
```

Response:
```json
{
  "session_id": "user-alice-session-1",
  "status": "started",
  "created_at": "2024-01-15T10:00:00Z",
  "user_id": null
}
```

### With User Context (OAuth)

When using OAuth authentication:

```bash
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_OAUTH_TOKEN" \
  -d '{
    "session_id": "work-session"
  }'
```

The session will be automatically scoped to the authenticated user.

## Step 2: Using Sessions

### Call Tools Within a Session

```bash
curl -X POST http://localhost:8000/session/user-alice-session-1/tools/create_entities \
  -H "Content-Type: application/json" \
  -d '{
    "entities": [
      {
        "name": "project_alpha",
        "entityType": "project",
        "observations": ["High priority project", "Due next month"]
      }
    ]
  }'
```

### Retrieve Session State

```bash
curl -X POST http://localhost:8000/session/user-alice-session-1/tools/read_graph \
  -H "Content-Type: application/json" \
  -d '{}'
```

The session maintains all entities created within it.

## Step 3: Managing Multiple Users

### User Isolation Example

```bash
# User Alice starts a session
curl -X POST http://localhost:8000/session/start \
  -H "X-User-Id: alice" \
  -d '{"session_id": "alice-session"}'

# Alice creates an entity
curl -X POST http://localhost:8000/session/alice-session/tools/create_entities \
  -H "X-User-Id: alice" \
  -d '{
    "entities": [{
      "name": "alice_secret",
      "entityType": "note",
      "observations": ["This is Alice'\''s private note"]
    }]
  }'

# User Bob starts a separate session
curl -X POST http://localhost:8000/session/start \
  -H "X-User-Id: bob" \
  -d '{"session_id": "bob-session"}'

# Bob cannot access Alice's data
curl -X POST http://localhost:8000/session/bob-session/tools/read_graph \
  -H "X-User-Id: bob" \
  -d '{}'
# Returns empty graph - Bob's session is isolated from Alice's
```

### Concurrent Sessions per User

A single user can have multiple concurrent sessions:

```bash
# Alice starts session for work
curl -X POST http://localhost:8000/session/start \
  -H "Authorization: Bearer ALICE_TOKEN" \
  -d '{"session_id": "alice-work"}'

# Alice starts session for personal tasks
curl -X POST http://localhost:8000/session/start \
  -H "Authorization: Bearer ALICE_TOKEN" \
  -d '{"session_id": "alice-personal"}'

# Both sessions are active and isolated from each other
```

## Step 4: Session Lifecycle Management

### Keep Sessions Alive

Sessions timeout after inactivity. Keep them alive with pings:

```bash
# Ping every 5 minutes to prevent timeout
curl -X POST http://localhost:8000/session/user-alice-session-1/ping
```

Response:
```json
{
  "session_id": "user-alice-session-1",
  "status": "active",
  "last_ping": "2024-01-15T10:30:00Z"
}
```

### Explicit Session Closure

Always close sessions when done:

```bash
curl -X POST http://localhost:8000/session/user-alice-session-1/close
```

Response:
```json
{
  "session_id": "user-alice-session-1",
  "status": "closed",
  "duration": "30m15s"
}
```

?> Closing sessions explicitly helps free resources and prevents memory leaks.

### Automatic Cleanup

The bridge automatically cleans up sessions that:
- Haven't received pings within the timeout period (default: 30 minutes)
- Have been explicitly closed
- Belong to revoked OAuth tokens

## Step 5: Group-Based Data Access

### Configure Group Access

When using OAuth with group membership:

```bash
# User in 'engineering' group
curl -X POST http://localhost:8000/session/start \
  -H "Authorization: Bearer TOKEN_WITH_ENGINEERING_GROUP" \
  -d '{"session_id": "eng-session"}'

# Data path automatically resolves to group-specific location
# e.g., /data/engineering/memory.json
```

### Group Isolation Example

```python
# Configure group-based paths in your deployment
MCP_SERVER_COMMAND="npx -y @modelcontextprotocol/server-memory /data/{group_id}/memory.json"
```

Users in different groups will have separate data stores:
- Engineering group → `/data/engineering/memory.json`
- Marketing group → `/data/marketing/memory.json`

![Group Access Control](../mcp-memory-access-denied.png)

## Step 6: Best Practices

### Session Naming

Use descriptive, hierarchical session IDs:

```bash
# Good examples
"user-{user_id}-{purpose}-{timestamp}"
"alice-project-alpha-20240115"
"team-engineering-planning-session-1"

# Avoid
"session1"
"temp"
"test"
```

### Resource Management

```python
# Pattern: Use context managers or try/finally
import requests

session_id = "my-important-session"
base_url = "http://localhost:8000"

try:
    # Start session
    response = requests.post(
        f"{base_url}/session/start",
        json={"session_id": session_id}
    )
    
    # Use session
    result = requests.post(
        f"{base_url}/session/{session_id}/tools/my_tool",
        json={"param": "value"}
    )
    
finally:
    # Always close session
    requests.post(f"{base_url}/session/{session_id}/close")
```

### Ping Strategy

```python
import time
import threading

def keep_alive(session_id, interval=300):
    """Ping session every interval seconds"""
    while True:
        try:
            requests.post(f"http://localhost:8000/session/{session_id}/ping")
            time.sleep(interval)
        except:
            break

# Start keep-alive in background
thread = threading.Thread(
    target=keep_alive,
    args=("my-session", 300),
    daemon=True
)
thread.start()
```

### Monitoring Sessions

```bash
# Get session status
curl http://localhost:8000/session/user-alice-session-1/status

# List all sessions (requires admin privileges)
curl http://localhost:8000/sessions \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

## Step 7: Scaling Sessions

### In-Memory Session Manager (Default)

- Simple and fast
- Limited to single-instance deployments
- Sessions lost on restart

### Redis Session Manager (Production)

For multi-instance deployments, use Redis:

```python
# Configure in your deployment
SESSION_MANAGER_TYPE="redis"
REDIS_URL="redis://localhost:6379"
```

Benefits:
- Sessions shared across instances
- Horizontal scaling
- Session persistence
- Distributed locking

## Real-World Example

### Multi-User Customer Support System

```python
import requests
import uuid

class SupportSession:
    def __init__(self, base_url, customer_id):
        self.base_url = base_url
        self.customer_id = customer_id
        self.session_id = f"support-{customer_id}-{uuid.uuid4()}"
        
    def __enter__(self):
        # Start session
        response = requests.post(
            f"{self.base_url}/session/start",
            json={"session_id": self.session_id},
            headers={"X-Customer-Id": self.customer_id}
        )
        return self
    
    def __exit__(self, *args):
        # Close session
        requests.post(
            f"{self.base_url}/session/{self.session_id}/close"
        )
    
    def add_note(self, note):
        return requests.post(
            f"{self.base_url}/session/{self.session_id}/tools/create_entities",
            json={
                "entities": [{
                    "name": f"note-{uuid.uuid4()}",
                    "entityType": "support_note",
                    "observations": [note]
                }]
            }
        )
    
    def get_history(self):
        return requests.post(
            f"{self.base_url}/session/{self.session_id}/tools/read_graph",
            json={}
        )

# Usage
with SupportSession("http://localhost:8000", "customer-12345") as session:
    session.add_note("Customer reported login issue")
    session.add_note("Reset password sent")
    history = session.get_history()
    print(history.json())
```

## Troubleshooting

### Session Not Found

```json
{"error": "Session not found", "session_id": "..."}
```

**Causes:**
- Session timed out due to inactivity
- Session was explicitly closed
- Invalid session ID

**Solution:** Start a new session

### Session Already Exists

```json
{"error": "Session already exists", "session_id": "..."}
```

**Solution:** Use a different session ID or close the existing one first

### Permission Denied

```json
{"error": "Permission denied"}
```

**Causes:**
- Trying to access another user's session
- Invalid or expired OAuth token

**Solution:** Verify authentication and use correct session ID

## Summary

In this tutorial, you learned:

✅ How to start and manage sessions  
✅ How to handle multiple concurrent users  
✅ Session lifecycle best practices  
✅ Group-based data isolation  
✅ Resource management strategies  
✅ Scaling considerations  

## Next Steps

- [Configure OAuth](../how-to/configure-oauth.md) for production authentication
- [Deploy to Production](../how-to/deploy-production.md) with session persistence
- [Monitor Your Deployment](../how-to/monitoring.md) to track session metrics

## Resources

- [Session Management API](../reference/endpoints.md#sessions)
- [Architecture Overview](../explanation/sessions.md)
- [Security Model](../explanation/security.md)
