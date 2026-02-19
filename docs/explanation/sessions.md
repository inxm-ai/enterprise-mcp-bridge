# Session Management

Understanding how sessions work in the Enterprise MCP Bridge.

## Overview

Sessions are the core mechanism for maintaining stateful interactions between clients and MCP servers. They enable:

- Long-running conversations
- Persistent context across multiple requests
- User-specific state isolation
- Resource lifecycle management

## Session Lifecycle

```
1. Client → POST /session/start
   ↓
2. Bridge creates new session
   ↓
3. MCP server process spawned
   ↓
4. Session ID returned to client
   ↓
5. Client makes requests with session ID
   ↓
6. MCP server maintains state
   ↓
7. Client → POST /session/{id}/ping (keep alive)
   ↓
8. Client → POST /session/{id}/close
   ↓
9. MCP server process terminated
   ↓
10. Resources cleaned up
```

## Session Modes

### Stateless Mode

For one-off tool calls without maintaining state:

```bash
# Call tool directly (no session)
curl -X POST http://localhost:8000/tools/my_tool \
  -d '{"param": "value"}'
```

**Characteristics:**
- No session created
- MCP server spawned temporarily
- Process terminated after response
- No state persisted
- Lower resource usage
- Higher latency (process startup)

**Use Cases:**
- Simple, independent tool calls
- Read-only operations
- Stateless APIs
- High-throughput scenarios

### Stateful Mode

For maintaining conversation context:

```bash
# Start session
curl -X POST http://localhost:8000/session/start \
  -d '{"session_id": "my-session"}'

# Call tools within session
curl -X POST http://localhost:8000/session/my-session/tools/my_tool \
  -d '{"param": "value"}'

# Close session
curl -X POST http://localhost:8000/session/my-session/close
```

**Characteristics:**
- Persistent MCP server process
- State maintained across requests
- Lower per-request latency
- Higher resource usage
- Supports complex workflows

**Use Cases:**
- Conversational AI
- Multi-step workflows
- Collaborative editing
- Complex business logic

## Session Storage

### In-Memory Manager (Default)

Simple, fast, but not distributed:

```python
# Configuration
SESSION_MANAGER_TYPE=memory
```

**Pros:**
- Fast access
- No external dependencies
- Simple setup

**Cons:**
- Not distributed (single instance)
- Lost on restart
- Limited by RAM
- No persistence

**Best For:**
- Development
- Testing
- Single-instance deployments
- Ephemeral sessions

### Redis Manager (Production)

Distributed, persistent session storage:

```python
# Configuration
SESSION_MANAGER_TYPE=redis
REDIS_URL=redis://redis:6379
```

**Pros:**
- Distributed (multi-instance)
- Persists across restarts
- Scalable
- Shared state

**Cons:**
- Requires Redis
- Slightly higher latency
- External dependency

**Best For:**
- Production
- Multi-instance deployments
- High availability
- Persistent sessions

## Session Isolation

### Per-User Isolation

Each user has isolated sessions:

```
User A:
  Session 1 → MCP Process 1 → Data A
  Session 2 → MCP Process 2 → Data A

User B:
  Session 3 → MCP Process 3 → Data B
  Session 4 → MCP Process 4 → Data B
```

**Implementation:**
- Sessions tagged with user_id
- Cross-user access blocked
- Enforced at session manager level

### Per-Group Isolation

Sessions can be group-scoped:

```
Engineering Group:
  Session 1 → /data/engineering/

Marketing Group:
  Session 2 → /data/marketing/
```

**Implementation:**
- Group extracted from OAuth token
- Path placeholders resolved
- Data partitioned by group

## Session Timeout

### Inactivity Timeout

Sessions timeout after inactivity:

```python
# Default: 30 minutes
SESSION_TIMEOUT_SECONDS=1800
```

**Mechanism:**
- Last activity timestamp tracked
- Periodic cleanup job checks timestamps
- Expired sessions terminated

### Keep-Alive Mechanism

Prevent timeout with ping:

```bash
# Ping every 5 minutes
curl -X POST http://localhost:8000/session/my-session/ping
```

**Response:**
```json
{
  "session_id": "my-session",
  "status": "active",
  "last_ping": "2024-01-15T10:30:00Z",
  "expires_at": "2024-01-15T11:00:00Z"
}
```

### Cleanup Process

Automatic cleanup runs periodically:

```python
# Cleanup interval: 5 minutes
SESSION_CLEANUP_INTERVAL_SECONDS=300
```

**Process:**
1. Check all sessions
2. Find expired sessions
3. Terminate MCP processes
4. Remove from session store
5. Release resources

## Resource Management

### Per-Session Resources

Each session consumes:

```
- 1 MCP server process
- Memory for process state
- File descriptors
- Network connections
- Storage for session data
```

### Resource Limits

Recommended limits:

```yaml
# Container limits
resources:
  limits:
    memory: 512Mi  # Per pod
    cpu: 500m
  requests:
    memory: 256Mi
    cpu: 250m

# Session limits
MAX_SESSIONS_PER_USER: 10
MAX_TOTAL_SESSIONS: 1000
```

### Monitoring

Track session metrics:

```
# Prometheus metrics
mcp_sessions_active{user_id="user123"} 3
mcp_sessions_created_total 150
mcp_sessions_closed_total 145
mcp_sessions_expired_total 2
```

## Session Operations

### Create Session

```bash
POST /session/start
{
  "session_id": "my-session-123"
}
```

**Response:**
```json
{
  "session_id": "my-session-123",
  "status": "started",
  "created_at": "2024-01-15T10:00:00Z",
  "user_id": "user123",
  "group_id": "engineering"
}
```

### Use Session

```bash
POST /session/{session_id}/tools/{tool_name}
{
  "param": "value"
}
```

### Ping Session

```bash
POST /session/{session_id}/ping
```

### Close Session

```bash
POST /session/{session_id}/close
```

**Response:**
```json
{
  "session_id": "my-session-123",
  "status": "closed",
  "duration": "15m30s",
  "total_requests": 42
}
```

## Best Practices

### 1. Explicit Session Management

Always close sessions when done:

```python
session_id = create_session()
try:
    # Use session
    result = call_tool(session_id, "my_tool", {...})
finally:
    # Always close
    close_session(session_id)
```

### 2. Session Naming

Use descriptive session IDs:

```python
# Good
session_id = f"user-{user_id}-{purpose}-{timestamp}"
session_id = "alice-project-alpha-20240115-101530"

# Bad
session_id = "session1"
session_id = "temp"
```

### 3. Keep-Alive Strategy

Implement automatic ping:

```python
def keep_alive(session_id, interval=300):
    """Ping session every interval seconds"""
    while session_active(session_id):
        ping_session(session_id)
        time.sleep(interval)

# Start in background
threading.Thread(
    target=keep_alive,
    args=(session_id, 300),
    daemon=True
).start()
```

### 4. Error Handling

Handle session errors gracefully:

```python
try:
    result = call_tool(session_id, "tool", {...})
except SessionNotFoundError:
    # Session expired or closed
    session_id = create_session()
    result = call_tool(session_id, "tool", {...})
except SessionPermissionError:
    # Wrong user accessing session
    logger.error(f"Permission denied for session {session_id}")
    raise
```

### 5. Resource Cleanup

Implement cleanup for abandoned sessions:

```python
# Periodic cleanup
def cleanup_stale_sessions():
    for session in get_all_sessions():
        if session.last_activity < now() - timedelta(hours=1):
            close_session(session.id)
```

## Scaling Considerations

### Horizontal Scaling

For multiple bridge instances:

1. **Use Redis:** Shared session storage
2. **Load Balancer:** Distribute requests
3. **Session Affinity:** Optional sticky sessions

```
Client → Load Balancer → Bridge Instance 1 → Redis
                      → Bridge Instance 2 → Redis
                      → Bridge Instance 3 → Redis
```

### Vertical Scaling

For single instance:

1. **Increase Resources:** More CPU/memory
2. **Tune Timeouts:** Adjust session timeout
3. **Limit Sessions:** Set max sessions per user

## Troubleshooting

### Session Not Found

**Symptoms:**
```json
{"error": "Session not found", "session_id": "..."}
```

**Causes:**
- Session expired
- Session closed
- Invalid session ID
- Wrong bridge instance (without Redis)

**Solutions:**
- Create new session
- Implement keep-alive
- Check session ID
- Use Redis for multi-instance

### High Memory Usage

**Symptoms:**
- Container OOM kills
- Slow performance
- High memory metrics

**Causes:**
- Too many active sessions
- Memory leaks in MCP server
- No session cleanup

**Solutions:**
- Reduce session timeout
- Limit sessions per user
- Fix MCP server leaks
- Enable cleanup

### Session Hijacking

**Symptoms:**
- Unauthorized access
- Session used by wrong user

**Prevention:**
- Validate user_id on each request
- Use cryptographically random session IDs
- Implement session binding
- Enable audit logging

## Summary

Session management in Enterprise MCP Bridge:

✅ Enables stateful interactions  
✅ Supports both stateless and stateful modes  
✅ Provides user and group isolation  
✅ Includes automatic cleanup  
✅ Scales horizontally with Redis  
✅ Offers comprehensive monitoring  

## Next Steps

- [Architecture Overview](architecture.md)
- [Security Model](security.md)
- [Multi-User Sessions Tutorial](../tutorials/multi-user-sessions.md)
- [Configuration Reference](../reference/configuration.md)
