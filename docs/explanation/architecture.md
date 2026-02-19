# Architecture Overview

Understanding the design and architecture of the Enterprise MCP Bridge.

## System Architecture

The Enterprise MCP Bridge sits between REST API clients and Model Context Protocol (MCP) servers, providing enterprise-grade features for production deployments.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            REST API Clients                              │
│                    (Browser, CLI, Apps, Services)                        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ HTTP/JSON
                             ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE MCP BRIDGE                                │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              FastAPI Application Layer                          │    │
│  │  • REST Endpoints  • OpenAPI Docs  • CORS  • Middleware        │    │
│  └────────────────────────┬────────────────────────────────────────┘    │
│                           │                                              │
│  ┌────────────────────────┴────────────────────────────────────────┐    │
│  │         OAuth & Authentication Layer (Optional)                 │    │
│  │  • Token Validation  • Token Exchange  • Group Extraction       │    │
│  │  • Token Refresh     • Token Injection into MCP calls          │    │
│  └────────────────────────┬────────────────────────────────────────┘    │
│                           │                                              │
│  ┌────────────────────────┴────────────────────────────────────────┐    │
│  │              Session Management Layer                           │    │
│  │  • Session Creation/Lifecycle  • Multi-tenancy Isolation        │    │
│  │  • Pluggable Backend (Memory/Redis)  • Auto-cleanup            │    │
│  └────────────────┬────────────────────────────────────────────────┘    │
│                   │                                                      │
│  ┌────────────────┴─────────────────────────────────────────────────┐   │
│  │              Tool Routing & Discovery Layer                      │   │
│  │  • Auto-discovery  • Endpoint Generation  • Schema Mapping      │   │
│  │  • Parameter Validation  • Response Transformation             │   │
│  └────────────────┬─────────────────────────────────────────────────┘   │
│                   │                                                      │
│  ┌────────────────┴─────────────────────────────────────────────────┐   │
│  │              MCP Server Manager                                  │   │
│  │  • Process Spawning  • stdio Communication  • Error Recovery    │   │
│  │  • Resource Limits   • Protocol Handling                        │   │
│  └────────────────┬─────────────────────────────────────────────────┘   │
│                   │ MCP Protocol (stdio)                                 │
└───────────────────┼──────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ↓                       ↓
┌───────────────────┐   ┌───────────────────┐
│   MCP Server      │   │   MCP Server      │
│   (Process 1)     │   │   (Process 2)     │
│                   │   │                   │
│  • User A Session │   │  • User B Session │
│  • Isolated Data  │   │  • Isolated Data  │
└───────────────────┘   └───────────────────┘

Additional Features:
┌─────────────────────────────────────────────────────────────────────────┐
│  UI Generation (AI-Powered)          Workflow Engine                    │
│  • LLM-based HTML/JS generation      • Multi-agent orchestration        │
│  • pfusch components                 • Conditional routing              │
│  • User/group scoped apps            • State management                 │
└─────────────────────────────────────────────────────────────────────────┘

External Integrations:
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│  OAuth Provider │   │  Session Store   │   │  External APIs      │
│  (Keycloak/     │   │  (Redis/         │   │  (GitHub/M365/      │
│   Entra ID)     │   │   Memory)        │   │   Atlassian)        │
└─────────────────┘   └──────────────────┘   └─────────────────────┘
```

## Core Components

### 1. FastAPI Application Layer

The outermost layer provides:

- **HTTP/REST Interface:** Standard REST endpoints for all operations
- **SSE Streaming Support:** Server-Sent Events for real-time streaming at `/sse`
- **OpenAPI Documentation:** Auto-generated API docs at `/docs` and `/redoc`
- **Request/Response Handling:** JSON serialization, validation, error handling
- **CORS Support:** Cross-origin request handling
- **Middleware:** Logging, metrics, authentication
- **LLM Integration:** Chat completions with simple workflows
- **UI Generation:** AI-powered web application generation from prompts

**Key Files:**
- `app/server.py` - Main FastAPI application
- `app/routes.py` - Route definitions

### 2. Session Management Layer

Handles user sessions and MCP server instances:

- **Session Creation:** Spin up isolated MCP server processes per session
- **Session Lifecycle:** Start, ping (keep-alive), close operations
- **Resource Cleanup:** Automatic cleanup of idle sessions
- **Multi-tenancy:** Isolated sessions per user/group
- **Pluggable Backend:** In-memory or Redis-based storage

**Key Files:**
- `app/session_manager/` - Session manager implementations
- `app/session/` - Session handling logic

### 3. MCP Server Manager

Manages the MCP server processes:

- **Process Management:** Spawn, monitor, terminate MCP servers
- **stdio Communication:** Bidirectional communication over stdin/stdout
- **Protocol Handling:** MCP protocol serialization/deserialization
- **Error Recovery:** Restart failed processes
- **Resource Limits:** CPU, memory, timeout constraints

**Key Files:**
- `app/mcp_server/` - MCP server process management

### 4. OAuth & Authentication Layer

Provides enterprise authentication and authorization:

- **Token Validation:** Verify OAuth tokens from identity providers
- **Token Exchange:** Exchange user tokens for service tokens
- **Token Refresh:** Automatic token renewal
- **Group Extraction:** Extract user groups from tokens
- **Token Injection:** Inject tokens into MCP server calls

**Key Files:**
- `app/oauth/` - OAuth implementation
- `app/oauth/token_retriever.py` - Token exchange logic

### 5. Tool Routing Layer

Maps REST endpoints to MCP tools:

- **Auto-discovery:** Detect available tools from MCP servers
- **Endpoint Generation:** Create REST endpoints for each tool
- **Schema Mapping:** Convert MCP schemas to OpenAPI
- **Parameter Validation:** Validate tool inputs
- **Response Transformation:** Convert MCP responses to REST

## Request Flow

### Stateless Tool Call

```
1. Client → HTTP POST /tools/my_tool
           ↓
2. FastAPI validates request
           ↓
3. OAuth layer validates token (if configured)
           ↓
4. MCP Server Manager spawns temporary MCP process
           ↓
5. Tool call sent to MCP server via stdio
           ↓
6. MCP server executes tool
           ↓
7. Response returned via stdio
           ↓
8. MCP process terminated
           ↓
9. Response transformed to JSON
           ↓
10. Client ← HTTP 200 + JSON response
```

### Stateful Session Call

```
1. Client → HTTP POST /session/start
           ↓
2. Session Manager creates new session
           ↓
3. MCP Server Manager spawns persistent MCP process
           ↓
4. Session ID returned to client
           ↓
5. Client → HTTP POST /session/{id}/tools/my_tool
           ↓
6. Tool call sent to existing MCP process
           ↓
7. Response returned
           ↓
8. Session state maintained
           ↓
9. [Repeat steps 5-8 as needed]
           ↓
10. Client → HTTP POST /session/{id}/close
           ↓
11. Session Manager terminates MCP process
           ↓
12. Resources cleaned up
```

## Data Flow

### Tool Discovery

```
MCP Server → list_tools() → Bridge → Generate OpenAPI schema → /docs
```

### Tool Invocation

```
Client → REST JSON → Bridge → MCP Protocol → MCP Server
MCP Server → MCP Protocol → Bridge → REST JSON → Client
```

### Token Exchange

```
Client Token → OAuth Provider → Service Token → MCP Server
     ↓              ↓               ↓              ↓
User Auth     Token Exchange   Downstream    Protected
                                   API        Resource
```

## Deployment Architectures

### Single Instance (Development)

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ HTTP
     ▼
┌─────────────────┐
│  MCP Bridge     │
│  - FastAPI      │
│  - In-memory    │
│    sessions     │
│  - Local MCP    │
│    process      │
└─────────────────┘
```

### High Availability (Production)

```
                    ┌──────────┐
                    │ Client 1 │
                    └────┬─────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Load Balancer│ │              │ │              │
└──────┬───────┘ │              │ │              │
       │         │              │ │              │
   ┌───┴────┬────┴───┬─────────┴─┴──────────┐   │
   │        │        │                       │   │
   ▼        ▼        ▼                       ▼   ▼
┌─────┐  ┌─────┐  ┌─────┐               ┌─────────┐
│Bridge│  │Bridge│  │Bridge│               │  Redis  │
│ Pod 1│  │ Pod 2│  │ Pod 3│               │Sessions │
└──┬───┘  └──┬───┘  └──┬───┘               └─────────┘
   │         │         │
   └─────────┼─────────┘
             │
             ▼
      ┌──────────────┐
      │ MCP Server   │
      │ (Remote)     │
      └──────────────┘
```

### Microservices Architecture

```
┌──────────────────────────────────────────────┐
│              API Gateway                     │
└────┬────────────┬────────────┬───────────────┘
     │            │            │
     ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ GitHub  │  │Atlassian│  │  M365   │
│  MCP    │  │   MCP   │  │  MCP    │
│ Bridge  │  │ Bridge  │  │ Bridge  │
└─────────┘  └─────────┘  └─────────┘
     │            │            │
     └────────────┼────────────┘
                  │
                  ▼
          ┌───────────────┐
          │   OAuth/      │
          │   Keycloak    │
          └───────────────┘
```

## Security Architecture

### Authentication Flow

```
1. User → Login → Keycloak
2. Keycloak ← OAuth Token
3. User → Request + Token → Bridge
4. Bridge → Validate Token → Keycloak
5. Bridge → Extract Claims (user_id, groups)
6. Bridge → Execute with Context
```

### Token Exchange Flow

```
1. User Token → Bridge
2. Bridge → Exchange Request → Keycloak
3. Keycloak → Service Token
4. Bridge → MCP Call + Service Token → MCP Server
5. MCP Server → API Call + Service Token → External API
```

### Group-Based Isolation

```
User A (group: engineering)
  ↓
Bridge → /data/engineering/memory.json

User B (group: marketing)
  ↓
Bridge → /data/marketing/memory.json

✓ Data isolation enforced at runtime
✓ Groups extracted from OAuth token
✓ Paths resolved dynamically
```

## Design Patterns

### 1. Adapter Pattern

The bridge adapts MCP protocol to REST:

```python
# MCP Protocol
{"jsonrpc": "2.0", "method": "tools/call", "params": {...}}

# Transformed to REST
POST /tools/tool_name
Content-Type: application/json
{...}
```

### 2. Factory Pattern

Session managers created via factory:

```python
SessionManagerFactory.create(type="redis")
SessionManagerFactory.create(type="memory")
```

### 3. Proxy Pattern

Bridge proxies requests to MCP servers:

```python
client_request → bridge → mcp_server
client ← bridge ← mcp_server
```

### 4. Strategy Pattern

Different authentication strategies:

```python
- BearerTokenAuth
- OAuth2TokenExchange
- AnonymousAuth
- TokenForwarding
```

## Scaling Considerations

### Vertical Scaling

- Increase CPU/memory per instance
- More workers per instance
- Larger session capacity

### Horizontal Scaling

- Multiple bridge instances
- Shared Redis for sessions
- Load balancer distribution
- Stateless operation mode

### Session Distribution

```
Session Affinity (Sticky Sessions):
Client → Always same bridge instance

Session Sharing (Redis):
Client → Any bridge instance → Shared session store
```

## Performance Characteristics

### Latency

- **Stateless:** ~50-200ms (includes MCP process spawn)
- **Stateful:** ~10-50ms (reuses existing process)
- **Remote:** +network latency to remote server

### Throughput

- **Single Instance:** ~100-500 req/s (depends on MCP server)
- **Multi-Instance:** Scales linearly with instances

### Resource Usage

- **Memory:** ~50-200MB base + sessions
- **CPU:** Depends on MCP server workload
- **Storage:** Session state (in-memory or Redis)

## Observability

### Metrics Exposed

```
# HTTP metrics
http_requests_total
http_request_duration_seconds
http_requests_in_progress

# Session metrics  
mcp_sessions_active
mcp_sessions_created_total
mcp_sessions_closed_total

# MCP server metrics
mcp_tool_calls_total
mcp_tool_call_duration_seconds
mcp_tool_errors_total
```

### Logging

Structured JSON logs with:
- Request ID for tracing
- User ID and session ID
- Tool names and parameters
- Errors and stack traces
- Performance metrics

### Tracing

OpenTelemetry distributed tracing:
- Request path through components
- MCP server communication
- OAuth token exchange
- External API calls

## Extension Points

### Custom Session Managers

```python
class CustomSessionManager(SessionManagerBase):
    def create_session(self, session_id, user_id):
        # Custom implementation
        pass
```

### Custom Token Retrievers

```python
class CustomTokenRetriever(TokenRetrieverBase):
    def get_token(self, user_token):
        # Custom token exchange
        pass
```

### Custom MCP Servers

Any stdio-based MCP server works:
- Python MCP servers
- Node.js MCP servers
- Go MCP servers
- Custom implementations

## Technology Stack

- **Framework:** FastAPI (Python 3.11+)
- **MCP SDK:** Python MCP library
- **Session Store:** In-memory or Redis
- **Authentication:** OAuth2 / Keycloak
- **Observability:** Prometheus, OpenTelemetry
- **Container:** Docker
- **Orchestration:** Kubernetes (optional)

## Summary

The Enterprise MCP Bridge architecture:

✅ Separates concerns into clear layers  
✅ Provides multiple deployment options  
✅ Scales horizontally and vertically  
✅ Integrates enterprise security  
✅ Offers comprehensive observability  

## Next Steps

- [Security Model](security.md)
- [Session Management](sessions.md)
- [Design Decisions](design-decisions.md)
