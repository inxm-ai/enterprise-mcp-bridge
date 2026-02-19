# API Reference

Complete REST API reference for the Enterprise MCP Bridge.

## Base URL

All API endpoints are relative to the base URL:

```
http://localhost:8000
```

Or with custom base path:
```
http://localhost:8000/api/v1/mcp
```

## Authentication

Most endpoints support optional OAuth2 authentication:

```http
Authorization: Bearer YOUR_ACCESS_TOKEN
```

## Endpoints

### Health & Metadata

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.4.2",
  "uptime": "1h23m45s"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is unhealthy

---

#### GET /

Root endpoint with service information.

**Response:**
```json
{
  "name": "Enterprise MCP Bridge",
  "version": "0.4.2",
  "documentation": "/docs"
}
```

---

### Tools

#### GET /tools

List all available tools from the MCP server.

**Response:**
```json
{
  "tools": [
    {
      "name": "create_entities",
      "description": "Create new entities in memory",
      "inputSchema": {
        "type": "object",
        "properties": {
          "entities": {
            "type": "array",
            "items": {...}
          }
        }
      }
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Success
- `500 Internal Server Error` - MCP server error

---

#### GET /tools/{tool_name}/schema

Get the input schema for a specific tool.

**Parameters:**
- `tool_name` (path) - Name of the tool

**Response:**
```json
{
  "name": "create_entities",
  "description": "Create new entities in memory",
  "inputSchema": {
    "type": "object",
    "properties": {...},
    "required": [...]
  }
}
```

**Status Codes:**
- `200 OK` - Success
- `404 Not Found` - Tool not found

---

#### POST /tools/{tool_name}

Call a tool (stateless mode).

**Parameters:**
- `tool_name` (path) - Name of the tool

**Request Body:**
```json
{
  "param1": "value1",
  "param2": "value2"
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Tool execution result"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Tool not found
- `500 Internal Server Error` - Execution error

---

### Sessions

#### POST /session/start

Start a new session.

**Request Body:**
```json
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
  "user_id": "user-123"
}
```

**Status Codes:**
- `200 OK` - Session started
- `400 Bad Request` - Invalid session ID
- `409 Conflict` - Session already exists

---

#### POST /session/{session_id}/ping

Keep a session alive.

**Parameters:**
- `session_id` (path) - Session identifier

**Response:**
```json
{
  "session_id": "my-session-123",
  "status": "active",
  "last_ping": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200 OK` - Success
- `404 Not Found` - Session not found

---

#### POST /session/{session_id}/close

Close a session.

**Parameters:**
- `session_id` (path) - Session identifier

**Response:**
```json
{
  "session_id": "my-session-123",
  "status": "closed",
  "duration": "30m15s"
}
```

**Status Codes:**
- `200 OK` - Session closed
- `404 Not Found` - Session not found

---

#### POST /session/{session_id}/tools/{tool_name}

Call a tool within a session (stateful mode).

**Parameters:**
- `session_id` (path) - Session identifier
- `tool_name` (path) - Tool name

**Request Body:**
```json
{
  "param1": "value1",
  "param2": "value2"
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Tool execution result"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Session or tool not found
- `500 Internal Server Error` - Execution error

---

### OAuth

#### GET /oauth/login

Initiate OAuth login flow.

**Query Parameters:**
- `redirect_url` (optional) - URL to redirect after login

**Response:**
- Redirects to OAuth provider login page

---

#### GET /oauth/callback

OAuth callback endpoint.

**Query Parameters:**
- `code` - Authorization code
- `state` - State parameter

**Response:**
- Sets authentication cookie
- Redirects to `redirect_url` or `/`

---

#### GET /oauth/userinfo

Get current user information.

**Headers:**
```
Authorization: Bearer ACCESS_TOKEN
```

**Response:**
```json
{
  "sub": "user-id-123",
  "email": "user@example.com",
  "groups": ["engineering", "users"],
  "preferred_username": "john.doe"
}
```

**Status Codes:**
- `200 OK` - Success
- `401 Unauthorized` - Invalid or missing token

---

#### POST /oauth/logout

Log out the current user.

**Response:**
```json
{
  "status": "logged_out"
}
```

---

### Metrics

#### GET /metrics

Prometheus metrics endpoint.

**Response:**
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/tools"} 1234

# HELP mcp_tool_calls_total Total MCP tool calls
# TYPE mcp_tool_calls_total counter
mcp_tool_calls_total{tool="create_entities"} 567
...
```

**Status Codes:**
- `200 OK` - Success

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "status_code": 400
}
```

### Common Error Codes

| Status Code | Meaning |
|------------|---------|
| `400` | Bad Request - Invalid parameters |
| `401` | Unauthorized - Missing or invalid authentication |
| `403` | Forbidden - Insufficient permissions |
| `404` | Not Found - Resource not found |
| `409` | Conflict - Resource already exists |
| `500` | Internal Server Error - Server error |
| `503` | Service Unavailable - Service is down |

## OpenAPI Specification

The complete OpenAPI specification is available at:

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **OpenAPI JSON:** `/openapi.json`

## Examples

### Call a Tool (Stateless)

```bash
curl -X POST http://localhost:8000/tools/create_entities \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{
    "entities": [
      {
        "name": "project_alpha",
        "entityType": "project",
        "observations": ["High priority"]
      }
    ]
  }'
```

### Session-based Flow

```bash
# Start session
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session"}'

# Call tool in session
curl -X POST http://localhost:8000/session/my-session/tools/create_entities \
  -H "Content-Type: application/json" \
  -d '{"entities": [...]}'

# Keep session alive
curl -X POST http://localhost:8000/session/my-session/ping

# Close session
curl -X POST http://localhost:8000/session/my-session/close
```

## Client Libraries

### Python

```python
import requests

class MCPBridgeClient:
    def __init__(self, base_url, token=None):
        self.base_url = base_url
        self.headers = {}
        if token:
            self.headers['Authorization'] = f'Bearer {token}'
    
    def list_tools(self):
        response = requests.get(
            f'{self.base_url}/tools',
            headers=self.headers
        )
        return response.json()
    
    def call_tool(self, tool_name, **kwargs):
        response = requests.post(
            f'{self.base_url}/tools/{tool_name}',
            json=kwargs,
            headers=self.headers
        )
        return response.json()

# Usage
client = MCPBridgeClient('http://localhost:8000', token='YOUR_TOKEN')
tools = client.list_tools()
result = client.call_tool('create_entities', entities=[...])
```

### JavaScript

```javascript
class MCPBridgeClient {
  constructor(baseUrl, token = null) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json'
    };
    if (token) {
      this.headers['Authorization'] = `Bearer ${token}`;
    }
  }

  async listTools() {
    const response = await fetch(`${this.baseUrl}/tools`, {
      headers: this.headers
    });
    return response.json();
  }

  async callTool(toolName, params) {
    const response = await fetch(`${this.baseUrl}/tools/${toolName}`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(params)
    });
    return response.json();
  }
}

// Usage
const client = new MCPBridgeClient('http://localhost:8000', 'YOUR_TOKEN');
const tools = await client.listTools();
const result = await client.callTool('create_entities', {entities: [...]});
```

## Next Steps

- [Configuration Reference](configuration.md)
- [Error Codes](error-codes.md)
- [Examples](examples.md)
