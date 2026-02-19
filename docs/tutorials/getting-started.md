# Getting Started with Enterprise MCP Bridge

This tutorial will guide you through setting up and running your first Enterprise MCP Bridge instance. By the end, you'll have a working bridge exposing an MCP server via REST API.

## What You'll Learn

- How to install and run the Enterprise MCP Bridge
- How to verify your installation
- How to make your first API call
- How to explore the auto-generated API documentation

## Prerequisites

Before you begin, ensure you have:

- Python 3.11 or higher installed
- Basic familiarity with REST APIs
- A terminal/command line interface
- 10-15 minutes of time

## Step 1: Clone the Repository

First, clone the Enterprise MCP Bridge repository:

```bash
git clone https://github.com/inxm-ai/enterprise-mcp-bridge.git
cd enterprise-mcp-bridge
```

## Step 2: Install Dependencies

Install the bridge application and its dependencies:

```bash
pip install app
```

This will install all necessary packages including FastAPI, MCP SDK, and other dependencies.

## Step 3: Start the Server

Launch the bridge with the default configuration:

```bash
uvicorn app.server:app --reload
```

You should see output similar to:

```
INFO:     Will watch for changes in these directories: ['/path/to/enterprise-mcp-bridge']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

?> The `--reload` flag enables auto-reloading during development. Remove it in production.

## Step 4: Verify Installation

Open your browser and navigate to:

**http://localhost:8000/docs**

You should see the auto-generated Swagger UI documentation. This interactive interface lets you explore and test all available endpoints.

![Swagger UI](../example-application.png)

## Step 5: Test the API

Let's test the bridge by listing available tools. You can do this in several ways:

### Using the Swagger UI

1. Navigate to http://localhost:8000/docs
2. Find the `GET /tools` endpoint
3. Click "Try it out"
4. Click "Execute"
5. View the response

### Using curl

```bash
curl http://localhost:8000/tools
```

### Using Python

```python
import requests

response = requests.get("http://localhost:8000/tools")
print(response.json())
```

You should receive a JSON response listing the available tools from the default MCP server.

## Step 6: Start a Session

For stateful interactions, you can start a session:

```bash
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-first-session"}'
```

Response:
```json
{
  "session_id": "my-first-session",
  "status": "started",
  "created_at": "2024-01-15T10:00:00Z"
}
```

## Step 7: Call a Tool

Now let's call a tool (using the default memory MCP server):

```bash
curl -X POST http://localhost:8000/tools/create_entities \
  -H "Content-Type: application/json" \
  -d '{
    "entities": [
      {
        "name": "test_entity",
        "entityType": "concept",
        "observations": ["This is a test observation"]
      }
    ]
  }'
```

## Understanding What Happened

1. **The Bridge** started and loaded the default MCP server (memory server)
2. **MCP Tools** were discovered automatically and mapped to REST endpoints
3. **API Documentation** was generated automatically via FastAPI
4. **Tool Calls** were translated from HTTP/JSON to MCP protocol and back

## Next Steps

Congratulations! You've successfully set up and tested your first Enterprise MCP Bridge instance.

Continue your learning journey:

- [Quick Start Guide](quick-start.md) - Deploy with different MCP servers
- [Your First MCP Server](first-mcp-server.md) - Create a custom MCP server
- [Multi-User Sessions](multi-user-sessions.md) - Handle multiple users

Or explore how-to guides:
- [Deploy to Production](../how-to/deploy-production.md)
- [Run in Docker](../how-to/docker.md)

## Troubleshooting

### Port Already in Use

If port 8000 is already in use, specify a different port:

```bash
uvicorn app.server:app --reload --port 8080
```

### Python Version Error

Ensure you're using Python 3.11 or higher:

```bash
python --version
```

If needed, use a specific Python version:

```bash
python3.11 -m pip install app
python3.11 -m uvicorn app.server:app --reload
```

### Import Errors

If you encounter import errors, ensure you're in the correct directory:

```bash
cd enterprise-mcp-bridge
pip install -e app
```

## Summary

In this tutorial, you:

✅ Installed the Enterprise MCP Bridge  
✅ Started the server with default configuration  
✅ Explored the auto-generated API documentation  
✅ Made your first API calls  
✅ Started a session and called tools  

You're now ready to explore more advanced features!
