# Your First MCP Server

Learn how to create and deploy a custom MCP server with the Enterprise MCP Bridge.

## What You'll Build

In this tutorial, you'll create a simple MCP server that provides calculation tools and deploy it using the bridge.

## Prerequisites

- Completed the [Getting Started](getting-started.md) tutorial
- Basic Python knowledge
- Text editor or IDE

## Step 1: Understanding MCP Server Structure

An MCP server needs to:
1. Define tools with their input schemas
2. Handle tool invocation requests
3. Communicate over stdio using the MCP protocol

## Step 2: Create Your MCP Server

Create a new directory for your MCP server:

```bash
mkdir -p mcp/calculator
cd mcp/calculator
```

Create `server.py`:

```python
#!/usr/bin/env python3
"""
A simple calculator MCP server
"""
import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Initialize the MCP server
app = Server("calculator-server")

# Define available tools
TOOLS = [
    Tool(
        name="add",
        description="Add two numbers together",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    ),
    Tool(
        name="multiply",
        description="Multiply two numbers",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    ),
    Tool(
        name="power",
        description="Raise a number to a power",
        inputSchema={
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "Base number"},
                "exponent": {"type": "number", "description": "Exponent"}
            },
            "required": ["base", "exponent"]
        }
    )
]

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available tools"""
    return TOOLS

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocation"""
    
    if name == "add":
        result = arguments["a"] + arguments["b"]
        return [TextContent(
            type="text",
            text=f"The sum of {arguments['a']} and {arguments['b']} is {result}"
        )]
    
    elif name == "multiply":
        result = arguments["a"] * arguments["b"]
        return [TextContent(
            type="text",
            text=f"The product of {arguments['a']} and {arguments['b']} is {result}"
        )]
    
    elif name == "power":
        result = arguments["base"] ** arguments["exponent"]
        return [TextContent(
            type="text",
            text=f"{arguments['base']} raised to the power of {arguments['exponent']} is {result}"
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 3: Create Requirements File

Create `requirements.txt`:

```
mcp
```

## Step 4: Test Your MCP Server Locally

First, install the MCP CLI:

```bash
pip install mcp
```

Test your server:

```bash
python server.py
```

The server should start and wait for input. Press Ctrl+C to stop it.

## Step 5: Connect to the Bridge

Now let's run your MCP server through the bridge:

```bash
cd ../../  # Back to project root
export MCP_SERVER_COMMAND="python mcp/calculator/server.py"
uvicorn app.server:app --reload
```

## Step 6: Verify Tools Are Available

Check that your tools are exposed:

```bash
curl http://localhost:8000/tools
```

You should see your three calculator tools: `add`, `multiply`, and `power`.

## Step 7: Test Your Tools

### Addition

```bash
curl -X POST http://localhost:8000/tools/add \
  -H "Content-Type: application/json" \
  -d '{"a": 5, "b": 3}'
```

Expected response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "The sum of 5 and 3 is 8"
    }
  ]
}
```

### Multiplication

```bash
curl -X POST http://localhost:8000/tools/multiply \
  -H "Content-Type: application/json" \
  -d '{"a": 4, "b": 7}'
```

### Power

```bash
curl -X POST http://localhost:8000/tools/power \
  -H "Content-Type: application/json" \
  -d '{"base": 2, "exponent": 10}'
```

## Step 8: Deploy with Docker

Create a `Dockerfile` in your calculator directory:

```dockerfile
FROM ghcr.io/inxm-ai/enterprise-mcp-bridge:latest

# Copy your MCP server
COPY . /mcp/calculator/

# Install dependencies
RUN pip install -r /mcp/calculator/requirements.txt

# Set the command to run your server
ENV MCP_SERVER_COMMAND="python /mcp/calculator/server.py"
```

Build and run:

```bash
docker build -t my-calculator-bridge .
docker run -it -p 8000:8000 my-calculator-bridge
```

## Step 9: Add More Features

### Adding Input Validation

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocation with validation"""
    
    if name == "add":
        a, b = arguments["a"], arguments["b"]
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise ValueError("Both arguments must be numbers")
        result = a + b
        return [TextContent(
            type="text",
            text=f"The sum of {a} and {b} is {result}"
        )]
    # ... rest of the tools
```

### Adding Error Handling

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocation with error handling"""
    
    try:
        if name == "power":
            base = arguments["base"]
            exponent = arguments["exponent"]
            
            # Prevent dangerous calculations
            if abs(exponent) > 1000:
                raise ValueError("Exponent too large")
            
            result = base ** exponent
            return [TextContent(
                type="text",
                text=f"{base} raised to the power of {exponent} is {result}"
            )]
    except OverflowError:
        return [TextContent(
            type="text",
            text="Error: Result too large to compute"
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]
```

## Understanding What You Built

Your MCP server:

1. **Defines Tools:** Using the MCP Tool schema
2. **Handles Requests:** Via the `@app.call_tool()` decorator
3. **Communicates via stdio:** Using the MCP protocol
4. **Gets Wrapped:** By the Enterprise MCP Bridge into REST endpoints

The Bridge automatically:
- Maps each tool to a REST endpoint (`/tools/{tool_name}`)
- Generates OpenAPI documentation
- Handles HTTP request/response transformation
- Manages errors and status codes

## Next Steps

Enhance your MCP server:

- Add more complex tools
- Integrate with external APIs
- Add state management
- Implement authentication

Continue learning:
- [Multi-User Sessions](multi-user-sessions.md)
- [Configure OAuth](../how-to/configure-oauth.md)
- [Deploy to Production](../how-to/deploy-production.md)

## Resources

- [MCP Protocol Documentation](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Example MCP Servers](https://github.com/modelcontextprotocol/servers)

## Summary

In this tutorial, you:

✅ Created a custom MCP server from scratch  
✅ Defined multiple tools with input schemas  
✅ Connected your server to the bridge  
✅ Tested tools via REST API  
✅ Learned to deploy with Docker  

You can now build any MCP server and expose it as a REST API!
