from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("mcp-EXAMPLE-server")


# Two simple demo examples

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def hello() -> str:
    """Write hello"""
    return "Hello"

call_count = 0

@mcp.tool()
def call_counter() -> int:
    """Get the number of times this tool has been called"""
    global call_count
    call_count += 1
    return call_count

@mcp.tool()
def error(message: str) -> str:
    """Always return an error message"""
    raise ValueError(message)

@mcp.prompt()
def greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
