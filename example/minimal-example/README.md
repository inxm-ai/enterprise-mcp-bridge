# Minimal Example

This minimal example uses the default mcp server, and calls the add tool with a simple payload.

## Usage

```
❯ ./start.sh
Default mcp example
Starting docker container
Calling the add tool with {"a": 2, "b": 1}
{"isError":false,"content":[{"text":"3","structuredContent":null}],"structuredContent":{"result":3}}
```

If the result is `3`, then the add tool is working correctly.