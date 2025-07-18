#!/bin/sh

if [ "$#" -gt 0 ]; then
  export MCP_SERVER_COMMAND="$*"
fi
uvicorn app.server:app --host 0.0.0.0 --port 8000