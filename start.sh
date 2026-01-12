#!/bin/bash

if [ "$#" -gt 0 ]; then
  export MCP_SERVER_COMMAND="$*"
fi

# for development purposes another pip install can be triggered in the /mcp directory
if [ "$ENV" == "dev" ]; then
  echo "Development mode: checking for /mcp directory"
  if [ -d "/mcp" ]; then
    if [ -f "/mcp/install.sh" ]; then
      echo "Running /mcp/install.sh"
      sh /mcp/install.sh
    fi
    if [ -f "/mcp/requirements.txt" ]; then
      echo "Installing additional requirements from /mcp/requirements.txt"
      pip install -r /mcp/requirements.txt
    fi
    if [ -f "/mcp/requirements-dev.txt" ]; then
      echo "Installing additional development requirements from /mcp/requirements-dev.txt"
      pip install -r /mcp/requirements-dev.txt
    fi
    if [ -f "/mcp/pyproject.toml" ]; then
      echo "Installing additional requirements from /mcp/pyproject.toml"
      pip install -e /mcp
    fi
  fi
fi

# If MCP_RUN_ON_START is set, run its command before starting the server
if [ -n "$MCP_RUN_ON_START" ]; then
  echo "Running MCP_RUN_ON_START: $MCP_RUN_ON_START"
  # run in a clean environment and set HOME to a tmp dir so npm uses a separate cache
  env -i HOME=/tmp PATH="$PATH" bash -lc "$MCP_RUN_ON_START"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "MCP_RUN_ON_START command failed with exit code $rc"
  fi
fi

uvicorn app.server:app --host 0.0.0.0 --port 8000