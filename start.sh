#!/bin/bash

if [ "$#" -gt 0 ]; then
  export MCP_SERVER_COMMAND="$*"
fi

# for development purposes another pip install can be triggered in the /mcp directory
if [ "$ENV" == "dev" ]; then
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
uvicorn app.server:app --host 0.0.0.0 --port 8000