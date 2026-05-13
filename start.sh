#!/bin/bash

if [ -n "$MCP_RUN_ON_START" ] && [ -n "$MCP_RUN_ON_START_INHERIT_ENVIRONMENT" ]; then
  echo "MCP_RUN_ON_START and MCP_RUN_ON_START_INHERIT_ENVIRONMENT cannot both be set"
  exit 1
fi

if [ "$#" -gt 0 ]; then
  export MCP_SERVER_COMMAND="$*"
fi

if [ -n "$MCP_GIT_CLONE" ]; then
  echo "Cloning MCP repository from $MCP_GIT_CLONE..."
  apt-get update && apt-get install -y git
  rm -rf /mcp
  git clone "$MCP_GIT_CLONE" /mcp
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

# If MCP_RUN_ON_START_INHERIT_ENVIRONMENT is set, run its command before starting the server
if [ -n "$MCP_RUN_ON_START_INHERIT_ENVIRONMENT" ]; then
  echo "Running MCP_RUN_ON_START_INHERIT_ENVIRONMENT: $MCP_RUN_ON_START_INHERIT_ENVIRONMENT"
  # run in the current environment and set HOME to a tmp dir so npm uses a separate cache
  env HOME=/tmp PATH="$PATH" bash -lc "$MCP_RUN_ON_START_INHERIT_ENVIRONMENT"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "MCP_RUN_ON_START_INHERIT_ENVIRONMENT command failed with exit code $rc"
  fi
fi

APP_PORT="${PORT:-8000}"
uvicorn app.server:app --host 0.0.0.0 --port "$APP_PORT"
