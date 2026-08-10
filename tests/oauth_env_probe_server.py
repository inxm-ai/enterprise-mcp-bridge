"""Test-only probe server for the bridge's own conformance suite.

Exists solely to prove that OAUTH_ENV token propagation actually reaches the
child process (see test_bridge_conformance.py). Kept separate from
mcp/server.py — the real default fallback MCP server — so that production
default does not gain an env-var-echoing tool.
"""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-oauth-env-probe-server")


@mcp.tool()
def read_env(name: str) -> str:
    """Return the value of an environment variable in this process, or "" if unset."""
    return os.environ.get(name, "")


if __name__ == "__main__":
    mcp.run()
