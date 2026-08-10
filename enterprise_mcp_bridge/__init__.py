"""Consumer-facing package for the enterprise MCP bridge.

The bridge application itself lives in the ``app`` package (``app.server:app``,
exactly what production serves via uvicorn). This package holds the small,
intentional surface exposed to other repositories — currently only
:mod:`enterprise_mcp_bridge.testing`.
"""
