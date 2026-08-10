"""Test support for repositories that serve their MCP through the bridge.

Install with the ``testing`` extra::

    pip install "enterprise-mcp-bridge[testing] @ git+https://github.com/inxm-ai/enterprise-mcp-bridge@<tag>"

and use the one public entrypoint::

    from enterprise_mcp_bridge.testing import bridge_client

    with bridge_client(f"{sys.executable} tests/conformance_server.py") as client:
        response = client.get("/tools")

What it does
------------
Creates the *real* bridge FastAPI application in-process (the same
``app.server:app`` object production serves) and yields a
``fastapi.testclient.TestClient`` against it. There is no test-only app, no
test-only route, and no faked transport: the bridge resolves
``MCP_SERVER_COMMAND`` per request/session and spawns the MCP server as a
stdio child process exactly as it does in production.

Child-process lifecycle
-----------------------
The bridge owns the child. Sessionless requests spawn a child for the duration
of the request; ``POST /session/start`` spawns one that lives until the session
is closed or the client context exits. Nothing needs Kubernetes or the network
— everything stays on this machine, offline.

Supported configuration
-----------------------
``mcp_server_command``
    Required. The exact ``MCP_SERVER_COMMAND`` production would use, e.g.
    ``f"{sys.executable} path/to/server.py"``. Prefer absolute paths and
    ``sys.executable`` so the test does not depend on the working directory
    or a ``python`` on PATH.
``env``
    Optional extra environment variables (e.g. ``EFFECT_TOOLS``, ``MCP_ENV``).
    Applied before the app is imported and restored on exit.

Caveat: the bridge reads a few variables at *import* time (``EFFECT_TOOLS``,
OTLP settings, ``SERVICE_NAME``), and the app module is imported once per
process. The first ``bridge_client(...)`` in a pytest run therefore fixes those
values for the whole run; ``MCP_SERVER_COMMAND`` itself is read per request and
can differ between contexts.
"""

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from fastapi.testclient import TestClient

MCP_SERVER_COMMAND_VAR = "MCP_SERVER_COMMAND"


@contextmanager
def bridge_client(
    mcp_server_command: str,
    env: Mapping[str, str] | None = None,
) -> Iterator[TestClient]:
    """Yield a ``TestClient`` for the real bridge app wired to the given MCP server."""
    overrides = {MCP_SERVER_COMMAND_VAR: mcp_server_command, **(env or {})}
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        # Imported lazily so `overrides` is in place for the bridge's
        # import-time configuration reads on first use.
        from app.server import app

        with TestClient(app) as client:
            yield client
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
