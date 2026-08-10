"""Test support for repositories that serve their MCP through the bridge.

Install with the ``testing`` extra::

    pip install "enterprise-mcp-bridge[testing] @ git+https://github.com/inxm-ai/enterprise-mcp-bridge@<tag>"

and use the one public entrypoint::

    from enterprise_mcp_bridge.testing import bridge_client

    command = " ".join(shlex.quote(p) for p in (sys.executable, "tests/conformance_server.py"))
    with bridge_client(command) as client:
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
of the request. ``POST /session/start`` spawns a child that lives until you
call ``POST /session/close`` with the session id — exiting the
``bridge_client`` context does NOT close open sessions, so close every session
your test starts (a ``finally`` block or fixture teardown). Nothing needs
Kubernetes or the network — everything stays on this machine, offline.

Supported configuration
-----------------------
``mcp_server_command``
    Required. The exact ``MCP_SERVER_COMMAND`` production would use. The
    bridge parses it with ``shlex.split()``, so quote each part yourself
    (``shlex.quote``) — ``sys.executable`` and repository paths can contain
    spaces. Prefer absolute paths and ``sys.executable`` so the test does not
    depend on the working directory or a ``python`` on PATH.
``env``
    Optional extra environment variables (e.g. ``EFFECT_TOOLS``, ``MCP_ENV``).
    Applied before the app is imported and restored on exit.

Caveat: the bridge reads a few variables at *import* time (``EFFECT_TOOLS``,
OTLP settings, ``SERVICE_NAME``), and the app module is imported once per
process. The first ``bridge_client(...)`` in a pytest run therefore fixes those
values for the whole run; ``MCP_SERVER_COMMAND`` itself is read per request and
can differ between contexts.

Caveat: ``app`` is a common top-level package name, and pytest puts a
consumer's own repository root on ``sys.path``. ``bridge_client`` resolves its
own ``app`` package by location (a sibling of this module, not by search
order) and inserts it first, so the bridge's app wins the first import in the
process. If a *different* ``app`` package was already imported earlier in the
same process, that is an unresolvable naming collision and ``bridge_client``
raises ``RuntimeError`` rather than silently picking one.
"""

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

MCP_SERVER_COMMAND_VAR = "MCP_SERVER_COMMAND"
# Sibling of this package in both an installed layout (pyproject.toml packages
# `app*` and `enterprise_mcp_bridge*` as top-level siblings) and a checkout of
# this repo. Resolving the bridge's own `app` package from here — rather than
# relying on sys.path search order — is what keeps `from app.server import
# app` from picking up a consumer repository's own top-level `app` package.
_BRIDGE_ROOT = Path(__file__).resolve().parent.parent


def _is_bridges_own_app_module(module) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    return Path(module_file).resolve().is_relative_to(_BRIDGE_ROOT)


@contextmanager
def bridge_client(
    mcp_server_command: str,
    env: Mapping[str, str] | None = None,
) -> Iterator[TestClient]:
    """Yield a ``TestClient`` for the real bridge app wired to the given MCP server."""
    if not mcp_server_command or not mcp_server_command.strip():
        raise ValueError(
            "mcp_server_command must not be blank: the bridge falls back to its "
            "own default demo server when MCP_SERVER_COMMAND is unset, which "
            "would silently test the wrong MCP server."
        )

    existing_app = sys.modules.get("app")
    if existing_app is not None and not _is_bridges_own_app_module(existing_app):
        raise RuntimeError(
            "A module named `app` is already imported and is not the bridge's "
            f"own `app` package (expected under {_BRIDGE_ROOT}). This usually "
            "means the consumer repository also has a top-level `app` package "
            "that pytest imported first; import `enterprise_mcp_bridge.testing` "
            "before your own `app` package, or rename one of the two."
        )

    overrides = {MCP_SERVER_COMMAND_VAR: mcp_server_command, **(env or {})}
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    bridge_root_str = str(_BRIDGE_ROOT)
    path_inserted = bridge_root_str not in sys.path
    if path_inserted:
        sys.path.insert(0, bridge_root_str)
    try:
        # Imported lazily so `overrides` and the sys.path insertion above are
        # in place for the bridge's import-time configuration reads and for
        # resolving `app` to the bridge's own package on first use.
        from app.server import app

        with TestClient(app) as client:
            yield client
    finally:
        if path_inserted:
            sys.path.remove(bridge_root_str)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
