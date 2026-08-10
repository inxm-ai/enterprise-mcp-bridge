# Test Your MCP Through the Bridge

Run your MCP repository's tests against the *real* bridge — real FastAPI app,
real routes, real stdio child process — without Kubernetes or network access.
The first time your server meets the bridge should be CI, not the production
deploy.

## Install

```bash
pip install "enterprise-mcp-bridge[testing] @ git+https://github.com/inxm-ai/enterprise-mcp-bridge@<tag>"
```

Pin a tag. The `testing` extra adds pytest; the test-support API itself only
needs the package's runtime dependencies.

## Use

```python
import shlex
import sys

import pytest

from enterprise_mcp_bridge.testing import bridge_client

CONFORMANCE_SERVER_COMMAND = " ".join(
    shlex.quote(part) for part in (sys.executable, "tests/conformance_server.py")
)


@pytest.fixture(scope="module")
def bridge():
    with bridge_client(CONFORMANCE_SERVER_COMMAND) as client:
        yield client


def test_my_tool(bridge):
    response = bridge.post("/tools/search", json={"query": "inxm"})
    assert response.status_code == 200
```

`bridge_client(mcp_server_command, env=None)` is the whole API:

| Parameter | Meaning |
|---|---|
| `mcp_server_command` | Exactly what production sets as `MCP_SERVER_COMMAND`. Use `sys.executable` and absolute paths. The bridge parses it with `shlex.split()`, so quote each part yourself (`shlex.quote`) — paths with spaces otherwise split incorrectly. Must not be blank; `bridge_client` raises `ValueError` if it is, since a blank value would silently fall back to the bridge's own default demo server. |
| `env` | Extra environment variables (e.g. `EFFECT_TOOLS`), applied before the app loads and restored afterwards. |

`app` is a common top-level package name, and pytest puts your repository root
on `sys.path`. `bridge_client` resolves its own `app` package by location, not
by search order, so it wins the first import even if your repository also has
a top-level `app` package — but if that package was already imported earlier
in the same test process, the collision is unresolvable and `bridge_client`
raises `RuntimeError` rather than guessing. Import `enterprise_mcp_bridge.testing`
before your own `app` package if you hit this.

It yields a [`fastapi.testclient.TestClient`](https://fastapi.tiangolo.com/reference/testclient/)
bound to the same `app.server:app` object production serves. The bridge spawns
your server as a stdio child per request, or per session via
`POST /session/start` — identical to production.

Session children are yours to end: a session's child process stays alive until
`POST /session/close` is called with its id — leaving the `bridge_client`
context does **not** close it. Close every session your test starts:

```python
session_id = bridge.post("/session/start").json()["x-inxm-mcp-session"]
try:
    ...
finally:
    bridge.post("/session/close", headers={"x-inxm-mcp-session": session_id})
```

## Offline conformance servers

Real tool calls often reach real APIs. Keep the suite offline by exposing a
second entrypoint that registers the same tools with a scripted worker:
production uses `mcp/server.py` → `create_server(real_worker)`; tests use
`tests/conformance_server.py` → `create_server(scripted_worker)`. Point
`bridge_client` at the conformance entrypoint — same registration code, same
stdio path, only the infrastructure edge is substituted.

## What you don't need to test

The generic bridge behaviours — startup and tool listing, session isolation,
the dry-run gate, child tool exception → HTTP status mapping — are proven once
in this repo (`tests/test_bridge_conformance.py`). Your suite only adds cases
for your own tools: at least one success case, plus repo-specific failure
cases where useful.

## Caveats

- The app module is imported once per process, and a few settings
  (`EFFECT_TOOLS`, OTLP endpoints, `SERVICE_NAME`) are read at import time.
  The first `bridge_client(...)` in a pytest run fixes those for the whole
  run; `MCP_SERVER_COMMAND` itself is re-read per request and may differ
  between contexts.
- The child process needs its dependencies importable by the interpreter you
  put in `mcp_server_command` — in CI, install your MCP package into the same
  environment.
- **stdio only.** `bridge_client` exercises the bridge's `MCP_SERVER_COMMAND`
  child-process path exclusively. The bridge's separate remote-MCP mode
  (`MCP_REMOTE_SERVER`, see [Use Remote MCP Servers](remote-mcp-servers.md)) is
  a different client strategy — HTTP + OAuth to a hosted server, no child
  process — and is not supported by this fixture.
