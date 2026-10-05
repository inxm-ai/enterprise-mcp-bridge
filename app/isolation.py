"""Run each local MCP server child as its own unprivileged user.

Several stdio servers can share one bridge process (``MCP_SERVERS``). Run as
the bridge's user, any child could read a sibling's ``/proc/<pid>/environ``
(where the bridge puts the caller's exchanged token, see ``OAUTH_ENV``) or
attach to it with ptrace. Isolated, each server gets a stable uid of its own,
a private 0700 home, no capabilities, ``no_new_privs`` and only its own
environment, so the kernel keeps children (and third-party packages run with
``npx``/``uvx``) out of each other and out of the bridge.

The bridge must run as root to switch users; isolation that cannot be
applied fails the request instead of silently running the child as root.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

UID_MIN = 20000
UID_SPAN = 40000

# Bridge-owned state children must not reach (e.g. the tools cache).
PRIVATE_DIR = Path(os.environ.get("MCP_BRIDGE_PRIVATE_DIR", "/var/lib/mcp-bridge"))
# Per-server homes, one 0700 directory per uid.
CHILD_ROOT = Path(os.environ.get("MCP_CHILD_ROOT", "/var/lib/mcp-children"))

# The only bridge environment an isolated child inherits; everything else
# it gets is its own (server env, token variable, MCP_ENV_* expansions).
_INHERITED = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


class IsolationError(RuntimeError):
    """Isolation was requested but cannot be applied."""


def enabled_globally() -> bool:
    return os.environ.get("MCP_ISOLATE_CHILDREN", "").strip().lower() == "true"


def uid_for(server_id: str) -> int:
    """A stable uid for a server: the same id always maps to the same user."""
    digest = hashlib.sha256(server_id.encode()).digest()
    return UID_MIN + int.from_bytes(digest[:4], "big") % UID_SPAN


def check_unique_uids(server_ids: Iterable[str]) -> None:
    seen: dict[int, str] = {}
    for server_id in server_ids:
        uid = uid_for(server_id)
        if uid in seen:
            raise ValueError(
                f"MCP servers {seen[uid]!r} and {server_id!r} would share uid {uid}; "
                "rename one of them"
            )
        seen[uid] = server_id


def _child_env(env: dict[str, str], bridge_env: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in env.items()
        if key in _INHERITED or bridge_env.get(key) != value
    }


def _prepare_home(uid: int) -> Path:
    CHILD_ROOT.mkdir(mode=0o711, parents=True, exist_ok=True)
    os.chmod(CHILD_ROOT, 0o711)
    home = CHILD_ROOT / str(uid)
    for path in (home, home / "tmp", home / "cache"):
        path.mkdir(mode=0o700, exist_ok=True)
        os.chown(path, uid, uid)
        os.chmod(path, 0o700)
    return home


def isolate(
    server_id: str,
    command: str,
    args: list[str],
    env: dict[str, str],
    bridge_env: Optional[dict[str, str]] = None,
) -> tuple[str, list[str], dict[str, str], Path]:
    """The command, arguments, environment and cwd that run ``command`` as
    ``server_id``'s own user."""
    if os.geteuid() != 0:
        raise IsolationError(
            f"MCP server {server_id!r} is isolated, but the bridge does not run "
            "as root and cannot switch users"
        )
    setpriv = shutil.which("setpriv")
    if not setpriv:
        raise IsolationError(
            f"MCP server {server_id!r} is isolated, but setpriv is not installed"
        )
    uid = uid_for(server_id)
    home = _prepare_home(uid)
    child_env = _child_env(env, os.environ if bridge_env is None else bridge_env)
    cache = str(home / "cache")
    child_env.update(
        {
            "HOME": str(home),
            "USER": f"mcp-{uid}",
            "TMPDIR": str(home / "tmp"),
            "XDG_CACHE_HOME": cache,
            "npm_config_cache": str(home / "cache" / "npm"),
            "UV_CACHE_DIR": str(home / "cache" / "uv"),
        }
    )
    wrapped = [
        f"--reuid={uid}",
        f"--regid={uid}",
        "--clear-groups",
        "--no-new-privs",
        "--inh-caps=-all",
        "--",
        command,
        *args,
    ]
    return setpriv, wrapped, child_env, home
