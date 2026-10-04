import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from app import isolation, multi_server
from app.mcp_server import server_params


@pytest.fixture
def as_root(monkeypatch, tmp_path):
    """Pretend to be root without touching real users or /var/lib."""
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 0)
    monkeypatch.setattr(isolation.shutil, "which", lambda _: "/usr/bin/setpriv")
    monkeypatch.setattr(isolation.os, "chown", lambda *_: None)
    monkeypatch.setattr(isolation, "CHILD_ROOT", tmp_path / "children")
    return tmp_path / "children"


def test_each_server_has_a_stable_uid_of_its_own():
    uid = isolation.uid_for("mcp-m365-mail-server")
    assert uid == isolation.uid_for("mcp-m365-mail-server")
    assert isolation.UID_MIN <= uid < isolation.UID_MIN + isolation.UID_SPAN
    assert uid != isolation.uid_for("mcp-wikipedia-server")


def test_servers_sharing_a_uid_are_refused(monkeypatch):
    monkeypatch.setattr(isolation, "uid_for", lambda _: 20001)
    with pytest.raises(ValueError, match="would share uid"):
        isolation.check_unique_uids(["a", "b"])


def test_isolation_fails_closed_without_root(monkeypatch):
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 1000)
    with pytest.raises(isolation.IsolationError, match="root"):
        isolation.isolate("x", "npx", [], {})


def test_isolation_fails_closed_without_setpriv(monkeypatch):
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 0)
    monkeypatch.setattr(isolation.shutil, "which", lambda _: None)
    with pytest.raises(isolation.IsolationError, match="setpriv"):
        isolation.isolate("x", "npx", [], {})


def test_an_isolated_child_runs_unprivileged_with_only_its_own_env(as_root):
    bridge_env = {"PATH": "/usr/bin", "MCP_SERVERS": "[...]", "SECRET": "bridge"}
    env = {
        **bridge_env,
        "OAUTH_ENV": "MS365_MCP_OAUTH_TOKEN",
        "MS365_MCP_OAUTH_TOKEN": "user-token",
    }
    command, args, child_env, home = isolation.isolate(
        "mcp-m365-mail-server", "npx", ["-y", "pkg@1.0.0"], env, bridge_env
    )
    uid = isolation.uid_for("mcp-m365-mail-server")
    assert command == "/usr/bin/setpriv"
    assert args[:5] == [
        f"--reuid={uid}",
        f"--regid={uid}",
        "--clear-groups",
        "--no-new-privs",
        "--inh-caps=-all",
    ]
    assert args[5:] == ["--", "npx", "-y", "pkg@1.0.0"]
    assert child_env["MS365_MCP_OAUTH_TOKEN"] == "user-token"
    assert child_env["PATH"] == "/usr/bin"
    assert "SECRET" not in child_env and "MCP_SERVERS" not in child_env
    assert child_env["HOME"] == str(home) == str(as_root / str(uid))
    assert child_env["npm_config_cache"].startswith(str(home))
    assert child_env["UV_CACHE_DIR"].startswith(str(home))
    assert oct(home.stat().st_mode & 0o777) == "0o700"


def test_isolate_is_parsed_per_server_and_uid_clashes_refused(monkeypatch):
    with pytest.raises(ValueError, match="isolate must be a boolean"):
        multi_server.parse_servers(
            json.dumps([{"id": "x", "base_path": "/x", "command": "x", "isolate": 1}])
        )
    monkeypatch.setattr(isolation, "uid_for", lambda _: 20001)
    servers = [
        {"id": "a", "base_path": "/a", "command": "a", "isolate": True},
        {"id": "b", "base_path": "/b", "command": "b", "isolate": True},
    ]
    with pytest.raises(ValueError, match="would share uid"):
        multi_server.parse_servers(json.dumps(servers))
    # Only isolated local servers need a uid of their own.
    servers[1]["isolate"] = False
    assert len(multi_server.parse_servers(json.dumps(servers))) == 2


def test_an_isolated_server_is_spawned_through_setpriv(monkeypatch, as_root):
    (server,) = multi_server.parse_servers(
        json.dumps(
            [
                {
                    "id": "mcp-wiki-server",
                    "base_path": "/api/mcp-wiki-server",
                    "command": "uvx wikipedia-mcp@1.5.0",
                    "isolate": True,
                }
            ]
        )
    )
    token = multi_server.bind_server(server)
    try:
        params = server_params.get_server_params()
    finally:
        multi_server.reset_server(token)
    assert params.command == "/usr/bin/setpriv"
    assert params.args[-2:] == ["uvx", "wikipedia-mcp@1.5.0"]
    assert str(params.cwd) == params.env["HOME"]

    (open_server,) = multi_server.parse_servers(
        json.dumps([{"id": "y", "base_path": "/y", "command": "uvx y@1"}])
    )
    token = multi_server.bind_server(open_server)
    try:
        assert server_params.get_server_params().command == "uvx"
    finally:
        multi_server.reset_server(token)


def test_the_bridge_starts_with_isolated_servers_configured():
    """MCP_SERVERS is parsed at import time: the whole module must load."""
    servers = [
        {"id": "a", "base_path": "/a", "command": "a", "isolate": True},
        {"id": "b", "base_path": "/b", "url": "https://b.example/mcp"},
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app import multi_server; print(len(multi_server.SERVERS))",
        ],
        env={**os.environ, "MCP_SERVERS": json.dumps(servers)},
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"


@pytest.mark.skipif(
    os.geteuid() != 0 or not shutil.which("setpriv"),
    reason="needs root and setpriv (runs in the Docker testing stage)",
)
def test_the_kernel_keeps_isolated_children_apart(monkeypatch):
    """Two real isolated children: neither can read the other's token."""
    # Not tmp_path: pytest's root-only parent would keep children out of
    # their own homes.
    base = Path(tempfile.mkdtemp())
    os.chmod(base, 0o711)
    monkeypatch.setattr(isolation, "CHILD_ROOT", base / "children")
    command, args, env, home = isolation.isolate(
        "holder",
        sys.executable,
        ["-c", "import time; time.sleep(30)"],
        {"PATH": os.environ["PATH"], "TOKEN": "secret-user-token"},
        {"PATH": os.environ["PATH"]},
    )
    holder = subprocess.Popen([command, *args], env=env, cwd=home)
    try:
        environ = f"/proc/{holder.pid}/environ"
        holder_uid = str(isolation.uid_for("holder"))
        status = ""
        for _ in range(50):
            status = open(f"/proc/{holder.pid}/status").read()
            if f"Uid:\t{holder_uid}" in status:
                break
            time.sleep(0.1)
        assert f"Uid:\t{holder_uid}" in status, "the holder runs as its own user"
        probe = (
            "import os, sys\n"
            f"try:\n    open({environ!r}, 'rb').read()\n"
            "except PermissionError:\n    sys.exit(0)\n"
            "sys.exit(1)\n"
        )
        command, args, env, home = isolation.isolate(
            "package",
            sys.executable,
            ["-c", probe],
            {"PATH": os.environ["PATH"]},
            {"PATH": os.environ["PATH"]},
        )
        result = subprocess.run([command, *args], env=env, cwd=home, timeout=30)
        assert result.returncode == 0, "a sibling read the holder's environment"
        whoami = subprocess.run(
            [command, *args[:-2], "-c", "import os; print(os.getuid())"],
            env=env,
            cwd=home,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert whoami.stdout.strip() == str(isolation.uid_for("package"))
    finally:
        holder.kill()
        holder.wait()
        shutil.rmtree(base, ignore_errors=True)
