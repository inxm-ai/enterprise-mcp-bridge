import os
import sys
import types
import pytest
from mcp_server import server_params
from mcp import StdioServerParameters

class DummyStdioServerParameters:
    def __init__(self, command, args, env):
        self.command = command
        self.args = args
        self.env = env

def patch_module(monkeypatch, env=None, argv=None):
    monkeypatch.setattr(server_params, "StdioServerParameters", DummyStdioServerParameters)
    monkeypatch.setattr(os, "environ", env or {})
    monkeypatch.setattr(sys, "argv", argv or ["main.py"])

def test_env_command(monkeypatch):
    env = {"MCP_SERVER_COMMAND": "python myserver.py --foo bar"}
    patch_module(monkeypatch, env=env)
    params = server_params.get_server_params()
    assert params.command == "python"
    assert params.args == ["myserver.py", "--foo", "bar"]
    assert params.env["MCP_SERVER_COMMAND"] == "python myserver.py --foo bar"

def test_sys_argv(monkeypatch):
    env = {}
    argv = ["main.py", "--", "bash", "-c", "echo hello"]
    patch_module(monkeypatch, env=env, argv=argv)
    params = server_params.get_server_params()
    assert params.command == "bash"
    assert params.args == ["-c", "echo hello"]
    assert params.env == env

def test_sys_argv_missing_args(monkeypatch):
    env = {}
    argv = ["main.py", "--"]
    patch_module(monkeypatch, env=env, argv=argv)
    params = server_params.get_server_params()
    assert params.command == "python"
    assert "server.py" in params.args[0]

def test_default(monkeypatch):
    env = {}
    argv = ["main.py"]
    patch_module(monkeypatch, env=env, argv=argv)
    params = server_params.get_server_params()
    assert params.command == "python"
    assert "server.py" in params.args[0]

def test_env_command_empty(monkeypatch):
    env = {"MCP_SERVER_COMMAND": ""}
    patch_module(monkeypatch, env=env)
    params = server_params.get_server_params()
    assert params.command == "python"
    assert "server.py" in params.args[0]
