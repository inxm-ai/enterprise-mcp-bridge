import os
import sys
import pytest
from app.mcp_server import server_params


class DummyStdioServerParameters:
    def __init__(self, command, args, env):
        self.command = command
        self.args = args
        self.env = env


def patch_module(monkeypatch, env=None, argv=None):
    monkeypatch.setattr(
        server_params, "StdioServerParameters", DummyStdioServerParameters
    )
    monkeypatch.setattr(os, "environ", env or {})
    monkeypatch.setattr(sys, "argv", argv or ["main.py"])


class DummyTokenRetriever:
    def __init__(self, token_value):
        self.token_value = token_value

    def retrieve_token(self, token):
        return {"access_token": self.token_value}


class DummyTokenRetrieverFactory:
    def __init__(self, token_value):
        self.token_value = token_value

    def get(self, *args, **kwargs):
        return DummyTokenRetriever(self.token_value)


def test_oauth_env_set_with_token(monkeypatch, mock_token_retriever_factory):
    env = {"OAUTH_ENV": "OAUTH_TOKEN_VAR"}
    patch_module(monkeypatch, env=env)

    # Call get_server_params with access_token instead of oauth_token
    params = server_params.get_server_params(access_token="test_access_token")
    assert "OAUTH_TOKEN_VAR" in params.env
    assert params.env["OAUTH_TOKEN_VAR"] == "test_access_token"


def test_oauth_env_set_missing_token(monkeypatch):
    env = {"OAUTH_ENV": "OAUTH_TOKEN_VAR"}
    patch_module(monkeypatch, env=env)
    monkeypatch.setattr(
        server_params,
        "TokenRetrieverFactory",
        lambda: DummyTokenRetrieverFactory("dummy"),
    )
    with pytest.raises(ValueError):
        server_params.get_server_params()


def test_oauth_env_not_set(monkeypatch):
    env = {}
    patch_module(monkeypatch, env=env)
    monkeypatch.setattr(
        server_params,
        "TokenRetrieverFactory",
        lambda: DummyTokenRetrieverFactory("dummy"),
    )
    params = server_params.get_server_params()
    assert "OAUTH_TOKEN_VAR" not in params.env


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
