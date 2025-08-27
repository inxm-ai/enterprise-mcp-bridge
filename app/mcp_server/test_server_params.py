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


def test_process_template_basic(monkeypatch):
    from app.mcp_server.server_params import process_template

    def mock_get_data_access_manager():
        class MockDataManager:
            def resolve_data_resource(self, access_token, requested_group):
                return "mock_data_path"

        return MockDataManager()

    def mock_extract_user_info(self, access_token):
        return {"user_id": "mock_user_id"}

    monkeypatch.setattr(
        "app.mcp_server.server_params.get_data_access_manager",
        mock_get_data_access_manager,
    )
    monkeypatch.setattr(
        "app.oauth.user_info.UserInfoExtractor.extract_user_info",
        mock_extract_user_info,
    )

    template = "command --data {data_path} --user {user_id} --group {group_id}"
    result = process_template(template, "mock_access_token", "mock_group_id")

    assert (
        result
        == "command --data mock_data_path --user mock_user_id --group mock_group_id"
    )


def test_process_template_no_placeholders():
    from app.mcp_server.server_params import process_template

    template = "command --no-placeholders"
    result = process_template(template, "mock_access_token", "mock_group_id")

    assert result == "command --no-placeholders"


def test_process_template_error_handling(monkeypatch):
    from app.mcp_server.server_params import process_template

    def mock_get_data_access_manager():
        raise Exception("Mocked exception")

    monkeypatch.setattr(
        "app.mcp_server.server_params.get_data_access_manager",
        mock_get_data_access_manager,
    )

    template = "command --data {data_path}"
    result = process_template(template, "mock_access_token", "mock_group_id")

    assert result == "command --data {data_path}"
