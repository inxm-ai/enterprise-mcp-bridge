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
    assert params.command == sys.executable
    assert "server.py" in params.args[0]


def test_default(monkeypatch):
    env = {}
    argv = ["main.py"]
    patch_module(monkeypatch, env=env, argv=argv)
    params = server_params.get_server_params()
    assert params.command == sys.executable
    assert "server.py" in params.args[0]


def test_env_command_empty(monkeypatch):
    env = {"MCP_SERVER_COMMAND": ""}
    patch_module(monkeypatch, env=env)
    params = server_params.get_server_params()
    assert params.command == sys.executable
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


def _record_templated_tokens(monkeypatch):
    seen_tokens = []

    def mock_get_data_access_manager():
        class MockDataManager:
            def resolve_data_resource(self, access_token, requested_group):
                seen_tokens.append(access_token)
                return f"u/{access_token}"

        return MockDataManager()

    monkeypatch.setattr(
        server_params, "get_data_access_manager", mock_get_data_access_manager
    )
    return seen_tokens


def test_mcp_env_template_uses_caller_token_not_exchanged_token(monkeypatch):
    env = {
        "OAUTH_ENV": "SLACK_TOKEN",
        "MCP_ENV_CACHE_PATH": "/cache/{data_path}/cache.json",
    }
    patch_module(monkeypatch, env=env)
    monkeypatch.setattr(
        server_params,
        "TokenRetrieverFactory",
        lambda: DummyTokenRetrieverFactory("provider_token"),
    )
    seen_tokens = _record_templated_tokens(monkeypatch)

    params = server_params.get_server_params(access_token="caller_jwt")

    assert params.env["SLACK_TOKEN"] == "provider_token"
    assert params.env["CACHE_PATH"] == "/cache/u/caller_jwt/cache.json"
    assert seen_tokens == ["caller_jwt"]


def test_command_template_uses_caller_token_not_exchanged_token(monkeypatch):
    env = {
        "OAUTH_ENV": "SLACK_TOKEN",
        "MCP_SERVER_COMMAND": "server --cache /cache/{data_path}",
    }
    patch_module(monkeypatch, env=env)
    monkeypatch.setattr(
        server_params,
        "TokenRetrieverFactory",
        lambda: DummyTokenRetrieverFactory("provider_token"),
    )
    seen_tokens = _record_templated_tokens(monkeypatch)

    params = server_params.get_server_params(access_token="caller_jwt")

    assert params.command == "server"
    assert params.args == ["--cache", "/cache/u/caller_jwt"]
    assert seen_tokens == ["caller_jwt"]


class RaisingTokenRetrieverFactory:
    def __init__(self, error):
        self.error = error

    def get(self, *args, **kwargs):
        error = self.error

        class Retriever:
            def retrieve_token(self, token):
                raise error

        return Retriever()


def test_an_expired_provider_link_never_falls_back_to_the_callers_token(monkeypatch):
    from app.oauth.token_exchange import ProviderTokenExpiredError

    patch_module(monkeypatch, env={"OAUTH_ENV": "AZURE_MGMT_OAUTH_TOKEN"})
    monkeypatch.setattr(
        server_params,
        "TokenRetrieverFactory",
        lambda: RaisingTokenRetrieverFactory(
            ProviderTokenExpiredError("azure-management")
        ),
    )

    with pytest.raises(ProviderTokenExpiredError) as raised:
        server_params.get_server_params(access_token="callers-keycloak-token")
    assert "Reconnect azure-management" in raised.value.message


def test_other_retrieval_failures_still_pass_the_raw_value_through(monkeypatch):
    # Cookie set-ups rely on this: the broker refuses the cookie value, and
    # the server wants that value as it is.
    patch_module(monkeypatch, env={"OAUTH_ENV": "OAUTH_TOKEN_VAR"})
    monkeypatch.setattr(
        server_params,
        "TokenRetrieverFactory",
        lambda: RaisingTokenRetrieverFactory(RuntimeError("broker said no")),
    )

    params = server_params.get_server_params(access_token="cookie-value")
    assert params.env["OAUTH_TOKEN_VAR"] == "cookie-value"


class _FailingRetriever:
    def retrieve_token(self, token):
        raise RuntimeError("no linked account")


class _FailingFactory:
    def get(self, *args, **kwargs):
        return _FailingRetriever()


def test_a_provider_bound_server_never_gets_the_callers_own_token(monkeypatch):
    patch_module(monkeypatch, env={"OAUTH_ENV": "VENDOR_TOKEN"})
    monkeypatch.setattr(server_params, "TokenRetrieverFactory", _FailingFactory)
    monkeypatch.setattr(
        server_params, "current_keycloak_provider_alias", lambda default: "vendor"
    )
    with pytest.raises(RuntimeError):
        server_params.defined_env({"OAUTH_ENV": "VENDOR_TOKEN"}, "platform-token")


def test_a_server_without_a_provider_still_gets_the_value_as_it_came(monkeypatch):
    patch_module(monkeypatch, env={"OAUTH_ENV": "COOKIE_VALUE"})
    monkeypatch.setattr(server_params, "TokenRetrieverFactory", _FailingFactory)
    monkeypatch.setattr(
        server_params, "current_keycloak_provider_alias", lambda default: ""
    )
    env = server_params.defined_env({"OAUTH_ENV": "COOKIE_VALUE"}, "cookie-value")
    env = env[0] if isinstance(env, tuple) else env
    assert env["COOKIE_VALUE"] == "cookie-value"


@pytest.mark.parametrize("provider", ["user-api-key", "token-exchange"])
@pytest.mark.parametrize("result", [{"error": "missing"}, {"access_token": ""}, None])
def test_a_non_keycloak_provider_never_falls_back_to_the_caller_token(
    monkeypatch, provider, result
):
    class Retriever:
        def retrieve_token(self, token):
            if result is None:
                raise RuntimeError("provider unavailable")
            return result

    monkeypatch.setattr(
        server_params, "current_auth_provider", lambda default: provider
    )
    monkeypatch.setattr(server_params, "current_keycloak_provider_alias", lambda _: "")
    monkeypatch.setattr(
        server_params.TokenRetrieverFactory, "get", lambda _: Retriever()
    )
    env = {"OAUTH_ENV": "VENDOR_TOKEN"}
    with pytest.raises(RuntimeError if result is None else ValueError):
        server_params.defined_env(env, "platform-token")
    assert env == {"OAUTH_ENV": "VENDOR_TOKEN"}
