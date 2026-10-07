"""The scope assertion must fail closed and never start the wrong tenant."""

from unittest.mock import Mock

import jwt
import pytest
from fastapi import HTTPException

from app.memory_scope import memory_tenant
from app.session.client_strategy import build_mcp_client_strategy

CHAT = "c/550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_SECRET", "hub-secret")
    monkeypatch.setenv("MCP_ENV_MEMORY_TENANT", "{data_path}")
    monkeypatch.setenv("MCP_SERVER_COMMAND", "node memory.js")
    monkeypatch.setenv("MCP_ISOLATE", "false")
    monkeypatch.delenv("OAUTH_ENV", raising=False)
    return jwt.encode(
        {"sub": "max", "groups": ["/engineering"]}, "test", algorithm="HS256"
    )


def headers(scope=CHAT, secret="hub-secret"):
    return {"X-INXM-Memory-Scope": scope, "X-Internal-Secret": secret}


def test_chat_scope_reaches_only_the_memory_tenant_environment(configured):
    strategy = build_mcp_client_strategy(
        access_token=configured,
        requested_group=None,
        incoming_headers=headers(),
    )
    assert strategy.server_params.env["MEMORY_TENANT"] == CHAT
    assert strategy.server_params.command == "node"


@pytest.mark.parametrize("secret", ["", "wrong"])
def test_scope_without_the_internal_secret_never_starts_a_child(
    configured, monkeypatch, secret
):
    start = Mock()
    monkeypatch.setattr("app.session.client_strategy.get_server_params", start)
    with pytest.raises(HTTPException) as error:
        build_mcp_client_strategy(
            access_token=configured,
            requested_group=None,
            incoming_headers=headers(secret=secret),
        )
    assert error.value.status_code == 403
    start.assert_not_called()


def test_unset_secret_cannot_match_an_empty_assertion(configured, monkeypatch):
    monkeypatch.delenv("INTERNAL_API_SECRET")
    with pytest.raises(HTTPException):
        memory_tenant(headers(secret=""), configured, None)


@pytest.mark.parametrize(
    "scope",
    [
        "u/max",
        "g/engineering",
        "c/not-a-uuid",
        "c/550E8400-E29B-41D4-A716-446655440000",
        "p/other",
        "../shared",
    ],
)
def test_malformed_or_mismatched_scopes_are_refused(configured, scope):
    with pytest.raises(HTTPException) as error:
        memory_tenant(headers(scope), configured, None)
    assert error.value.status_code == 400


def test_projects_match_the_group_and_check_membership(configured):
    assert (
        memory_tenant(headers("p/engineering"), configured, "engineering")
        == "p/engineering"
    )
    with pytest.raises(HTTPException) as error:
        memory_tenant(headers("p/finance"), configured, "finance")
    assert error.value.status_code == 403


def test_project_encoding_keeps_nested_groups_distinct(configured):
    token = jwt.encode(
        {"sub": "max", "groups": ["/a/b", "/ab", "/a ~*"]}, "test", algorithm="HS256"
    )
    assert memory_tenant(headers("p/a%2Fb"), token, "a/b") == "p/a%2Fb"
    assert memory_tenant(headers("p/ab"), token, "ab") == "p/ab"
    assert memory_tenant(headers("p/a+%7E*"), token, "a ~*") == "p/a+%7E*"


def test_assertion_cannot_override_another_servers_data_path(configured, monkeypatch):
    monkeypatch.delenv("MCP_ENV_MEMORY_TENANT")
    with pytest.raises(HTTPException) as error:
        memory_tenant(headers(), configured, None)
    assert error.value.status_code == 400


def test_scoped_memory_refuses_persistent_sessions_or_missing_credentials(configured):
    with pytest.raises(HTTPException) as error:
        memory_tenant(headers(), configured, None, persistent=True)
    assert error.value.status_code == 400
    with pytest.raises(HTTPException) as error:
        memory_tenant(headers(), None, None)
    assert error.value.status_code == 401
    with pytest.raises(HTTPException):
        memory_tenant(headers(), configured, "engineering")


def test_normal_user_and_group_calls_still_use_existing_templates(configured):
    assert memory_tenant({}, configured, None) is None
    assert memory_tenant({}, configured, "engineering") is None


def test_scoped_endpoint_rejects_missing_or_bad_assertions_before_downstream_contact(
    configured, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routes import router

    start = Mock()
    monkeypatch.setattr("app.session.client_strategy.get_server_params", start)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    for assertion, expected in [({}, 400), (headers(secret="wrong"), 403)]:
        response = client.post(
            "/memory/tools/search_nodes",
            json={"query": "release"},
            headers={"authorization": f"Bearer {configured}", **assertion},
        )
        assert response.status_code == expected, response.text
    start.assert_not_called()


@pytest.mark.parametrize(
    "source", ["MEMORY_SCOPE_SECRET", "MCP_ENV_MEMORY_SCOPE_SECRET"]
)
def test_hosted_scope_uses_its_server_secret_setting_without_sending_it_to_the_child(
    configured, monkeypatch, source
):
    from app.multi_server import ServerConfig, bind_server, reset_server

    monkeypatch.delenv("INTERNAL_API_SECRET")
    monkeypatch.setenv(source, "hub-secret")
    server = ServerConfig.from_mapping(
        {
            "id": "mcp-memory-server",
            "base_path": "/api/mcp-memory-server",
            "command": "node memory.js",
            "env": {"MCP_ENV_MEMORY_TENANT": "{data_path}"},
            "settings_from": {"INTERNAL_API_SECRET": source},
            "env_from": {"ALIASED_SCOPE_SECRET": source},
            "isolate": False,
        }
    )
    binding = bind_server(server)
    try:
        strategy = build_mcp_client_strategy(
            access_token=configured, requested_group=None, incoming_headers=headers()
        )
        assert strategy.server_params.env["MEMORY_TENANT"] == CHAT
        assert "INTERNAL_API_SECRET" not in strategy.server_params.env
        assert "MEMORY_SCOPE_SECRET" not in strategy.server_params.env
        assert "MCP_ENV_MEMORY_SCOPE_SECRET" not in strategy.server_params.env
        assert "ALIASED_SCOPE_SECRET" not in strategy.server_params.env
        assert "hub-secret" not in strategy.server_params.env.values()
    finally:
        reset_server(binding)


@pytest.mark.parametrize(
    "declaration", [{}, {"env": {"MCP_ENV_MEMORY_TENANT": "fixed"}}]
)
def test_global_memory_template_does_not_enable_scopes_on_other_bound_servers(
    configured, monkeypatch, declaration
):
    from app.multi_server import ServerConfig, bind_server, reset_server

    server = ServerConfig.from_mapping(
        {
            "id": "other-server",
            "base_path": "/api/other-server",
            "command": "node other.js",
            **declaration,
        }
    )
    binding = bind_server(server)
    start = Mock()
    monkeypatch.setattr("app.session.client_strategy.get_server_params", start)
    try:
        with pytest.raises(HTTPException) as error:
            build_mcp_client_strategy(
                access_token=configured,
                requested_group=None,
                incoming_headers=headers(),
            )
        assert error.value.status_code == 400
        start.assert_not_called()
    finally:
        reset_server(binding)


def test_a_bound_server_can_explicitly_reference_its_memory_template(
    configured, monkeypatch
):
    from app.multi_server import ServerConfig, bind_server, reset_server

    monkeypatch.setenv("MEMORY_TEMPLATE", "{data_path}")
    server = ServerConfig.from_mapping(
        {
            "id": "memory",
            "base_path": "/api/memory",
            "command": "node memory.js",
            "env_from": {"MCP_ENV_MEMORY_TENANT": "MEMORY_TEMPLATE"},
            "isolate": False,
        }
    )
    binding = bind_server(server)
    try:
        strategy = build_mcp_client_strategy(
            access_token=configured, requested_group=None, incoming_headers=headers()
        )
        assert strategy.server_params.env["MEMORY_TENANT"] == CHAT
    finally:
        reset_server(binding)


def test_start_session_rejects_scope_headers_before_starting_a_personal_child(
    configured, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routes import router

    start = Mock()
    monkeypatch.setattr("app.session.client_strategy.get_server_params", start)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.post(
        "/session/start", headers={"authorization": f"Bearer {configured}", **headers()}
    )
    assert response.status_code == 400, response.text
    assert "sessionless" in response.text
    start.assert_not_called()
