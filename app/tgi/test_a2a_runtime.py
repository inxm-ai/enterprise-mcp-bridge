from app.tgi.a2a_runtime import _access_token_from_headers, build_a2a_app


def test_a2a_app_exposes_standard_jsonrpc_and_agent_card_routes():
    app = build_a2a_app()
    paths = {getattr(route, "path", None) for route in app.routes}

    assert "/tgi/v1/a2a" in paths
    assert "/.well-known/agent-card.json" in paths
    assert "/.well-known/agent.json" in paths


def test_a2a_auth_accepts_standard_bearer_token():
    assert (
        _access_token_from_headers({"authorization": "Bearer test-token"})
        == "test-token"
    )


def test_legacy_custom_prompt_payload_is_not_accepted_as_a2a():
    from fastapi.testclient import TestClient

    client = TestClient(build_a2a_app())
    response = client.post(
        "/tgi/v1/a2a",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "enterprise-mcp-bridge",
            "params": {"prompt": "Say hello"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "1"
    assert payload["error"]["code"] == -32601
