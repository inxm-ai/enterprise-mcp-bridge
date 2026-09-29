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
