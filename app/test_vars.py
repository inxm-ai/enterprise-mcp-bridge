import importlib


def test_mcp_map_header_to_input_parsing(monkeypatch):
    monkeypatch.setenv(
        "MCP_MAP_HEADER_TO_INPUT", "userId=x-auth-user-id,email=x-auth-user-email"
    )
    import app.vars as vars_module

    importlib.reload(vars_module)

    mapping = getattr(vars_module, "MCP_MAP_HEADER_TO_INPUT", None)
    assert isinstance(mapping, dict)
    assert mapping.get("userId") == "x-auth-user-id"
    assert mapping.get("email") == "x-auth-user-email"
