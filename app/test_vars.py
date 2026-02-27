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


def test_app_ui_patch_only_defaults_true(monkeypatch):
    monkeypatch.delenv("APP_UI_PATCH_ONLY", raising=False)
    import app.vars as vars_module

    importlib.reload(vars_module)

    assert vars_module.APP_UI_PATCH_ONLY is True


def test_generated_ui_gateway_role_args_valid_json(monkeypatch):
    monkeypatch.setenv(
        "GENERATED_UI_GATEWAY_ROLE_ARGS",
        '{"list_tools":{"prompt":"${prompt}"},"get_tool":{"server_id":"${server_id}","tool_name":"${tool_name}"},"call_tool":{"server_id":"${server_id}"}}',
    )
    import app.vars as vars_module

    importlib.reload(vars_module)

    mapping = getattr(vars_module, "GENERATED_UI_GATEWAY_ROLE_ARGS", None)
    assert isinstance(mapping, dict)
    assert mapping["list_tools"]["prompt"] == "${prompt}"
    assert mapping["get_tool"]["server_id"] == "${server_id}"
    assert mapping["call_tool"]["server_id"] == "${server_id}"


def test_generated_ui_gateway_role_args_invalid_json(monkeypatch):
    monkeypatch.setenv("GENERATED_UI_GATEWAY_ROLE_ARGS", "{this-is-invalid")
    import app.vars as vars_module

    importlib.reload(vars_module)

    mapping = getattr(vars_module, "GENERATED_UI_GATEWAY_ROLE_ARGS", None)
    assert mapping == {}


def test_generated_ui_gateway_role_args_ignores_non_object_roles(monkeypatch):
    monkeypatch.setenv(
        "GENERATED_UI_GATEWAY_ROLE_ARGS",
        '{"list_tools":"bad","unknown_role":{"x":1},"list_servers":{"prompt":"${prompt}"}}',
    )
    import app.vars as vars_module

    importlib.reload(vars_module)

    mapping = getattr(vars_module, "GENERATED_UI_GATEWAY_ROLE_ARGS", None)
    assert mapping == {"list_servers": {"prompt": "${prompt}"}}


def test_generated_ui_gateway_server_id_fields_parsing(monkeypatch):
    monkeypatch.setenv(
        "GENERATED_UI_GATEWAY_SERVER_ID_FIELDS",
        "server_id,meta.mcp_server_id,url",
    )
    import app.vars as vars_module

    importlib.reload(vars_module)

    fields = getattr(vars_module, "GENERATED_UI_GATEWAY_SERVER_ID_FIELDS", None)
    assert fields == ["server_id", "meta.mcp_server_id", "url"]


def test_generated_ui_gateway_server_id_fields_defaults(monkeypatch):
    monkeypatch.delenv("GENERATED_UI_GATEWAY_SERVER_ID_FIELDS", raising=False)
    import app.vars as vars_module

    importlib.reload(vars_module)

    fields = getattr(vars_module, "GENERATED_UI_GATEWAY_SERVER_ID_FIELDS", None)
    assert isinstance(fields, list)
    assert "url" in fields
    assert "meta.mcp_server_id" in fields
