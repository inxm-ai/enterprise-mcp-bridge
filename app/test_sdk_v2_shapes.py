"""The bridge reads real SDK v2 objects, not only the camelCase doubles the older tests use.

Every path here once read a camelCase attribute that a v2 model does not
have and fell back to a default without failing; these pin the real shapes.
"""

from mcp import types
from mcp.shared.exceptions import MCPError

from app.models import RunToolsResult
from app.tgi.services.tools import tools_map
from app.utils import mcp_fields
from app.utils.mcp_operation import (
    ERROR_TYPE_DOWNSTREAM_TRANSIENT,
    ERROR_TYPE_UPSTREAM_TIMEOUT,
    classify_error_result,
    classify_exception,
    is_mcp_error,
)

SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
}


def test_tgi_tool_mapping_keeps_the_schemas_of_a_v2_tool(monkeypatch):
    monkeypatch.setattr(tools_map.vars_module, "INCLUDE_TOOLS", [], raising=False)
    monkeypatch.setattr(tools_map.vars_module, "EXCLUDE_TOOLS", [], raising=False)
    tool = types.Tool(
        name="upload_file",
        description="d",
        inputSchema=SCHEMA,
        outputSchema={"type": "object"},
    )

    mapped = tools_map.map_tools([tool])

    function = mapped[0]["function"]
    assert function["parameters"] == SCHEMA
    assert mcp_fields.output_schema(tool) == {"type": "object"}


def test_the_rest_envelope_reads_a_v2_result_and_any_structured_shape():
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text="busy (retryable)")],
        isError=True,
        structuredContent=["not", "an", "object"],
    )

    envelope = RunToolsResult(result)

    assert envelope.isError is True
    assert envelope.content[0].text == "busy (retryable)"
    assert envelope.structuredContent == ["not", "an", "object"]


def test_the_rest_envelope_accepts_a_dict_shaped_result():
    envelope = RunToolsResult({"isError": True, "content": [{"text": "denied"}]})

    assert envelope.isError is True
    assert [c.text for c in envelope.content] == ["denied"]
    assert RunToolsResult({"isError": True}).content[0].text == "An error occurred"


def test_error_results_are_classified_by_payload_then_text():
    transient = types.CallToolResult(
        content=[types.TextContent(type="text", text="busy")],
        isError=True,
        structuredContent={"result": {"error": {"code": "busy", "retryable": True}}},
    )
    timeout = types.CallToolResult(
        content=[types.TextContent(type="text", text="Request timed out")], isError=True
    )

    assert classify_error_result(transient) == ERROR_TYPE_DOWNSTREAM_TRANSIENT
    assert classify_error_result(timeout) == ERROR_TYPE_UPSTREAM_TIMEOUT


def test_the_sdk_protocol_error_is_recognised_under_its_v2_name():
    error = MCPError(code=-32001, message="Request timed out")

    assert is_mcp_error(error)
    assert classify_exception(error) == ERROR_TYPE_UPSTREAM_TIMEOUT
