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


def _downloaded_file_result():
    """What github-mcp-server's get_file_contents returns: a note plus the file."""
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text", text="successfully downloaded text file (SHA: e89dc91)"
            ),
            types.EmbeddedResource(
                type="resource",
                resource=types.TextResourceContents(
                    uri="repo://inxm-ai/int-llm-skills/contents/SKILL.md",
                    mimeType="text/markdown",
                    text="# Principal developer",
                ),
            ),
        ]
    )


def test_the_rest_envelope_carries_the_text_of_an_embedded_resource():
    envelope = RunToolsResult(_downloaded_file_result())

    note, file = envelope.content
    assert note.text.startswith("successfully downloaded")
    assert note.type is None
    assert file.text == "# Principal developer"
    assert file.type == "resource"
    assert file.uri == "repo://inxm-ai/int-llm-skills/contents/SKILL.md"
    assert file.mimeType == "text/markdown"
    assert file.data is None


def test_the_rest_envelope_keeps_binary_payloads_and_describes_them():
    result = types.CallToolResult(
        content=[
            types.EmbeddedResource(
                type="resource",
                resource=types.BlobResourceContents(
                    uri="repo://o/r/contents/logo.png",
                    mimeType="image/png",
                    blob="iVBORw0=",
                ),
            ),
            types.ImageContent(type="image", data="AAAA", mimeType="image/png"),
            types.ResourceLink(
                type="resource_link", name="big.zip", uri="https://example.com/big.zip"
            ),
        ]
    )

    blob, image, link = RunToolsResult(result).content

    assert blob.text == "[resource: repo://o/r/contents/logo.png, image/png, 5 bytes]"
    assert blob.data == "iVBORw0="
    assert image.text == "[image: image/png, 3 bytes]"
    assert image.data == "AAAA"
    assert link.text == "[resource link: https://example.com/big.zip]"
    assert link.uri == "https://example.com/big.zip"
    assert link.data is None


def test_a_result_that_already_passed_a_bridge_keeps_its_file():
    once = RunToolsResult(_downloaded_file_result())

    twice = RunToolsResult(once.model_dump())

    assert twice.content[1].text == "# Principal developer"
    assert twice.content[1].uri == "repo://inxm-ai/int-llm-skills/contents/SKILL.md"
    assert twice.content[1].mimeType == "text/markdown"


def test_an_empty_binary_file_keeps_its_empty_payload():
    result = types.CallToolResult(
        content=[
            types.EmbeddedResource(
                type="resource",
                resource=types.BlobResourceContents(
                    uri="repo://o/r/contents/empty.bin",
                    mimeType="application/octet-stream",
                    blob="",
                ),
            )
        ]
    )

    (file,) = RunToolsResult(result).content

    assert file.data == ""
    assert file.text == (
        "[resource: repo://o/r/contents/empty.bin, application/octet-stream, 0 bytes]"
    )


def test_content_text_reads_wire_shaped_resources():
    item = {
        "type": "resource",
        "resource": {"uri": "file:///a.txt", "mimeType": "text/plain", "text": "hi"},
    }

    assert mcp_fields.content_text(item) == "hi"
    assert mcp_fields.content_text({"text": "plain"}) == "plain"
    assert mcp_fields.content_text({"type": "image"}) == "[image]"


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


def test_a_dict_shaped_error_result_is_classified_by_its_text():
    timeout = {"isError": True, "content": [{"text": "Request timed out"}]}

    assert classify_error_result(timeout) == ERROR_TYPE_UPSTREAM_TIMEOUT


def test_falsy_structured_content_counts_as_present():
    for value in ({}, [], 0, False, ""):
        assert mcp_fields.has_structured_content({"structuredContent": value})
        assert mcp_fields.has_structured_content(
            types.CallToolResult(content=[], structuredContent=value)
        )
    assert not mcp_fields.has_structured_content({})
    assert not mcp_fields.has_structured_content(types.CallToolResult(content=[]))


def test_the_sdk_protocol_error_is_recognised_under_its_v2_name():
    error = MCPError(code=-32001, message="Request timed out")

    assert is_mcp_error(error)
    assert classify_exception(error) == ERROR_TYPE_UPSTREAM_TIMEOUT
