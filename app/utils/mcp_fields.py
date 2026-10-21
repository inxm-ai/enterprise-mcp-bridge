"""Field access on MCP results and tools across SDK lines and the bridge's own shapes.

MCP SDK v2 models expose snake_case attributes (``is_error``,
``structured_content``, ``input_schema``, ``mime_type``); the wire, the
bridge's JSON envelopes and the duck-typed fakes in this repo use the
protocol's camelCase names. Every read goes through here, so a result may
be an SDK model of either line, one of the bridge's own models, or a dict.
"""

from typing import Any

from pydantic import BaseModel

MISSING = object()


def read(obj: Any, snake: str, camel: str, default: Any = None) -> Any:
    """The field under its camelCase or snake_case name, from an object or a dict.

    camelCase is tried first: an SDK v2 model has no such attribute and falls
    through to its snake_case field, while a test double that sets the
    protocol name explicitly (a Mock would otherwise invent a truthy attribute
    for any name asked of it) answers with what it was given.
    """
    if isinstance(obj, dict):
        for key in (camel, snake):
            if key in obj:
                return obj[key]
        return default
    for name in (camel, snake):
        value = getattr(obj, name, MISSING)
        if value is not MISSING:
            return value
    return default


def dump(model: Any, **kwargs: Any) -> Any:
    """A pydantic model dumped as the protocol spells it (SDK v2 fields are snake_case); other dumpers as they are."""
    if isinstance(model, BaseModel):
        return model.model_dump(by_alias=True, **kwargs)
    return model.model_dump(**kwargs)


def is_error(result: Any) -> bool:
    return bool(read(result, "is_error", "isError", False))


def structured_content(result: Any) -> Any:
    return read(result, "structured_content", "structuredContent")


def has_structured_content(result: Any) -> bool:
    """Whether the result carries structured content, whatever its value.

    Structured content is any JSON value, so ``{}``, ``[]``, ``0`` and
    ``false`` are results in their own right and must not read as absent.
    ``None`` is: an SDK model always has the attribute and defaults it to
    ``None``, and the protocol has no null structured content.
    """
    value = read(result, "structured_content", "structuredContent", MISSING)
    return value is not MISSING and value is not None


def content(result: Any) -> list:
    """The result's content items, from an object or a dict; empty when there are none."""
    return read(result, "content", "content") or []


def first_text(result: Any) -> str:
    """The text of the first content item, or ``""`` when there is none."""
    items = content(result)
    if not items:
        return ""
    return read(items[0], "text", "text") or ""


NON_TEXT_TYPES = ("resource", "resource_link", "image", "audio")


def is_text_item(item: Any) -> bool:
    """Whether the item is the tool's own words rather than a file or media.

    Only these speak for the tool: parsing them for structured data or an
    error signal is sound, while a downloaded JSON file saying
    ``"success": false`` is just a file.
    """
    return read(item, "type", "type") not in NON_TEXT_TYPES


def content_text(item: Any) -> str:
    """What a content item says as text, whatever its type.

    Only text items carry ``text`` at the top level. An embedded resource
    (a downloaded file, say) holds its text under ``resource``, and binary
    payloads (a blob, an image, audio) are described rather than inlined, so
    no item reads as empty just because it is not a text item.
    """
    if isinstance(item, str):
        return item
    item_type = read(item, "type", "type")
    if item_type == "resource":
        resource = resource_contents(item)
        text = read(resource, "text", "text")
        if isinstance(text, str):
            return text
        return _binary_note(
            "resource",
            read(resource, "uri", "uri"),
            mime_type(resource),
            binary_data(resource),
        )
    if item_type == "resource_link":
        return _binary_note(
            "resource link", read(item, "uri", "uri"), mime_type(item), None
        )
    if item_type in ("image", "audio"):
        return _binary_note(
            item_type, None, mime_type(item), read(item, "data", "data")
        )
    text = read(item, "text", "text", "")
    return "" if text is None else str(text)


def resource_contents(item: Any) -> Any:
    """The contents of an embedded resource item.

    The protocol nests them under ``resource``; the bridge's own envelope
    has them flattened onto the item, so a result that already passed
    through a bridge reads the same.
    """
    return read(item, "resource", "resource") or item


def binary_data(contents: Any) -> Any:
    """The base64 payload of a blob resource, an image or audio, if any."""
    return read(contents, "blob", "blob") or read(contents, "data", "data")


def _binary_note(kind: str, uri: Any, mime: Any, data: Any) -> str:
    details = [str(part) for part in (uri, mime) if part]
    if isinstance(data, str) and data:
        # The decoded size without decoding: a blob can be megabytes.
        details.append(f"{len(data) * 3 // 4 - data[-2:].count('=')} bytes")
    return f"[{kind}: {', '.join(details)}]" if details else f"[{kind}]"


def set_structured_content(result: Any, value: Any) -> None:
    """Write the structured content back under whichever name the object carries."""
    if isinstance(result, dict):
        result["structuredContent"] = value
        return
    for name in ("structured_content", "structuredContent"):
        if getattr(result, name, MISSING) is not MISSING:
            setattr(result, name, value)
            return
    setattr(result, "structuredContent", value)


def input_schema(tool: Any) -> Any:
    return read(tool, "input_schema", "inputSchema")


def output_schema(tool: Any) -> Any:
    return read(tool, "output_schema", "outputSchema")


def mime_type(contents: Any) -> Any:
    return read(contents, "mime_type", "mimeType")
