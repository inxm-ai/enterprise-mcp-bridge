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
