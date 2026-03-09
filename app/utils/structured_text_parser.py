"""Parsers for detecting and converting structured text to Python objects.

Each parser is a callable that accepts a string and returns a parsed value
(dict, list, etc.) or ``None`` if the text does not look like that format.
"""

import json
import xml.etree.ElementTree as ET
from typing import Any


def _xml_element_to_dict(element: ET.Element) -> Any:
    children = list(element)
    if not children:
        text = (element.text or "").strip()
        if element.attrib:
            result: dict = dict(element.attrib)
            if text:
                result["_text"] = text
            return result
        return text if text else None

    child_dict: dict = {}
    for child in children:
        value = _xml_element_to_dict(child)
        tag = child.tag
        if tag in child_dict:
            existing = child_dict[tag]
            if not isinstance(existing, list):
                child_dict[tag] = [existing]
            child_dict[tag].append(value)
        else:
            child_dict[tag] = value

    if element.attrib:
        child_dict.update(element.attrib)

    return child_dict


def _parse_json(text: str) -> Any | None:
    stripped = text.lstrip()
    if not stripped or stripped[0] not in ("{", "["):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_xml(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped.startswith("<"):
        return None
    try:
        root = ET.fromstring(stripped)
        return {root.tag: _xml_element_to_dict(root)}
    except ET.ParseError:
        return None


_PARSERS = [_parse_json, _parse_xml]


def try_parse_structured_text(text: str) -> Any | None:
    """Try each registered parser in order.

    Returns the first successfully parsed value, or ``None`` if no parser
    succeeds.  The original text is never mutated.
    """
    if not text or not isinstance(text, str):
        return None
    return next(
        (result for parser in _PARSERS if (result := parser(text)) is not None), None
    )
