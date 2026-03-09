"""Parsers for detecting and converting structured text to Python objects.

Each parser is a callable that accepts a string and returns a parsed value
(dict, list, etc.) or raises an exception if the text cannot be parsed.
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


def _parse_json(text: str) -> Any:
    return json.loads(text)


def _parse_xml(text: str) -> Any:
    stripped = text.strip()
    if not stripped.startswith("<"):
        raise ValueError("Not XML")
    root = ET.fromstring(stripped)
    return {root.tag: _xml_element_to_dict(root)}


_PARSERS = [_parse_json, _parse_xml]


def try_parse_structured_text(text: str) -> Any | None:
    """Try each registered parser in order.

    Returns the first successfully parsed value, or ``None`` if no parser
    succeeds.  The original text is never mutated.
    """
    if not text or not isinstance(text, str):
        return None
    for parser in _PARSERS:
        try:
            return parser(text)
        except Exception:
            pass
    return None
