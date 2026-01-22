from typing import Any, Dict, Optional


# JSON Schema describing the expected structure of the generated UI logic.
generation_logic_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "GeneratedLogic",
    "type": "object",
    "properties": {
        "service_script": {
            "type": "string",
            "description": "The Javascript code for the McpService class that wraps tool calls. Leave empty if McpService is inlined elsewhere.",
        },
        "components_script": {
            "type": "string",
            "description": "The Javascript code for the Pfusch components.",
        },
        "test_script": {
            "type": "string",
            "description": "The node:test suite enabling checking of the components.",
        },
    },
    "required": ["components_script", "test_script"],
    "additionalProperties": False,
}

# JSON Schema describing the expected structure of the generated dummy data module.
generation_dummy_data_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "GeneratedDummyData",
    "type": "object",
    "properties": {
        "dummy_data": {
            "type": "string",
            "description": "A Javascript module that exports dummy data used by tests.",
        }
    },
    "required": ["dummy_data"],
    "additionalProperties": False,
}
# JSON Schema describing the expected structure of the generated UI presentation.
generation_presentation_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "GeneratedPresentation",
    "type": "object",
    "properties": {
        "html": {
            "type": "object",
            "description": "HTML output. Either a full `page` (document) or a `snippet` (embed) is acceptable.",
            "properties": {
                "page": {
                    "type": "string",
                    "description": 'Complete HTML document as a string. Do NOT inline snippet or script content. Use <!-- include:snippet --> where the snippet should render, and include a <script type="module"> block containing <!-- include:service_script --> and <!-- include:components_script -->.',
                },
                "snippet": {
                    "type": "string",
                    "description": 'HTML snippet as a string. Should not include <html> or <body> tags. Do NOT inline scripts; use <!-- include:service_script --> and <!-- include:components_script --> inside a <script type="module"> block.',
                },
            },
            "required": ["page", "snippet"],
            "additionalProperties": False,
        },
        "metadata": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "scope": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "id": {"type": "string"},
                    },
                    "required": ["type", "id"],
                    "additionalProperties": False,
                },
                "requirements": {"type": "string"},
                "original_requirements": {"type": "string"},
                "components": {"type": "array", "items": {"type": "string"}},
                "pfusch_components": {"type": "array", "items": {"type": "string"}},
                "created_by": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
                "history": {"type": "array", "items": {"type": "object"}},
            },
            "additionalProperties": True,
        },
    },
    "required": ["html"],
    "additionalProperties": True,
}

# Legacy schema for fallback/compatibility
generation_ui_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "GeneratedUi",
    "type": "object",
    "properties": {
        **generation_logic_schema["properties"],
        **generation_presentation_schema["properties"],
    },
    "required": ["html"],
    "additionalProperties": True,
}


def generation_response_format(
    schema: Optional[Dict[str, Any]] = None, name: str = "generated_ui"
) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema or generation_ui_schema,
        },
    }
