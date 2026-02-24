import json
import copy
import asyncio
from typing import Any, Dict, List, Optional
import logging

from fastapi import HTTPException
from genson import SchemaBuilder
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.app_facade.generated_schemas import generation_response_format

logger = logging.getLogger("uvicorn.error")

DUMMY_DATA_SYSTEM_PROMPT = (
    "You are an expert software engineer generating realistic test data. "
    "Your task is to generate dummy data for the provided tools. "
    "The output must strictly follow the JSON schema derived from the tools' output schemas "
    "and represent the resolved value returned by the service helper."
)

SCHEMA_DERIVATION_SYSTEM_PROMPT = (
    "You are a strict JSON Schema expert. "
    "Given observed tool output data, infer a JSON Schema Draft-07 compatible schema "
    "for the resolved service value returned to callers. "
    "Return schema fields from this subset only: "
    "type, properties, required, items, additionalProperties, enum."
)

MISSING_OUTPUT_SCHEMA_FALLBACK = {
    "anyOf": [
        {"type": "object", "additionalProperties": True},
        {
            "type": "array",
            "items": {
                "type": ["object", "array", "string", "number", "boolean", "null"]
            },
        },
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "null"},
    ]
}

SCHEMA_DERIVATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "object",
                        "array",
                        "string",
                        "number",
                        "integer",
                        "boolean",
                        "null",
                    ],
                },
                "properties": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "object",
                                    "array",
                                    "string",
                                    "number",
                                    "integer",
                                    "boolean",
                                    "null",
                                ],
                            },
                            "enum": {"type": "array", "items": {"type": "string"}},
                            "items": {"type": "object", "additionalProperties": True},
                            "properties": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                            },
                            "required": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "additionalProperties": {"type": "boolean"},
                        },
                        "additionalProperties": True,
                    },
                },
                "required": {"type": "array", "items": {"type": "string"}},
                "items": {"type": "object", "additionalProperties": True},
                "additionalProperties": {"type": "boolean"},
                "enum": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": True,
        }
    },
    "required": ["schema"],
    "additionalProperties": False,
}


class DummyDataGenerator:
    def __init__(self, tgi_service: Any):
        self.tgi_service = tgi_service

    @staticmethod
    def _json_type_for_value(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    @staticmethod
    def _value_matches_type(value: Any, declared_type: str) -> bool:
        actual_type = DummyDataGenerator._json_type_for_value(value)
        if declared_type == "number" and actual_type in {"number", "integer"}:
            return True
        if declared_type == "integer" and actual_type == "integer":
            return True
        return actual_type == declared_type

    @staticmethod
    def _preferred_type_from_union(values: List[Any]) -> Optional[str]:
        normalized = [value for value in values if isinstance(value, str)]
        if not normalized:
            return None
        for preferred in (
            "object",
            "array",
            "string",
            "number",
            "integer",
            "boolean",
            "null",
        ):
            if preferred in normalized:
                return preferred
        return normalized[0]

    @classmethod
    def _sanitize_output_schema(cls, schema: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(schema, dict) or not schema:
            return None

        def walk(node: Any) -> Any:
            if not isinstance(node, dict):
                return node

            current: Dict[str, Any] = copy.deepcopy(node)

            properties = current.get("properties")
            if isinstance(properties, dict):
                current["properties"] = {
                    str(key): walk(value) for key, value in properties.items()
                }

            if "items" in current and isinstance(current.get("items"), dict):
                current["items"] = walk(current["items"])

            if "additionalProperties" in current and isinstance(
                current.get("additionalProperties"), dict
            ):
                current["additionalProperties"] = walk(current["additionalProperties"])

            for keyword in ("anyOf", "oneOf", "allOf"):
                branch = current.get(keyword)
                if isinstance(branch, list):
                    current[keyword] = [walk(item) for item in branch]

            node_type = current.get("type")
            if isinstance(node_type, list):
                chosen = cls._preferred_type_from_union(node_type)
                current["type"] = chosen or "string"
            elif node_type is not None and not isinstance(node_type, str):
                current["type"] = "string"

            if "type" not in current:
                if any(
                    key in current
                    for key in ("properties", "required", "additionalProperties")
                ):
                    current["type"] = "object"
                elif "items" in current:
                    current["type"] = "array"
                elif isinstance(current.get("enum"), list):
                    enum_types = {
                        cls._json_type_for_value(enum_value)
                        for enum_value in current.get("enum") or []
                    }
                    if len(enum_types) == 1:
                        current["type"] = next(iter(enum_types))

            declared_type = current.get("type")
            enum_values = current.get("enum")
            if isinstance(declared_type, str) and isinstance(enum_values, list):
                filtered_enum = [
                    enum_value
                    for enum_value in enum_values
                    if cls._value_matches_type(enum_value, declared_type)
                ]
                if filtered_enum:
                    current["enum"] = filtered_enum
                else:
                    current.pop("enum", None)

            declared_type = current.get("type")
            if declared_type == "object":
                object_properties = current.get("properties")
                if not isinstance(object_properties, dict):
                    object_properties = {}
                    current["properties"] = object_properties
                required_props = current.get("required")
                if isinstance(required_props, list):
                    current["required"] = [
                        item
                        for item in required_props
                        if isinstance(item, str) and item in object_properties
                    ]
                current.pop("items", None)
            elif declared_type == "array":
                if not isinstance(current.get("items"), dict):
                    current["items"] = {"type": "string"}
                current.pop("properties", None)
                current.pop("required", None)
                current.pop("additionalProperties", None)
            elif isinstance(declared_type, str):
                current.pop("properties", None)
                current.pop("required", None)
                current.pop("additionalProperties", None)
                current.pop("items", None)

            return current

        sanitized = walk(schema)
        return sanitized if isinstance(sanitized, dict) and sanitized else None

    def _build_dummy_data_schema(
        self, tool_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Builds a JSON schema for the dummy data response based on the output schemas of the tools.
        """
        properties = {}
        for tool in tool_specs:
            name = tool.get("name")
            output_schema = tool.get("outputSchema")

            if not name:
                continue

            if isinstance(output_schema, dict) and output_schema:
                properties[name] = (
                    self._sanitize_output_schema(output_schema) or output_schema
                )
            else:
                properties[name] = copy.deepcopy(MISSING_OUTPUT_SCHEMA_FALLBACK)

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "GeneratedDummyData",
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "required": list(properties.keys()),
        }

    def _convert_to_js_module(
        self, data: Dict[str, Any], schema_hints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Converts the JSON data dictionary to a Javascript module string.
        """
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        hints_str = json.dumps(schema_hints or {}, indent=2, ensure_ascii=False)
        return (
            f"export const dummyData = {json_str};\n"
            f"export const dummyDataSchemaHints = {hints_str};"
        )

    def _infer_schema_from_sample(
        self,
        *,
        sample: Any,
        tool_name: str = "unknown_tool",
    ) -> Optional[Dict[str, Any]]:
        """Infer JSON schema from a sample value using genson.

        This is much faster than asking an LLM to derive the schema,
        since genson infers the schema instantly from the actual data.
        """
        try:
            # Use genson to infer schema from the sample data
            builder = SchemaBuilder()
            builder.add_object(sample)
            inferred = builder.to_schema()

            if isinstance(inferred, dict) and inferred:
                # Genson doesn't infer required fields, so we explicitly mark all
                # properties as required (since they're present in the sample).
                # This prevents responses mode from doing its own normalization,
                # which would cause a mismatch with what the LLM generates.
                inferred = self._add_required_fields_to_schema(inferred)

                # Sanitize and validate the inferred schema
                sanitized = self._sanitize_output_schema(inferred)
                return sanitized or inferred
        except Exception as exc:
            logger.warning(
                "Failed to infer schema from sample for tool '%s': %s",
                tool_name,
                exc,
            )
        return None

    @classmethod
    def _add_required_fields_to_schema(cls, schema: Any) -> Any:
        """Recursively add required fields to all objects in the schema.

        Genson infers the structure but doesn't mark fields as required.
        We explicitly mark all properties as required (since they exist in the sample)
        to ensure responses mode doesn't alter the schema unexpectedly.
        """
        if not isinstance(schema, dict):
            return schema

        result = copy.deepcopy(schema)

        # Add required array if this is an object with properties
        if result.get("type") == "object" and "properties" in result:
            properties = result.get("properties", {})
            if isinstance(properties, dict) and properties:
                result["required"] = list(properties.keys())
            # Ensure strict mode
            if "additionalProperties" not in result:
                result["additionalProperties"] = False

        # Recursively process nested properties
        if "properties" in result and isinstance(result["properties"], dict):
            result["properties"] = {
                key: cls._add_required_fields_to_schema(value)
                for key, value in result["properties"].items()
            }

        # Recursively process array items
        if "items" in result and isinstance(result["items"], dict):
            result["items"] = cls._add_required_fields_to_schema(result["items"])

        return result

    def _derive_schema_from_sample(
        self,
        *,
        tool_spec: Dict[str, Any],
        sample: Any,
        ui_model_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Derive schema from sample using fast genson inference instead of LLM.

        Previously this made an LLM call which was slow. Now we use automatic
        schema inference from the actual sample data, which is instant.
        """
        tool_name = str(tool_spec.get("name") or "unknown_tool")

        # Use fast schema inference instead of LLM
        schema = self._infer_schema_from_sample(
            sample=sample,
            tool_name=tool_name,
        )

        if schema:
            return schema

        # Fallback: return None if inference fails, parent will use MISSING_OUTPUT_SCHEMA_FALLBACK
        logger.warning(
            "Could not infer schema from sample for tool '%s'",
            tool_name,
        )
        return None

    async def _enrich_tool_specs_with_derived_schemas(
        self,
        *,
        tool_specs: List[Dict[str, Any]],
        ui_model_headers: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        async def process_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
            spec_copy = copy.deepcopy(spec)
            output_schema = spec_copy.get("outputSchema")
            if isinstance(output_schema, dict) and output_schema:
                spec_copy["outputSchema"] = (
                    self._sanitize_output_schema(output_schema) or output_schema
                )
                output_schema = spec_copy.get("outputSchema")
            observed = spec_copy.get("sampleResolvedValue")
            if observed is None:
                observed = spec_copy.get("sampleStructuredContent")
            schema_status = "provided_output_schema"
            if not output_schema and observed is not None:
                derived_schema = self._derive_schema_from_sample(
                    tool_spec=spec_copy,
                    sample=observed,
                    ui_model_headers=ui_model_headers,
                )
                if derived_schema:
                    spec_copy["outputSchema"] = derived_schema
                    schema_status = "derived_output_schema"
                else:
                    schema_status = "missing_output_schema"
            elif not output_schema:
                schema_status = "missing_output_schema"
            spec_copy["_schema_status"] = schema_status
            return spec_copy

        return await asyncio.gather(*(process_spec(spec) for spec in tool_specs))

    def _normalize_tool_value(self, value: Any) -> Any:
        if isinstance(value, dict) and "structuredContent" in value:
            return value.get("structuredContent")
        return value

    def _is_error_observed_sample(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        if value.get("_dummy_data_error") is True:
            return True
        if value.get("isError") is True:
            return True
        explicit_error = value.get("error")
        if isinstance(explicit_error, (str, dict, list)) and explicit_error:
            return True
        return False

    def _normalize_payload_for_tools(
        self, payload: Dict[str, Any], tool_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        normalized: Dict[str, Any] = {}
        for spec in tool_specs:
            tool_name = spec.get("name")
            if not tool_name or tool_name not in payload:
                continue
            normalized[tool_name] = self._normalize_tool_value(payload.get(tool_name))
        return normalized

    def _apply_observed_samples(
        self, payload: Dict[str, Any], tool_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        for spec in tool_specs:
            tool_name = spec.get("name")
            observed = spec.get("sampleResolvedValue")
            if observed is None:
                observed = spec.get("sampleStructuredContent")
            if not tool_name or observed is None:
                continue
            if self._is_error_observed_sample(observed):
                continue
            payload[tool_name] = observed
        return payload

    def _build_schema_hints(self, tool_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        hints: Dict[str, Any] = {}
        for spec in tool_specs:
            if spec.get("_schema_status") != "missing_output_schema":
                continue
            tool_name = spec.get("name")
            if not tool_name:
                continue
            hints[tool_name] = {
                "schema_status": "missing_output_schema",
                "next_action": "ask_for_schema_then_regenerate_dummy_data",
                "fallback_mode": "llm_generated_any_schema",
                "tool": tool_name,
            }
        return hints

    def _parse_dummy_data_response(
        self, *, tool_name: str, content: str
    ) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            stripped = content.strip()
            try:
                parsed, end_idx = json.JSONDecoder().raw_decode(stripped)
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse dummy data JSON for {tool_name}: {exc}. Content: {content}"
                )
                return None

            trailing = stripped[end_idx:].strip()
            if trailing and any(char != "}" for char in trailing):
                logger.error(
                    "Failed to parse dummy data JSON for %s: unrecoverable trailing content after JSON object. Content: %s",
                    tool_name,
                    content,
                )
                return None
            if trailing:
                logger.warning(
                    "Recovered dummy data JSON for %s by trimming %d trailing brace character(s)",
                    tool_name,
                    len(trailing),
                )

        if not isinstance(parsed, dict):
            logger.error(
                "Dummy data response payload for %s must be an object. Content: %s",
                tool_name,
                content,
            )
            return None
        return parsed

    async def generate_dummy_data(
        self,
        *,
        prompt: str,
        tool_specs: List[Dict[str, Any]],
        ui_model_headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generates dummy data for the given tool specifications.
        """
        if not tool_specs:
            return self._convert_to_js_module({}, {})

        enriched_tool_specs = await self._enrich_tool_specs_with_derived_schemas(
            tool_specs=tool_specs,
            ui_model_headers=ui_model_headers,
        )

        non_stream = getattr(self.tgi_service.llm_client, "non_stream_completion", None)
        if not callable(non_stream):
            raise HTTPException(
                status_code=502,
                detail="LLM client does not support dummy data generation",
            )

        async def generate_for_tool(tool_spec: Dict[str, Any]) -> Dict[str, Any]:
            tool_name = tool_spec["name"]
            schema = self._build_dummy_data_schema([tool_spec])

            user_message_content = json.dumps(
                {
                    "instruction": f"Generate dummy data for the tool: {tool_name}.",
                    "original_prompt": prompt,
                    "tools_context": [tool_spec],
                    "requirement": (
                        "Ensure the data is realistic and conforms exactly to the output schema of the tool. "
                        "When sampleResolvedValue or sampleStructuredContent is present, "
                        "prefer those values directly."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )

            messages = [
                Message(role=MessageRole.SYSTEM, content=DUMMY_DATA_SYSTEM_PROMPT),
                Message(role=MessageRole.USER, content=user_message_content),
            ]

            chat_request = ChatCompletionRequest(
                messages=messages,
                tools=None,
                stream=False,
                response_format=generation_response_format(
                    schema=schema,
                    name=f"generated_dummy_data_{tool_name}",
                ),
                extra_headers=ui_model_headers,
            )

            try:
                response = await non_stream(chat_request, "", None)
            except Exception as e:
                logger.error(f"Failed to generate dummy data for {tool_name}: {e}")
                return {}

            if not response.choices:
                logger.error(f"Dummy data response choices were empty for {tool_name}")
                return {}

            choice = response.choices[0]
            content = choice.message.content or ""

            if not content and choice.message.tool_calls:
                content = choice.message.tool_calls[0].function.arguments

            if not content:
                logger.error(f"Dummy data response content was empty for {tool_name}")
                return {}

            parsed = self._parse_dummy_data_response(
                tool_name=tool_name, content=content
            )
            if parsed is None:
                return {}
            return parsed

        results = await asyncio.gather(
            *(generate_for_tool(spec) for spec in enriched_tool_specs)
        )

        payload = {}
        for res in results:
            payload.update(res)

        payload = self._normalize_payload_for_tools(payload, enriched_tool_specs)
        payload = self._apply_observed_samples(payload, enriched_tool_specs)
        schema_hints = self._build_schema_hints(enriched_tool_specs)

        return self._convert_to_js_module(payload, schema_hints)
