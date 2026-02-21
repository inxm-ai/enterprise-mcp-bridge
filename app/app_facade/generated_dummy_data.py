import json
import copy
import asyncio
from typing import Any, Dict, List, Optional
import logging

from fastapi import HTTPException
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
    "for the resolved service value returned to callers."
)

MISSING_OUTPUT_SCHEMA_FALLBACK = {
    "anyOf": [
        {"type": "object", "additionalProperties": True},
        {"type": "array", "items": {}},
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
                "type": {"type": "string"},
                "properties": {"type": "object"},
                "required": {"type": "array", "items": {"type": "string"}},
                "items": {"type": ["object", "boolean"]},
                "additionalProperties": {"type": ["object", "boolean"]},
                "oneOf": {"type": "array"},
                "anyOf": {"type": "array"},
                "allOf": {"type": "array"},
                "enum": {"type": "array"},
            },
            "required": ["type"],
            "additionalProperties": True,
        }
    },
    "required": ["schema"],
    "additionalProperties": False,
}


class DummyDataGenerator:
    def __init__(self, tgi_service: Any):
        self.tgi_service = tgi_service

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
                properties[name] = output_schema
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

    async def _derive_schema_from_sample(
        self,
        *,
        tool_spec: Dict[str, Any],
        sample: Any,
        ui_model_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not getattr(self.tgi_service.llm_client, "client", None):
            return None

        tool_name = str(tool_spec.get("name") or "unknown_tool")
        tool_context = {
            "name": tool_name,
            "description": tool_spec.get("description"),
            "inputSchema": tool_spec.get("inputSchema"),
            "sampleResolvedValue": sample,
        }
        messages = [
            Message(role=MessageRole.SYSTEM, content=SCHEMA_DERIVATION_SYSTEM_PROMPT),
            Message(
                role=MessageRole.USER,
                content=(
                    "Derive output schema for the resolved service value from this observed tool data:\n"
                    + json.dumps(tool_context, ensure_ascii=False)
                    + "\nReturn only JSON."
                ),
            ),
        ]

        chat_request = ChatCompletionRequest(
            messages=messages,
            tools=None,
            stream=False,
            response_format=generation_response_format(
                schema=SCHEMA_DERIVATION_RESPONSE_SCHEMA,
                name=f"{tool_name}_derived_output_schema",
            ),
            extra_headers=ui_model_headers,
        )

        try:
            response = await self.tgi_service.llm_client.client.chat.completions.create(
                **self.tgi_service.llm_client._build_request_params(chat_request)
            )
        except Exception as exc:
            logger.warning(
                "Failed to derive schema from observed sample for tool '%s': %s",
                tool_name,
                exc,
            )
            return None

        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return None
            choice = choices[0]
            content = getattr(getattr(choice, "message", None), "content", None) or ""
            if not content and getattr(
                getattr(choice, "message", None), "tool_calls", None
            ):
                content = choice.message.tool_calls[0].function.arguments
            parsed = json.loads(content) if content else {}
            schema = parsed.get("schema") if isinstance(parsed, dict) else None
            if isinstance(schema, dict) and schema:
                return schema
        except Exception as exc:
            logger.warning(
                "Failed to parse derived schema for tool '%s': %s",
                tool_name,
                exc,
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
            observed = spec_copy.get("sampleResolvedValue")
            if observed is None:
                observed = spec_copy.get("sampleStructuredContent")
            schema_status = "provided_output_schema"
            if not output_schema and observed is not None:
                derived_schema = await self._derive_schema_from_sample(
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

        if not getattr(self.tgi_service.llm_client, "client", None):
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
                response = (
                    await self.tgi_service.llm_client.client.chat.completions.create(
                        **self.tgi_service.llm_client._build_request_params(
                            chat_request
                        )
                    )
                )
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

            try:
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    logger.error(
                        "Dummy data response payload for %s must be an object. Content: %s",
                        tool_name,
                        content,
                    )
                    return {}
                return parsed
            except json.JSONDecodeError as exc:
                logger.error(
                    f"Failed to parse dummy data JSON for {tool_name}: {exc}. Content: {content}"
                )
                return {}

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
