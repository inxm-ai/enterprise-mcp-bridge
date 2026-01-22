import json
from typing import Any, Dict, List, Optional
import logging

from fastapi import HTTPException
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.app_facade.generated_schemas import generation_response_format

logger = logging.getLogger("uvicorn.error")

DUMMY_DATA_SYSTEM_PROMPT = (
    "You are an expert software engineer generating realistic test data. "
    "Your task is to generate dummy data for the provided tools. "
    "The output must strictly follow the JSON schema derived from the tools' output schemas."
)


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

            if output_schema:
                properties[name] = {
                    "type": "object",
                    "properties": {"structuredContent": output_schema},
                    "required": ["structuredContent"],
                    "additionalProperties": False,
                }
            else:
                properties[name] = {
                    "type": "object",
                    "properties": {"structuredContent": {"type": "null"}},
                    "required": ["structuredContent"],
                    "additionalProperties": False,
                }

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "GeneratedDummyData",
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "required": list(properties.keys()),
        }

    def _convert_to_js_module(self, data: Dict[str, Any]) -> str:
        """
        Converts the JSON data dictionary to a Javascript module string.
        """
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        return f"export const dummyData = {json_str};"

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
            return "export const dummyData = {};\n"

        schema = self._build_dummy_data_schema(tool_specs)
        tool_names = [spec["name"] for spec in tool_specs]

        # Construct a detailed prompt including tool definitions to help the LLM understand context unique constraints
        # even if the structure is enforced by schema.
        user_message_content = json.dumps(
            {
                "instruction": f"Generate dummy data for the following tools: {', '.join(tool_names)}.",
                "original_prompt": prompt,
                "tools_context": tool_specs,
                "requirement": "Ensure the data is realistic and conforms exactly to the output schema of each tool.",
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
                name="generated_dummy_data",
            ),
            extra_headers=ui_model_headers,
        )

        if not getattr(self.tgi_service.llm_client, "client", None):
            raise HTTPException(
                status_code=502,
                detail="LLM client does not support dummy data generation",
            )

        try:
            response = await self.tgi_service.llm_client.client.chat.completions.create(
                **self.tgi_service.llm_client._build_request_params(chat_request)
            )
        except Exception as e:
            logger.error(f"Failed to generate dummy data: {e}")
            raise HTTPException(
                status_code=502, detail=f"LLM generation failed: {str(e)}"
            )

        if not response.choices:
            raise HTTPException(
                status_code=502, detail="Dummy data response choices were empty"
            )

        choice = response.choices[0]
        content = choice.message.content or ""

        if not content and choice.message.tool_calls:
            content = choice.message.tool_calls[0].function.arguments

        if not content:
            raise HTTPException(
                status_code=502, detail="Dummy data response content was empty"
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse dummy data JSON: {exc}. Content: {content}")
            raise HTTPException(
                status_code=502, detail=f"Failed to parse dummy data JSON: {exc}"
            )

        return self._convert_to_js_module(payload)
