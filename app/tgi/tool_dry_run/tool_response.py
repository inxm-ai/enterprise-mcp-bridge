import os

from app.json.schema_validation import validate_schema
from app.session.session import MCPSessionBase
from app.tgi.clients.llm_client import LLMClient
from app.tgi.models.models import ChatCompletionRequest
from app.tgi.services.prompt_service import PromptService


def get_tool_dry_run_response(
    session: MCPSessionBase, tool: dict, tool_input: dict
) -> dict:
    if not os.environ.get("TGI_URL", None):
        raise ValueError(
            "TGI_URL environment variable is not set, cannot create dry run response."
        )

    input_schema = tool.get("inputSchema", None)
    if input_schema:
        validate_schema(input_schema, tool_input)
    output_schema = tool.get("outputSchema", None)

    prompts = PromptService()
    prompt = prompts.find_prompt_by_name_or_role(
        session, f"dryrun_{tool.get('name', 'unknown')}"
    )
    if not prompt:
        prompt = prompts.find_prompt_by_name_or_role(session, "dryrun_default")
    if not prompt:
        prompt = "You are a helpful assistant that provides mock responses for tools."

    request = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Provide a mock response for the tool '{tool.get('name', 'unknown')}' with the following input: {tool_input}. ",
            },
        ],
    )
    if output_schema:
        request.response_format = {
            "type": "json_schema",
            "json_schema": output_schema,
        }
    client = LLMClient()
    client.stream_completion(request, None, None)
    pass
