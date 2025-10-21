import os
import json
from types import SimpleNamespace
from typing import Any

from app.json.schema_validation import validate_schema
from app.session.session import MCPSessionBase
from app.tgi.clients.llm_client import LLMClient
from app.tgi.models.models import ChatCompletionRequest
from app.tgi.services.prompt_service import PromptService
from app.tgi.protocols.chunk_reader import chunk_reader


async def get_tool_dry_run_response(
    session: MCPSessionBase, tool: dict, tool_input: dict
) -> Any:
    """
    Create a dry-run response for a tool by streaming a mock response from the LLM,
    aggregating the chunks and returning an MCP-style result object.

    Behavior:
    - If the tool declares an outputSchema, the stream is expected to yield a
      JSON payload (possibly streamed). The function will attempt to parse the
      aggregated stream into structured JSON and return it in
      `structuredContent`.
    - If no outputSchema is declared, the stream is aggregated as plain text and
      returned in the `content` list as a single entry with a `text` field.
    """

    if not os.environ.get("TGI_URL", None):
        raise ValueError(
            "TGI_URL environment variable is not set, cannot create dry run response."
        )

    input_schema = tool.get("inputSchema", None)
    if input_schema:
        try:
            validate_schema(input_schema, tool_input)
        except Exception as exc:
            # Failed to parse structured JSON
            return SimpleNamespace(
                isError=True,
                content=[
                    SimpleNamespace(
                        text=f"Failed to validate input against schema: {exc}"
                    )
                ],
                structuredContent=None,
            )
    output_schema = tool.get("outputSchema", None)

    prompts = PromptService()
    # PromptService APIs are async; be tolerant and await if coroutine returned.
    prompt = await prompts.find_prompt_by_name_or_role(
        session, f"dryrun_{tool.get('name', 'unknown')}"
    )
    if not prompt:
        prompt = await prompts.find_prompt_by_name_or_role(session, "dryrun_default")
    if not prompt:
        prompt = "You are a helpful assistant that provides mock responses for tools."

    # If prompt is an object (dict/model), try to extract textual content
    if not isinstance(prompt, str) and hasattr(prompts, "get_prompt_content"):
        try:
            prompt = await prompts.get_prompt_content(session, prompt)
        except Exception:
            # Best-effort only; fall back to whatever prompt value we have
            try:
                prompt = str(prompt)
            except Exception:
                prompt = "You are a helpful assistant that provides mock responses for tools."

    request = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Provide a mock response for the tool '{tool.get('name', 'unknown')}' with the following input: {tool_input}. ",
            },
        ],
        stream=True,
    )
    if output_schema:
        request.response_format = {
            "type": "json_schema",
            "json_schema": output_schema,
        }

    client = LLMClient()

    stream_source = client.stream_completion(request, None, None)

    # Aggregate stream into either structured JSON or plain text
    aggregated = ""
    async with chunk_reader(stream_source) as reader:
        async for parsed in reader.as_parsed():
            if parsed.is_done:
                break
            if parsed.content:
                aggregated += parsed.content
            elif parsed.parsed:
                try:
                    aggregated += json.dumps(parsed.parsed, ensure_ascii=False)
                except Exception:
                    aggregated += str(parsed.parsed)

    aggregated = aggregated.strip()

    # Build MCP-style result object
    if output_schema:
        # Expecting JSON object/array as structured result
        if not aggregated:
            return SimpleNamespace(isError=True, content=[], structuredContent=None)

        try:
            decoder = json.JSONDecoder()
            parsed_obj, idx = decoder.raw_decode(aggregated)
            return SimpleNamespace(
                isError=False, content=[], structuredContent=parsed_obj
            )
        except Exception as exc:
            # Failed to parse structured JSON
            return SimpleNamespace(
                isError=True,
                content=[
                    SimpleNamespace(text=f"Failed to parse structured result: {exc}")
                ],
                structuredContent=None,
            )

    # No output schema: return the aggregated text as a chat/plain response
    return SimpleNamespace(
        isError=False,
        content=[SimpleNamespace(text=aggregated)],
        structuredContent=None,
    )
