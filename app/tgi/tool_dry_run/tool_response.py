import asyncio
import logging
import os
import json
import re
from types import SimpleNamespace
from typing import Any, Optional
from fastapi import HTTPException

from app.json.schema_validation import validate_schema
from app.session.session import MCPSessionBase
from app.session_manager.session_context import map_tools
from app.tgi.clients.llm_client import LLMClient
from app.tgi.models.models import ChatCompletionRequest
from app.tgi.services.prompt_service import PromptService
from app.tgi.protocols.chunk_reader import chunk_reader
from app.utils.effect_tools import dry_run_effect
from app.vars import SIMULATED_META_KEY, is_dry_run_requested

logger = logging.getLogger("uvicorn.error")


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

    def _normalize_schema_name(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip()).strip("_")
        return cleaned or "tool_response"

    prompts = PromptService()
    default_prompt = (
        "You are a helpful assistant that provides mock responses for tools."
    )
    # PromptService APIs are async; be tolerant and await if coroutine returned.
    # Prompt lookup depends on the downstream MCP server's list_prompts, which may
    # error out for servers that don't implement it cleanly. Never let that fail
    # the dry run - fall back to the default mock-response prompt instead.
    try:
        prompt = await prompts.find_prompt_by_name_or_role(
            session, f"dryrun_{tool.get('name', 'unknown')}"
        )
        if not prompt:
            prompt = await prompts.find_prompt_by_name_or_role(
                session, "dryrun_default"
            )
    except Exception as exc:
        logger.warning(
            f"[DryRun] Failed to look up dry-run prompt for tool "
            f"'{tool.get('name', 'unknown')}', falling back to default: {exc}"
        )
        prompt = None
    if not prompt:
        prompt = default_prompt

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
                "content": f"Provide a mock response for the tool '{tool.get('name', 'unknown')}'\nDescription: {tool.get('description', 'No description available')}\nInput:\n```json\n{tool_input}\n```\n",
            },
        ],
        stream=True,
    )
    if output_schema:
        schema_name = _normalize_schema_name(tool.get("name") or "tool_response")
        request.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": output_schema,
            },
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


async def resolve_dry_run_effect_call(
    session: MCPSessionBase,
    dry_run_header: Optional[str],
    tool_name: str,
    effect_tools: list[str],
    simulate_header: Optional[str] = None,
) -> tuple[bool, Optional[list[dict]]]:
    """Decide whether a tool call is answered with a dry-run result.

    Unknown effects cannot become live calls. Explicit simulation can also
    mock reads when the caller's arguments came from simulated data.
    """
    if dry_run_header is not None and dry_run_header.lower() not in {"true", "false"}:
        raise HTTPException(400, "Invalid X-Inxm-Dry-Run")
    if not is_dry_run_requested(dry_run_header):
        return False, None
    tools = map_tools(await session.list_tools())
    tool_def = next((tool for tool in tools if tool.get("name") == tool_name), None)
    if tool_def is None:
        raise HTTPException(404, "Tool not found")
    if is_dry_run_requested(simulate_header):
        return True, tools
    effect = dry_run_effect(tool_name, tool_def, effect_tools)
    if effect is None:
        raise HTTPException(422, "Dry run unsupported: tool effect is unknown")
    return effect, tools


async def dry_run_tool_result(
    session: MCPSessionBase,
    tool_name: str,
    tool_input: dict,
    tools: Optional[list[dict]] = None,
) -> Any:
    """Answer a tool call with a simulated result instead of executing the tool.

    ``tools`` is an already mapped tool listing to reuse instead of listing.
    """
    if tools is None:
        tools = map_tools(await session.list_tools())
    tool = next((tool for tool in tools if tool.get("name") == tool_name), None)
    # Tests and other callsites may patch get_tool_dry_run_response with a
    # sync function, so await only when a coroutine comes back.
    maybe_result = get_tool_dry_run_response(session, tool, tool_input)
    if asyncio.iscoroutine(maybe_result):
        maybe_result = await maybe_result
    return mark_simulated(maybe_result)


def mark_simulated(result: Any) -> Any:
    """Mark a simulated answer in its MCP `_meta`, so a caller can tell a
    made-up result (and the ids in it) from a real one, also on a stream."""
    if isinstance(result, dict):
        meta = result.get("_meta")
        result["_meta"] = {
            **(meta if isinstance(meta, dict) else {}),
            SIMULATED_META_KEY: True,
        }
        return result
    if "meta" in getattr(type(result), "model_fields", {}):
        # An MCP SDK result: its `meta` field is sent as `_meta`.
        meta = getattr(result, "meta", None)
        result.meta = {
            **(meta if isinstance(meta, dict) else {}),
            SIMULATED_META_KEY: True,
        }
        return result
    try:
        meta = getattr(result, "_meta", None)
        setattr(
            result,
            "_meta",
            {**(meta if isinstance(meta, dict) else {}), SIMULATED_META_KEY: True},
        )
    except (AttributeError, TypeError):
        pass
    return result
