from app.session.session import mcp_session
from app.session_manager.session_context import map_tools
from app.tgi.models import ChatCompletionRequest
from opentelemetry import trace
from app.tgi.llm_client import LLMClient
from app.vars import (
    HOST,
    MCP_BASE_PATH,
    PORT,
    SERVICE_NAME,
    OAUTH_ENV,
    DEFAULT_MODEL,
    AGENT_CARD_CACHE_FILE,
)
import json
import os
import tempfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()
tracer = trace.get_tracer(__name__)
_agent_card_cache = None
_running = False


def _load_agent_card_from_file() -> dict | None:
    """Load agent card from AGENT_CARD_CACHE_FILE if present and valid."""
    path = AGENT_CARD_CACHE_FILE
    if not path:
        return None
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        # If the file is corrupt or unreadable, ignore and regenerate
        return None


def _save_agent_card_to_file(card: dict) -> None:
    """Atomically write the agent card JSON to AGENT_CARD_CACHE_FILE."""
    path = AGENT_CARD_CACHE_FILE
    if not path:
        return
    dirpath = os.path.dirname(path) or "."
    try:
        os.makedirs(dirpath, exist_ok=True)
        # write to a temp file then rename
        fd, tmp_path = tempfile.mkstemp(dir=dirpath)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(card, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    except Exception:
        # Do not fail agent creation if we cannot write the cache
        return


def get_description(reply: str) -> str:
    return reply.strip(" \"'")


def get_as_list(reply: str) -> list[str]:
    if not reply:
        return []
    lines = reply.split("\n")
    items = []
    current = None
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("-") or stripped.startswith("*"):
            content = stripped[1:].strip()
            if current is not None:
                items.append(current)
            current = content
        elif current is not None:
            if line.strip() == "":
                current += "\n"
            else:
                current += "\n" + line
    if current is not None:
        items.append(current)
    return [item.strip() for item in items]


@router.get("/.well-known/agent.json")
async def get_agent_card():
    global _agent_card_cache
    global _running
    with tracer.start_as_current_span("get_agent_card") as span:
        if not DEFAULT_MODEL:
            span.set_attribute("agent_card.available", False)
            raise HTTPException(status_code=404, detail="Not Found")

        span.set_attribute("agent_card.available", True)
        # try load from in-memory cache first
        if _agent_card_cache is not None:
            span.set_attribute("agent_card.cache.hit", True)
            return JSONResponse(content=_agent_card_cache)

        if _running:
            raise HTTPException(
                status_code=429, detail="Agent card generation in progress"
            )

        _running = True
        span.set_attribute("agent_card.cache.hit", False)

        llm = LLMClient()

        async with mcp_session(anon=True) as session:
            tools = map_tools(await session.list_tools())

            tools_descriptions = [tool["description"] for tool in tools]

            base_request = ChatCompletionRequest(model=DEFAULT_MODEL, messages=[])

            summary = await llm.ask(
                base_prompt="You are an MCP concise summarization engine. Summarize the following content, suitable for a service description, giving a good overview of the different tools and capabilities. Return only the summary.",
                question="Summarize the following description: "
                + "\n".join(tools_descriptions),
                base_request=base_request,
                outer_span=span,
            )

            # Base agent card
            agent_card = {
                "name": SERVICE_NAME,
                "description": get_description(summary),
                "url": f"https://{HOST}{':' + str(PORT) if PORT else ''}{MCP_BASE_PATH}/tgi/v1/a2a",
                "version": "1.0.0",
                "capabilities": {"streaming": True, "pushNotifications": False},
                "authentication": {"schemes": ["bearer"] if OAUTH_ENV else []},
                "defaultInputModes": ["data"],
                "defaultOutputModes": ["text", "data"],
                "skills": [],
            }

            # Add default skill
            default_skill_description = await llm.ask(
                base_prompt="You are an Agent capability description summarizer. Based on the user's question, create a description of the agent's capabilities. Return only the description, do not return any other text. Call it Agent.",
                question=f"What can this do: {summary}. tools_descriptions: {tools_descriptions}",
                base_request=base_request,
                outer_span=span,
            )
            default_examples = await llm.ask(
                base_prompt="You are an Agent prompt example generator. Based on the user's question, create a few example prompts for how the agent can be used. Create the examples as unordered markdown list, and only return the examples.",
                question=f"Generate examples for: {default_skill_description}.",
                base_request=base_request,
                outer_span=span,
            )

            default_examples = get_as_list(default_examples)
            default_skill = {
                "id": SERVICE_NAME.lower().replace(" ", "-"),
                "name": SERVICE_NAME,
                "description": get_description(default_skill_description),
                "tags": ["default"],
                "examples": default_examples,
                "inputModes": ["text"],
                "outputModes": ["text"],
                "parameter_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The input prompt for the skill.",
                        }
                    },
                    "required": ["prompt"],
                },
            }
            agent_card["skills"].append(default_skill)

            # Add skills from MCP tools
            for index, tool in enumerate(tools):
                tool_tags = await llm.ask(
                    base_prompt="You are an Agent tool tag generator. Based on the tool description, generate a list of tags that are relevant to the tool. Create the examples as unordered markdown list, and only return the tags.",
                    question=f"Generate tags for: {tool['description']}.",
                    base_request=base_request,
                    outer_span=span,
                )
                if "inputSchema" in tool:
                    tool_examples = await llm.ask(
                        base_prompt=r"You are an Agent prompt example generator. Based on the tool description and inputSchema, generate a few examples (show the example json) for how the tool can be used. If the properties are empty, just return a single example with `\{\}` Create the examples as unordered markdown list, and only return the examples.",
                        question=f"Generate examples for: {tool['description']}\n\nInput Schema:\n```json\n{tool['inputSchema']}\n```",
                        base_request=base_request,
                        outer_span=span,
                    )
                else:
                    tool_examples = await llm.ask(
                        base_prompt="You are an Agent prompt example generator. Based on the tool description, generate a few example prompts for how the tool can be used. Create the examples as unordered markdown list, and only return the examples.",
                        question=f"Generate examples for: {tool['description']}.",
                        base_request=base_request,
                        outer_span=span,
                    )
                tool_examples = get_as_list(tool_examples)

                skill = {
                    "id": tool["id"] if "id" in tool else index,
                    "name": tool["name"],
                    "description": tool["description"],
                    "tags": get_as_list(tool_tags),
                    "examples": tool_examples if tool_examples else [],
                    "inputModes": ["data"] if "inputSchema" in tool else ["text"],
                    "outputModes": ["data"] if "outputSchema" in tool else ["text"],
                }
                if "inputSchema" in tool:
                    skill["parameter_schema"] = tool["inputSchema"]
                agent_card["skills"].append(skill)

            _agent_card_cache = agent_card

            try:
                _save_agent_card_to_file(agent_card)
            except Exception:
                pass
            _running = False
            return JSONResponse(content=agent_card)


# Try to populate the in-memory cache from disk
try:
    _agent_card_cache = _load_agent_card_from_file()
except Exception:
    _agent_card_cache = None
