from dataclasses import dataclass
import json
import logging
import os
from typing import Optional

from fastapi import HTTPException

from app.models import RunPromptResult


logger = logging.getLogger("uvicorn.error")


@dataclass
class Template:
    role: str
    content: str


@dataclass
class Prompt:
    name: str
    title: str
    description: str
    arguments: list[dict]
    template: dict  # Changed from str to dict to match expected structure

    def to_dict(self) -> dict:
        """Convert the prompt to a dictionary."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "arguments": self.arguments,
            "template": self.template,
        }


def default_prompt_list_result(prompts: list[Prompt]) -> dict[str, list[dict]]:
    """Convert prompts to the default MCP list format."""
    return {"prompts": [prompt.to_dict() for prompt in prompts]}


def system_defined_prompts() -> list[Prompt]:
    # System defined prompt looks like this:
    # { "name": "greeting", "title": "Hello You", "description": "Get a personalized greeting.", "arguments": [{ "name": "name" }], "template": "Hello, {name}!" }
    prompts = json.loads(os.environ.get("SYSTEM_DEFINED_PROMPTS", "[]"))
    return [Prompt(**p) for p in prompts]


async def list_prompts(list_prompts: any):
    system_prompts = system_defined_prompts()
    try:
        prompts = await list_prompts()
        prompts.prompts += [prompt.to_dict() for prompt in system_prompts]
    except Exception as e:
        logger.warning(f"[PromptHelper] Error listing prompts: {str(e)}")
        # Not every MCP has list_prompts, so deal with it friendly
        if (
            hasattr(e, "__class__")
            and e.__class__.__name__ == "McpError"
            and "Method not found" in str(e)
        ):
            if len(system_prompts) < 1:
                logger.info("[PromptHelper] No system prompts available")
                raise HTTPException(status_code=404, detail="Method not found")
            else:
                logger.info("[PromptHelper] Returning system prompts")
                logger.debug(f"[PromptHelper] System prompts: {system_prompts}")
                prompts = default_prompt_list_result(system_prompts)
        else:
            raise HTTPException(status_code=500, detail="Loading prompts failed")
    return prompts


async def call_prompt(call: any, prompt_name: str, args: Optional[dict] = []):
    system_prompts = system_defined_prompts()
    prompt = next((p for p in system_prompts if p.name == prompt_name), None)
    if prompt:
        logger.info(f"Using system prompt: {prompt.name}")
        expanded = prompt.template["content"].format(**args)

        class SystemDefinedPromptResult:
            def __init__(self, description, messages):
                self.description = description
                self.messages = messages

        return RunPromptResult(
            SystemDefinedPromptResult(
                description=prompt.description,
                messages=[Template(content=expanded, role=prompt.template["role"])],
            )
        )

    # Call the prompt with the provided arguments
    logger.info(f"Calling prompt: {prompt_name} with args: {args}")
    return await call(prompt_name, args)
