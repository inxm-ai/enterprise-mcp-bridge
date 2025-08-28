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
    template: str


def default_prompt_list_result(prompts: list[Prompt] = []):
    return {
        "_meta": None,
        "nextCursor": None,
        "prompts": [
            {k: v for k, v in prompt.__dict__.items() if k != "template"}
            for prompt in prompts
        ],
    }


def system_defined_prompts() -> list[Prompt]:
    # System defined prompt looks like this:
    # { "name": "greeting", "title": "Hello You", "description": "Get a personalized greeting.", "arguments": [{ "name": "name" }], "template": "Hello, {name}!" }
    prompts = json.loads(os.environ.get("SYSTEM_DEFINED_PROMPTS", "[]"))
    return [Prompt(**p) for p in prompts]


async def list_prompts(list_prompts: any):
    system_prompts = system_defined_prompts()
    try:
        prompts = await list_prompts()
        prompts.prompts += [
            {k: v for k, v in prompt.__dict__.items() if k != "template"}
            for prompt in system_prompts
        ]
    except Exception as e:
        logger.error(f"Error listing prompts: {str(e)}")
        # Not every MCP has list_prompts, so deal with it friendly
        if (
            hasattr(e, "__class__")
            and e.__class__.__name__ == "McpError"
            and "Method not found" in str(e)
        ):
            if len(system_prompts) < 1:
                raise HTTPException(status_code=404, detail="Method not found")
            else:
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
