import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, Optional, Union

from app.tgi.models import ChatCompletionRequest, Message, MessageRole

from app.app_facade.generated_schemas import (
    generation_presentation_schema,
    generation_response_format,
)


logger = logging.getLogger("uvicorn.error")


async def run_phase2_attempt(
    *,
    system_prompt: str,
    prompt: str,
    logic_payload: Dict[str, Any],
    access_token: Optional[str],
    instruction: str,
    tgi_service: Any,
    parse_json: Callable[[str], Dict[str, Any]],
    chunk_reader: Callable[..., Any],
    ui_model_headers: Optional[Dict[str, str]],
) -> AsyncIterator[Union[bytes, Dict[str, Any]]]:
    """
    Executes Phase 2: Presentation generation.
    Generate template parts for the HTML shell that uses Phase 1 components.
    """
    logger.info("[stream_generate_ui] Phase 2: Presentation")
    payload = json.dumps({"message": "Phase 2: Generating template parts"})
    yield f"event: log\ndata: {payload}\n\n".encode("utf-8")

    service_script = logic_payload.get("service_script", "")
    components_script = logic_payload.get("components_script", "")

    context_content = (
        f"Original Request: {prompt}\n\n"
        f"We have generated the following components:\n"
        f"```javascript\n{components_script}\n```\n\n"
        f"And service logic:\n"
        f"```javascript\n{service_script}\n```\n\n"
        f"{instruction}\n\n"
        "Return template_parts only."
    )

    messages = [
        Message(role=MessageRole.SYSTEM, content=system_prompt),
        Message(role=MessageRole.USER, content=context_content),
    ]

    chat_request = ChatCompletionRequest(
        messages=messages,
        tools=None,
        stream=True,
        response_format=generation_response_format(
            schema=generation_presentation_schema, name="generated_presentation"
        ),
        extra_headers=ui_model_headers,
    )

    content = ""
    try:
        stream_source = tgi_service.llm_client.stream_completion(
            chat_request, access_token or "", None
        )
    except Exception as exc:
        payload = json.dumps({"error": str(exc)})
        yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
        yield {"type": "result", "success": False, "error": str(exc)}
        return

    async with chunk_reader(stream_source) as reader:
        async for parsed in reader.as_parsed():
            if getattr(parsed, "is_keepalive", False):
                yield parsed.raw.encode("utf-8")
                continue
            if parsed.is_done:
                break

            if getattr(parsed, "content", None):
                content_piece = parsed.content
                content += content_piece
                payload = json.dumps({"chunk": content_piece})
                yield f"data: {payload}\n\n".encode("utf-8")
            else:
                yield b":\n\n"

    presentation_payload = parse_json(content)
    yield {"type": "result", "success": True, "payload": presentation_payload}
