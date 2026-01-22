import asyncio
import json
import logging
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from fastapi import HTTPException

from app.tgi.models import ChatCompletionRequest, Message, MessageRole

from app.app_facade.generated_schemas import (
    generation_logic_schema,
    generation_response_format,
)


logger = logging.getLogger("uvicorn.error")


async def run_phase1_attempt(
    *,
    attempt: int,
    max_attempts: int,
    messages: List[Message],
    allowed_tools: List[Dict[str, Any]],
    dummy_data: Optional[str],
    access_token: Optional[str],
    tgi_service: Any,
    parse_json: Callable[[str], Dict[str, Any]],
    run_tests: Callable[[str, str, str, Optional[str]], Tuple[bool, str]],
    iterative_test_fix: Callable[
        ...,
        Awaitable[Tuple[bool, str, str, str, Optional[str], List[Message]]],
    ],
    chunk_reader: Callable[..., Any],
    ui_model_headers: Optional[Dict[str, str]],
) -> AsyncIterator[Union[bytes, Dict[str, Any]]]:
    """
    Executes a single attempt of Phase 1 logic generation.
    """
    logger.info(f"[stream_generate_ui] Phase 1 Attempt {attempt}/{max_attempts}")

    yield f"event: log\ndata: {json.dumps({'message': f'Phase 1: Generating logic and tests (Attempt {attempt})'})}\n\n".encode(
        "utf-8"
    )

    chat_request = ChatCompletionRequest(
        messages=messages,
        tools=allowed_tools if allowed_tools else None,
        stream=True,
        response_format=generation_response_format(
            schema=generation_logic_schema, name="generated_logic"
        ),
        extra_headers=ui_model_headers,
    )

    content = ""
    tool_calls_accumulated: Dict[int, dict] = {}
    tool_calls_seen = 0
    last_finish_reason = None

    try:
        stream_source = tgi_service.llm_client.stream_completion(
            chat_request, access_token or "", None
        )
    except Exception as exc:
        logger.error(f"[stream_generate_ui] LLM stream creation failed: {exc}")
        payload = json.dumps({"error": str(exc)})
        yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
        yield {"type": "result", "success": False, "error": str(exc)}
        return

    error_in_attempt = False
    try:
        async with chunk_reader(stream_source) as reader:
            async for parsed in reader.as_parsed():
                if getattr(parsed, "is_keepalive", False):
                    yield parsed.raw.encode("utf-8")
                    continue
                if parsed.is_done:
                    break
                if getattr(parsed, "content", None):
                    content += parsed.content
                if parsed.tool_calls:
                    tool_calls_seen += len(parsed.tool_calls)
                if parsed.accumulated_tool_calls:
                    tool_calls_accumulated = parsed.accumulated_tool_calls
                if parsed.finish_reason:
                    last_finish_reason = parsed.finish_reason
    except Exception as exc:
        logger.error(f"Streaming failed: {exc}")
        error_in_attempt = True

    if error_in_attempt:
        payload = json.dumps({"error": "Streaming failed"})
        yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
        yield {"type": "result", "success": False, "error": "Streaming failed"}
        return

    if not content and tool_calls_accumulated:
        tool_call_names = sorted(
            {
                call.get("name")
                for call in tool_calls_accumulated.values()
                if call.get("name")
            }
        )
        logger.info(
            "[stream_generate_ui] Phase 1 stream returned tool calls without content. "
            "tool_calls=%s, seen=%s, finish_reason=%s",
            tool_call_names or "unknown",
            tool_calls_seen,
            last_finish_reason or "unknown",
        )
        for idx in sorted(tool_calls_accumulated.keys()):
            args = tool_calls_accumulated[idx].get("arguments")
            if args is None:
                continue
            if isinstance(args, dict) and isinstance(args.get("input"), dict):
                content = json.dumps(args["input"], ensure_ascii=False)
                break
            if isinstance(args, str):
                if not args:
                    continue
                content = args
                break
            content = json.dumps(args, ensure_ascii=False)
            break

    if not content:
        payload = json.dumps({"error": "No content generated"})
        yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
        yield {"type": "result", "success": False, "error": "No content generated"}
        return

    try:
        current_payload = parse_json(content)
        service_script = current_payload.get("service_script")
        components_script = current_payload.get("components_script")
        test_script = current_payload.get("test_script")

        if not (components_script and test_script):
            raise HTTPException(
                status_code=502, detail="Missing required scripts in generation"
            )

        yield f"event: log\ndata: {json.dumps({'message': 'Running initial tests...'})}\n\n".encode(
            "utf-8"
        )

        success, output = run_tests(
            service_script,
            components_script,
            test_script,
            dummy_data,
        )

        if not success:
            yield f"event: log\ndata: {json.dumps({'message': 'Tests failed, starting iterative fix with tools...'})}\n\n".encode(
                "utf-8"
            )

            messages.append(Message(role=MessageRole.ASSISTANT, content=content))
            tool_events_queue: asyncio.Queue = asyncio.Queue()

            async def stream_tool_events() -> Tuple[
                bool,
                str,
                str,
                str,
                Optional[str],
                List[Message],
            ]:
                return await iterative_test_fix(
                    service_script=service_script,
                    components_script=components_script,
                    test_script=test_script,
                    dummy_data=dummy_data,
                    messages=messages,
                    allowed_tools=allowed_tools,
                    access_token=access_token,
                    max_attempts=25,
                    event_queue=tool_events_queue,
                )

            fix_task = asyncio.create_task(stream_tool_events())

            updated_messages = messages
            fix_success = False
            fixed_service = ""
            fixed_components = ""
            fixed_test = ""
            fixed_dummy_data = dummy_data

            try:
                while not fix_task.done():
                    try:
                        event_data = tool_events_queue.get_nowait()
                        if event_data.get("event") == "tool_start":
                            payload = {
                                "tool": event_data["tool"],
                                "description": event_data["description"],
                            }
                            yield f"event: tool_start\ndata: {json.dumps(payload)}\n\n".encode(
                                "utf-8"
                            )
                        elif event_data.get("event") == "test_result":
                            yield f"event: test_result\ndata: {json.dumps(event_data)}\n\n".encode(
                                "utf-8"
                            )
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.05)

                (
                    fix_success,
                    fixed_service,
                    fixed_components,
                    fixed_test,
                    fixed_dummy_data,
                    updated_messages,
                ) = await fix_task

                while not tool_events_queue.empty():
                    event_data = tool_events_queue.get_nowait()
                    if event_data.get("event") == "tool_start":
                        payload = {
                            "tool": event_data["tool"],
                            "description": event_data["description"],
                        }
                        yield f"event: tool_start\ndata: {json.dumps(payload)}\n\n".encode(
                            "utf-8"
                        )
                    elif event_data.get("event") == "test_result":
                        yield f"event: test_result\ndata: {json.dumps(event_data)}\n\n".encode(
                            "utf-8"
                        )

            except Exception:
                fix_task.cancel()
                raise

            if fix_success:
                yield f"event: log\ndata: {json.dumps({'message': 'Tests fixed and passing!'})}\n\n".encode(
                    "utf-8"
                )
                logic_payload = {
                    "service_script": fixed_service,
                    "components_script": fixed_components,
                    "test_script": fixed_test,
                    "dummy_data": fixed_dummy_data or dummy_data,
                }

                if (
                    updated_messages
                    and updated_messages[-1].role == MessageRole.ASSISTANT
                ):
                    clean_content = json.dumps(
                        {
                            "service_script": fixed_service,
                            "components_script": fixed_components,
                            "test_script": fixed_test,
                        },
                        ensure_ascii=False,
                    )
                    updated_messages[-1].content = clean_content

                yield {
                    "type": "result",
                    "success": True,
                    "payload": logic_payload,
                    "messages": updated_messages,
                }
                return
            yield f"event: log\ndata: {json.dumps({'message': 'Tool-based fix failed, regenerating from scratch...'})}\n\n".encode(
                "utf-8"
            )
            yield {"type": "result", "success": False, "messages": updated_messages}
            return

        yield f"event: log\ndata: {json.dumps({'message': 'Tests passed!'})}\n\n".encode(
            "utf-8"
        )
        logic_payload = {
            **current_payload,
            "dummy_data": dummy_data,
        }
        messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        yield {
            "type": "result",
            "success": True,
            "payload": logic_payload,
            "messages": messages,
        }
        return

    except (json.JSONDecodeError, HTTPException) as exc:
        messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        messages.append(
            Message(role=MessageRole.USER, content=f"Error: {exc}. Fix it.")
        )
        yield {"type": "result", "success": False, "messages": messages}
        return
