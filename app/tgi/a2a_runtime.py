"""Standards-compliant A2A transport backed by the existing bridge runtime.

AG2 owns the A2A server/task protocol while Enterprise MCP Bridge keeps
responsibility for authentication, MCP sessions, workflows, and tool execution.
"""

import logging
import os
from http.cookies import SimpleCookie
from typing import Any, Optional

from ag2 import Agent
from ag2.a2a import A2AServer, build_card
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCard, AgentSkill, Part, Task, TaskState, TaskStatus
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.session import session_id, try_get_session_id
from app.session_manager import mcp_session_context
from app.tgi.models import ChatCompletionRequest
from app.tgi.protocols.chunk_reader import accumulate_content
from app.tgi.routes import (
    _extract_completion_content,
    _is_async_iterable,
    sessions,
    tgi_service,
)
from app.vars import (
    DEFAULT_MODEL,
    MCP_BASE_PATH,
    SERVICE_NAME,
    SESSION_FIELD_NAME,
    TOKEN_COOKIE_NAME,
    TOKEN_NAME,
    TOKEN_SOURCE,
)

logger = logging.getLogger("uvicorn.error")


def _cookie_values(headers: dict[str, str]) -> dict[str, str]:
    cookie = SimpleCookie()
    try:
        cookie.load(headers.get("cookie", ""))
    except Exception:
        return {}
    return {key: morsel.value for key, morsel in cookie.items()}


def _access_token_from_headers(headers: dict[str, str]) -> Optional[str]:
    """Resolve auth exactly like the HTTP routes, using A2A request headers."""
    cookies = _cookie_values(headers)

    if TOKEN_SOURCE == "cookie":
        token = cookies.get(TOKEN_COOKIE_NAME)
        if token:
            return token

    token = headers.get(TOKEN_NAME.lower())
    if token:
        return token

    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]

    return None


def _session_from_headers(
    headers: dict[str, str], access_token: Optional[str]
) -> Optional[str]:
    cookies = _cookie_values(headers)
    raw_session = try_get_session_id(
        headers.get(SESSION_FIELD_NAME.lower()),
        cookies.get(SESSION_FIELD_NAME),
    )
    return session_id(raw_session, access_token)


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    return value if value not in ("", None) else None


def _rpc_path() -> str:
    base = "/" + MCP_BASE_PATH.strip("/") if MCP_BASE_PATH.strip("/") else ""
    return f"{base}/tgi/v1/a2a"


def _card_path() -> str:
    base = "/" + MCP_BASE_PATH.strip("/") if MCP_BASE_PATH.strip("/") else ""
    return f"{base}/.well-known/agent-card.json"


def _legacy_card_path() -> str:
    base = "/" + MCP_BASE_PATH.strip("/") if MCP_BASE_PATH.strip("/") else ""
    return f"{base}/.well-known/agent.json"


def _public_a2a_url(request: Optional[Request] = None) -> str:
    explicit = os.getenv("A2A_PUBLIC_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")

    if request is not None:
        forwarded = request.headers.get("forwarded", "")
        forwarded_values: dict[str, str] = {}
        if forwarded:
            for item in forwarded.split(",", 1)[0].split(";"):
                key, sep, value = item.strip().partition("=")
                if sep:
                    forwarded_values[key.lower()] = value.strip().strip('"')

        scheme = (
            forwarded_values.get("proto")
            or request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
            or request.url.scheme
        )
        host = (
            forwarded_values.get("host")
            or request.headers.get("x-forwarded-host", "").split(",", 1)[0].strip()
            or request.headers.get("host", "")
        )
        forwarded_port = request.headers.get("x-forwarded-port", "").split(",", 1)[0].strip()
        if forwarded_port and host and ":" not in host:
            default_port = (scheme == "https" and forwarded_port == "443") or (
                scheme == "http" and forwarded_port == "80"
            )
            if not default_port:
                host = f"{host}:{forwarded_port}"
        if host:
            return f"{scheme}://{host}{_rpc_path()}"

    # Internal placeholder used by the handler; discovery responses replace
    # this with a request-derived public URL when A2A_PUBLIC_URL is unset.
    return f"http://localhost{_rpc_path()}"


class BridgeA2AExecutor(A2AAgentExecutor):
    """Delegate a real A2A task to the bridge's existing MCP/LLM runtime."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        task_id = context.task_id
        context_id = context.context_id
        if message is None or not task_id or not context_id:
            return

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[message],
            )
        )
        await updater.start_work()

        if not DEFAULT_MODEL:
            await updater.failed(
                updater.new_agent_message(
                    parts=[Part(text="No default model configured.")]
                )
            )
            return

        headers = {
            str(key).lower(): str(value)
            for key, value in (context.call_context.state.get("headers") or {}).items()
        }
        access_token = _access_token_from_headers(headers)
        user_token = headers.get(TOKEN_NAME.lower()) or access_token
        session_key = _session_from_headers(headers, access_token)

        metadata = context.metadata or {}
        group = _metadata_value(metadata, "group")
        prompt = _metadata_value(metadata, "prompt")
        use_workflow = _metadata_value(metadata, "use_workflow")
        workflow_execution_id = _metadata_value(metadata, "workflow_execution_id")

        chat_request = ChatCompletionRequest(
            messages=[{"role": "user", "content": context.get_user_input()}],
            model=DEFAULT_MODEL,
            stream=False,
            use_workflow=use_workflow,
            workflow_execution_id=workflow_execution_id,
        )

        try:
            async with mcp_session_context(
                sessions,
                session_key,
                access_token,
                group,
                headers,
            ) as session:
                result = await tgi_service.chat_completion(
                    session,
                    chat_request,
                    user_token,
                    access_token,
                    prompt,
                )
                if _is_async_iterable(result):
                    response_text = await accumulate_content(result)
                else:
                    response_text = _extract_completion_content(result)

            await updater.add_artifact(
                parts=[Part(text=response_text)],
                name="response",
                last_chunk=True,
            )
            await updater.complete()
        except Exception:
            logger.exception("[A2A] Agent execution failed")
            await updater.failed(
                updater.new_agent_message(
                    parts=[Part(text="Agent execution failed.")]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            return
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.cancel()


def _build_agent_card(agent: Agent, url: str) -> AgentCard:
    description = (
        "Enterprise MCP Bridge agent backed by MCP tools, authentication, "
        "sessions, and workflow orchestration."
    )
    return build_card(
        agent,
        url=url,
        description=description,
        skills=[
            AgentSkill(
                id=SERVICE_NAME.lower().replace(" ", "-"),
                name=SERVICE_NAME,
                description=description,
                tags=["mcp", "enterprise"],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ],
    )


def build_a2a_app(executor: Optional[A2AAgentExecutor] = None):
    """Build AG2's standards-compliant JSON-RPC A2A ASGI application."""
    agent = Agent(name=SERVICE_NAME)
    server = A2AServer(agent, executor=executor or BridgeA2AExecutor())

    internal_card = _build_agent_card(agent, _public_a2a_url())
    app = server.build_jsonrpc(
        url=_public_a2a_url(),
        card=internal_card,
        rpc_url=_rpc_path(),
        card_url=_card_path(),
        legacy_card_url=_legacy_card_path(),
    )

    async def _serve_public_card(request: Request):
        card = _build_agent_card(agent, _public_a2a_url(request))
        return JSONResponse(agent_card_to_dict(card))

    card_paths = {_card_path(), _legacy_card_path()}
    app.routes[:] = [
        route for route in app.routes if getattr(route, "path", None) not in card_paths
    ]
    app.routes.extend(
        [
            Route(_card_path(), endpoint=_serve_public_card, methods=["GET"]),
            Route(_legacy_card_path(), endpoint=_serve_public_card, methods=["GET"]),
        ]
    )
    return app
