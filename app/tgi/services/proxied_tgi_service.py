import logging
import os
import time
from typing import List, Optional, Union, AsyncGenerator
from opentelemetry import trace


from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
    Tool,
)
from app.session import MCPSessionBase
from app.tgi.services.prompt_service import PromptService
from app.tgi.services.tool_service import (
    ToolService,
)
from app.tgi.services.message_summarization_service import MessageSummarizationService
from app.tgi.services.tool_chat_runner import ToolChatRunner
from app.tgi.clients.llm_client import LLMClient
from app.tgi.models.model_formats import BaseModelFormat, get_model_format_for
from app.tgi.workflows import WorkflowEngine, WorkflowRepository, WorkflowStateStore
from app.vars import TGI_MODEL_NAME
from app.tgi.behaviours.well_planned_orchestrator import WellPlannedOrchestrator

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class ProxiedTGIService:
    """Service that orchestrates chat completions with tool support."""

    def __init__(self, model_format: Optional[BaseModelFormat] = None):
        self.logger = logger
        self.model_format = model_format or get_model_format_for()
        self.prompt_service = PromptService()
        self.llm_client = LLMClient(self.model_format)
        self.tool_service = ToolService(
            model_format=self.model_format, llm_client=self.llm_client
        )
        self.message_summarization_service = MessageSummarizationService(
            llm_client=self.llm_client
        )
        self.tool_chat_runner = ToolChatRunner(
            llm_client=self.llm_client,
            tool_service=self.tool_service,
            tool_resolution=self.model_format.create_tool_resolution_strategy(),
            message_summarization_service=self.message_summarization_service,
            logger_obj=self.logger,
        )
        self.tool_resolution = self.model_format.create_tool_resolution_strategy()
        self.workflow_engine: Optional[WorkflowEngine] = None
        # pass a lambda that looks up the current _non_stream_chat_with_tools
        # attribute at call time so tests can monkeypatch it after
        # construction.
        self.well_planned_orchestrator = WellPlannedOrchestrator(
            llm_client=self.llm_client,
            prompt_service=self.prompt_service,
            tool_service=self.tool_service,
            non_stream_chat_with_tools_callable=(
                lambda *a, **k: self._non_stream_chat_with_tools(*a, **k)
            ),
            stream_chat_with_tools_callable=(
                lambda *a, **k: self._stream_chat_with_tools(*a, **k)
            ),
            tool_resolution=self.tool_resolution,
            logger_obj=self.logger,
            model_name=TGI_MODEL_NAME,
        )
        try:
            repo = WorkflowRepository()
            db_path = os.environ.get("WORKFLOW_DB_PATH") or repo.base_path.joinpath(
                "workflow_state.db"
            )
            self.workflow_engine = WorkflowEngine(
                repository=repo,
                state_store=WorkflowStateStore(db_path),
                llm_client=self.llm_client,
                prompt_service=self.prompt_service,
                tool_service=self.tool_service,
            )
        except Exception as exc:
            self.logger.debug(f"[ProxiedTGI] Workflow engine not initialized: {exc}")

    async def one_off_chat_completion(
        self,
        session: MCPSessionBase,
        request: ChatCompletionRequest,
        access_token: Optional[str] = None,
        prompt: Optional[str] = None,
        span: trace.Span = None,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """Handle chat completion requests with optional tool support."""
        span.set_attribute("chat.strategy", "one-off")

        # Prepare messages including system prompt if provided
        messages = await self.prompt_service.prepare_messages(
            session, request.messages, prompt, span
        )
        request.messages = []

        # Get available tools from the session
        available_tools = await self.tool_service.get_all_mcp_tools(session, span)

        if request.stream:
            # Return streaming async generator
            return self._stream_chat_with_tools(
                session, messages, available_tools, request, access_token, span
            )
        else:
            # Return complete response
            return await self._non_stream_chat_with_tools(
                session, messages, available_tools, request, access_token, span
            )

    async def well_planned_chat_completion(
        self,
        session: MCPSessionBase,
        request: ChatCompletionRequest,
        access_token: Optional[str] = None,
        prompt: Optional[str] = None,
        span: trace.Span = None,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        # Delegate the well-planned flow to the orchestrator. This keeps the
        # public API unchanged while moving the implementation elsewhere.
        return await self.well_planned_orchestrator.well_planned_chat_completion(
            session, request, access_token, prompt, span
        )

    async def chat_completion(
        self,
        session: MCPSessionBase,
        request: ChatCompletionRequest,
        user_token: Optional[str] = None,
        access_token: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """Handle chat completion requests with optional tool support."""
        with tracer.start_as_current_span("chat_completion") as span:
            span.set_attribute("chat.model", request.model or "unknown")
            span.set_attribute("chat.streaming", request.stream or False)
            span.set_attribute("chat.messages_count", len(request.messages))
            span.set_attribute("chat.has_tools", bool(request.tools))
            span.set_attribute("chat.tool_choice", bool(request.tool_choice))
            span.set_attribute("chat.use_workflow", bool(request.use_workflow))

            logger.info(
                f"[ProxiedTGI] Received chat completion request: model={request.model}, use workflow={request.use_workflow}, tool_choice={request.tool_choice}, tools_provided={bool(request.tools)}"
            )

            if self.workflow_engine and request.use_workflow:
                if user_token is None:
                    raise ValueError(
                        "User token is required for workflow-based chat completions"
                    )
                request.stream = True
                workflow_result = await self.workflow_engine.start_or_resume_workflow(
                    session, request, user_token, access_token, span
                )
                if workflow_result is not None:
                    return workflow_result

            if request.tools:
                span.set_attribute("chat.tools_count", len(request.tools))

            # Route to well-planned orchestrator when tool_choice is explicitly set to
            # "auto". This is the orchestrator trigger used by the well-planned flow.
            use_well_planned = request.tool_choice and request.tool_choice == "auto"

            # Always await the chosen handler so that it performs any
            # necessary synchronous setup (prepare messages, tool discovery)
            # and then returns either a ChatCompletionResponse or an
            # async-generator object for streaming results.
            if use_well_planned:
                res = await self.well_planned_chat_completion(
                    session, request, access_token, prompt, span
                )
            else:
                res = await self.one_off_chat_completion(
                    session, request, access_token, prompt, span
                )

            # The awaited call returns either a response object or an
            # async-generator. Return it directly; do not await the result
            # again (awaiting an async-generator raises TypeError).
            return res

    async def _stream_chat_with_tools(
        self,
        session: MCPSessionBase,
        messages: List[Message],
        available_tools: List[dict],
        chat_request: ChatCompletionRequest,
        access_token: Optional[str],
        parent_span,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion with tool handling, supporting tool calls."""
        async for chunk in self.tool_chat_runner.stream_chat_with_tools(
            session=session,
            messages=messages,
            available_tools=available_tools,
            chat_request=chat_request,
            access_token=access_token,
            parent_span=parent_span,
        ):
            yield chunk

    def _deduplicate_retry_hints(self, messages: List[Message]) -> None:
        """Keep only the most recent retry instruction to prevent payload bloat."""
        seen_hint = False
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if getattr(msg, "name", None) == "mcp_tool_retry_hint":
                if seen_hint:
                    del messages[idx]
                else:
                    seen_hint = True

    async def _non_stream_chat_with_tools(
        self,
        session: MCPSessionBase,
        messages: List[Message],
        available_tools: List[Tool],
        chat_request: ChatCompletionRequest,
        access_token: Optional[str],
        parent_span,
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion with tool handling."""
        from app.tgi.models import ChatCompletionResponse

        messages_history = messages.copy()
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(f"[ProxiedTGI] Non-stream chat iteration {iteration}")

            # Create request for LLM
            llm_request = ChatCompletionRequest(
                messages=messages_history,
                model=chat_request.model or TGI_MODEL_NAME,
                tools=available_tools if available_tools else None,
                tool_choice=chat_request.tool_choice,
                stop=chat_request.stop,
                stream=False,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                top_p=chat_request.top_p,
            )

            # Call actual LLM
            response = await self.llm_client.non_stream_completion(
                llm_request, access_token, parent_span
            )

            # Check if response contains tool calls
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.tool_calls
            ):

                # Add assistant message with tool calls to history
                messages_history.append(response.choices[0].message)

                tool_results, success = await self.tool_service.execute_tool_calls(
                    session,
                    response.choices[0].message.tool_calls,
                    access_token,
                    parent_span,
                    available_tools=available_tools,
                )

                # Add tool results to history
                messages_history.extend(tool_results)

                # Summarize messages if history is getting too long
                if self.message_summarization_service.should_summarize(
                    messages_history
                ):
                    self.logger.info(
                        f"[ProxiedTGI] Summarizing {len(messages_history)} messages to avoid token limits"
                    )
                    messages_history = (
                        await self.message_summarization_service.summarize_messages(
                            messages_history, access_token, parent_span
                        )
                    )
                    self.logger.info(
                        f"[ProxiedTGI] Message history reduced to {len(messages_history)} messages"
                    )
            else:
                # No tool calls, return final response
                return response

        # If we reach here, we hit max iterations
        return ChatCompletionResponse(
            id=self.llm_client.create_completion_id(),
            object="chat.completion",
            created=int(time.time()),
            model=chat_request.model or TGI_MODEL_NAME,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content="Maximum conversation iterations reached.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=self.llm_client.create_usage_stats(0, 0),
        )
