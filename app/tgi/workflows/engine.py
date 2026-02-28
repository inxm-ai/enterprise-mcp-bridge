import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator, Optional

from app.oauth.user_info import UserInfoExtractor
from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.protocols.chunk_reader import chunk_reader
from app.tgi.services.prompt_service import PromptService
from app.tgi.services.tool_service import ToolService
from app.tgi.services.tool_chat_runner import ToolChatRunner
from app.tgi.services.tools.tool_resolution import ToolResolutionStrategy
from app.tgi.workflows.agent import AgentExecutor
from app.tgi.workflows.arg_injector import (
    ArgResolutionError,
    ToolResultCapture,
    build_arg_injector_fn,
)
from app.tgi.workflows.chunk_formatter import WorkflowChunkFormatter
from app.tgi.workflows.models import (
    WorkflowAgentDef,
    WorkflowDefinition,
    WorkflowExecutionState,
)
from app.tgi.workflows.passthrough_filter import PassThroughFilter
from app.tgi.workflows.repository import WorkflowRepository
from app.tgi.workflows.state import WorkflowStateStore
from app.vars import TGI_MODEL_NAME, WORKFLOW_MAX_PARALLEL_AGENTS

# Extracted modules
from app.tgi.workflows import (
    agent_dispatch,
    agent_result,
    context_builder,
    error_analysis,
    reroute,
    state_management,
    stream_processor,
    tag_parser,
)
from app.tgi.workflows.feedback import FeedbackService
from app.tgi.workflows.reroute import RoutingService

logger = logging.getLogger("uvicorn.error")


class WorkflowEngine:
    """
    Executes workflows by streaming agent outputs and persisting state for resumption.
    """

    MAX_RETURN_RETRIES = 3
    WORKFLOW_OWNER_KEY = "_workflow_owner_id"
    CONTINUE_PLACEHOLDER = "[continue]"
    END_PLACEHOLDER = "[end]"
    WORKFLOW_DESCRIPTION_TOOL_NAMES = {"plan"}
    WORKFLOW_DESCRIPTION_CONTEXT_KEY = "_workflow_description"
    FEEDBACK_PAUSE_NOTICE_KEY = "_feedback_pause_notice_sent"

    def __init__(
        self,
        repository: WorkflowRepository,
        state_store: WorkflowStateStore,
        llm_client: Any,
        prompt_service: Optional[PromptService] = None,
        tool_service: Optional[ToolService] = None,
        chunk_formatter: Optional[WorkflowChunkFormatter] = None,
        tool_chat_runner: Optional[ToolChatRunner] = None,
        agent_executor: Optional[AgentExecutor] = None,
    ):
        self.repository = repository
        self.state_store = state_store
        self.llm_client = llm_client
        self.prompt_service = prompt_service or PromptService()
        self.tool_service = tool_service or ToolService()
        self.chunk_formatter = chunk_formatter or WorkflowChunkFormatter()
        self.agent_executor = agent_executor or AgentExecutor()
        self.user_info_extractor = UserInfoExtractor()
        self.passthrough_filter = PassThroughFilter(
            llm_client=self.llm_client,
            model_name=TGI_MODEL_NAME,
            logger_obj=logger,
        )
        self.tool_chat_runner = tool_chat_runner or ToolChatRunner(
            llm_client=self.llm_client,
            tool_service=self.tool_service,
            tool_resolution=(
                self.tool_service.model_format.create_tool_resolution_strategy()
                if self.tool_service
                else ToolResolutionStrategy()
            ),
            logger_obj=logger,
            message_summarization_service=None,
        )
        self.max_parallel_agents = max(1, int(WORKFLOW_MAX_PARALLEL_AGENTS))
        self.feedback_service = FeedbackService(llm_client=self.llm_client)
        self.routing_service = RoutingService(
            llm_client=self.llm_client,
            prompt_service=self.prompt_service,
            state_store=self.state_store,
        )

    async def start_or_resume_workflow(
        self,
        session: Any,
        request: ChatCompletionRequest,
        user_token: str,
        access_token: Optional[str],
        span,
        workflow_chain: Optional[list[str]] = None,
        handoff: bool = False,
    ) -> Optional[AsyncGenerator[str, None]]:
        execution_id = request.workflow_execution_id or str(uuid.uuid4())
        state = self.state_store.load_execution(execution_id)
        state_loaded = state is not None
        workflow_def = None
        if state_loaded and not handoff:
            try:
                workflow_def = self.repository.get(state.flow_id)
            except Exception:
                workflow_def = None
            if not workflow_def:
                return None
            requested_workflow = getattr(request, "use_workflow", None)
            if (
                requested_workflow not in (None, True)
                and str(requested_workflow) != state.flow_id
            ):
                logger.warning(
                    "[WorkflowEngine] Ignoring requested workflow '%s' for resumed execution '%s'; "
                    "using stored flow '%s'.",
                    requested_workflow,
                    execution_id,
                    state.flow_id,
                )
        else:
            workflow_def = self._select_workflow(request)
            if not workflow_def:
                return None
        if state_loaded:
            self._enforce_workflow_owner(state, user_token)
            if handoff:
                state_management.reset_state_for_handoff(
                    state, workflow_def.flow_id, save_fn=self.state_store.save_state
                )
        else:
            state = WorkflowExecutionState.new(execution_id, workflow_def.flow_id)
            self._enforce_workflow_owner(state, user_token)
        state.context.setdefault("agents", {})
        state.context["_workflow_loop"] = bool(workflow_def.loop)
        chain = list(workflow_chain or state.context.get("_workflow_chain") or [])
        if workflow_def.flow_id not in chain:
            chain.append(workflow_def.flow_id)
        state.context["_workflow_chain"] = chain
        persist_inner_thinking = state_management.should_persist_inner_thinking(
            request.persist_inner_thinking, state
        )
        if not persist_inner_thinking:
            state_management.prune_inner_thinking(state, workflow_def)
        self.chunk_formatter.ensure_task_id(state)
        if state_loaded and state_management.strip_continue_placeholder(request):
            user_message = None
        else:
            user_message = context_builder.extract_user_message(request)

        if user_message and user_message.strip().lower() == self.END_PLACEHOLDER:
            state_management.mark_workflow_success(
                state, save_fn=self.state_store.save_state, reason="user_end"
            )

            async def _done_only():
                yield "data: [DONE]\n\n"

            return _done_only()
        if workflow_def.loop and user_message and not state.awaiting_feedback:
            state_management.reset_state_for_loop_turn(
                state, ensure_task_id_fn=self.chunk_formatter.ensure_task_id
            )
        state_management.append_user_message(state, user_message)

        if request.start_with and not state.context.get("_start_with_applied"):
            state_management.apply_start_with(
                state,
                workflow_def,
                request.start_with,
                save_fn=self.state_store.save_state,
                complete_deps_fn=state_management.complete_dependencies_for_agent,
            )

        self.state_store.save_state(state)
        no_reroute = tag_parser.has_no_reroute(user_message)
        return_full_state = bool(request.return_full_state)
        handoff_state: dict[str, bool] = {"workflow_handoff": False}

        async def _runner():
            user_message_local = user_message
            emit_execution_id = not state_loaded or user_message_local is not None
            if emit_execution_id:
                yield self.chunk_formatter.format_chunk(
                    state=state,
                    content=f'<workflow_execution_id for="{workflow_def.flow_id}">{execution_id}</workflow_execution_id>\n',
                    status="submitted",
                    role="system",
                )
            if return_full_state:
                for event in state.events:
                    yield event

            if state.completed:
                yield "data: [DONE]\n\n"
                return

            preresolved_tools = None
            if len(workflow_def.agents) == 1:
                modified, original = await context_builder.resolve_tools(
                    session,
                    workflow_def.agents[0],
                    self.tool_service,
                    self.agent_executor,
                )
                preresolved_tools = modified  # Use modified schema for routing

            routing_check = await self.routing_service.routing_intent_check(
                session,
                workflow_def,
                user_message_local,
                request,
                access_token,
                span,
                routing_tools=preresolved_tools,
                execution_id=execution_id,
            )
            if routing_check and routing_check.get("reroute") and not no_reroute:
                reason = routing_check["reroute"]
                state.completed = True
                self.state_store.save_state(state)
                yield self._record_event(
                    state, f"<reroute>{reason}</reroute>\n", status="reroute"
                )
                yield "data: [DONE]\n\n"
                return
            if preresolved_tools:
                try:
                    if hasattr(self.llm_client, "call_count"):
                        self.llm_client.call_count = max(
                            0, getattr(self.llm_client, "call_count", 0) - 1
                        )
                    rt = getattr(self.llm_client, "request_tools", None)
                    if isinstance(rt, list) and rt and rt[0] is None:
                        rt.pop(0)
                except Exception:
                    pass

            if state.awaiting_feedback:
                if user_message_local:
                    await self.feedback_service.merge_feedback(
                        state,
                        user_message_local,
                        request,
                        access_token,
                        span,
                        save_fn=self.state_store.save_state,
                    )
                    if "_resume_agent" not in state.context:
                        resume_target = state_management.find_feedback_agent(state)
                        if resume_target:
                            state.context["_resume_agent"] = resume_target
                            self.state_store.save_state(state)
                    self.chunk_formatter.ensure_task_id(state, reset=True)
                    self.state_store.save_state(state)
                    yield self._record_event(
                        state,
                        f"Received feedback: {user_message_local}",
                        status="feedback_received",
                    )
                else:
                    if not state.context.get(self.FEEDBACK_PAUSE_NOTICE_KEY):
                        state.context[self.FEEDBACK_PAUSE_NOTICE_KEY] = True
                        self.state_store.save_state(state)
                        if not return_full_state:
                            yield self.chunk_formatter.format_chunk(
                                state=state,
                                content="Workflow paused awaiting user feedback.",
                                status="waiting_for_feedback",
                                role="assistant",
                            )
                    yield "data: [DONE]\n\n"
                    return

            if not state.events:
                yield self._record_event(
                    state,
                    f"Routing workflow {workflow_def.flow_id}\n",
                    status="in_progress",
                )

            async for event in self._run_agents(
                workflow_def,
                state,
                session,
                request,
                user_token,
                access_token,
                span,
                persist_inner_thinking=persist_inner_thinking,
                no_reroute=no_reroute,
                workflow_chain=chain,
                handoff_state=handoff_state,
            ):
                yield event

            if not handoff_state.get("workflow_handoff"):
                if state.completed or (
                    workflow_def.loop and not state.awaiting_feedback
                ):
                    yield "data: [DONE]\n\n"

        return _runner()

    def _resolve_request_user_id(self, user_token: str) -> str:
        return state_management.resolve_request_user_id(
            user_token, self.user_info_extractor
        )

    def _enforce_workflow_owner(
        self, state: WorkflowExecutionState, user_token: str
    ) -> None:
        state_management.enforce_workflow_owner(
            state, user_token, self.user_info_extractor
        )

    def list_workflows(
        self,
        user_token: str,
        *,
        limit: int,
        before: Optional[str] = None,
        before_id: Optional[str] = None,
        after: Optional[str] = None,
        after_id: Optional[str] = None,
    ) -> list[WorkflowExecutionState]:
        """
        Return workflow executions for the user identified by the token.
        """
        if not user_token:
            raise PermissionError("Access token required to list workflows.")
        user_id = self._resolve_request_user_id(user_token)
        if not user_id:
            return []
        return self.state_store.list_workflows(
            owner_id=user_id,
            limit=limit,
            before=before,
            before_id=before_id,
            after=after,
            after_id=after_id,
        )

    def _select_workflow(
        self, request: ChatCompletionRequest
    ) -> Optional[WorkflowDefinition]:
        use_workflow = getattr(request, "use_workflow", None)
        if not use_workflow:
            return None
        user_message = context_builder.extract_user_message(request)
        if use_workflow is True:
            return self.repository.find_best_match(user_message)
        try:
            return self.repository.get(use_workflow)
        except Exception:
            return None

    async def _collect_agent_execution(
        self,
        workflow_def,
        agent_def,
        state,
        session,
        request,
        access_token,
        span,
        persist_inner_thinking,
        no_reroute,
    ) -> dict[str, Any]:
        """Collect execution results for a single agent (used in parallel runs)."""
        chunks: list[str] = []
        statuses: list[dict[str, Any]] = []
        async for result in self._execute_agent(
            workflow_def,
            agent_def,
            state,
            session,
            request,
            access_token,
            span,
            persist_inner_thinking,
            no_reroute,
        ):
            (chunks if isinstance(result, str) else statuses).append(result)
        return {"agent": agent_def, "chunks": chunks, "status_events": statuses}

    async def _handle_workflow_handoff(
        self,
        result: dict[str, Any],
        state: WorkflowExecutionState,
        request: ChatCompletionRequest,
        user_token: str,
        access_token: Optional[str],
        span,
        workflow_chain: list[str],
        handoff_state: dict[str, bool],
        session: Any,
    ) -> tuple[list[str], bool]:
        """Handle a workflow_reroute status result.

        Returns ``(events, should_return)`` where *events* is a list of
        SSE chunks to yield and *should_return* indicates whether the
        caller should stop iterating (``return`` from ``_run_agents``).
        """
        target_workflow = result.get("target_workflow")
        start_with_payload = result.get("start_with")
        start_with_payload = reroute.augment_workflow_handoff_start_with(
            start_with_payload, state
        )
        metadata: dict[str, Any] = {"target_workflow": target_workflow}
        if start_with_payload is not None:
            metadata["start_with"] = start_with_payload

        events: list[str] = []

        if not target_workflow:
            state.completed = True
            self.state_store.save_state(state)
            events.append(
                self._record_event(
                    state,
                    "\nWorkflow reroute target was not provided.\n",
                    status="error",
                    metadata=metadata,
                )
            )
            handoff_state["workflow_handoff"] = True
            events.append("data: [DONE]\n\n")
            return events, True

        events.append(
            self._record_event(
                state,
                f"\nRerouting to workflow '{target_workflow}'\n",
                status="reroute",
                metadata=metadata,
            )
        )

        if target_workflow in workflow_chain:
            state.completed = True
            self.state_store.save_state(state)
            events.append(
                self._record_event(
                    state,
                    (
                        f"\nWorkflow reroute loop detected for '{target_workflow}'. "
                        "Aborting.\n"
                    ),
                    status="error",
                    metadata={
                        "target_workflow": target_workflow,
                        "workflow_chain": workflow_chain,
                    },
                )
            )
            handoff_state["workflow_handoff"] = True
            events.append("data: [DONE]\n\n")
            return events, True

        state.completed = True
        self.state_store.save_state(state)
        new_chain = list(workflow_chain)
        new_chain.append(target_workflow)
        new_request = reroute.build_workflow_reroute_request(
            request,
            target_workflow,
            start_with_payload,
            execution_id=state.execution_id,
        )
        new_stream = await self.start_or_resume_workflow(
            session,
            new_request,
            user_token,
            access_token,
            span,
            workflow_chain=new_chain,
            handoff=True,
        )
        if not new_stream:
            events.append(
                self._record_event(
                    state,
                    f"\nWorkflow '{target_workflow}' is not defined.\n",
                    status="error",
                    metadata={"target_workflow": target_workflow},
                )
            )
            handoff_state["workflow_handoff"] = True
            events.append("data: [DONE]\n\n")
            return events, True

        handoff_state["workflow_handoff"] = True
        async for new_event in new_stream:
            events.append(new_event)
        return events, True

    async def _run_agents(
        self,
        workflow_def: WorkflowDefinition,
        state: WorkflowExecutionState,
        session: Any,
        request: ChatCompletionRequest,
        user_token: str,
        access_token: Optional[str],
        span,
        persist_inner_thinking: bool = False,
        no_reroute: bool = False,
        workflow_chain: Optional[list[str]] = None,
        handoff_state: Optional[dict[str, bool]] = None,
    ) -> AsyncGenerator[str, None]:
        workflow_chain = list(
            workflow_chain or state.context.get("_workflow_chain") or []
        )
        handoff_state = handoff_state or {"workflow_handoff": False}
        completed_agents = state_management.get_completed_agents(workflow_def, state)
        forced_next: Optional[str] = None

        resume_agent = state.context.pop("_resume_agent", None)
        if resume_agent:
            logger.info(
                "[_run_agents] Resume agent '%s' (completed=%s)",
                resume_agent,
                resume_agent in completed_agents,
            )
        if resume_agent and resume_agent not in completed_agents:
            forced_next = resume_agent

        last_visible_output: Optional[str] = None

        async def _handoff_fn(result):
            return await self._handle_workflow_handoff(
                result,
                state,
                request,
                user_token,
                access_token,
                span,
                workflow_chain,
                handoff_state,
                session,
            )

        async def _status(result):
            return await agent_dispatch.process_agent_status(
                result,
                workflow_def,
                state,
                save_fn=self.state_store.save_state,
                record_event_fn=self._record_event,
                handle_workflow_handoff_fn=_handoff_fn,
            )

        def _stop(agent_def):
            return agent_dispatch.check_stop_point(
                agent_def,
                state,
                self.state_store.save_state,
                self._record_event,
            )

        while len(completed_agents) < len(workflow_def.agents) or forced_next:
            if state.completed:
                yield "data: [DONE]\n\n"
                return

            progress_made = False
            retry_triggered = False

            if not forced_next and self.max_parallel_agents > 1:
                runnable = agent_dispatch.find_runnable_agents(
                    workflow_def.agents,
                    completed_agents,
                )
                batch = agent_dispatch.select_parallel_batch(
                    runnable,
                    self.max_parallel_agents,
                )
                if batch:
                    condition_checks = await asyncio.gather(
                        *[
                            self._condition_met(
                                agent_def,
                                state.context,
                                session,
                                request,
                                access_token,
                                span,
                                workflow_def,
                                execution_id=state.execution_id,
                            )
                            for agent_def in batch
                        ]
                    )
                    runnable_batch: list[WorkflowAgentDef] = []
                    for agent_def, is_allowed in zip(batch, condition_checks):
                        if is_allowed:
                            runnable_batch.append(agent_def)
                            continue
                        state_management.skip_agent(
                            agent_def,
                            state,
                            completed_agents,
                            self.state_store.save_state,
                        )
                        progress_made = True
                        yield self._record_event(
                            state,
                            f"Skipping {agent_def.agent} (condition not met)",
                            status="skipped",
                        )

                    if len(runnable_batch) > 1:
                        outcomes = await asyncio.gather(
                            *[
                                self._collect_agent_execution(
                                    workflow_def,
                                    agent_def,
                                    state,
                                    session,
                                    request,
                                    access_token,
                                    span,
                                    persist_inner_thinking,
                                    no_reroute,
                                )
                                for agent_def in runnable_batch
                            ]
                        )

                        by_agent = {o["agent"].agent: o for o in outcomes}
                        for agent_def in runnable_batch:
                            outcome = by_agent.get(agent_def.agent)
                            if not outcome:
                                continue
                            for chunk in outcome["chunks"]:
                                yield chunk

                            for result in outcome["status_events"]:
                                action = await _status(result)
                                for e in action.get("events", []):
                                    yield e
                                forced_next, last_visible_output, retry, _ = (
                                    agent_dispatch.apply_status_action(
                                        action,
                                        forced_next,
                                        last_visible_output,
                                    )
                                )
                                if action.get("should_return"):
                                    return
                                if retry:
                                    retry_triggered = True
                                    progress_made = True
                                if action.get("should_break"):
                                    break

                            if state.completed:
                                yield "data: [DONE]\n\n"
                                return
                            if retry_triggered:
                                break

                            completed_agents.add(agent_def.agent)
                            progress_made = True
                            for e in _stop(agent_def):
                                yield e
                            if state.completed:
                                yield "data: [DONE]\n\n"
                                return

                    if retry_triggered:
                        continue
                    if progress_made:
                        continue

            for agent_def in workflow_def.agents:
                if agent_def.agent in completed_agents:
                    continue
                if forced_next and agent_def.agent != forced_next:
                    continue
                if agent_def.depends_on and not set(agent_def.depends_on).issubset(
                    completed_agents
                ):
                    continue

                if not await self._condition_met(
                    agent_def,
                    state.context,
                    session,
                    request,
                    access_token,
                    span,
                    workflow_def,
                    execution_id=state.execution_id,
                ):
                    state_management.skip_agent(
                        agent_def,
                        state,
                        completed_agents,
                        self.state_store.save_state,
                    )
                    progress_made = True
                    yield self._record_event(
                        state,
                        f"Skipping {agent_def.agent} (condition not met)",
                        status="skipped",
                    )
                    continue

                async for result in self._execute_agent(
                    workflow_def,
                    agent_def,
                    state,
                    session,
                    request,
                    access_token,
                    span,
                    persist_inner_thinking,
                    no_reroute,
                ):
                    if isinstance(result, str):
                        yield result
                        continue
                    action = await _status(result)
                    for e in action.get("events", []):
                        yield e
                    forced_next, last_visible_output, retry, _ = (
                        agent_dispatch.apply_status_action(
                            action,
                            forced_next,
                            last_visible_output,
                        )
                    )
                    if action.get("should_return"):
                        return
                    if retry:
                        retry_triggered = True
                        progress_made = True
                    if action.get("should_break"):
                        break

                if state.completed:
                    yield "data: [DONE]\n\n"
                    return
                if retry_triggered:
                    break

                completed_agents.add(agent_def.agent)
                progress_made = True
                for e in _stop(agent_def):
                    yield e
                if state.completed:
                    yield "data: [DONE]\n\n"
                    return

            if retry_triggered:
                continue

            if not progress_made:
                message, metadata = error_analysis.build_deadlock_error(
                    workflow_def.agents,
                    completed_agents,
                    forced_next,
                    state.context,
                )
                state.completed = True
                self.state_store.save_state(state)
                yield self._record_event(
                    state, message, status="error", metadata=metadata
                )
                yield "data: [DONE]\n\n"
                return

        for e in agent_dispatch.build_workflow_completion(
            state,
            completed_agents,
            workflow_def,
            last_visible_output,
            self._record_event,
            self.state_store.save_state,
        ):
            yield e

    async def _execute_agent(
        self,
        workflow_def: WorkflowDefinition,
        agent_def: WorkflowAgentDef,
        state: WorkflowExecutionState,
        session: Any,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
        persist_inner_thinking: bool = False,
        no_reroute: bool = False,
    ) -> AsyncGenerator[Any, None]:
        state.current_agent = agent_def.agent

        logger.info("[_execute_agent] Starting agent '%s'", agent_def.agent)

        start_text = tag_parser.start_text(agent_def)
        yield self._record_event(state, start_text)

        agent_context = state.context.setdefault("agents", {}).setdefault(
            agent_def.agent,
            {"content": "", "pass_through": agent_def.pass_through},
        )
        agent_context["pass_through"] = agent_def.pass_through
        agent_context["completed"] = False
        was_awaiting_feedback = bool(agent_context.pop("awaiting_feedback", None))
        had_feedback = bool(agent_context.pop("had_feedback", None))

        logger.info(
            "[_execute_agent] Agent '%s' feedback_state=(%s, %s)",
            agent_def.agent,
            was_awaiting_feedback,
            had_feedback,
        )

        pending_user_reroute = agent_context.pop("pending_user_reroute", None)
        if pending_user_reroute:
            result = reroute.process_pending_user_reroute(
                pending_user_reroute,
                agent_context,
                state.context,
                append_user_message_fn=lambda msg: state_management.append_user_message(
                    state, msg, update_user_query=False
                ),
            )
            is_self_reroute = (
                isinstance(result, dict)
                and result.get("status") == "reroute"
                and result.get("target") == agent_def.agent
            )
            if is_self_reroute:
                logger.info(
                    "[_execute_agent] Agent '%s' received feedback self-reroute; continuing execution in-place.",
                    agent_def.agent,
                )
                agent_context["completed"] = False
                self.state_store.save_state(state)
            else:
                self.state_store.save_state(state)
                yield result
                return

        system_prompt = await self._resolve_prompt(session, agent_def)

        if had_feedback:
            system_prompt = (
                f"{system_prompt}\n\n"
                "IMPORTANT: You have already requested user feedback and the user has responded. "
                "Your previous content and the user's response are included in your context. "
                "DO NOT ask for feedback again. Instead, process the user's response, extract any "
                "required information (like IDs, selections, etc.), and emit the appropriate "
                "<reroute> and <return> tags to continue the workflow."
            )
        agent_context_payload = context_builder.build_agent_context(
            agent_def,
            state,
            resolve_arg_reference=self.agent_executor.resolve_arg_reference,
            get_original_user_prompt=state_management.get_original_user_prompt,
        )
        scoped_context = isinstance(agent_def.context, list)
        context_summary = context_builder.create_context_summary(
            agent_context_payload, scoped=scoped_context
        )

        user_prompt = (
            state.context.get("user_query")
            or context_builder.extract_user_message(request)
            or ""
        )
        user_content = context_builder.build_user_content(
            agent_def.agent,
            workflow_def.root_intent,
            user_prompt,
            context_summary,
            state.context,
            is_loop=bool(workflow_def.loop),
        )

        modified_tools, original_tools = await context_builder.resolve_tools(
            session, agent_def, self.tool_service, self.agent_executor
        )
        tools_available = context_builder.normalize_tools(modified_tools or [])
        tools_for_validation = context_builder.normalize_tools(original_tools or [])
        needs_workflow_description = context_builder.needs_workflow_description(
            tools_available
        )
        if needs_workflow_description:
            system_prompt = f"{system_prompt}\n\n{context_builder.workflow_description_instruction()}"

        tool_configs = self.agent_executor.get_tool_configs_for_agent(agent_def)
        streaming_tools = {
            tc.name for tc in tool_configs if getattr(tc, "streaming", False)
        }

        agent_request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_content),
            ],
            model=request.model or TGI_MODEL_NAME,
            stream=True,
            tools=tools_available,
            stop=request.stop,
        )

        content_text = ""

        original_prompt = (
            state_management.get_original_user_prompt(state) or user_prompt
        )
        arg_injector_fn = build_arg_injector_fn(
            agent_def,
            state.context,
            needs_workflow_description,
            original_prompt,
            workflow_description_tool_names=self.WORKFLOW_DESCRIPTION_TOOL_NAMES,
            workflow_description_context_key=self.WORKFLOW_DESCRIPTION_CONTEXT_KEY,
            ensure_workflow_description_arg_fn=context_builder.ensure_workflow_description_arg,
        )

        result_capture = (
            ToolResultCapture(agent_def.agent, agent_def.returns)
            if agent_def.returns
            else None
        )

        stop_after_tool_results = None
        if not no_reroute and reroute.has_tool_reroute(agent_def.reroute):

            def stop_after_tool_results(raw_results: list[dict[str, Any]]) -> bool:
                return reroute.should_stop_after_tool_results(
                    agent_def.reroute, raw_results
                )

        runner_stream = self.tool_chat_runner.stream_chat_with_tools(
            session=session,
            messages=[],
            available_tools=tools_available,
            chat_request=agent_request,
            access_token=access_token or "",
            parent_span=span,
            emit_think_messages=agent_def.pass_through,
            arg_injector=arg_injector_fn,
            tools_for_validation=tools_for_validation,
            streaming_tools=streaming_tools if streaming_tools else None,
            stop_after_tool_results=stop_after_tool_results,
        )

        stream_result = stream_processor.StreamResult()
        user_msg_for_progress = context_builder.extract_user_message(request) or ""

        async def _progress_handler_fn(payload, *, passthrough_history=None):
            return await stream_processor.run_progress_handler(
                state=state,
                agent_def=agent_def,
                progress_payload=payload,
                user_message=user_msg_for_progress,
                model_name=request.model or TGI_MODEL_NAME,
                passthrough_history=passthrough_history,
                llm_stream_fn=lambda req: self.llm_client.stream_completion(
                    req, access_token or "", span
                ),
                chunk_reader_fn=chunk_reader,
                record_event_fn=self._record_event,
                format_chunk_fn=self.chunk_formatter.format_chunk,
            )

        try:
            async for event in stream_processor.process_agent_stream(
                runner_stream=runner_stream,
                chunk_reader_fn=chunk_reader,
                agent_def=agent_def,
                state=state,
                agent_context=agent_context,
                result_capture=result_capture,
                result=stream_result,
                record_event_fn=self._record_event,
                run_progress_handler_fn=_progress_handler_fn,
            ):
                yield event
            content_text = stream_result.content_text
        except ArgResolutionError as exc:
            logger.error("[WorkflowEngine] Argument resolution failed: %s", exc)
            agent_context["content"] = ""
            agent_context["completed"] = True
            error_text = (
                "Workflow configuration error while preparing tool arguments.\n"
                f"{exc}"
            )
            state.completed = True
            self.state_store.save_state(state)
            yield self._record_event(state, error_text, status="error")
            yield "data: [DONE]\n\n"
            return

        async def _routing_decide(ad, reason, ctx, req, at, sp, *, execution_id=None):
            return await self.routing_service.routing_decide_next_agent(
                session,
                workflow_def,
                ad,
                reason,
                ctx,
                req,
                at,
                sp,
                execution_id=execution_id,
            )

        async for event in agent_result.finalize_agent_result(
            agent_def=agent_def,
            workflow_def=workflow_def,
            state=state,
            agent_context=agent_context,
            content_text=content_text,
            tool_errors=stream_result.tool_errors,
            tool_outcomes=stream_result.tool_outcomes,
            passthrough_history=stream_result.passthrough_history,
            persist_inner_thinking=persist_inner_thinking,
            was_awaiting_feedback=was_awaiting_feedback,
            had_feedback=had_feedback,
            result_capture=result_capture,
            no_reroute=no_reroute,
            request=request,
            access_token=access_token,
            span=span,
            record_event_fn=self._record_event,
            save_fn=self.state_store.save_state,
            render_feedback_question_fn=self.feedback_service.render_feedback_question,
            routing_decide_fn=_routing_decide,
            max_return_retries=self.MAX_RETURN_RETRIES,
        ):
            yield event

    async def _resolve_prompt(self, session: Any, agent_def: WorkflowAgentDef) -> str:
        prompt_text = agent_def.description or f"You are the {agent_def.agent} agent."
        try:
            prompt = await self.prompt_service.find_prompt_by_name_or_role(
                session, agent_def.agent
            )
            if prompt:
                prompt_text = await self.prompt_service.get_prompt_content(
                    session, prompt
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                f"[WorkflowEngine] Using default prompt for {agent_def.agent}: {exc}"
            )
        return tag_parser.append_agent_guidelines(
            prompt_text or f"You are the {agent_def.agent} agent.",
            agent_def,
        )

    async def _condition_met(
        self,
        agent_def: WorkflowAgentDef,
        context: dict,
        session: Any,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
        workflow_def: WorkflowDefinition,
        execution_id: Optional[str] = None,
    ) -> bool:
        return await reroute.evaluate_condition(
            agent_def,
            context,
            session,
            request,
            access_token,
            span,
            workflow_def,
            routing_service=self.routing_service,
            execution_id=execution_id,
        )

    def _record_event(
        self,
        state: WorkflowExecutionState,
        text: str,
        status: str = "in_progress",
        role: str = "assistant",
        finish_reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        event = self.chunk_formatter.format_chunk(
            state=state,
            content=text,
            status=status,
            role=role,
            finish_reason=finish_reason,
            metadata=metadata,
        )
        state.events.append(event)
        self.state_store.save_state(state)
        return event

    def cancel_execution(self, execution_id: str, reason: Optional[str] = None) -> bool:
        state = self.state_store.load_execution(execution_id)
        if not state:
            return False
        if state.completed:
            return True
        state.completed = True
        state.awaiting_feedback = False
        message = reason or "Workflow cancelled."
        self._record_event(state, message, status="cancelled", role="system")
        return True
