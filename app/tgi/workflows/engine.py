import json
import logging
import re
import uuid
from typing import Any, AsyncGenerator, Optional

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.protocols.chunk_reader import chunk_reader
from app.tgi.services.prompt_service import PromptService
from app.tgi.workflows.models import (
    WorkflowAgentDef,
    WorkflowDefinition,
    WorkflowExecutionState,
)
from app.tgi.workflows.repository import WorkflowRepository
from app.tgi.workflows.state import WorkflowStateStore
from app.vars import TGI_MODEL_NAME

logger = logging.getLogger("uvicorn.error")


class WorkflowEngine:
    """
    Executes workflows by streaming agent outputs and persisting state for resumption.
    """

    def __init__(
        self,
        repository: WorkflowRepository,
        state_store: WorkflowStateStore,
        llm_client: Any,
        prompt_service: Optional[PromptService] = None,
    ):
        self.repository = repository
        self.state_store = state_store
        self.llm_client = llm_client
        self.prompt_service = prompt_service or PromptService()

    async def start_or_resume_workflow(
        self,
        session: Any,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
    ) -> Optional[AsyncGenerator[str, None]]:
        workflow_def = self._select_workflow(request)
        if not workflow_def:
            return None

        execution_id = request.workflow_execution_id or str(uuid.uuid4())
        state = self.state_store.get_or_create(execution_id, workflow_def.flow_id)
        state.context.setdefault("agents", {})

        async def _runner():
            # Replay stored events for resume
            for event in state.events:
                yield event

            if state.completed:
                yield "data: [DONE]\n\n"
                return

            user_message = self._extract_user_message(request)

            # If we are waiting for user feedback, either resume with it or pause again
            if state.awaiting_feedback:
                if user_message:
                    self._merge_feedback(state, user_message)
                    yield self._record_event(
                        state, f"Received feedback: {user_message}"
                    )
                else:
                    yield self._record_event(
                        state, "Workflow paused awaiting user feedback."
                    )
                    yield "data: [DONE]\n\n"
                    return

            # Add a start marker once
            if not state.events:
                yield self._record_event(
                    state, f"Routing workflow {workflow_def.flow_id}"
                )

            async for event in self._run_agents(
                workflow_def, state, session, request, access_token, span
            ):
                yield event

            if state.completed:
                yield "data: [DONE]\n\n"

        return _runner()

    def _select_workflow(
        self, request: ChatCompletionRequest
    ) -> Optional[WorkflowDefinition]:
        use_workflow = getattr(request, "use_workflow", None)
        if not use_workflow:
            return None

        user_message = self._extract_user_message(request)
        if use_workflow is True:
            return self.repository.match_workflow(user_message or "")

        try:
            return self.repository.get(str(use_workflow))
        except Exception:
            return None

    async def _run_agents(
        self,
        workflow_def: WorkflowDefinition,
        state: WorkflowExecutionState,
        session: Any,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
    ) -> AsyncGenerator[str, None]:
        completed_agents = set(state.context.get("agents", {}).keys())
        forced_next: Optional[str] = None

        last_visible_output: Optional[str] = None

        while len(completed_agents) < len(workflow_def.agents):
            progress_made = False
            for agent_def in workflow_def.agents:
                if agent_def.agent in completed_agents:
                    continue
                if forced_next and agent_def.agent != forced_next:
                    continue

                if agent_def.depends_on and not set(agent_def.depends_on).issubset(
                    completed_agents
                ):
                    continue

                if not self._condition_met(agent_def, state.context):
                    state.context["agents"][agent_def.agent] = {
                        "content": "",
                        "pass_through": agent_def.pass_through,
                        "skipped": True,
                        "reason": "condition_not_met",
                    }
                    self.state_store.save_state(state)
                    completed_agents.add(agent_def.agent)
                    progress_made = True
                    yield self._record_event(
                        state, f"Skipping {agent_def.agent} (condition not met)"
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
                ):
                    if isinstance(result, str):
                        yield result
                        continue

                    status = result.get("status")
                    if (
                        status == "done"
                        and result.get("content")
                        and not result.get("pass_through")
                    ):
                        last_visible_output = result.get("content")
                    if status == "feedback":
                        return
                    if status == "reroute":
                        forced_next = result.get("target")
                        if forced_next:
                            yield self._record_event(
                                state, f"Rerouting to {forced_next}"
                            )
                    else:
                        forced_next = None

                completed_agents.add(agent_def.agent)
                progress_made = True

            if not progress_made:
                # Avoid infinite loop if dependencies cannot be satisfied
                break

        if not state.awaiting_feedback and len(completed_agents) == len(
            workflow_def.agents
        ):
            if last_visible_output:
                yield self._record_event(
                    state, f"Result: {last_visible_output.strip()}"
                )
            state.completed = True
            yield self._record_event(state, "Workflow complete")
            self.state_store.save_state(state)

    async def _execute_agent(
        self,
        workflow_def: WorkflowDefinition,
        agent_def: WorkflowAgentDef,
        state: WorkflowExecutionState,
        session: Any,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
    ) -> AsyncGenerator[Any, None]:
        state.current_agent = agent_def.agent

        start_text = self._start_text(agent_def)
        yield self._record_event(state, start_text)

        system_prompt = await self._resolve_prompt(session, agent_def)
        context_json = json.dumps(state.context, ensure_ascii=False, default=str)
        user_prompt = self._extract_user_message(request) or ""
        user_content = (
            f"<agent:{agent_def.agent}> Goal: {workflow_def.root_intent}\n"
            f"User request: {user_prompt}\n"
            f"Context: {context_json}"
        )

        agent_request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_content),
            ],
            model=request.model or TGI_MODEL_NAME,
            stream=True,
        )

        stream = self.llm_client.stream_completion(
            agent_request, access_token or "", span
        )
        content_text = ""

        async with chunk_reader(stream, enable_tracing=False) as reader:
            async for parsed in reader.as_parsed():
                if parsed.is_done:
                    break
                if parsed.content:
                    content_text += parsed.content
                    if agent_def.pass_through:
                        yield self._record_event(state, parsed.content)

        reroute_reason = self._extract_tag(content_text, "reroute")
        feedback_needed = bool(self._extract_tag(content_text, "user_feedback_needed"))
        cleaned_content = self._strip_tags(content_text)

        state.context["agents"][agent_def.agent] = {
            "content": cleaned_content.strip(),
            "pass_through": agent_def.pass_through,
        }

        if reroute_reason:
            state.context["agents"][agent_def.agent]["reroute_reason"] = reroute_reason

        if feedback_needed:
            state.awaiting_feedback = True
            self.state_store.save_state(state)
            yield self._record_event(state, "User feedback needed before continuing.")
            yield {"status": "feedback", "content": cleaned_content}
            return

        if reroute_reason and agent_def.reroute:
            reroute_on = agent_def.reroute.get("on") or []
            if reroute_reason in reroute_on:
                self.state_store.save_state(state)
                yield {"status": "reroute", "target": agent_def.reroute.get("to")}
                return

        self.state_store.save_state(state)
        yield {
            "status": "done",
            "content": cleaned_content,
            "pass_through": agent_def.pass_through,
        }

    def _start_text(self, agent_def: WorkflowAgentDef) -> str:
        if agent_def.agent.startswith("get_"):
            noun = agent_def.agent.replace("get_", "").replace("_", " ")
            return f"Fetching your {noun}..."
        if agent_def.agent.startswith("ask_"):
            noun = agent_def.agent.replace("ask_", "").replace("_", " ")
            return f"Asking for {noun}..."
        if agent_def.description:
            return agent_def.description
        return f"Running agent {agent_def.agent}"

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
        return prompt_text or f"You are the {agent_def.agent} agent."

    def _extract_user_message(self, request: ChatCompletionRequest) -> Optional[str]:
        for message in reversed(request.messages):
            if message.role == MessageRole.USER:
                return message.content
        return None

    def _condition_met(self, agent_def: WorkflowAgentDef, context: dict) -> bool:
        if not agent_def.when:
            return True
        try:
            return bool(
                eval(agent_def.when, {"__builtins__": {}}, {"context": context})
            )
        except Exception:
            return False

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        match = re.search(
            f"<{tag}>(.*?)</{tag}>", text or "", re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return None

    def _strip_tags(self, text: str) -> str:
        stripped = re.sub(r"<(/?)(reroute|user_feedback_needed)>", "", text or "")
        return re.sub(r"<no_reroute>", "", stripped, flags=re.IGNORECASE)

    def _record_event(self, state: WorkflowExecutionState, text: str) -> str:
        event = f"data: {text}\n\n"
        state.events.append(event)
        self.state_store.save_state(state)
        return event

    def _merge_feedback(self, state: WorkflowExecutionState, feedback: str) -> None:
        state.awaiting_feedback = False
        if state.current_agent:
            agent_entry = state.context["agents"].setdefault(
                state.current_agent, {"content": "", "pass_through": False}
            )
            agent_entry["content"] = (
                agent_entry.get("content", "") + f" {feedback}"
            ).strip()
        state.context.setdefault("feedback", []).append(feedback)
        self.state_store.save_state(state)
