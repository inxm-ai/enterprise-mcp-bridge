import json
import logging
import re
import uuid
from typing import Any, AsyncGenerator, Optional

from app.tgi.models import ChatCompletionRequest, Message, MessageRole
from app.tgi.protocols.chunk_reader import chunk_reader
from app.tgi.services.prompt_service import PromptService
from app.tgi.services.tool_service import ToolService
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
        tool_service: Optional[ToolService] = None,
    ):
        self.repository = repository
        self.state_store = state_store
        self.llm_client = llm_client
        self.prompt_service = prompt_service or PromptService()
        self.tool_service = tool_service or ToolService()

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
        user_message = self._extract_user_message(request)
        no_reroute = self._has_no_reroute(user_message)

        async def _runner():
            # Replay stored events for resume
            for event in state.events:
                yield event

            if state.completed:
                yield "data: [DONE]\n\n"
                return

            user_message_local = user_message
            routing_check = await self._routing_intent_check(
                session,
                workflow_def,
                user_message_local,
                request,
                access_token,
                span,
            )
            if routing_check and routing_check.get("reroute") and not no_reroute:
                reason = routing_check["reroute"]
                state.completed = True
                self.state_store.save_state(state)
                yield f"data: <reroute>{reason}</reroute>\n\n"
                yield "data: [DONE]\n\n"
                return

            # If we are waiting for user feedback, either resume with it or pause again
            if state.awaiting_feedback:
                if user_message_local:
                    self._merge_feedback(state, user_message_local)
                    yield self._record_event(
                        state, f"Received feedback: {user_message_local}"
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
                workflow_def,
                state,
                session,
                request,
                access_token,
                span,
                no_reroute=no_reroute,
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
        no_reroute: bool = False,
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

                if not await self._condition_met(
                    agent_def,
                    state.context,
                    session,
                    request,
                    access_token,
                    span,
                    workflow_def,
                ):
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
                    no_reroute,
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
        no_reroute: bool,
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
            tools=await self._resolve_tools(session, agent_def),
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
        if reroute_reason and not no_reroute:
            if agent_def.reroute:
                reroute_on = agent_def.reroute.get("on") or []
                if reroute_reason in reroute_on:
                    self.state_store.save_state(state)
                    yield {"status": "reroute", "target": agent_def.reroute.get("to")}
                    return
            dynamic_target = await self._routing_decide_next_agent(
                session,
                workflow_def,
                agent_def,
                reroute_reason,
                state.context,
                request,
                access_token,
                span,
            )
            if dynamic_target:
                self.state_store.save_state(state)
                yield {"status": "reroute", "target": dynamic_target}
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
        return self._append_agent_guidelines(
            prompt_text or f"You are the {agent_def.agent} agent."
        )

    def _append_agent_guidelines(self, prompt_text: str) -> str:
        guidelines = (
            "Workflow guidelines:\n"
            "- If you need more info from the user, respond only with "
            "<user_feedback_needed>Your question</user_feedback_needed>.\n"
            "- If the request does not match the goal or cannot be solved by this workflow, respond only with "
            "<reroute>reason</reroute>.\n"
            "- Respect <no_reroute> if present in the latest user message; otherwise honor reroute signals.\n"
            "- Keep responses concise; include only the necessary tag when using these markers."
        )
        return f"{prompt_text}\n\n{guidelines}"

    async def _resolve_tools(
        self, session: Any, agent_def: WorkflowAgentDef
    ) -> Optional[list]:
        if not self.tool_service or not hasattr(session, "list_tools"):
            return None
        try:
            all_tools = await self.tool_service.get_all_mcp_tools(session)
        except Exception:
            return None
        if agent_def.tools is None:
            return all_tools
        if isinstance(agent_def.tools, list) and len(agent_def.tools) == 0:
            return []
        names = set(agent_def.tools or [])
        filtered = [
            tool
            for tool in all_tools
            if (
                tool.get("function", {}).get("name")
                if isinstance(tool, dict)
                else getattr(getattr(tool, "function", None), "name", None)
            )
            in names
        ]
        return filtered

    def _extract_user_message(self, request: ChatCompletionRequest) -> Optional[str]:
        for message in reversed(request.messages):
            if message.role == MessageRole.USER:
                return message.content
        return None

    def _has_no_reroute(self, content: Optional[str]) -> bool:
        return bool(content and "<no_reroute>" in content)

    async def _condition_met(
        self,
        agent_def: WorkflowAgentDef,
        context: dict,
        session: Any,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
        workflow_def: WorkflowDefinition,
    ) -> bool:
        if not agent_def.when:
            return True
        decision = await self._routing_when_check(
            session, agent_def, context, request, access_token, span, workflow_def
        )
        if decision is not None:
            return decision
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

    async def _routing_intent_check(
        self,
        session: Any,
        workflow_def: WorkflowDefinition,
        user_message: Optional[str],
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
    ) -> Optional[dict]:
        payload = (
            "ROUTING_INTENT_CHECK\n"
            f"root_intent={workflow_def.root_intent}\n"
            f"user_message={user_message or ''}\n"
            f"agents={[a.agent for a in workflow_def.agents]}"
        )
        response = await self._call_routing_agent(
            session, request, access_token, span, payload
        )
        reroute_reason = self._extract_tag(response, "reroute")
        if reroute_reason:
            return {"reroute": reroute_reason}
        return None

    async def _routing_when_check(
        self,
        session: Any,
        agent_def: WorkflowAgentDef,
        context: dict,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
        workflow_def: WorkflowDefinition,
    ) -> Optional[bool]:
        payload = (
            "ROUTING_WHEN_CHECK\n"
            f"agent={agent_def.agent}\n"
            f"when={agent_def.when}\n"
            f"root_intent={workflow_def.root_intent}\n"
            f"context={json.dumps(context, ensure_ascii=False)}\n"
            f"user_message={self._extract_user_message(request) or ''}"
        )
        response = await self._call_routing_agent(
            session, request, access_token, span, payload
        )
        run_value = self._extract_run_tag(response)
        return run_value

    async def _routing_decide_next_agent(
        self,
        session: Any,
        workflow_def: WorkflowDefinition,
        agent_def: WorkflowAgentDef,
        reroute_reason: str,
        context: dict,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
    ) -> Optional[str]:
        payload = (
            "ROUTING_REROUTE_DECISION\n"
            f"agent={agent_def.agent}\n"
            f"reason={reroute_reason}\n"
            f"available_agents={[a.agent for a in workflow_def.agents]}\n"
            f"context={json.dumps(context, ensure_ascii=False)}"
        )
        response = await self._call_routing_agent(
            session, request, access_token, span, payload
        )
        return self._extract_next_agent(response)

    async def _call_routing_agent(
        self,
        session: Any,
        request: ChatCompletionRequest,
        access_token: Optional[str],
        span,
        payload: str,
    ) -> str:
        routing_prompt = await self._resolve_routing_prompt(session)
        routing_request = ChatCompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=routing_prompt),
                Message(role=MessageRole.USER, content=payload),
            ],
            model=request.model or TGI_MODEL_NAME,
            stream=True,
        )
        text = ""
        stream = self.llm_client.stream_completion(
            routing_request, access_token or "", span
        )
        async with chunk_reader(stream, enable_tracing=False) as reader:
            async for parsed in reader.as_parsed():
                if parsed.is_done:
                    break
                if parsed.content:
                    text += parsed.content
        return text.strip()

    async def _resolve_routing_prompt(self, session: Any) -> str:
        default_prompt = (
            "You are the routing_agent. Decide whether a workflow matches and which agent to run next. "
            "Respond only with tags: <run>true/false</run>, <reroute>REASON</reroute>, "
            "and optional <next_agent>agent_name</next_agent>. "
            "Use provided context, root_intent, when expression, and user message."
        )
        try:
            prompt = await self.prompt_service.find_prompt_by_name_or_role(
                session, "routing_agent"
            )
            if prompt:
                return await self.prompt_service.get_prompt_content(session, prompt)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"[WorkflowEngine] Using default routing prompt: {exc}")
        return default_prompt

    def _extract_run_tag(self, text: str) -> Optional[bool]:
        match = re.search(r"<run>(true|false)</run>", text or "", re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        return None

    def _extract_next_agent(self, text: str) -> Optional[str]:
        next_agent = self._extract_tag(text, "next_agent")
        if next_agent:
            return next_agent.strip()
        return None
