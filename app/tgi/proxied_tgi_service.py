import logging
import json
import os
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
import aiohttp
from fastapi import HTTPException
from opentelemetry import trace

from app.tgi.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Message,
    MessageRole,
    Choice,
    DeltaMessage,
    Usage,
    Tool,
    ToolCall,
    FunctionDefinition,
)
from app.session import MCPSessionBase

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class ProxiedTGIService:
    """Service that proxies chat completions to an actual deployed LLM model."""

    def __init__(self):
        self.logger = logger
        self.tgi_url = os.environ.get("TGI_URL", "https://api.openai.com/v1")
        self.tgi_token = os.environ.get("TGI_TOKEN", "")

        # Ensure TGI_URL doesn't end with slash for consistent URL building
        if self.tgi_url.endswith("/"):
            self.tgi_url = self.tgi_url[:-1]

        self.logger.info(f"[ProxiedTGI] Initialized with URL: {self.tgi_url}")
        if self.tgi_token:
            self.logger.info(
                f"[ProxiedTGI] Using authentication token: {self.tgi_token[:10]}..."
            )
        else:
            self.logger.info("[ProxiedTGI] No authentication token configured")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "mcp-rest-server/1.0",
        }

        if self.tgi_token:
            headers["Authorization"] = f"Bearer {self.tgi_token}"

        return headers

    async def find_prompt_by_name_or_role(
        self, session: MCPSessionBase, prompt_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a prompt by name or by role=system, or return the first available prompt.

        Args:
            session: The MCP session to use
            prompt_name: Optional specific prompt name to search for

        Returns:
            The found prompt data or None if no prompts available
        """
        with tracer.start_as_current_span("find_prompt") as span:
            span.set_attribute("prompt.requested_name", prompt_name or "")

            try:
                # Get all available prompts
                prompts_result = await session.list_prompts()
                if not prompts_result or not hasattr(prompts_result, "prompts"):
                    self.logger.debug(
                        "[ProxiedTGI] No prompts available from MCP server"
                    )
                    span.set_attribute("prompt.found", False)
                    return None

                prompts = prompts_result.prompts
                self.logger.debug(
                    f"[ProxiedTGI] Found {len(prompts)} prompts available"
                )

                # If specific prompt name requested, search for it
                if prompt_name:
                    for prompt in prompts:
                        if prompt.name == prompt_name:
                            span.set_attribute("prompt.found_name", prompt.name)
                            span.set_attribute("prompt.found", True)
                            self.logger.debug(
                                f"[ProxiedTGI] Found requested prompt: {prompt_name}"
                            )
                            return prompt

                    # Prompt not found
                    span.set_attribute("prompt.found", False)
                    self.logger.warning(
                        f"[ProxiedTGI] Requested prompt '{prompt_name}' not found"
                    )
                    return None

                # Look for 'system' prompt or one with role=system
                for prompt in prompts:
                    if prompt.name == "system":
                        span.set_attribute("prompt.found_name", prompt.name)
                        span.set_attribute("prompt.found", True)
                        self.logger.debug("[ProxiedTGI] Found 'system' prompt")
                        return prompt

                # Look for any prompt with role information
                for prompt in prompts:
                    if hasattr(prompt, "description") and prompt.description:
                        if (
                            "role=system" in prompt.description.lower()
                            or "system" in prompt.description.lower()
                        ):
                            span.set_attribute("prompt.found_name", prompt.name)
                            span.set_attribute("prompt.found", True)
                            self.logger.debug(
                                f"[ProxiedTGI] Found system-role prompt: {prompt.name}"
                            )
                            return prompt

                # Return first available prompt as fallback
                if prompts:
                    first_prompt = prompts[0]
                    span.set_attribute("prompt.found_name", first_prompt.name)
                    span.set_attribute("prompt.found", True)
                    self.logger.debug(
                        f"[ProxiedTGI] Using first available prompt: {first_prompt.name}"
                    )
                    return first_prompt

                span.set_attribute("prompt.found", False)
                return None

            except Exception as e:
                self.logger.error(f"[ProxiedTGI] Error finding prompt: {str(e)}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

    async def get_prompt_content(
        self, session: MCPSessionBase, prompt: Dict[str, Any]
    ) -> str:
        """
        Get the actual content of a prompt by calling it.

        Args:
            session: The MCP session to use
            prompt: The prompt object to execute

        Returns:
            The prompt content as a string
        """
        with tracer.start_as_current_span("get_prompt_content") as span:
            span.set_attribute("prompt.name", prompt.name)

            try:
                # Call the prompt to get its content
                result = await session.call_prompt(prompt.name, {})

                if result.isError:
                    error_msg = f"Error getting prompt content: {result}"
                    self.logger.error(f"[ProxiedTGI] {error_msg}")
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)

                # Extract content from the result
                content = ""
                if hasattr(result, "messages") and result.messages:
                    for message in result.messages:
                        if hasattr(message, "content") and hasattr(
                            message.content, "text"
                        ):
                            content += message.content.text + "\n"
                        elif hasattr(message, "text"):
                            content += message.text + "\n"

                content = content.strip()
                self.logger.debug(
                    f"[ProxiedTGI] Retrieved prompt content: {len(content)} characters"
                )
                span.set_attribute("prompt.content_length", len(content))
                return content

            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"Error retrieving prompt content: {str(e)}"
                self.logger.error(f"[ProxiedTGI] {error_msg}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise HTTPException(status_code=500, detail=error_msg)

    async def get_all_mcp_tools(self, session: MCPSessionBase) -> List[Tool]:
        """
        Get all available tools from the MCP server as OpenAI-compatible tools.

        Args:
            session: The MCP session to use

        Returns:
            List of all available tools in OpenAI format
        """
        with tracer.start_as_current_span("get_all_mcp_tools") as span:
            try:
                # Get available tools from MCP server
                tools_result = await session.list_tools()
                if not tools_result or not hasattr(tools_result, "tools"):
                    self.logger.debug("[ProxiedTGI] No tools available from MCP server")
                    span.set_attribute("tools.count", 0)
                    return []

                # Convert MCP tools to OpenAI format
                openai_tools = []
                for mcp_tool in tools_result.tools:
                    tool = Tool(
                        type="function",
                        function=FunctionDefinition(
                            name=mcp_tool.name,
                            description=getattr(mcp_tool, "description", ""),
                            parameters=getattr(mcp_tool, "inputSchema", {}),
                        ),
                    )
                    openai_tools.append(tool)
                    self.logger.debug(f"[ProxiedTGI] Added MCP tool: {mcp_tool.name}")

                span.set_attribute("tools.count", len(openai_tools))
                self.logger.debug(
                    f"[ProxiedTGI] Retrieved {len(openai_tools)} MCP tools"
                )
                return openai_tools

            except Exception as e:
                self.logger.error(f"[ProxiedTGI] Error getting MCP tools: {str(e)}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

    async def execute_tool_call(
        self, session: MCPSessionBase, tool_call: ToolCall, access_token: Optional[str]
    ) -> Dict[str, Any]:
        """
        Execute a single tool call via MCP.

        Args:
            session: The MCP session to use
            tool_call: The tool call to execute
            access_token: Optional access token for authentication

        Returns:
            The result of the tool execution
        """
        with tracer.start_as_current_span("execute_tool_call") as span:
            span.set_attribute("tool.name", tool_call.function.name)
            span.set_attribute("tool.call_id", tool_call.id)

            try:
                # Parse tool arguments
                try:
                    args = (
                        json.loads(tool_call.function.arguments)
                        if tool_call.function.arguments
                        else {}
                    )
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid tool arguments JSON: {str(e)}"
                    self.logger.error(f"[ProxiedTGI] {error_msg}")
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", error_msg)
                    return {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps({"error": error_msg}),
                    }

                # Execute the tool
                result = await session.call_tool(
                    tool_call.function.name, args, access_token
                )

                if result.isError:
                    error_content = ""
                    if hasattr(result, "content") and result.content:
                        error_content = (
                            str(result.content[0].text)
                            if result.content
                            else str(result)
                        )

                    self.logger.error(
                        f"[ProxiedTGI] Tool execution error: {error_content}"
                    )
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", error_content)

                    return {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps({"error": error_content}),
                    }

                # Format successful result
                content = ""
                if hasattr(result, "structuredContent") and result.structuredContent:
                    content = json.dumps(result.structuredContent)
                elif hasattr(result, "content") and result.content:
                    if len(result.content) == 1 and hasattr(
                        result.content[0], "structuredContent"
                    ):
                        content = json.dumps(result.content[0].structuredContent)
                    else:
                        content = json.dumps(
                            [
                                {
                                    "text": (
                                        item.text
                                        if hasattr(item, "text")
                                        else str(item)
                                    )
                                }
                                for item in result.content
                            ]
                        )
                else:
                    content = json.dumps({"result": str(result)})

                self.logger.debug(
                    f"[ProxiedTGI] Tool '{tool_call.function.name}' executed successfully"
                )
                span.set_attribute("tool.success", True)

                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": content,
                }

            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}"
                self.logger.error(f"[ProxiedTGI] {error_msg}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps({"error": error_msg}),
                }

    async def prepare_messages(
        self,
        session: MCPSessionBase,
        messages: List[Message],
        prompt_name: Optional[str] = None,
    ) -> List[Message]:
        """
        Prepare messages by adding system prompt if needed.

        Args:
            session: The MCP session to use
            messages: Original messages from the request
            prompt_name: Optional specific prompt name to use

        Returns:
            Messages with system prompt prepended if found
        """
        with tracer.start_as_current_span("prepare_messages") as span:
            span.set_attribute("messages.original_count", len(messages))

            try:
                # Find appropriate prompt
                prompt = await self.find_prompt_by_name_or_role(session, prompt_name)

                prepared_messages = []

                # Add system prompt if found and no system message exists
                has_system_message = any(
                    msg.role == MessageRole.SYSTEM for msg in messages
                )

                if prompt and not has_system_message:
                    prompt_content = await self.get_prompt_content(session, prompt)
                    if prompt_content:
                        system_message = Message(
                            role=MessageRole.SYSTEM, content=prompt_content
                        )
                        prepared_messages.append(system_message)
                        self.logger.debug(
                            f"[ProxiedTGI] Added system prompt: {prompt.name}"
                        )
                        span.set_attribute("prompt.added", True)
                        span.set_attribute("prompt.name", prompt.name)

                # Add original messages
                prepared_messages.extend(messages)

                span.set_attribute("messages.final_count", len(prepared_messages))
                self.logger.debug(
                    f"[ProxiedTGI] Prepared {len(prepared_messages)} messages"
                )

                return prepared_messages

            except Exception as e:
                self.logger.error(f"[ProxiedTGI] Error preparing messages: {str(e)}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                # Return original messages on error
                return messages

    def create_completion_id(self) -> str:
        """Generate a unique completion ID."""
        return f"chatcmpl-{uuid.uuid4().hex[:29]}"

    def create_usage_stats(
        self, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> Usage:
        """Create usage statistics."""
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    async def chat_completion(
        self,
        session: MCPSessionBase,
        chat_request: ChatCompletionRequest,
        access_token: Optional[str] = None,
        prompt_name: Optional[str] = None,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """
        Execute chat completion with tool call handling, similar to chat.js flow.

        Args:
            session: The MCP session to use
            chat_request: The chat completion request
            access_token: Optional access token for tool execution
            prompt_name: Optional specific prompt name to use

        Returns:
            ChatCompletionResponse for non-streaming, AsyncGenerator for streaming
        """
        with tracer.start_as_current_span("chat_completion") as span:
            span.set_attribute("chat.model", chat_request.model or "unknown")
            span.set_attribute("chat.streaming", chat_request.stream)
            span.set_attribute("chat.messages_count", len(chat_request.messages))

            try:
                # Step 1: Prepare messages with system prompt
                prepared_messages = await self.prepare_messages(
                    session, chat_request.messages, prompt_name
                )

                # Step 2: Get available tools
                available_tools = await self.get_all_mcp_tools(session)
                span.set_attribute("chat.available_tools_count", len(available_tools))

                if chat_request.stream:
                    return self._stream_chat_with_tools(
                        session,
                        prepared_messages,
                        available_tools,
                        chat_request,
                        access_token,
                        span,
                    )
                else:
                    return await self._non_stream_chat_with_tools(
                        session,
                        prepared_messages,
                        available_tools,
                        chat_request,
                        access_token,
                        span,
                    )

            except Exception as e:
                self.logger.error(f"[ProxiedTGI] Error in chat completion: {str(e)}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

    async def _stream_chat_with_tools(
        self,
        session: MCPSessionBase,
        messages: List[Message],
        available_tools: List[Tool],
        chat_request: ChatCompletionRequest,
        access_token: Optional[str],
        parent_span,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion with tool handling."""
        messages_history = messages.copy()
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(f"[ProxiedTGI] Stream chat iteration {iteration}")

            # Create request for LLM
            llm_request = ChatCompletionRequest(
                messages=messages_history,
                model=chat_request.model,
                tools=available_tools if available_tools else None,
                tool_choice=chat_request.tool_choice,
                stream=True,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                top_p=chat_request.top_p,
            )

            # For streaming, we need to handle this differently
            # For now, let's implement a simpler approach that doesn't support tool calls in streaming
            async for chunk in self._stream_llm_completion(llm_request, parent_span):
                yield chunk
            break  # For streaming, we'll do one iteration for now

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
        messages_history = messages.copy()
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.logger.debug(f"[ProxiedTGI] Non-stream chat iteration {iteration}")

            # Create request for LLM
            llm_request = ChatCompletionRequest(
                messages=messages_history,
                model=chat_request.model,
                tools=available_tools if available_tools else None,
                tool_choice=chat_request.tool_choice,
                stream=False,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                top_p=chat_request.top_p,
            )

            # Call actual LLM
            response = await self._non_stream_llm_completion(llm_request, parent_span)

            # Check if response contains tool calls
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.tool_calls
            ):

                # Add assistant message with tool calls to history
                messages_history.append(response.choices[0].message)

                # Execute tool calls
                tool_results = await self._execute_tool_calls(
                    session,
                    response.choices[0].message.tool_calls,
                    access_token,
                    parent_span,
                )

                # Add tool results to history
                messages_history.extend(tool_results)

                # Continue to next iteration
                continue
            else:
                # No tool calls, return final response
                return response

        # If we reach here, we hit max iterations
        return ChatCompletionResponse(
            id=self.create_completion_id(),
            object="chat.completion",
            created=int(time.time()),
            model=chat_request.model or "unknown",
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
            usage=self.create_usage_stats(0, 0),
        )

    async def _stream_llm_completion(
        self, request: ChatCompletionRequest, parent_span
    ) -> AsyncGenerator[str, None]:
        """Stream completion from the actual LLM."""
        with tracer.start_as_current_span("stream_llm_completion") as span:
            span.set_attribute("llm.url", self.tgi_url)
            span.set_attribute("llm.model", request.model or "unknown")

            try:
                # Convert request to dict for JSON serialization
                payload = request.model_dump(exclude_none=True)

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.tgi_url}/chat/completions",
                        headers=self._get_headers(),
                        json=payload,
                    ) as response:

                        if not response.ok:
                            error_text = await response.text()
                            error_msg = f"LLM API error: {response.status} {error_text}"
                            self.logger.error(f"[ProxiedTGI] {error_msg}")
                            span.set_attribute("error", True)
                            span.set_attribute("error.message", error_msg)

                            # Return error as streaming response
                            error_chunk = ChatCompletionChunk(
                                id=self.create_completion_id(),
                                created=int(time.time()),
                                model=request.model or "unknown",
                                choices=[
                                    Choice(
                                        index=0,
                                        delta=DeltaMessage(
                                            content=f"Error: {error_msg}"
                                        ),
                                        finish_reason="stop",
                                    )
                                ],
                            )
                            yield f"data: {error_chunk.model_dump_json()}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        # Stream response
                        async for line in response.content:
                            line_str = line.decode("utf-8").strip()
                            if line_str:
                                yield f"{line_str}\n"

            except Exception as e:
                error_msg = f"Error streaming from LLM: {str(e)}"
                self.logger.error(f"[ProxiedTGI] {error_msg}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                # Return error as streaming response
                error_chunk = ChatCompletionChunk(
                    id=self.create_completion_id(),
                    created=int(time.time()),
                    model=request.model or "unknown",
                    choices=[
                        Choice(
                            index=0,
                            delta=DeltaMessage(content=f"Error: {error_msg}"),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

    async def _non_stream_llm_completion(
        self, request: ChatCompletionRequest, parent_span
    ) -> ChatCompletionResponse:
        """Get non-streaming completion from the actual LLM."""
        with tracer.start_as_current_span("non_stream_llm_completion") as span:
            span.set_attribute("llm.url", self.tgi_url)
            span.set_attribute("llm.model", request.model or "unknown")

            try:
                # Convert request to dict for JSON serialization, ensure stream=False
                payload = request.model_dump(exclude_none=True)
                payload["stream"] = False

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.tgi_url}/chat/completions",
                        headers=self._get_headers(),
                        json=payload,
                    ) as response:

                        if not response.ok:
                            error_text = await response.text()
                            error_msg = f"LLM API error: {response.status} {error_text}"
                            self.logger.error(f"[ProxiedTGI] {error_msg}")
                            span.set_attribute("error", True)
                            span.set_attribute("error.message", error_msg)
                            raise HTTPException(
                                status_code=response.status, detail=error_msg
                            )

                        # Parse response
                        response_data = await response.json()
                        return ChatCompletionResponse(**response_data)

            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"Error calling LLM: {str(e)}"
                self.logger.error(f"[ProxiedTGI] {error_msg}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise HTTPException(status_code=500, detail=error_msg)

    async def _execute_tool_calls(
        self,
        session: MCPSessionBase,
        tool_calls: List[ToolCall],
        access_token: Optional[str],
        parent_span,
    ) -> List[Message]:
        """Execute multiple tool calls and return tool result messages."""
        with tracer.start_as_current_span("execute_tool_calls") as span:
            span.set_attribute("tool_calls.count", len(tool_calls))

            tool_results = []

            for tool_call in tool_calls:
                result = await self.execute_tool_call(session, tool_call, access_token)

                tool_message = Message(
                    role=MessageRole.TOOL,
                    content=result["content"],
                    tool_call_id=result["tool_call_id"],
                    name=result["name"],
                )
                tool_results.append(tool_message)

            return tool_results
