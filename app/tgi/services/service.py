import logging
import json
import uuid
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from opentelemetry import trace

from app.tgi.models import (
    Message,
    MessageRole,
    Usage,
    Tool,
    ToolCall,
    FunctionDefinition,
)
from app.session import MCPSessionBase

logger = logging.getLogger("uvicorn.error")
tracer = trace.get_tracer(__name__)


class TGIService:
    """Service for handling TGI-compatible chat completions with MCP integration."""

    def __init__(self):
        self.logger = logger

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
                    self.logger.debug("[TGI] No prompts available from MCP server")
                    span.set_attribute("prompt.found", False)
                    return None

                prompts = prompts_result.prompts
                self.logger.debug(f"[TGI] Found {len(prompts)} prompts available")

                # If specific prompt name requested, search for it
                if prompt_name:
                    for prompt in prompts:
                        if prompt.name == prompt_name:
                            span.set_attribute("prompt.found_name", prompt.name)
                            span.set_attribute("prompt.found", True)
                            self.logger.debug(
                                f"[TGI] Found requested prompt: {prompt_name}"
                            )
                            return prompt

                    # Prompt not found
                    span.set_attribute("prompt.found", False)
                    self.logger.warning(
                        f"[TGI] Requested prompt '{prompt_name}' not found"
                    )
                    return None

                # Look for 'system' prompt or one with role=system
                for prompt in prompts:
                    if prompt.name == "system":
                        span.set_attribute("prompt.found_name", prompt.name)
                        span.set_attribute("prompt.found", True)
                        self.logger.debug("[TGI] Found 'system' prompt")
                        return prompt

                # Look for any prompt with role=system in its description or metadata
                for prompt in prompts:
                    # Check if prompt has role information
                    if hasattr(prompt, "description") and prompt.description:
                        if (
                            "role=system" in prompt.description.lower()
                            or "system" in prompt.description.lower()
                        ):
                            span.set_attribute("prompt.found_name", prompt.name)
                            span.set_attribute("prompt.found", True)
                            self.logger.debug(
                                f"[TGI] Found system-role prompt: {prompt.name}"
                            )
                            return prompt

                # Return first available prompt as fallback
                if prompts:
                    first_prompt = prompts[0]
                    span.set_attribute("prompt.found_name", first_prompt.name)
                    span.set_attribute("prompt.found", True)
                    self.logger.debug(
                        f"[TGI] Using first available prompt: {first_prompt.name}"
                    )
                    return first_prompt

                span.set_attribute("prompt.found", False)
                return None

            except Exception as e:
                self.logger.error(f"[TGI] Error finding prompt: {str(e)}")
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
                    self.logger.error(f"[TGI] {error_msg}")
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
                    f"[TGI] Retrieved prompt content: {len(content)} characters"
                )
                span.set_attribute("prompt.content_length", len(content))
                return content

            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"Error retrieving prompt content: {str(e)}"
                self.logger.error(f"[TGI] {error_msg}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise HTTPException(status_code=500, detail=error_msg)

    async def filter_available_tools(
        self, session: MCPSessionBase, requested_tools: List[Tool]
    ) -> List[Tool]:
        """
        Filter requested tools to only include those available on the MCP server.

        Args:
            session: The MCP session to use
            requested_tools: List of tools requested in the chat completion

        Returns:
            Filtered list of available tools
        """
        with tracer.start_as_current_span("filter_tools") as span:
            span.set_attribute("tools.requested_count", len(requested_tools))

            try:
                # Get available tools from MCP server
                tools_result = await session.list_tools()
                if not tools_result or not hasattr(tools_result, "tools"):
                    self.logger.debug("[TGI] No tools available from MCP server")
                    span.set_attribute("tools.available_count", 0)
                    span.set_attribute("tools.filtered_count", 0)
                    return []

                available_tool_names = {tool.name for tool in tools_result.tools}
                span.set_attribute("tools.available_count", len(available_tool_names))
                self.logger.debug(f"[TGI] Available MCP tools: {available_tool_names}")

                # Filter requested tools
                filtered_tools = []
                for tool in requested_tools:
                    if tool.function.name in available_tool_names:
                        filtered_tools.append(tool)
                        self.logger.debug(
                            f"[TGI] Tool '{tool.function.name}' is available"
                        )
                    else:
                        self.logger.warning(
                            f"[TGI] Tool '{tool.function.name}' not available on MCP server"
                        )

                span.set_attribute("tools.filtered_count", len(filtered_tools))
                self.logger.debug(
                    f"[TGI] Filtered to {len(filtered_tools)} available tools"
                )
                return filtered_tools

            except Exception as e:
                self.logger.error(f"[TGI] Error filtering tools: {str(e)}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

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
                    self.logger.debug("[TGI] No tools available from MCP server")
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
                    self.logger.debug(f"[TGI] Added MCP tool: {mcp_tool.name}")

                span.set_attribute("tools.count", len(openai_tools))
                self.logger.debug(f"[TGI] Retrieved {len(openai_tools)} MCP tools")
                return openai_tools

            except Exception as e:
                self.logger.error(f"[TGI] Error getting MCP tools: {str(e)}")
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
                    self.logger.error(f"[TGI] {error_msg}")
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

                    self.logger.error(f"[TGI] Tool execution error: {error_content}")
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
                    f"[TGI] Tool '{tool_call.function.name}' executed successfully"
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
                self.logger.error(f"[TGI] {error_msg}")
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
                        self.logger.debug(f"[TGI] Added system prompt: {prompt.name}")
                        span.set_attribute("prompt.added", True)
                        span.set_attribute("prompt.name", prompt.name)

                # Add original messages
                prepared_messages.extend(messages)

                span.set_attribute("messages.final_count", len(prepared_messages))
                self.logger.debug(f"[TGI] Prepared {len(prepared_messages)} messages")

                return prepared_messages

            except Exception as e:
                self.logger.error(f"[TGI] Error preparing messages: {str(e)}")
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
