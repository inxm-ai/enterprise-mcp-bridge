import pytest
from unittest.mock import Mock

from app.tgi.services.prompt_service import PromptService
from app.tgi.models import Message, MessageRole


class MockMCPSession:
    """Mock MCP session for testing."""

    def __init__(self, prompts=None):
        self.prompts = prompts or []

    async def list_prompts(self):
        """Mock list_prompts method."""
        return {"prompts": self.prompts}

    async def call_prompt(self, name, args):
        """Mock call_prompt method."""
        mock_result = Mock()
        mock_result.isError = False
        mock_result.messages = []

        # Find the prompt and return mock content
        for prompt in self.prompts:
            if prompt["name"] == name:
                mock_message = Mock()
                mock_message.content = Mock()
                mock_message.content.text = f"System prompt from {name}"
                mock_result.messages = [mock_message]
                break

        return mock_result


@pytest.fixture
def mock_prompts():
    """Create mock prompts for testing."""
    system_prompt = {
        "name": "system",
        "description": "System prompt with role=system",
        "template": {"role": "system", "content": "System prompt from system"},
    }

    custom_prompt = {
        "name": "custom",
        "description": "Custom prompt",
        "template": {"role": "system", "content": "Custom prompt content"},
    }

    return [system_prompt, custom_prompt]


@pytest.fixture
def prompt_service():
    """Create PromptService instance."""
    return PromptService()


class TestPromptService:
    """Test cases for PromptService."""

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_success(self, prompt_service, mock_prompts):
        """Test finding prompt by specific name."""
        session = MockMCPSession(prompts=mock_prompts)

        result = await prompt_service.find_prompt_by_name_or_role(session, "custom")

        assert result is not None
        assert result["name"] == "custom"

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_or_role_no_prompts(self, prompt_service):
        """Test finding prompt when no prompts are available."""
        session = MockMCPSession(prompts=[])
        result = await prompt_service.find_prompt_by_name_or_role(session)
        assert result is None

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_or_role_exception(self, prompt_service):
        """Test exception handling in find_prompt_by_name_or_role."""

        class BadSession:
            async def list_prompts(self):
                raise Exception("Session error")

        with pytest.raises(Exception):
            await prompt_service.find_prompt_by_name_or_role(BadSession(), "test")

    @pytest.mark.asyncio
    async def test_find_prompt_by_name_not_found(self, prompt_service, mock_prompts):
        """Test finding prompt by name that doesn't exist."""
        session = MockMCPSession(prompts=mock_prompts)

        result = await prompt_service.find_prompt_by_name_or_role(
            session, "nonexistent"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_prompt_content_error(self, prompt_service, mock_prompts):
        """Test get_prompt_content error handling."""

        class BadSession:
            async def call_prompt(self, name, args):
                mock_result = Mock()
                mock_result.isError = True
                return mock_result

        session = BadSession()
        prompt = Mock()
        prompt.name = "bad"
        with pytest.raises(Exception):
            await prompt_service.get_prompt_content(session, prompt)

    @pytest.mark.asyncio
    async def test_get_prompt_content_exception(self, prompt_service, mock_prompts):
        """Test get_prompt_content with unexpected exception."""

        class BadSession:
            async def call_prompt(self, name, args):
                raise RuntimeError("Tool crashed!")

        session = BadSession()
        prompt = Mock()
        prompt.name = "bad"
        with pytest.raises(Exception):
            await prompt_service.get_prompt_content(session, prompt)

    @pytest.mark.asyncio
    async def test_prepare_messages_error(self, prompt_service, mock_prompts):
        """Test prepare_messages error handling returns original messages."""

        class BadSession:
            async def list_prompts(self):
                raise Exception("Session error")

        user_message = Message(role=MessageRole.USER, content="Hello")
        messages = [user_message]
        prepared = await prompt_service.prepare_messages(
            BadSession(), messages, "system"
        )
        assert prepared == messages

    @pytest.mark.asyncio
    async def test_prepare_messages_with_system_prompt(
        self, prompt_service, mock_prompts
    ):
        """Test message preparation with system prompt addition."""
        session = MockMCPSession(prompts=mock_prompts)

        user_message = Message(role=MessageRole.USER, content="Hello")
        messages = [user_message]

        prepared = await prompt_service.prepare_messages(session, messages, "system")

        assert len(prepared) == 2
        assert prepared[0].role == MessageRole.SYSTEM
        assert prepared[0].content == "System prompt from system"
        assert prepared[1] == user_message

    @pytest.mark.asyncio
    async def test_prepare_messages_with_existing_system(
        self, prompt_service, mock_prompts
    ):
        """Test message preparation when system message already exists."""
        session = MockMCPSession(prompts=mock_prompts)

        system_message = Message(role=MessageRole.SYSTEM, content="Existing system")
        user_message = Message(role=MessageRole.USER, content="Hello")
        messages = [system_message, user_message]

        prepared = await prompt_service.prepare_messages(session, messages, "system")

        # Should not add another system message
        assert len(prepared) == 2
        assert prepared[0] == system_message
        assert prepared[1] == user_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
