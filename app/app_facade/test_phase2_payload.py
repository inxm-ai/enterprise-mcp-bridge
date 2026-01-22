import pytest
from unittest.mock import MagicMock, patch
from contextlib import asynccontextmanager
from app.app_facade.generated_service import GeneratedUIService, MessageRole


@pytest.mark.asyncio
async def test_phase_2_attempt_payload_size():
    # Mock services
    storage_mock = MagicMock()
    tgi_mock = MagicMock()
    llm_mock = MagicMock()
    tgi_mock.llm_client = llm_mock

    # Create the service
    service = GeneratedUIService(storage=storage_mock, tgi_service=tgi_mock)

    # Use simple classes to represent chunks
    class MockChunk:
        def __init__(self, content=None, is_done=False):
            self.content = content
            self.is_done = is_done
            self.tool_calls = None
            self.accumulated_tool_calls = None
            self.finish_reason = None

    # Setup mock stream completion
    async def mock_stream(*args, **kwargs):
        # Inspect the ChatCompletionRequest
        chat_request = args[0]

        # Verify tools are None (Phase 2 optimization)
        assert chat_request.tools is None

        # Verify messages length is small (reset history)
        assert len(chat_request.messages) == 2
        assert chat_request.messages[0].role == MessageRole.SYSTEM
        assert chat_request.messages[1].role == MessageRole.USER

        # Verify context contains optimization message
        content = chat_request.messages[1].content
        assert "We have generated the following components" in content

        yield MockChunk(content="", is_done=False)
        yield MockChunk(content='{"html": {"page": "<div></div>"}}', is_done=False)
        yield MockChunk(content=None, is_done=True)

    llm_mock.stream_completion = mock_stream

    # Mock chunk_reader to bypass parsing and just yield our MockChunks
    @asynccontextmanager
    async def mock_chunk_reader_fn(source):
        class MockReader:
            def as_parsed(self):
                # source is the async generator mock_stream
                return source

        yield MockReader()

    system_prompt = "System Prompt"
    user_prompt = "Generate a UI"
    instruction = "Generate HTML"
    logic_payload = {
        "service_script": "class Service {}",
        "components_script": "class Component {}",
        "test_script": "test()",
    }

    with patch(
        "app.app_facade.generated_service.chunk_reader",
        side_effect=mock_chunk_reader_fn,
    ):
        # Run _phase_2_attempt with new signature
        generator = service._phase_2_attempt(
            system_prompt=system_prompt,
            prompt=user_prompt,
            logic_payload=logic_payload,
            access_token="token",
            instruction=instruction,
        )

        result_payload = None
        async for item in generator:
            if isinstance(item, dict) and item.get("type") == "result":
                result_payload = item.get("payload")

    assert result_payload is not None
    assert "html" in result_payload
