import pytest
from unittest.mock import AsyncMock, Mock
from app.tgi.behaviours.well_planned_orchestrator import WellPlannedOrchestrator


@pytest.mark.asyncio
async def test_strip_think_tags_from_result():
    # Setup
    llm_client = Mock()
    prompt_service = Mock()
    tool_service = Mock()
    non_stream = AsyncMock()
    stream = AsyncMock()
    tool_resolution = Mock()

    orchestrator = WellPlannedOrchestrator(
        llm_client, prompt_service, tool_service, non_stream, stream, tool_resolution
    )

    # Test cases

    # 1. Think tags at the end
    text1 = (
        "Here is the result.\n\n<think>Executing...</think>\n\n<think>Success</think>"
    )
    cleaned1 = orchestrator._strip_think_tags(text1)
    assert cleaned1.strip() == "Here is the result."

    # 2. Think tags in the middle (should be removed)
    text2 = "Start <think>Thinking</think> End"
    cleaned2 = orchestrator._strip_think_tags(text2)
    assert (
        cleaned2.strip() == "Start End"
    )  # Note: spaces might remain if not matched by \s*

    # 3. Multiple think tags
    text3 = "Content<think>1</think><think>2</think>"
    cleaned3 = orchestrator._strip_think_tags(text3)
    assert cleaned3 == "Content"

    # 4. No think tags
    text4 = "Just content"
    cleaned4 = orchestrator._strip_think_tags(text4)
    assert cleaned4 == "Just content"

    # 5. Think tags with newlines
    text5 = "Content\n<think>Multi\nLine</think>\nMore"
    cleaned5 = orchestrator._strip_think_tags(text5)
    assert cleaned5.strip() == "Content\nMore"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
