import pytest

from app.tgi.workflows.passthrough_filter import PassThroughFilter


class DummyLLM:
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    async def ask(
        self, base_prompt, base_request, question, access_token=None, outer_span=None
    ):
        self.calls.append(
            {
                "prompt": base_prompt,
                "question": question,
            }
        )
        if self.replies:
            return self.replies.pop(0)
        return "ALLOW"


@pytest.mark.asyncio
async def test_blocks_exact_repeats():
    filt = PassThroughFilter(DummyLLM(), cooldown_seconds=0)
    msg = "I have just started generating the plan phases!"

    first = await filt.should_emit(
        msg,
        agent_name="planner",
        history=[],
        user_message="",
        pass_through_guideline=None,
        access_token="",
        span=None,
    )
    assert first

    second = await filt.should_emit(
        msg,
        agent_name="planner",
        history=[first.strip()],
        user_message="",
        pass_through_guideline=None,
        access_token="",
        span=None,
    )
    assert second is None


@pytest.mark.asyncio
async def test_blocks_semantic_variations():
    filt = PassThroughFilter(DummyLLM(replies=["BLOCK", "BLOCK"]), cooldown_seconds=2)
    history: list[str] = []
    msg1 = (
        "Testing is underway - I'm verifying the email checking functionality and draft "
        "response generation. Currently 1 out of 10 test tasks complete. The system is properly "
        "connecting to your email service and analyzing message types that would require responses."
    )
    msg2 = (
        "The test is running smoothly - I'm currently processing the workflow logic and validating "
        "the email checking intervals. This will take a few moments to complete all the test scenarios..."
    )
    msg3 = (
        "Testing is progressing well - I've completed the first task and am working through the remaining "
        "9 steps. Currently validating the email inbox monitoring and response drafting components of your workflow."
    )

    allowed = await filt.should_emit(
        msg1,
        agent_name="tester",
        history=history,
        user_message="",
        pass_through_guideline=None,
        access_token="",
        span=None,
    )
    assert allowed
    history.append(allowed.strip())

    blocked_two = await filt.should_emit(
        msg2,
        agent_name="tester",
        history=history,
        user_message="",
        pass_through_guideline=None,
        access_token="",
        span=None,
    )
    assert blocked_two is None

    blocked_three = await filt.should_emit(
        msg3,
        agent_name="tester",
        history=history,
        user_message="",
        pass_through_guideline=None,
        access_token="",
        span=None,
    )
    assert blocked_three is None


@pytest.mark.asyncio
async def test_llm_blocks_borderline():
    llm = DummyLLM(replies=["BLOCK"])
    filt = PassThroughFilter(llm, cooldown_seconds=2)
    history = ["Processing batch 1 of 5 items"]

    borderline = await filt.should_emit(
        "Processing batch 1 of 5 items with extra detail",
        agent_name="runner",
        history=history,
        user_message="",
        pass_through_guideline=None,
        access_token="",
        span=None,
    )
    assert borderline is None
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_allows_numeric_progress():
    filt = PassThroughFilter(DummyLLM(), cooldown_seconds=5)
    history = []

    first = await filt.should_emit(
        "Progress: 1/10 tasks complete",
        agent_name="runner",
        history=history,
        user_message="",
        pass_through_guideline=None,
        access_token="",
        span=None,
    )
    assert first
    history.append(first.strip())

    # Immediately follow-up with numeric change should be allowed despite cooldown
    second = await filt.should_emit(
        "Progress: 2/10 tasks complete",
        agent_name="runner",
        history=history,
        user_message="",
        pass_through_guideline=None,
        access_token="",
        span=None,
    )
    assert second
