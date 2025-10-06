from app.tgi.behaviours.well_planned_orchestrator import WellPlannedOrchestrator
from app.tgi.behaviours.todos.todo_manager import TodoItem


class DummyLLM:
    def create_completion_id(self):
        return "dummy"

    def create_usage_stats(self):
        return {}


class DummyPromptService:
    async def prepare_messages(self, *args, **kwargs):
        return []


class DummyToolService:
    async def get_all_mcp_tools(self, *args, **kwargs):
        return []


async def dummy_non_stream(*args, **kwargs):
    return {"ok": True}


def test_select_tools_for_todo_matches_by_name():
    llm = DummyLLM()
    prompt = DummyPromptService()
    tool_service = DummyToolService()

    orchestrator = WellPlannedOrchestrator(
        llm_client=llm,
        prompt_service=prompt,
        tool_service=tool_service,
        non_stream_chat_with_tools_callable=dummy_non_stream,
        stream_chat_with_tools_callable=dummy_non_stream,
        tool_resolution="none",
    )

    # Build available tools as in the user's prompt: a list of dicts with nested function.name
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "list-mail-folders",
                "description": "...",
            },
        },
        {
            "type": "function",
            "function": {"name": "group-emails-by-reason", "description": "..."},
        },
        {
            "type": "function",
            "function": {"name": "summarize-emails", "description": "..."},
        },
    ]

    todo = TodoItem(
        id="1",
        name="group and summarize",
        goal="process inbox",
        needed_info=None,
        tools=["group-emails-by-reason", "summarize-emails"],
    )

    selected = orchestrator._select_tools_for_todo(todo, available_tools)

    # We expect two tools selected and their names to match the requested ones.
    assert len(selected) == 2
    assert available_tools[0] not in selected
    assert available_tools[1] in selected
    assert available_tools[2] in selected
