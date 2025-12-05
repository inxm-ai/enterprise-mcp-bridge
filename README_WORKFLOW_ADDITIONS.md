## Replace the "Field reference:" section in README.md with this:

---

Field reference:
- `flow_id` (string): Unique id used by `use_workflow`.
- `root_intent` (string): Intent label; used for auto-selection when `use_workflow: true`.
- `agents` (array, ordered): Steps executed in order when their dependencies are met.
  - `agent` (string): Name of the agent/prompt to run (also used to look up a custom prompt by name).
  - `description` (string): Default prompt text if no custom prompt exists for this agent.
  - `pass_through` (bool | string, default false): Controls response visibility. If `true`, the agent's streamed content is shown to the user. If a string, it acts as a response guideline instruction added to the agent's system prompt (e.g., `"Return only the searches you are performing"`). When using a string guideline, agents must wrap visible content in `<passthrough></passthrough>` tags—only content inside these tags will be streamed to the user.
  - `depends_on` (array[string]): Agent names that must complete before this agent runs.
  - `when` (string, optional): Python-style expression evaluated against `context`; if falsy, the agent is skipped and marked with `reason: condition_not_met`.
  - `reroute` (object, optional): `{ "on": ["CODE1", ...], "to": "agent_name" }`. If the agent emits `<reroute>CODE1</reroute>`, the router jumps to the `to` agent next.
  - `tools` (array, optional): Limit available tools for this agent. Can be:
    - Array of strings: Simple tool names (e.g., `["get_weather", "search"]`)
    - Array of objects: Advanced tool configurations with settings and argument mappings (see below)
    - Empty array `[]`: Disable all tools for this agent
    - Omitted: All MCP tools are available
  - `returns` (array[string], optional): Field names to extract from tool results and store in context for use by subsequent agents.

##### Advanced tool configuration

Tools can be configured with settings and argument mappings:

```json
{
  "agent": "create_plan",
  "description": "Create a detailed plan",
  "tools": [
    "simple_tool",
    {
      "plan": {
        "settings": { "streaming": true },
        "args": { "selected-tools": "select_tools.selected_tools" }
      }
    }
  ],
  "depends_on": ["select_tools"]
}
```

Tool configuration options:
- `settings.streaming` (bool): If `true`, the tool is called via the streaming endpoint (`POST /tools/{tool_name}/stream`) to receive progress updates.
- `args` (object): Maps tool input arguments to values from previous agent contexts. Format: `"input_arg": "agent_name.field_name"`. Mapped arguments are automatically injected and removed from the tool schema presented to the LLM.

##### Using `returns` to capture tool outputs

The `returns` field allows you to capture specific fields from tool results for use in subsequent agents:

```json
{
  "agent": "select_tools",
  "description": "Select appropriate tools for the task",
  "tools": ["select_tools"],
  "returns": ["selected_tools"],
  "pass_through": true
},
{
  "agent": "execute_plan",
  "description": "Execute the plan with selected tools",
  "tools": [
    {
      "run_task": {
        "args": { "tools": "select_tools.selected_tools" }
      }
    }
  ],
  "depends_on": ["select_tools"]
}
```

In this example:
1. `select_tools` agent runs a tool that returns `{ "selected_tools": ["tool1", "tool2"] }`
2. The `returns: ["selected_tools"]` captures this value into the agent's context
3. The `execute_plan` agent's `run_task` tool automatically receives `selected_tools` as its `tools` argument via the `args` mapping

##### String pass_through with `<passthrough>` tags

When `pass_through` is a string (guideline), the agent is instructed to wrap user-visible content in `<passthrough>` tags. Only content inside these tags is streamed to the user:

```json
{
  "agent": "search",
  "description": "Search for relevant documents",
  "pass_through": "Return only the searches you are performing",
  "tools": ["search"]
}
```

The agent might respond:
```
Analyzing query... <passthrough>Searching for: quarterly reports 2024</passthrough>
Processing results... <passthrough>Found 5 documents</passthrough>
```

The user sees only: `Searching for: quarterly reports 2024 Found 5 documents`

This allows agents to maintain internal reasoning while selectively showing progress to users.

---

## END - Replace up to (but not including) the "Prompt usage:" line
