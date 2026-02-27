# Workflows and Agents

Build complex multi-step automations using workflows that orchestrate multiple agents and MCP tools.

## Overview

The Enterprise MCP Bridge includes a powerful workflow engine that enables you to:

- 🔄 **Chain Multiple Agents** - Orchestrate sequences of LLM-powered agents
- 🔧 **Coordinate MCP Tools** - Use different tools at different workflow steps
- 🔀 **Conditional Logic** - Branch based on tool results or conditions
- 🔁 **Loops and Retry** - Handle failures with automatic retry
- 💾 **State Management** - Persist workflow state for long-running processes
- 🎯 **Context Passing** - Share data between agents automatically

## Key Concepts

### Workflow

A workflow defines a series of agents that execute in sequence or based on conditions.

```yaml
flow_id: "data-analysis"
root_intent: "Analyze customer data and generate report"
agents:
  - agent: "data-collector"
    description: "Fetch customer data from database"
    tools: ["query_database"]
  
  - agent: "analyzer"
    description: "Analyze data for insights"
    depends_on: ["data-collector"]
    tools: ["analyze_data"]
  
  - agent: "report-writer"
    description: "Generate formatted report"
    depends_on: ["analyzer"]
    tools: ["create_document"]
```

### Agent

An agent is a single step in a workflow that:
- Has a specific purpose (description)
- Can use specific MCP tools
- Receives context from previous agents
- Can pass results to following agents

### Execution State

Workflow execution state is persisted, allowing:
- Resume after interruption
- Human-in-the-loop feedback
- Audit trail of decisions
- Debugging and monitoring

## Quick Start

### 1. Define a Simple Workflow

```json
{
  "flow_id": "customer-support",
  "root_intent": "Handle customer support request",
  "agents": [
    {
      "agent": "classifier",
      "description": "Classify the customer request",
      "tools": ["categorize_issue"]
    },
    {
      "agent": "responder",
      "description": "Generate appropriate response",
      "depends_on": ["classifier"],
      "tools": ["draft_response"]
    }
  ]
}
```

### 2. Start Workflow Execution

```bash
curl -X POST http://localhost:8000/workflows/execute \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{
    "flow_id": "customer-support",
    "initial_context": {
      "customer_message": "My order hasn't arrived yet"
    }
  }'
```

### 3. Monitor Progress

```bash
# Get execution status
curl http://localhost:8000/workflows/executions/{execution_id}

# Stream updates
curl -N http://localhost:8000/workflows/executions/{execution_id}/stream
```

## Workflow Definition

### Basic Structure

```json
{
  "flow_id": "unique-workflow-id",
  "root_intent": "High-level goal of the workflow",
  "agents": [...],
  "loop": false
}
```

### Agent Configuration

```json
{
  "agent": "agent-name",
  "description": "What this agent does",
  "pass_through": false,
  "depends_on": ["previous-agent"],
  "when": "condition expression",
  "tools": ["tool1", "tool2"],
  "returns": ["extracted_field"],
  "context": true,
  "on_tool_error": "error-handler-agent",
  "stop_point": false,
  "reroute": [
    {
      "on": ["tool:tool1:success"],
      "ask": { "question": "...", "expected_responses": [...] }
    },
    {
      "on": ["tool:tool1:error"],
      "target": "error-handler"
    }
  ]
}
```

**Field reference:**

| Field | Type | Default | Description |
|---|---|---|---|
| `agent` | string | required | Unique identifier for this agent within the workflow |
| `description` | string | required | What the agent should do — sent as the system prompt to the LLM |
| `pass_through` | `false` / `true` / string | `false` | `false`: hide intermediate output. `true`: stream output to the user. String: stream output with this specific guideline/instruction appended to the system prompt |
| `depends_on` | string[] | `[]` | Agents that must complete before this one runs. Agents with no dependencies (or whose dependencies are all met) run in parallel |
| `when` | string | — | A condition expression evaluated against the workflow context. If it evaluates to falsy the agent is skipped |
| `tools` | `null` / `[]` / string[] / object[] | `null` | `null`: all available tools. `[]`: no tools. Array of names: only those tools. Array of objects: advanced config with `args`, `settings` (see [Advanced Tool Configuration](#advanced-tool-configuration)) |
| `returns` | string[] | — | Field names to extract from tool results and store in the agent's context for downstream use |
| `context` | `true` / `false` / `"user_prompt"` / string[] | `true` | Controls how much workflow context is injected. `true`: full context. `false`: none. `"user_prompt"`: only the original user message. Array of dot-path references (e.g. `["other_agent.result"]`): only those fields |
| `on_tool_error` | string | — | Agent to automatically reroute to when any tool call fails, even if the LLM doesn't emit a `<reroute>` tag |
| `stop_point` | boolean | `false` | If `true`, no further agents run after this one completes |
| `reroute` | object / object[] | — | Conditional routing rules triggered by tool outcomes or LLM-emitted tags. See [Reroute Configuration](#reroute-configuration) |

### Advanced Tool Configuration

```json
{
  "agent": "processor",
  "description": "Process the data",
  "tools": [
    {
      "process_data": {
        "streaming": true,
        "settings": {
          "timeout": 30
        },
        "args": {
          "input": "${previous_agent.result}"
        }
      }
    }
  ]
}
```

## Examples

### Sequential Workflow

Simple sequence where each agent depends on the previous:

```json
{
  "flow_id": "blog-post-generation",
  "root_intent": "Generate and publish blog post",
  "agents": [
    {
      "agent": "researcher",
      "description": "Research topic and gather information",
      "tools": ["web_search", "read_document"]
    },
    {
      "agent": "writer",
      "description": "Write the blog post content",
      "depends_on": ["researcher"],
      "tools": ["generate_text"]
    },
    {
      "agent": "editor",
      "description": "Review and edit for quality",
      "depends_on": ["writer"],
      "tools": ["edit_text", "check_grammar"]
    },
    {
      "agent": "publisher",
      "description": "Publish to blog platform",
      "depends_on": ["editor"],
      "tools": ["publish_post"]
    }
  ]
}
```

### Conditional Routing

Route based on tool results:

```json
{
  "flow_id": "code-review",
  "root_intent": "Review code changes",
  "agents": [
    {
      "agent": "linter",
      "description": "Run linter on code",
      "tools": ["run_linter"],
      "reroute": {
        "on": {
          "tool:run_linter:success": "security-scan",
          "tool:run_linter:error": "fix-lint-errors"
        }
      }
    },
    {
      "agent": "fix-lint-errors",
      "description": "Automatically fix linting errors",
      "tools": ["fix_lint"],
      "reroute": {
        "on": {
          "tool:fix_lint:success": "linter"
        }
      }
    },
    {
      "agent": "security-scan",
      "description": "Scan for security issues",
      "depends_on": ["linter"],
      "tools": ["security_scanner"]
    }
  ]
}
```

### Parallel Execution

Agents without dependencies run in parallel:

```json
{
  "flow_id": "multi-source-analysis",
  "root_intent": "Analyze data from multiple sources",
  "agents": [
    {
      "agent": "fetch-db",
      "description": "Fetch from database",
      "tools": ["query_database"]
    },
    {
      "agent": "fetch-api",
      "description": "Fetch from external API",
      "tools": ["call_api"]
    },
    {
      "agent": "fetch-files",
      "description": "Read from files",
      "tools": ["read_file"]
    },
    {
      "agent": "combine",
      "description": "Combine all data sources",
      "depends_on": ["fetch-db", "fetch-api", "fetch-files"],
      "tools": ["merge_data"]
    }
  ]
}
```

### Loop Workflow

Workflow that repeats until completion:

```json
{
  "flow_id": "iterative-improvement",
  "root_intent": "Iteratively improve code quality",
  "loop": true,
  "agents": [
    {
      "agent": "analyzer",
      "description": "Analyze code quality metrics",
      "tools": ["analyze_code"]
    },
    {
      "agent": "improver",
      "description": "Make improvements",
      "depends_on": ["analyzer"],
      "tools": ["refactor_code"],
      "when": "quality_score < 90"
    }
  ]
}
```

## Context and Data Flow

### Passing Data Between Agents

Agents automatically receive context from previous agents:

```json
{
  "agent": "summarizer",
  "description": "Summarize the results from researcher",
  "depends_on": ["researcher"],
  "tools": [
    {
      "summarize": {
        "args": {
          "text": "${researcher.result.content}"
        }
      }
    }
  ]
}
```

**Available variables:**
- `${agent_name.result}` - Result from previous agent
- `${initial_context.field}` - From initial workflow context
- `${tool_name.output}` - Output from specific tool

### Initial Context

Provide data when starting workflow:

```bash
curl -X POST http://localhost:8000/workflows/execute \
  -d '{
    "flow_id": "my-workflow",
    "initial_context": {
      "user_id": "123",
      "task": "Generate report",
      "parameters": {
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
      }
    }
  }'
```

## Execution Modes

### Foreground vs Background

Workflows can execute in two different modes, each with distinct characteristics and use cases.

#### Foreground Execution (Default)

**Overview:**
- Workflow runs inline with the HTTP request
- Client waits for completion or timeout
- Simpler for short-running workflows
- Direct error handling

**Characteristics:**
```bash
# Standard foreground execution
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "messages": [{"role": "user", "content": "Analyze this data"}],
    "use_workflow": true,
    "workflow_id": "data-analysis",
    "stream": true
  }'
```

**Advantages:**
- ✅ Immediate results streamed to client
- ✅ Simple error handling
- ✅ No need to poll for completion
- ✅ Client maintains direct connection

**Disadvantages:**
- ❌ Connection must stay open entire time
- ❌ Request timeout limitations apply
- ❌ Client must handle reconnection if dropped
- ❌ Not suitable for long-running workflows

**Best For:**
- Quick analyses (< 1 minute)
- Interactive sessions
- Development and testing
- Single-user scenarios

#### Background Execution

**Overview:**
- Workflow runs in a separate background task
- Client can disconnect and reconnect
- Ideal for long-running workflows
- Multiple clients can subscribe to same execution

**Characteristics:**
```bash
# Start background execution with special header
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "X-INXM-Workflow-Background: true" \
  -d '{
    "messages": [{"role": "user", "content": "Generate comprehensive report"}],
    "use_workflow": true,
    "workflow_id": "report-generation",
    "workflow_execution_id": "exec-12345",
    "stream": true
  }'
```

**Requirements:**
- Must include `X-INXM-Workflow-Background: true` header
- Must be a streaming request (`stream: true`)
- Must use workflows (`use_workflow: true`)
- Recommended to provide `workflow_execution_id` for resumption

**Advantages:**
- ✅ Workflow continues even if client disconnects
- ✅ Multiple clients can subscribe to same execution
- ✅ Can reconnect with same execution ID
- ✅ No timeout limitations
- ✅ Better for long-running tasks
- ✅ Client can monitor progress at intervals

**Disadvantages:**
- ❌ Slightly more complex setup
- ❌ Need to manage execution IDs
- ❌ Requires SSE streaming support

**Best For:**
- Long-running workflows (> 1 minute)
- Report generation
- Batch processing
- Multi-step automations
- Workflows requiring human feedback at intervals
- Production deployments with reliability requirements

#### Reconnecting to Background Workflows

If you disconnect from a background workflow, reconnect using the same execution ID:

```bash
# Reconnect to existing background execution
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "X-INXM-Workflow-Background: true" \
  -d '{
    "messages": [{"role": "user", "content": "Continue"}],
    "use_workflow": true,
    "workflow_id": "report-generation",
    "workflow_execution_id": "exec-12345",
    "stream": true
  }'
```

**What happens:**
1. Bridge checks if workflow with this execution ID is already running
2. If running, subscribes you to existing execution stream
3. You receive all events from the current point forward
4. If paused for feedback, you can provide it

#### Canceling Background Workflows

Stop a running background workflow:

```bash
curl -X POST http://localhost:8000/workflows/exec-12345/cancel \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "cancelled": true,
  "execution_id": "exec-12345"
}
```

#### Monitoring Background Workflows

Check workflow state without streaming:

```bash
# Get current workflow state
curl http://localhost:8000/workflows/exec-12345/state \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "execution_id": "exec-12345",
  "flow_id": "report-generation",
  "current_agent": "summarizer",
  "completed_agents": ["data_collector", "analyzer"],
  "awaiting_feedback": false,
  "completed": false,
  "error": null,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

#### Comparison Table

| Feature | Foreground | Background |
|---------|-----------|------------|
| **Connection** | Must stay open | Can disconnect/reconnect |
| **Timeout** | HTTP timeout applies | No timeout |
| **Multiple subscribers** | No | Yes |
| **Complexity** | Simple | Moderate |
| **Best for** | Quick tasks | Long tasks |
| **Cancellation** | Close connection | API endpoint |
| **State persistence** | No | Yes |
| **Resume after failure** | No | Yes |

#### Best Practices

**Use Foreground When:**
- Workflow completes in under 30 seconds
- You need immediate synchronous results
- Developing or debugging
- Single user/client scenario

**Use Background When:**
- Workflow takes > 1 minute
- You need reliability and fault tolerance
- Multiple users might monitor progress
- Human feedback required at intervals
- Production environment with load balancing

**Execution ID Management:**
```python
import uuid

# Generate unique execution ID
execution_id = f"exec-{uuid.uuid4()}"

# Use consistent ID for reconnection
# Store in database or client session
```

## Human-in-the-Loop

### Feedback Requests

Workflows can pause for human feedback:

```json
{
  "agent": "approval-needed",
  "description": "Wait for human approval",
  "pass_through": "Request approval: ${draft.content}",
  "tools": []
}
```

### Providing Feedback

```bash
curl -X POST http://localhost:8000/workflows/executions/{id}/feedback \
  -d '{
    "feedback": "Approved with minor changes",
    "continue": true
  }'
```

### Reroute Configuration

`reroute` defines rules for how an agent hands off control after completing. It can be a single rule object or a list of rule objects. Each rule has an `on` trigger list and either an `ask` (pause for user input) or an automatic target.

```json
"reroute": [
  {
    "on": ["tool:select_tools:success"],
    "ask": {
      "question": "Instruction for the LLM to generate a user-facing question",
      "expected_responses": [ { "<id>": { "to": "...", "with": [...] } } ]
    }
  },
  {
    "on": ["tool:select_tools:error"],
    "target": "error_handler"
  }
]
```

**Trigger formats for `on`:**

| Trigger | When it fires |
|---|---|
| `"tool:<name>:success"` | The named tool completed without an error |
| `"tool:<name>:error"` | The named tool returned an error |
| `"ASK_USER"` | The LLM emitted `<reroute>ASK_USER</reroute>` |
| Any string | Matched against whatever is inside `<reroute>...</reroute>` in the LLM's output |

**Rule fields:**

| Field | Description |
|---|---|
| `on` | List of trigger strings. Rule fires when any of them match |
| `ask` | Pause for user input. See [ask config](#ask-config) |
| `target` | Automatically route to this agent name without pausing |

### Conditional Routing Based on Feedback

The Enterprise MCP Bridge supports conditional routing where workflows pause to ask the user a question, then route to different agents based on their choice.

#### How It Works

1. **Agent emits trigger** — The agent completes its work and emits `<reroute>TRIGGER_NAME</reroute>`
2. **Trigger matches** — The engine checks if any `on` entry in the agent's `reroute` list matches
3. **Workflow pauses** — If the matched entry has an `ask` field, the workflow pauses (`awaiting_feedback: true`) and an MCP-compliant elicitation payload is sent to the client
4. **User responds** — The user selects a choice (and optionally fills in input fields)
5. **Workflow routes** — The engine routes to the agent in the selected choice's `to` field
6. **Context copied** — Fields listed in `with` are copied from agent context into the shared workflow context

#### ask Config

The `ask` config specifies what question to present to the user and which choices are available.

```json
"ask": {
  "question": "Instruction for the LLM to generate a user-facing question (not shown directly)",
  "expected_responses": [
    {
      "<choice_id>": {
        "to": "target_agent",
        "with": ["context_field"],
        "value": "Human-readable label for this choice",
        "input": {
          "field_name": { "type": "string", "description": "Label/placeholder" }
        }
      }
    }
  ]
}
```

**`ask` fields:**

| Field | Description |
|---|---|
| `question` | Instruction passed to the LLM to generate the user-facing question text. This is rendered by the LLM — write it as a directive, not as the question itself |
| `expected_responses` | List of choice group objects. Each object maps choice IDs to their routing config |

**Choice entry fields** (values inside `expected_responses`):

| Field | Type | Description |
|---|---|---|
| `to` | string | Target agent name, `"workflows[flow_id]"` to hand off to another workflow, or `"external[name]"` to trigger an external system |
| `with` | string[] | Context field names to copy into the shared workflow context when this choice is selected. For `external[...]` targets, these are resolved to their actual values at elicitation time |
| `value` | string | Human-readable label for this choice, forwarded to the client in the elicitation payload |
| `each` | string | A context key pointing to a dynamic list — the engine expands one choice entry per item at runtime |
| `input` | object | **Per-choice input field definitions.** A map of `field_name → JSON Schema` object. When present, the UI should show an input form for this field only when the user selects this choice. See [Per-Choice Input Fields](#per-choice-input-fields) |

#### Example

```json
{
  "agent": "find_plan",
  "description": "Search for existing deployment plans",
  "reroute": [
    {
      "on": ["PLAN_FOUND"],
      "ask": {
        "question": "Present the found plan and ask whether to proceed, keep searching, or abort.",
        "expected_responses": [
          {
            "choose_plan": { "to": "execute_deployment", "with": ["plan_id"], "value": "Use this plan" },
            "continue_searching": { "to": "find_alternative_plans", "value": "Keep searching" },
            "abort": { "to": "cancel_workflow", "value": "Abort" }
          }
        ]
      }
    },
    {
      "on": ["NO_PLANS_FOUND"],
      "ask": {
        "question": "No plan found. Ask whether to create a new one.",
        "expected_responses": [
          {
            "create": { "to": "create_plan", "value": "Create a new plan" },
            "abort": { "to": "cancel_workflow", "value": "Abort" }
          }
        ]
      }
    }
  ]
}
```

The agent emits the trigger after completing its tool call:

```xml
<return name="plan_id">deploy-123</return>
<reroute>PLAN_FOUND</reroute>
```

#### Submitting User Feedback

Feedback is submitted as part of the next chat message. The client wraps the selection in a `<user_feedback>` tag:

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "<user_feedback>{\"action\":\"accept\",\"content\":{\"selection\":\"choose_plan\"}}</user_feedback>"}],
    "use_workflow": true,
    "workflow_execution_id": "exec-12345",
    "stream": true
  }'
```

For decline/cancel:

```bash
# content is just the keyword — no JSON wrapper needed
"content": "<user_feedback>cancel</user_feedback>"
```

### Per-Choice Input Fields

Some choices require the user to provide additional text alongside their selection — for example, sending back adjustment instructions when they choose to modify a previous step. The `input` key on a choice entry defines one or more typed input fields that the UI should display conditionally for that choice.

#### Workflow Config

```json
{
  "agent": "select_tools",
  "description": "Select the appropriate integrations.",
  "tools": ["select_tools"],
  "returns": ["selected_tools"],
  "reroute": [
    {
      "on": ["tool:select_tools:success"],
      "ask": {
        "question": "Present the selected integrations and ask whether to proceed or adjust.",
        "expected_responses": [
          {
            "proceed": {
              "to": "create_plan",
              "with": ["selected_tools"],
              "value": "Proceed with these integrations"
            },
            "adjust": {
              "to": "select_tools",
              "value": "Adjust the selection",
              "input": {
                "feedback": {
                  "type": "string",
                  "description": "What would you like to adjust?"
                }
              }
            }
          }
        ]
      }
    }
  ]
}
```

#### Elicitation Payload (SSE)

The client receives the following inside the `<user_feedback_needed>` tag:

```json
{
  "message": "I found 5 integrations. Proceed or adjust?",
  "requestedSchema": {
    "type": "object",
    "properties": {
      "selection": {
        "type": "string",
        "enum": ["proceed", "adjust"],
        "description": "Selected response id."
      },
      "value": {
        "oneOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"}]
      },
      "feedback": {
        "type": "string",
        "description": "What would you like to adjust?"
      }
    },
    "required": ["selection"],
    "additionalProperties": false
  },
  "meta": {
    "expected_responses": [
      {"id": "proceed", "to": "create_plan", "with": ["selected_tools"], "value": "Proceed with these integrations"},
      {"id": "adjust", "to": "select_tools", "value": "Adjust the selection",
       "input": {"feedback": {"type": "string", "description": "What would you like to adjust?"}}}
    ],
    "input_fields": {
      "feedback": {"for_selection": "adjust"}
    }
  }
}
```

Key points for client rendering:

- `requestedSchema.properties` contains all fields — both the `selection` enum and any per-choice input fields.
- `meta.input_fields` maps each input field name to `{"for_selection": "<choice_id>"}`. Use this to show the field **only when that choice is selected**.
- Input field definitions (type, description) come from `requestedSchema.properties["<field_name>"]`.
- `required` only ever contains `"selection"` — input fields are always optional in the schema.

#### Submitting Feedback With Input

When the user selects `"adjust"` and fills in the input field:

```
<user_feedback>{"action":"accept","content":{"selection":"adjust","feedback":"Remove the Jira integration"}}</user_feedback>
```

When the user selects a choice that has no input fields (`"proceed"`):

```
<user_feedback>{"action":"accept","content":{"selection":"proceed"}}</user_feedback>
```

#### What the Engine Does With Input Values

1. The input field value (`"Remove the Jira integration"`) is stored in the agent's context under its field name (`feedback`).
2. It is also appended as a user message so the target agent's LLM sees it as natural language — allowing the agent to act on the user's instructions without special prompt engineering.
3. The choice's `to` target (`select_tools`) is set as the next agent to run.

#### Multiple Input Fields

A single choice can define multiple input fields:

```json
"refine": {
  "to": "search_agent",
  "value": "Refine the search",
  "input": {
    "keywords": { "type": "string", "description": "New keywords to add" },
    "exclude": { "type": "string", "description": "Terms to exclude" }
  }
}
```

Both `keywords` and `exclude` will appear in `requestedSchema.properties` and in `meta.input_fields` with `"for_selection": "refine"`. Their values are concatenated into a single user message.

> **Note**: Field names must not collide with the reserved names `selection` or `value`. The engine will log a warning and skip any that do.

## Monitoring and Debugging

### List Executions

```bash
# All executions
curl http://localhost:8000/workflows/executions

# For specific workflow
curl http://localhost:8000/workflows/executions?flow_id=my-workflow

# For specific user
curl http://localhost:8000/workflows/executions?owner_id=user123
```

### View Execution Details

```bash
curl http://localhost:8000/workflows/executions/{execution_id}
```

Response:
```json
{
  "execution_id": "exec-123",
  "flow_id": "my-workflow",
  "current_agent": "processor",
  "completed": false,
  "context": {...},
  "events": [
    "Started: researcher",
    "Completed: researcher",
    "Started: processor"
  ],
  "created_at": "2024-01-15T10:00:00Z"
}
```

### Stream Progress

```bash
curl -N http://localhost:8000/workflows/executions/{id}/stream
```

Returns Server-Sent Events with real-time updates.

## Best Practices

### 1. Clear Agent Descriptions

Make agent purposes explicit:

```
❌ Bad: "Process the data"
✅ Good: "Extract customer names and email addresses from the uploaded CSV file"
```

### 2. Minimal Tool Access

Only give agents the tools they need:

```json
{
  "agent": "reader",
  "tools": ["read_file"],  // Not all tools
  "description": "Read configuration file"
}
```

### 3. Error Handling

Always plan for failures:

```json
{
  "agent": "api-call",
  "tools": ["call_external_api"],
  "reroute": {
    "on": {
      "tool:call_external_api:error": "retry-handler"
    }
  }
}
```

### 4. State Management

Keep workflow state clean:
- Only store necessary data in context
- Clear large data after use
- Use external storage for big files

## API Reference

### Create Workflow

```http
POST /workflows
Content-Type: application/json

{
  "flow_id": "my-workflow",
  "root_intent": "Workflow purpose",
  "agents": [...]
}
```

### Execute Workflow

```http
POST /workflows/execute
Content-Type: application/json

{
  "flow_id": "my-workflow",
  "initial_context": {...}
}
```

### Get Execution

```http
GET /workflows/executions/{execution_id}
```

### Provide Feedback

```http
POST /workflows/executions/{execution_id}/feedback
Content-Type: application/json

{
  "feedback": "User input",
  "continue": true
}
```

### List Workflows

```http
GET /workflows
```

### Delete Workflow

```http
DELETE /workflows/{flow_id}
```

## Troubleshooting

### Workflow Stuck

If workflow doesn't progress:
- Check current agent in execution state
- Review tool outputs for errors
- Verify dependencies are met
- Check if waiting for feedback

### Tool Failures

If tools keep failing:
- Verify tool is available
- Check tool parameters are correct
- Review argument mapping syntax
- Test tool directly via REST API

### Context Not Passing

If agents don't receive data:
- Verify `depends_on` is set correctly
- Check variable reference syntax: `${agent.result}`
- Review execution context
- Ensure previous agent completed successfully

## Limitations

- Maximum 50 agents per workflow
- Execution timeout: 30 minutes
- Loop workflows limited to 100 iterations
- Context size limited to 1MB

## Next Steps

- [API Reference](../reference/api.md)
- [Security Model](../explanation/security.md)
- [Deploy to Production](deploy-production.md)

## Resources

- [Workflow Examples](../reference/examples.md)
- [Agent Best Practices](#best-practices)
