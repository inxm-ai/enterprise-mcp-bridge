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
  "reroute": {
    "on": {
      "tool:tool1:success": "next-agent",
      "tool:tool1:error": "error-handler"
    }
  }
}
```

**Parameters:**

- `agent` - Unique identifier for this agent
- `description` - What the agent should do (sent to LLM)
- `pass_through` - Whether to show agent output (boolean or guideline string)
- `depends_on` - List of agents that must complete first
- `when` - Condition for whether agent should run
- `tools` - Which MCP tools this agent can use
- `reroute` - Conditional routing based on tool results

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
