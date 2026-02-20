# UI Generation

Generate web applications and dashboards from natural language prompts using AI-powered code generation.

## Overview

The Enterprise MCP Bridge includes a powerful AI-powered UI generation feature that creates complete, production-ready web applications directly from natural language descriptions. These applications automatically integrate with your MCP tools.

**Key Features:**
- 🤖 **LLM-Powered Generation** - Uses your configured LLM to translate prompts into working code
- ⚡ **Progressive Enhancement** - Built on [pfusch](https://matthiaskainer.github.io/pfusch/), a lightweight reactive framework
- 🔌 **MCP Integration** - Automatically connects to MCP tools with proper authentication
- 👥 **User/Group Scoping** - Each application is isolated to specific users or groups
- 📚 **Version History** - Maintains full generation history for auditing and rollback
- 🚀 **No Build Step** - Applications run directly in the browser

## Prerequisites

- LLM configured (OpenAI, Azure OpenAI, or compatible)
- MCP tools available
- OAuth authentication configured (for user scoping)

## Quick Start

### 1. Generate Your First App

```bash
# Generate a dashboard for memory MCP tools
curl -X POST http://localhost:8000/apps/generate \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a knowledge graph dashboard that displays entities and their relationships. Include search and filtering.",
    "app_id": "knowledge-dashboard",
    "version": "1.0.0"
  }'
```

### 2. Access the Generated App

```bash
# View the generated HTML
curl http://localhost:8000/apps/knowledge-dashboard/html

# Or open in browser
open http://localhost:8000/apps/knowledge-dashboard
```

### 3. Update the App

```bash
# Iterate with a new prompt
curl -X POST http://localhost:8000/apps/generate \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Add a graph visualization using D3.js to show entity relationships",
    "app_id": "knowledge-dashboard",
    "version": "1.1.0"
  }'
```

## How It Works

### Architecture

```
User Prompt → LLM (GPT-4/etc) → Generated HTML/JS → pfusch Components → MCP Tools
                ↓
         Validation & Testing
                ↓
         Stored Application
```

### Generation Process

1. **Prompt Analysis** - LLM analyzes your requirements
2. **Tool Discovery** - Available MCP tools are identified
3. **Code Generation** - HTML, CSS, and JavaScript generated
4. **Component Creation** - pfusch web components created
5. **MCP Integration** - Service layer connects to tools
6. **Testing** - Generated tests validate functionality
7. **Deployment** - App stored and made available

### Conversational Container Editing (Bridge-Hosted)

The bridge can now host a conversational editing container for generated apps (feature-flagged).

Feature flags:
- `APP_CONVERSATIONAL_UI_ENABLED=true`
- `APP_UI_SESSION_TTL_MINUTES=120` (default)
- `APP_UI_PATCH_ENABLED=true` (default)

Flow:
1. Create a draft session:
```bash
curl -X POST http://localhost:8000/app/_generated/user=user123/my-ui/main/chat/sessions \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -d '{}'
```
2. Send conversational update message (SSE):
```bash
curl -N -X POST http://localhost:8000/app/_generated/user=user123/my-ui/main/chat/sessions/{session_id}/messages \
  -H "Authorization: ******" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message":"Add a KPI card section and move filters to the top."}'
```
3. Load draft HTML in editor session:
```bash
curl "http://localhost:8000/app/_generated/user=user123/my-ui/main/draft?session_id={session_id}&as=page"
```
4. Publish draft to shared canonical version:
```bash
curl -X POST http://localhost:8000/app/_generated/user=user123/my-ui/main/chat/sessions/{session_id}/publish \
  -H "Authorization: ******"
```
5. Discard draft session:
```bash
curl -X DELETE http://localhost:8000/app/_generated/user=user123/my-ui/main/chat/sessions/{session_id} \
  -H "Authorization: ******"
```

Container endpoint:
```bash
open http://localhost:8000/app/_generated/user=user123/my-ui/main/container
```

Behavior:
- Live updates apply to draft only (editor-session scoped).
- Publish uses optimistic version checks to prevent stale overwrites.
- Update strategy is patch-first with automatic regenerate fallback.

### Progressive Enhancement with pfusch

Generated applications use [pfusch](https://matthiaskainer.github.io/pfusch/), a minimal framework for progressive enhancement:

```javascript
// Example generated component
const { html, state, script, trigger, helpers } = pfusch();

export default () => {
  const initialState = {
    entities: [],
    loading: false
  };

  return html.define('entity-list', ({ state }) => {
    return state.loading 
      ? html.div('Loading...')
      : html.ul(
          state.entities.map(e => 
            html.li(e.name)
          )
        );
  }, { state: initialState });
};
```

## Advanced Features

### User and Group Scoping

Applications are automatically scoped to users/groups:

```bash
# User-specific app
curl -X POST http://localhost:8000/apps/generate \
  -H "Authorization: ******" \
  -d '{
    "prompt": "Personal task manager",
    "app_id": "my-tasks",
    "scope": "user"
  }'

# Group-specific app
curl -X POST http://localhost:8000/apps/generate \
  -H "Authorization: ******" \
  -d '{
    "prompt": "Team dashboard",
    "app_id": "team-dashboard",
    "scope": "group"
  }'
```

### Version Management

Track changes and rollback if needed:

```bash
# List versions
curl http://localhost:8000/apps/knowledge-dashboard/versions

# Access specific version
curl http://localhost:8000/apps/knowledge-dashboard/html?version=1.0.0

# Rollback to previous version
curl -X POST http://localhost:8000/apps/knowledge-dashboard/rollback \
  -d '{"version": "1.0.0"}'
```

### Custom Styling

Provide styling requirements in prompts:

```json
{
  "prompt": "Create a dark-themed analytics dashboard with orange accents. Use a modern, minimal design with cards and charts.",
  "app_id": "analytics",
  "version": "1.0.0"
}
```

## Best Practices

### 1. Clear Prompts

Be specific about requirements:

```
❌ Bad: "Make a dashboard"
✅ Good: "Create a dashboard showing all entities from the memory MCP server. Include a search bar, table view with sorting, and detail panel. Use a card-based layout."
```

### 2. Iterative Development

Start simple, then enhance:

```
v1.0: "Basic entity list with search"
v1.1: "Add filtering by entity type"
v1.2: "Add graph visualization"
v1.3: "Add export to CSV functionality"
```

### 3. Test Generated Code

Always test before production:

```bash
# The generated code includes tests
# Review them at:
curl http://localhost:8000/apps/my-app/tests

# Run tests if test runner is available
```

### 4. Security Review

Review generated code for security issues:

```bash
# Check for:
# - Proper input escaping
# - XSS vulnerabilities
# - CSRF protection
# - Sensitive data exposure

curl http://localhost:8000/apps/my-app/html | grep -i "innerHTML\|eval"
```

## Example Prompts

### Knowledge Graph Dashboard

```
Create a knowledge graph dashboard for the memory MCP server. Features:
- List all entities with their types
- Search by name or observation
- Click to view entity details and relationships
- Add new entities with a form
- Visual graph showing entity connections
- Dark theme with blue accents
```

### Task Management App

```
Build a task management application using memory MCP tools. Include:
- Task list with status (todo, in progress, done)
- Create, edit, delete tasks
- Filter by status and priority
- Due date tracking
- Simple calendar view
- Responsive mobile-friendly design
```

### Analytics Dashboard

```
Design an analytics dashboard that:
- Shows key metrics in cards (total entities, types, recent activity)
- Line chart of entity creation over time
- Pie chart of entity distribution by type
- Searchable table of all entities
- Export data to CSV
- Professional look with subtle animations
```

## Troubleshooting

### Generation Fails

**Error:** `LLM generation failed`

**Solutions:**
- Check LLM is configured correctly
- Verify API key is valid
- Ensure prompt is clear and not too complex
- Try simplifying the prompt

### App Doesn't Load

**Error:** Blank page or JavaScript errors

**Solutions:**
- Check browser console for errors
- Verify MCP tools are available
- Check authentication is working
- Review generated code for syntax errors

### MCP Tools Not Working

**Error:** Tools return errors in generated app

**Solutions:**
- Test tools directly via REST API
- Check OAuth tokens are being passed
- Verify tool parameters are correct
- Review tool schema compatibility

## Security Considerations

⚠️ **Review Generated Code** - Always review AI-generated code before production

⚠️ **XSS Risks** - Generated HTML may not properly escape user input

⚠️ **CSRF Protection** - Ensure forms include CSRF tokens

⚠️ **Input Validation** - Validate all user input on server side

See [UI Generation Security](../explanation/security.md#ui-generation-security) for details.

## API Reference

### Generate Application

```http
POST /apps/generate
Authorization: ******
Content-Type: application/json

{
  "prompt": "Application description",
  "app_id": "unique-app-identifier",
  "version": "1.0.0",
  "scope": "user|group"
}
```

### Get Application HTML

```http
GET /apps/{app_id}/html?version={version}
```

### List Versions

```http
GET /apps/{app_id}/versions
```

### Delete Application

```http
DELETE /apps/{app_id}
```

## Limitations

- Generated code requires review for security
- Complex UIs may need manual refinement
- Limited to browser-based applications
- Requires modern browser with JavaScript enabled
- LLM quality affects generated code quality

## Next Steps

- [Security Considerations](../explanation/security.md#ui-generation-security)
- [Deploy to Production](deploy-production.md)
- [Configure OAuth](configure-oauth.md)

## Resources

- [pfusch Documentation](https://matthiaskainer.github.io/pfusch/)
- [Example Applications](../reference/examples.md)
