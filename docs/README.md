# Enterprise MCP Bridge Documentation

This directory contains the documentation for Enterprise MCP Bridge, built with [Docsify](https://docsify.js.org/).

## Structure

The documentation follows the [Diataxis framework](https://diataxis.fr/):

```
docs/
├── index.html           # Docsify configuration
├── home.md              # Homepage
├── _sidebar.md          # Navigation sidebar
├── .nojekyll            # GitHub Pages configuration
│
├── tutorials/           # Learning-oriented guides
│   ├── getting-started.md
│   ├── quick-start.md
│   ├── first-mcp-server.md
│   └── multi-user-sessions.md
│
├── how-to/              # Problem-oriented guides
│   ├── deploy-production.md
│   ├── configure-oauth.md
│   ├── remote-mcp-servers.md
│   └── ...
│
├── reference/           # Information-oriented documentation
│   ├── api.md
│   ├── configuration.md
│   ├── environment-variables.md
│   └── ...
│
└── explanation/         # Understanding-oriented content
    ├── architecture.md
    ├── security.md
    ├── sessions.md
    └── ...
```

## Diataxis Framework

### Tutorials (Learning-Oriented)

- **Purpose:** Help newcomers learn by doing
- **Content:** Step-by-step guides with clear outcomes
- **Audience:** Beginners learning the system
- **Example:** "Getting Started with Enterprise MCP Bridge"

### How-To Guides (Problem-Oriented)

- **Purpose:** Show how to solve specific problems
- **Content:** Practical steps to achieve a goal
- **Audience:** Users with specific tasks
- **Example:** "How to Deploy to Production"

### Reference (Information-Oriented)

- **Purpose:** Provide technical descriptions
- **Content:** Specifications, API docs, configuration options
- **Audience:** Users who know what they're looking for
- **Example:** "Configuration Reference"

### Explanation (Understanding-Oriented)

- **Purpose:** Explain concepts and design decisions
- **Content:** Background, context, alternatives
- **Audience:** Users wanting to understand "why"
- **Example:** "Architecture Overview"

## Local Development

### Preview Documentation

1. Install Docsify CLI:
   ```bash
   npm install -g docsify-cli
   ```

2. Serve the docs:
   ```bash
   docsify serve docs
   ```

3. Open http://localhost:3000

### Live Reload

Docsify automatically reloads when you save changes to markdown files.

## Writing Guidelines

### General

- Use clear, concise language
- Write in present tense
- Use active voice
- Keep paragraphs short
- Use code examples liberally

### Markdown Features

Docsify supports:
- Standard Markdown
- Code syntax highlighting
- Alerts/callouts
- Embedded content
- Custom CSS

### Code Blocks

Use fenced code blocks with language specification:

````markdown
```bash
docker run -p 8000:8000 ghcr.io/inxm-ai/enterprise-mcp-bridge:latest
```

```python
def hello():
    print("Hello, World!")
```
````

### Alerts

Use Docsify alerts for important information:

```markdown
?> This is a tip

!> This is a warning
```

### Links

- Internal links: `[Getting Started](tutorials/getting-started.md)`
- External links: `[Docsify](https://docsify.js.org/)`
- Anchors: `[Configuration](#configuration)`

### Images

Place images in the `docs/` directory:

```markdown
![Architecture](architecture.png)
```

## File Naming

- Use lowercase
- Use hyphens for spaces
- Use `.md` extension
- Be descriptive but concise

Examples:
- ✅ `getting-started.md`
- ✅ `configure-oauth.md`
- ❌ `GettingStarted.md`
- ❌ `oauth_config.md`

## Contributing to Docs

### Adding New Pages

1. Create the markdown file in the appropriate directory
2. Add it to `_sidebar.md`
3. Test locally with `docsify serve`
4. Submit a pull request

### Updating Existing Pages

1. Edit the markdown file
2. Preview changes locally
3. Submit a pull request

### Adding New Sections

When adding a new section:

1. Create a new directory (e.g., `docs/advanced/`)
2. Add an index file (e.g., `advanced/README.md`)
3. Update `_sidebar.md`
4. Add sub-pages as needed

## Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

### GitHub Pages Configuration

The site is configured to:
- Deploy from the `docs/` directory
- Use Docsify for rendering
- Serve at: https://inxm-ai.github.io/enterprise-mcp-bridge

### Manual Deployment

If needed, you can manually trigger deployment:

1. Go to GitHub Actions
2. Select "Deploy Documentation to GitHub Pages"
3. Click "Run workflow"

## Docsify Configuration

The main configuration is in `docs/index.html`:

```html
window.$docsify = {
  name: 'Enterprise MCP Bridge',
  repo: 'inxm-ai/enterprise-mcp-bridge',
  loadSidebar: true,
  subMaxLevel: 3,
  // ... more options
}
```

### Plugins

Current plugins:
- **Search:** Full-text search
- **Copy Code:** Copy code blocks
- **Pagination:** Page navigation
- **Syntax Highlighting:** Prism.js

## Resources

- [Docsify Documentation](https://docsify.js.org/)
- [Diataxis Framework](https://diataxis.fr/)
- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Pages Docs](https://docs.github.com/en/pages)

## Questions?

- Open an issue: https://github.com/inxm-ai/enterprise-mcp-bridge/issues
- Start a discussion: https://github.com/inxm-ai/enterprise-mcp-bridge/discussions
