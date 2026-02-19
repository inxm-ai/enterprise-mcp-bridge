# Contributing

We welcome contributions to the Enterprise MCP Bridge! This guide helps you get started.

## Quick Start

1. **Fork & Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/enterprise-mcp-bridge.git
   cd enterprise-mcp-bridge
   ```

2. **Create a Branch**
   ```bash
   git switch -c feat/my-feature
   ```

3. **Set Up Environment**
   ```bash
   # Create virtual environment (Python 3.11+)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install with dev dependencies
   pip install -e ./app[dev]
   ```

4. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

5. **Test Your Changes**
   ```bash
   pytest -q
   ```

6. **Submit Pull Request**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   git push origin feat/my-feature
   ```

   Then open a PR on GitHub!

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Keep functions small and focused
- Write clear docstrings

### Testing

- Write tests for all new features
- Maintain test coverage above 80%
- Use pytest fixtures for common setup
- Test both success and failure cases

### Commit Messages

Use conventional commit format:

```
feat: add new feature
fix: resolve bug
docs: update documentation
test: add tests
chore: update dependencies
```

### Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

## Project Structure

```
app/
  server.py          # FastAPI app
  routes.py          # API endpoints
  session/           # Session management
  session_manager/   # Session registry
  oauth/             # OAuth & token exchange
  mcp_server/        # MCP server management
  test_*.py          # Tests
```

## Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test
```bash
pytest app/test_app.py::test_function_name
```

### Run with Coverage
```bash
pytest --cov=app --cov-report=html
```

## Documentation

We use Docsify for documentation. To preview locally:

1. Install Docsify CLI:
   ```bash
   npm install -g docsify-cli
   ```

2. Serve docs:
   ```bash
   docsify serve docs
   ```

3. Open http://localhost:3000

### Documentation Structure

Follow the [Diataxis framework](https://diataxis.fr/):

- **Tutorials:** Learning-oriented, step-by-step guides
- **How-To Guides:** Problem-oriented, practical solutions
- **Reference:** Information-oriented, technical descriptions
- **Explanation:** Understanding-oriented, conceptual discussions

## Getting Help

- **Documentation:** https://inxm-ai.github.io/enterprise-mcp-bridge
- **Issues:** https://github.com/inxm-ai/enterprise-mcp-bridge/issues
- **Discussions:** https://github.com/inxm-ai/enterprise-mcp-bridge/discussions

## Code of Conduct

Please read our [Code of Conduct](code-of-conduct.md) before contributing.

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 License.

---

For detailed contribution guidelines, see [CONTRIBUTING.md](https://github.com/inxm-ai/enterprise-mcp-bridge/blob/main/CONTRIBUTING.md) in the repository root.
