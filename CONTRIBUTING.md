# Contributing to MCP REST Server

Thanks for your interest in contributing! This document explains how to propose changes, report issues, add features, improve docs, and release updates.

## 📌 Quick Start (TL;DR)
1. Fork & clone
2. Create a branch: `git switch -c feat/my-feature`
3. Create a virtual env (Python 3.10+)
4. Install locally (incl. dev deps): `pip install -e ./app[dev]`
5. Run tests: `pytest -q`
6. Make changes + add/adjust tests
7. Run lint & format (see below)
8. Commit using conventional message style
9. Open a Pull Request (PR)

---
## ✅ Requirements & Scope
This project exposes a FastAPI REST layer over an MCP server (spawned as a subprocess) with:
- Tool discovery (`/tools`)
- Tool execution (`/tools/{name}`)
- Session lifecycle (`/session/start`, `/session/close`)
- Optional OAuth token exchange & injection

Contributions should align with these goals: simplicity, security awareness, small surface area, test coverage, minimal external dependencies.

---
## 🗂 Project Layout
```
app/
  server.py          # FastAPI app wiring
  routes.py          # REST endpoints & request/session orchestration
  models.py          # Pydantic-style data models (if added later)
  session/           # Session abstraction & lifecycle mgmt
  session_manager/   # Session registry
  oauth/             # Token exchange + decorators
  mcp_server/        # Server parameter helpers
  test_*.py          # Tests (pytest)
```
Docker / deployment assets live at repository root.

---
## 🧪 Testing
We use `pytest`.
- Start by running the whole suite: `pytest -q`
- Write tests for every new feature or bug fix (regression tests).
- Favor small, explicit tests. Use fixtures when subprocess / network needed (see `test_app.py`).
- Parallel / race-sensitive logic: add deterministic tests (e.g. tool calls & sessions).

### Test Guidelines
- Keep external network out of unit tests. If you must mock, isolate in a narrow function and patch there.
- Tests that spawn the server should assert success fast (use existing `wait_for_port`).
- Add at least one negative test per feature (bad args, missing session, unknown tool, token failure).

---
## 🧹 Code Style & Quality
- Follow standard Python style (PEP 8). Black / Ruff not yet configured—feel free to propose adding them.
- Keep functions small; log at INFO for high-level events, DEBUG for internals.
- Avoid broad `except:`; catch specific exceptions.
- Type hints encouraged for new/changed public functions.
- Avoid adding heavy dependencies; discuss before introducing anything > ~1MB or with transitive risk.

### Suggested (Optional) Local Tooling
If you add tooling, document it—example (future):
```
pip install ruff black
ruff check .
black .
```
(Do not fail CI on style until the repo adopts a formatter baseline.)

---
## 🔐 Security & Privacy
- Never log raw OAuth tokens or secrets—mask them.
- Validate and sanitize any user-provided command / env values.
- Avoid expanding attack surface (e.g., arbitrary file reads).
- Prefer explicit allowlists over denylists.

Report vulnerabilities privately (see Security Reporting below) before opening a public issue.

---
## 🧩 Adding / Modifying Endpoints
1. Define request/response schema (consider creating / updating models in `models.py`).
2. Add routing logic in `routes.py` (keep orchestration only—push logic into helpers if it grows).
3. Add tests covering success + failure paths.
4. Update `README.md` (usage examples) if it affects API behavior.
5. Consider impact on sessions & token injection.

---
## 🔁 Session & MCP Interaction Patterns
- Prefer reusing session tasks when a session header/cookie is provided.
- Ensure proper cleanup on `/session/close` (await stop()).
- For background tasks, ensure they won't leak after test completion.

---
## 🔑 OAuth / Token Exchange
- New providers: implement retriever in `oauth/token_exchange.py` and register it in the factory.
- Decorators should remain idempotent and only augment args when required.
- Add tests for: valid exchange, invalid token, missing env.

---
## 🐞 Reporting Issues
Include:
- Environment (OS, Python version)
- Install method (pip editable install, Docker, etc.)
- Repro steps (exact curl / code)
- Expected vs actual behavior
- Logs (trim sensitive data)

For security issues: email the maintainer (see `pyproject.toml` author email) with subject: `[SECURITY] <short description>`.

---
## 🚀 Feature Requests
Open an issue describing:
- Problem / use-case
- Proposed solution (API shape, example request/response)
- Alternatives considered
- Backward compatibility impact

Small, additive features may go straight to PR if low-risk and well-tested.

---
## 🧱 Breaking Changes
- Must be discussed in an issue first.
- Provide migration notes in PR description.
- Update version (minor for added, patch for fixes, major for breaking once 1.0.0+).

Current version: `0.1.0` (pre-1.0: breaking changes may increment minor; still document clearly).

---
## 📝 Commit Messages
Use Conventional Commits (enables future automated changelogs):
- `feat: add session timeout config`
- `fix: handle unknown tool error masking`
- `docs: improve README with OAuth example`
- `test: add regression for call_counter session cookie`
- `refactor: extract tool mapping helper`
- `chore: bump version to 0.1.1`

Footer examples:
- `BREAKING CHANGE: renamed env var MCP_BASE_PATH -> MCP_ROOT_PATH`
- `Closes #42`

---
## 🔄 Pull Request Checklist
Before submitting:
- [ ] Branch is up to date with `main`
- [ ] Tests pass locally (`pytest -q`)
- [ ] Added / updated tests
- [ ] README / docs updated (if user-facing change)
- [ ] No secrets in diff
- [ ] Clear commit history (rebase / squash if noisy)

PR description should include: summary, motivation, test coverage, any follow-ups.

---
## 🧪 Continuous Integration (Future)
CI pipeline (to be added) could run:
1. Install deps
2. Run tests
3. Optional: lint / type check
4. Build Docker image & maybe run a smoke test

Contributors welcome to add a lightweight GitHub Actions workflow.

---
## 📦 Releasing
(Maintainers)
1. Update version in `pyproject.toml`
2. Tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
3. Push tag: `git push origin vX.Y.Z`
4. (Future) Build & publish to PyPI / build & push Docker image
5. Draft release notes (aggregate Conventional Commits if available)

---
## 📜 License
See `LICENSE.md`. By contributing, you agree that your contributions will be licensed under the same license.


---
## 🤖 AI-Assisted Contributions
You may use AI tools (code assistants, generation, refactor, documentation, test suggestion) while contributing, under these conditions:

### Expectations
- Human Review Required: Carefully read every generated line; you're the author of record.
- Test Everything: Run the full test suite (`pytest -q`) and add/adjust tests for AI-produced changes.
- Security & Privacy: Do not paste secrets, proprietary third‑party code, or internal system details into AI prompts.
- Licensing: Ensure output does not reproduce non-compatible licensed material. When in doubt, rewrite.
- Quality: Avoid blind bulk rewrites; keep diffs minimal and purposeful.
- Attribution (Optional): If > ~30% of a PR was AI-assisted, add a short note in the PR (e.g., "Portions drafted with AI, manually reviewed").

### Not Allowed
- Submitting unreviewed raw AI output.
- Introducing dependencies or architectural shifts justified only by AI suggestion without human rationale.
- Fabricated benchmarks, test results, or references.

### Good Practices
- Use AI to propose alternative designs; pick one and explain why.
- Let AI draft boilerplate, then you refine for clarity & logging consistency.
- Ask AI for additional edge cases—add those that are relevant.

### Maintainer Rights
Maintainers may request clarification on AI involvement or ask for additional tests / refactors if AI artifacts reduce readability or cohesion.

By submitting, you confirm you exercised due diligence and accept responsibility for the contribution.

---
## 🧭 Roadmap Ideas (Open for PRs)
- Add session timeout / eviction policy
- Optional Redis / external session backend
- Structured logging config & correlation IDs
- Type hints + mypy config
- Automated release workflow (GitHub Actions)
- Add OpenAPI examples
- E2E test via Docker

---
## 🙏 Thanks
Your time and contributions help make this project better. Feel free to open a draft PR early for feedback.
