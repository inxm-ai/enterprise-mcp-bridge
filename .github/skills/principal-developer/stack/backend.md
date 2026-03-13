# Backend — Rust, Kotlin, Airflow, MCPs

Read this for any task involving backend services, orchestration pipelines, or MCP tool development.

---

## Rust — Primary Services

Rust is the default for all new backend services.

| Concern | Library |
|---|---|
| HTTP server | `actix-web` |
| Async runtime | `tokio` |
| Serialisation | `serde` + `serde_json` — everywhere, no exceptions |
| Outgoing HTTP | `reqwest` |
| Domain errors | `thiserror` |
| Application-level error aggregation | `anyhow` |
| Tracing / observability | `tracing` + `tracing-opentelemetry` |

- **No gRPC** — REST only.
- All error handling follows ROP (`Result<T, E>`) — see `engineering-principles.md`.
- `?` operator is idiomatic and preferred for propagating errors.
- Use `thiserror` for library/domain error types; `anyhow` for top-level application aggregation where specific types don't matter to callers.
- Serialisation: `serde` on every type that crosses a boundary. No manual JSON construction.

---

## Kotlin — Glue Code Only

Kotlin is used minimally for glue/integration purposes. No large Kotlin services.

- `arrow-kt` for `Either`/`Result` style error handling where needed.
- Keep Kotlin surface area small; prefer delegating to Rust services.
- When writing Kotlin, apply the same ROP principles — errors as data, no swallowed exceptions.

---

## Apache Airflow + Python — Orchestration

Airflow is for orchestration only. Business logic does not live here.

- DAGs are **pure orchestration** — no business logic inside tasks or operators.
- Use classic DAG authoring syntax (no TaskFlow `@task` decorators) — evolve as needed.
- Each Airflow operator wraps exactly one concern (Shore Infrastructure Wrapper principle).
- Business logic lives in **separate Python modules** imported by tasks, not inline.
- DAGs are generated and evolved with LLM assistance — keep them readable and explicit.
- Treat each DAG as a thin vertical slice of orchestration; don't overload a single DAG.
- Observability: `opentelemetry-sdk` + `opentelemetry-exporter-otlp` — instrument task start, completion, and failure with wide events.

---

## MCPs — Python + enterprise-mcp-bridge

MCP servers are thin tools only. No orchestration logic inside tools.

- Deployed via [enterprise-mcp-bridge](https://github.com/inxm-ai/enterprise-mcp-bridge): FastAPI host providing multi-tenancy, OAuth, structured logging, and OTel tracing.
- MCP server code lives in `mcp/server.py` inside the bridge repo structure.
- **Each tool wraps one well-defined action** — single concern, thin interface.
- Each tool must be independently testable without an LLM — test inputs/outputs directly.
- OAuth token management and session handling is handled by the bridge — MCP tools never manage auth themselves.
- Structured logging is provided by the bridge — tools emit semantic log events, not raw print statements.
- Tools must be stateless where possible; stateful tools use the bridge's session manager abstraction.
- Apply the Logic Sandwich principle: pure logic in the tool handler, I/O at the edges via the bridge abstractions.
