Here is a draft plan to refine:

# Issue #89 — MCP spec 2026-07-28: impact analysis & implementation stories (markdown doc)

## Context

Issue #89 ("Update to 2026-07-28 of mcp") asks to identify the diffs introduced by the new MCP specification revision and create implementation stories. The bridge currently pins mcp[cli]>=1.22,<2 (app/pyproject.toml:12), which speaks spec **2025-06-18** — so it is **two revisions behind** (2025-06-18 → 2025-11-25 → 2026-07-28). The 2026-07-28 revision is a fundamental redesign (stateless protocol, sessions removed, elicitation/sampling replaced by Multi Round-Trip Requests, HTTP+SSE transport formally deprecated). The Python SDK that implements it is **v2, currently 2.0.0rc1** (pre-release, breaking changes possible between RCs; renames FastMCP → MCPServer, unified Client object, snake_case fields).

## Deliverable

One new file: **update-path.md** at the repo root (per user request — "for now"; can move into docs/ later). No code changes. Contents:

### 1. Spec diff summary (2025-06-18 → 2026-07-28, both changelogs merged)
- **Stateless core**: initialize handshake and Mcp-Session-Id removed; protocol version, capabilities, client/server identity travel in _meta per request; new mandatory server/discover RPC.
- **MRTR (Multi Round-Trip Requests)** replaces server-initiated elicitation/create / sampling/createMessage / roots/list: server returns resultType: "input_required" + inputRequests; client retries the original request with inputResponses. All results now carry required resultType.
- **Transports**: HTTP+SSE formally Deprecated; GET stream and resources/subscribe replaced by subscriptions/listen; SSE resumability (Last-Event-ID) removed; new required headers Mcp-Method/Mcp-Name on Streamable HTTP POSTs; x-mcp-header custom headers from tool params.
- **Deprecations**: Roots, Sampling, Logging features deprecated (per-request log level via _meta io.modelcontextprotocol/logLevel); ping, logging/setLevel removed; tasks moved to an official extension.
- **Caching**: required ttlMs/cacheScope on list/read results (CacheableResult); deterministic tools/list ordering recommended.
- **Auth**: RFC 9207 iss validation required; credentials keyed per issuer; DCR (RFC 7591) deprecated in favor of Client ID Metadata Documents; application_type required in DCR; OIDC discovery + incremental scope consent (from 2025-11-25).
- **From 2025-11-25 also**: icons metadata, enum/default-value elicitation schema extensions, JSON Schema 2020-12 dialect, resource-not-found error code -32002 → -32602, error-code allocation policy.

### 2. Per-area impact on the bridge (with file references from exploration)
| Area | Impact |
|---|---|
| SDK pin app/pyproject.toml:12 | Blocks v2; upgrade is prerequisite for everything protocol-level |
| Client strategies app/session/client_strategy.py | ClientSession+transport contexts → unified Client; snake_case; elicitation_callback removed (MRTR); logging_callback deprecated |
| Elicitation app/elicitation/coordinator.py, app/routes.py:497, app/sse/streaming.py:137 | Rework coordinator around InputRequiredResult/retry; the existing REST 409 + feedback_required SSE pattern maps naturally to MRTR; support 2025-11-25 enum/default schema extensions |
| MCP server facade app/sse/mcp_proxy.py | Uses deprecated HTTP+SSE; needs a stateless Streamable HTTP facade (+ subscriptions/listen for list-changed); good news: statelessness removes session-affinity concerns |
| Bridge sessions app/session/session.py, session_manager/ | Bridge's own X-INXM-MCP-Session layer is unaffected in concept (spec now blesses app-level handles), but downstream session lifecycle changes with stateless clients |
| Tool listing cache app/session_manager/session_context.py:212,250 | Honor ttlMs/cacheScope; emit deterministic ordering + cache fields from the facade |
| Results app/models.py, app/routes.py:470-483 | Surface/tolerate resultType; treat absent as "complete"; error-code change -32002→-32602 |
| OAuth app/session/client_strategy.py:153-226, app/oauth/*, app/well_known/oauth_metadata.py | Issuer-keyed token storage, iss validation, CIMD support, application_type in DCR, WWW-Authenticate with resource_metadata (already a gap) |
| Not implemented, now deprecated (roots, sampling, logging feature, tasks) | Do-nothing wins — record as explicitly out of scope |

### 3. Stories (numbered, each with goal / affected files / acceptance criteria / toggle behavior / dependency)
0. **Epic/tracking story** — adopt MCP 2026-07-28, feature-toggled.
1. **SDK v2 migration** (blocked on mcp 2.x stable; prep on RC in a branch): upgrade pin, Client API, snake_case, FastMCP→MCPServer in mcp/ example server. Prerequisite for stories 2–4.
2. **Feature toggle scaffolding**: e.g. MCP_SPEC_REVISION=2025-06-18|2026-07-28 env var in app/vars.py, threaded through facade + client strategy; default stays current behavior.
3. **MRTR elicitation rework** (toggle-dependent surface: keep 409 contract, change internals).
4. **Streamable HTTP server facade** replacing/alongside the SSE proxy (toggle selects exposed transport; SSE kept during deprecation window).
5. **Caching & determinism**: ttlMs/cacheScope passthrough + deterministic tool ordering (SDK-independent, can start now).
6. **Auth hardening**: issuer-keyed credentials, iss validation, CIMD, application_type, WWW-Authenticate challenge (largely SDK-independent, can start now).
7. **Logging/notifications migration**: per-request logLevel _meta, keep SSE log events working in both modes.
8. **Result model updates**: resultType, error codes, JSON Schema 2020-12 loosening for input/output schemas (app/json/schema_validation.py).
9. **Docs**: update docs/ + README for supported spec revisions and the toggle.

### 4. Sequencing note
Stories 5, 6, 8 (partially) are implementable on SDK 1.x now; 1–4, 7 gate on SDK v2 stable. Recommend skipping a 2025-11-25 stopover — several of its additions (tasks, elicitationId, URL-elicitation completion notification) were reverted or redesigned in 2026-07-28.

## Steps
1. Write update-path.md at the repo root per outline above (sources: both official changelogs at modelcontextprotocol.io, SDK v2 beta announcement).
3. Post a short comment on issue #89 linking the doc path/branch? — **No**: user chose "markdown doc first" for review; don't touch GitHub until they've reviewed.

## Verification
- Doc renders cleanly (markdown lint by eye), file references are real paths in the repo (spot-check each path:line cited).
- Every changelog entry from both revisions is either mapped to a story, marked no-impact, or marked out-of-scope — no silent omissions.
- User reviews the doc; GitHub issues creation happens afterwards as a follow-up on request.
