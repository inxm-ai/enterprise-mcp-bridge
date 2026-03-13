# Observability

OTel instrumentation, wide events, SLOs, sampling, and blameless postmortems. Read this for any task involving logging, tracing, metrics, alerting, or incident response.

---

## Philosophy

Observability is not a fire alarm you check when things break. It is a **continuous feedback loop** you use to validate every change after shipping. You build it, you own it in production — there is no ops team handoff.

**Today**: OTel three pillars (traces + metrics + logs) via the OTel SDK and collector pipeline.
**North star**: wide structured events as a single source of truth — high cardinality, high dimensionality, explorable. Every additional attribute makes every other attribute combinatorially more valuable.

When writing instrumentation, always ask: *could an LLM agent use this data to understand what happened, diagnose a failure, or detect a product usage pattern?* If not, the events are too thin.

---

## Observability-Driven Development (ODD)

Instrument as you write — not after. Treat instrumentation as part of the feature, not a follow-up task:
- Before shipping a feature, ask: *how will I know this is working correctly in production?*
- Add the spans, attributes, and events that answer that question *before* or *during* implementation.
- After shipping, check production to confirm the code behaves as expected — close the loop.

**Instrumentation Ownership**: The engineer who writes the feature writes the instrumentation. No exceptions, no delegation. Auto-instrumentation from frameworks is a floor, not a ceiling — always add custom attributes that carry business and domain context.

---

## Wide Events — the core unit of instrumentation

Every request, job, or meaningful operation emits a **single wide event** (a span or structured log) that carries the full context of that operation. Not a log line. Not a metric point. A rich blob.

**Minimum required fields on every span/event:**
```
service.name       — which service
service.version    — build/deploy version
trace_id           — distributed trace correlation
request_id         — unique per request
user_id            — who triggered this (where applicable)
build_id           — which build/commit is running
feature_flags      — active flags for this request
environment        — dev / staging / prod
```

Add domain-specific fields liberally. More context = more power. The goal is to answer *any* question about a request from a single event, without hopping between tools.

**Never pre-aggregate away the raw event.** Pre-aggregation destroys the relational seams between attributes and permanently forecloses future questions.

---

## High Cardinality by Default

High cardinality fields — `user_id`, `request_id`, `session_id`, `build_id`, `feature_flag_variant` — are first-class span attributes. They are the *most valuable* fields because they let you slice to a single user, request, or deployment.

Never sacrifice cardinality to reduce cost. Use sampling strategies (tail-based or head-based) to control volume while preserving full fidelity of sampled events.

---

## LLM-Friendly Instrumentation

Observability data will be consumed by LLMs for ops automation and product intelligence. Design with this in mind:

- **Semantic, self-describing field names**: `order.payment.status` not `p_stat`. An LLM (and a human) should understand the field without a schema lookup.
- **Consistent naming convention**: `<domain>.<entity>.<attribute>` — e.g. `user.subscription.tier`, `pipeline.run.duration_ms`.
- **Business-level events alongside technical events**: `user.report.exported`, `pipeline.dag.completed` — these feed product analytics and LLM-driven product intelligence.
- **Causal context**: include `triggered_by` (human action, scheduled job, LLM agent, upstream service) on every event.
- **No log-level filtering in production** — use sampling, not level suppression. A DEBUG event that gets sampled is infinitely more valuable than one that never existed.

---

## SLOs as the Entry Point — Not Dashboards

Dashboards only answer questions you predicted in advance.

- Define SLOs for every user-facing behaviour (latency p99, error rate, availability).
- When writing a new service or feature, ask: *what is the SLO for this?* before asking *what should I dashboard?*

When scaffolding a new service or feature, always define:
```
SLI:    what are we measuring? (e.g. % of requests completing in < 200ms)
SLO:    what's the target? (e.g. 99.5% over a rolling 28-day window)
Budget: how much can we burn? (e.g. 0.5% * 28 days = ~3.4 hours)
Alert:  at what burn rate do we page? (e.g. 2% budget consumed in 1h)
```

---

## SLO Burn Rate Alerts — Not Threshold Alerts

Threshold alerts (`error_rate > 1%`) are for known-unknowns only. They produce alert fatigue and miss slow-burning problems.

Alert on **error budget burn rate**:
- Define an error budget for each SLO (e.g. 99.9% availability = 43.8 min/month downtime budget).
- Alert when the burn rate is consuming the budget too fast to survive the month.
- Use a two-window burn alert: a short window (1h) catches fast burns, a long window (6h) catches slow burns — both must be elevated to page.
- SLO burn alerts page engineers. Infrastructure threshold alerts ticket.

---

## Sampling Strategy

Sampling is the correct lever for cost control — not reducing cardinality, not dropping log levels.

- **Tail-based sampling preferred**: make the sampling decision *after* the trace is complete, so you can keep 100% of errors, slow requests, and outliers while sampling routine traffic.
- **Head-based sampling**: acceptable when tail-based infrastructure isn't available. Use deterministic sampling keyed on `trace_id` so all spans of a trace are sampled consistently.
- Always record `sample_rate` as an attribute on every sampled event so downstream analysis can correctly reconstruct counts.

---

## Core Analysis Loop — Debugging from First Principles

When something goes wrong in production, never grep-and-pray. Use the core analysis loop:

1. **Form a hypothesis** — what do you believe is happening and why?
2. **Find the data** — query your observability backend to confirm or refute it.
3. **Refine** — if refuted, update your hypothesis and repeat. If confirmed, dig deeper.
4. **Resolve** — act only when you have data-backed confidence in the cause.

When helping debug a production issue, always apply this loop explicitly: state the hypothesis, identify the query/data that would test it, reason from evidence.

---

## Blameless Postmortems

Every significant incident — any event that meaningfully degrades user experience, burns error budget, or requires on-call response — produces a written postmortem within 48 hours.

**Structure:**
- **What happened** — timeline of events, factual, no editorialising.
- **Why it happened** — the system conditions that made this possible (not "who did it").
- **What we learned** — about the system, the instrumentation, the process.
- **What we're changing** — concrete follow-up actions with owners and dates.

**Rules:**
- Never name individuals as the cause. The system allowed the failure — fix the system.
- If the incident would have been caught faster with better observability, the postmortem must include an instrumentation improvement action.
- Postmortems are shared across the team — learning artifacts, not blame documents.
- If an incident recurs without a postmortem action having been completed, that is a process failure worth noting explicitly.

---

## Backend & Tooling

- **Internal**: Jaeger today, migrating toward Honeycomb.
- **Customer-facing**: tool-agnostic — emit standard OTel, let the backend be swappable.
- All instrumentation goes through the OTel SDK and collector pipeline — never vendor-specific SDKs directly in application code.

## Per-language Implementation

- **Rust**: `tracing` crate + `tracing-opentelemetry` subscriber + `opentelemetry-otlp` exporter.
- **Python** (Airflow, MCPs): `opentelemetry-sdk` + `opentelemetry-exporter-otlp`.
- **JavaScript/pfusch FE**: OTel browser SDK; instrument user interactions and component lifecycle errors.
- **Span naming**: `<service>.<domain>.<action>` — e.g. `orders.payment.charge`, `pipeline.dag.trigger`.
- **Never log sensitive data** (tokens, passwords, PII) — log IDs and reference keys only.
