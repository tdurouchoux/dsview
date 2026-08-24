---
status: "accepted"
date: "2026-08-24"
decision-makers: "tdurouchoux"
---

# Use Pydantic Logfire for dsview observability

## Context and Problem Statement

The dsview project needs an observability solution covering both LLM call tracing and general application metrics, with minimal components to maintain. Which solution should be adopted to instrument and monitor the application?

## Considered Options

* Pydantic Logfire
* OpenObserve (self-hosted, OTel-native)
* SigNoz (self-hosted)
* Datadog LLM Observability
* Grafana cloud
* Langfuse
* Opik / Comet
* Evidently AI

## Decision Outcome

Chosen option: "Pydantic Logfire", because it is a thin layer on top of OpenTelemetry (making future migration easy if needed), its hosted offering covers the project's log volume needs for free, it minimizes additional components to maintain, and it provides dedicated LLM features while still covering general application observability in a single tool.

### Consequences

* Good, because native OTel export avoids vendor lock-in and allows switching to another backend (e.g. Jaeger, SigNoz) without re-instrumenting the code
* Good, because the hosted offering removes the need to maintain dedicated observability infrastructure
* Bad, because the backend is a closed-source SaaS solution, with no full self-hosted option outside an enterprise license

## More Information

Integration delivered in PR #60.
