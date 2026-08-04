---
status: "accepted"
date: "2026-08-01"
decision-makers: "tdurouchoux"
---

# `/ingest` is fully asynchronous; clients must always poll `/ingest/status`

## Context and Problem Statement

ADR-0004 dispatched `/ingest` synchronously vs. in the background by checking
`isinstance(get_content_loader(link), PdfUrlLoader)` up front. A review (GitHub issue #47) found
this coupling caused real bugs: `ArxivContentLoader` subclasses `PdfUrlLoader` purely for code
reuse, so every arXiv link took the async `202` path even when it resolves in under a second via
alphaXiv. Duplicate detection happened after dispatch, so a duplicate PDF/arXiv link returned `202`
instead of the `409` every other content type got. The background task also reused the request's
DB `Session`, which FastAPI closes before background tasks run, and had no error handling around
the vault push.

This decision originally replaced that with a **timeout race**: every link ran through the same
code path, awaited with a bounded timeout (`INGEST_SYNC_TIMEOUT_SECONDS = 50.0`), returning
`200`/`409`/`500` if the pipeline finished in time or `202` if it didn't. This fixed every bug
above and removed all content-loader-type awareness from `dsview/api.py`.

In practice, that contract turned out to be **harder for API clients than the problem it solved**.
A single endpoint that could non-deterministically return either a completed result
(`200`/`409`/`500`) or an in-progress marker (`202`) for the *same link*, depending only on how
close the pipeline happened to land to the timeout, meant every caller had to handle both response
shapes on every call anyway — there was no way to know in advance which one a given request would
get. The race removed the old PDF-vs-everything-else special case at the cost of a new, more subtle
one: fast-today-slow-tomorrow non-determinism, per link, per request.

## Considered Options

* Always return `202`; clients poll `/ingest/status` for the outcome (chosen)
* Dispatch sync vs. async by content-loader type (ADR-0004's approach)
* Bounded timeout race: await the pipeline, return `200`/`409`/`500` if it finishes in time,
  `202` otherwise (this decision's original approach)

## Decision Outcome

Chosen option: "Always return `202`; clients poll `/ingest/status`", because a single endpoint
that could non-deterministically return either a completed result or an in-progress marker for
the same link meant every caller had to handle both response shapes anyway — dropping the race
removes that non-determinism at the cost of always polling, giving exactly one contract shape,
always.

Because there's no more wait window for the endpoint to observe the pipeline's own duplicate check
through, duplicate detection moves to an explicit, synchronous pre-check —
`IngestPipeline.get_existing_content(link, session)` — run before the background task is even
created. If it finds a match, `/ingest` raises `HTTPException(409)` immediately and never starts
the task. Without this pre-check, a duplicate submission would just return `202` like any other
request, and the caller would only ever observe the *original* ingestion's outcome via
`/ingest/status`, with no distinct signal that their submission was a no-op.

The background task (`_run_ingest_and_sync`) is unchanged from this decision's original approach:
its own `Session(engine)`, tracked in a module-level `_background_ingest_tasks` set with
done-callbacks so it isn't garbage-collected mid-flight. Any exception it raises — including a
genuine race condition where `ContentAlreadyExists` still reaches the pipeline despite the
pre-check — is now logged as an error like any other background failure; there's no special-casing
left, since the common case is caught synchronously beforehand and this path should be rare. A new
`@app.exception_handler(HTTPException)` standardizes every raised `HTTPException` (the `409` above,
`/relevance`'s `404`, `/health`'s `503`) into `{"status": "error", "detail": ...}`.
`/ingest/status`'s failed response also dropped `error_type`, down to `{"status": "failed",
"detail": failed.error_message}`.

### Consequences

* Good, because there is now exactly one `/ingest` contract, always: `202` immediately, poll
  `/ingest/status` for the real outcome. No caller — human, script, or the CLI/MCP server — can
  treat a `202` as "it's still running, but it might have already finished" anymore; it never has
  finished by the time the response comes back.
* Good, because `PdfUrlLoader`/`get_content_loader` are still not imported by `dsview/api.py`, and
  this is now even more true than under the timeout race: no content-loader type, and no timing
  behavior of any kind, is ever consulted to decide how `/ingest` responds. Any future slow (or
  fast) content type needs zero changes to the endpoint.
* Neutral, because `INGEST_SYNC_TIMEOUT_SECONDS`, the `asyncio.wait` race, and the nginx 60s
  read-timeout margin from ADR-0004 are no longer relevant to `/ingest` at all — the response is
  always near-instant regardless of pipeline duration.
* Bad, because duplicate detection is no longer "free": under the race, `save_content`'s duplicate
  check ran as the pipeline's first statement and reliably landed within the race window, needing
  no separate check. Now it's a deliberate, explicit pre-check duplicating (deliberately) the same
  lookup the pipeline would otherwise do — a small amount of redundancy in exchange for a `409` the
  caller can actually rely on seeing immediately.
* Bad, because in-process background work still isn't durable: if the API process restarts while a
  task is running, that ingestion is silently lost — no `FailedIngestion` row, and
  `/ingest/status` reports `pending` forever for that link. Same structural limitation as
  ADR-0004 and this decision's original timeout-race approach, still out of scope.
* Neutral, because the task-tracking set and vault lock (`async_pull_changes`/`async_upload_changes`,
  an `asyncio.Lock` shared with `/relevance`'s `api_sync_vault`) are per-process/per-event-loop,
  fine at the current `replicas: 1` but would need revisiting if that changes.

## More Information

This ADR supersedes ADR-0004. It also revises this ADR's own original decision (the timeout race
described in Context above) in place, rather than as a new numbered ADR, since that design never
shipped as a stable contract for external callers before being replaced.
