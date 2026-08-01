# ADR 0005: `/ingest` dispatches on a timeout race, not content-loader type

## Status

Accepted. Supersedes ADR 0004.

## Context

ADR 0004 dispatched `/ingest` synchronously vs. in the background by checking
`isinstance(get_content_loader(link), PdfUrlLoader)` up front. A review (GitHub issue #47) found
this coupling caused real bugs: `ArxivContentLoader` subclasses `PdfUrlLoader` purely for code
reuse, so every arXiv link took the async `202` path even when it resolves in under a second via
alphaXiv — contradicting ADR 0004's own stated intent to keep arXiv-via-alphaXiv synchronous.
Duplicate detection happened after dispatch, so a duplicate PDF/arXiv link returned `202` instead
of the `409` every other content type got. The background task also reused the request's DB
`Session`, which FastAPI closes before background tasks run (undocumented ordering, not the
documented guarantee ADR 0004 assumed), and had no error handling around the vault push.

More fundamentally, `isinstance(..., PdfUrlLoader)` makes the content-loader class hierarchy
respons­ible for an API-layer scheduling decision. Any future slow content type would need another
loader-type check wired into `/ingest`, repeating the coupling.

## Options considered

1. **Patch in place**: fresh session per background task, wrap the vault push in try/except, add
   a `get_content` pre-check for the 409 gap, and give loaders an explicit `IS_ASYNC` capability
   flag so arXiv-via-alphaXiv can opt out of the PDF branch. Fixes every reported bug but keeps
   the API layer inspecting content-loader type to decide how to run the pipeline — the
   architectural coupling stays, just with one more flag to keep in sync per content type.
2. **Timeout race**: every link runs through the exact same code path; the endpoint awaits the
   pipeline with a bounded timeout and returns `200`/`409`/`500` if it finishes in time, or `202`
   and lets it keep running detached if it doesn't. No content-loader type is consulted by
   `/ingest` at all.
3. **Message queue**: unavailable in this deployment (already ruled out in ADR 0004's context).

## Decision

Adopt the timeout race (option 2). `dsview/api.py` runs every submitted link through
`_run_ingest_and_sync`, awaited via `asyncio.wait({task}, timeout=INGEST_SYNC_TIMEOUT_SECONDS)`
(`50.0`, comfortably under the nginx 60s read timeout from ADR 0004). If the task is still
`pending` at the timeout, respond `202` and let it finish unattended; a module-level task set with
done-callbacks keeps it alive and logs any exception it eventually raises. Otherwise map its
outcome to `200`/`409`/`500` exactly as before.

`_run_ingest_and_sync` opens its own `Session(engine)` rather than reusing the request's, matching
the pattern already used by every MCP server tool. `save_content`'s duplicate check runs as the
pipeline's first statement, in milliseconds, so `ContentAlreadyExists` reliably lands within the
race window and maps to `409` regardless of link type — no separate pre-check needed.

Alongside this, `pull_changes`/`upload_changes` gained an `asyncio.Lock` (`async_pull_changes`/
`async_upload_changes` in `dsview/obsidian/sync_vault.py`, shared with `/relevance`'s
`api_sync_vault`), since concurrent ingests can now genuinely overlap in ways the old
`background_tasks.add_task` sequencing made less likely to surface. `/ingest/status` now checks
extraction results before a `FailedIngestion` row, so a link that failed once and later succeeded
no longer reports `failed` forever.

## Consequences

- `PdfUrlLoader`/`get_content_loader` are no longer imported by `dsview/api.py` — the API layer
  has no knowledge of content-loader types. Any future slow content type is handled automatically
  by the same race, with zero dispatch changes.
- `INGEST_SYNC_TIMEOUT_SECONDS = 50.0` sits inside the 45-57s large-PDF range from ADR 0004, so
  big PDFs will sometimes finish just under it (`200`) and sometimes just over (`202`),
  non-deterministically. Both outcomes are correct; it's no longer a clean PDF/non-PDF split even
  qualitatively.
- In-process background work still isn't durable: if the API process restarts while a task is
  running past the timeout, that ingestion is silently lost — no `FailedIngestion` row, and
  `/ingest/status` reports `pending` forever for that link. Same structural limitation as ADR
  0004, still out of scope.
- The task-tracking set and vault lock are per-process/per-event-loop, fine at the current
  `replicas: 1` but would need revisiting if that changes.
- `pdf_api_plan.md` is removed; this ADR and ADR 0004 are now the record of the `/ingest` design.
