# ADR 0004: Async `/ingest` for PDF content, synchronous for everything else

## Status

Superseded by [ADR 0005](0005-ingest-timeout-race-dispatch.md).

## Context

`/ingest` is called synchronously by external automations, behind an nginx Ingress with the
default 60s read timeout (no override, unlike the MCP ingress's `3600`). Getting real structure
out of a PDF (headings, figures — not just raw text) costs ~1.1-1.5s/page on the CPU-only
`dsview-api` deployment, a floor that more CPU doesn't move; a 37-page paper alone took 45-57s.
Long PDFs are therefore structurally incompatible with the 60s deadline, and there's no message
queue available to defer this work generically.

## Decision

Only PDF ingestion returns immediately (`202`) and finishes in a background task. Every other
content type (web, YouTube, arXiv-via-alphaXiv) stays exactly as synchronous as today, since it's
fast and callers expect a completed response. This removes the timeout risk for PDFs without
degrading extraction quality or changing the contract for other callers.

## Design

- **PDF detection**: `isinstance(get_content_loader(content.link), PdfUrlLoader)` — cheap and
  side-effect-free (no I/O until `.load()`). `ArxivContentLoader` subclasses `PdfUrlLoader`, so
  arXiv links go the async route too, even though most resolve fast via alphaXiv — erring toward
  async is always safe.
- **`/ingest` in `dsview/api.py`**: drops the `@api_sync_vault` decorator (can't conditionally
  defer parts of its fixed pull → await → push sequence) and inlines `pull_changes()` /
  `upload_changes()` directly. Non-PDF path is unchanged (pull → await pipeline → push → `200`).
  PDF path: pull stays sync, then the pipeline call *and* the vault push move into
  `BackgroundTasks.add_task`, returning `202 {"status": "processing"}` immediately. The push must
  move too since the pipeline is what writes the Obsidian notes being pushed.
- **DB session**: safe to reuse in the background task — FastAPI runs a `yield`-dependency's
  cleanup after `BackgroundTasks` finish, not after the response is sent.
- **Failures**: already handled — `IngestPipeline.async_ingest_content` wraps its body in
  try/except → `save_failed_ingestion` + rollback/commit, so no new error handling is needed.
- No changes to `content_loader.py`, `ingest_source.py`, `sync_vault.py`, the CLI, or the
  Kubernetes manifests — this is purely an API-layer workaround for the Ingress timeout.

## Testing

New `tests/test_api.py` (none exists today), via `TestClient` + dependency overrides, with
`async_ingest_content`/`pull_changes`/`upload_changes` mocked (hermetic, no real DB/git/network).
Two cases: non-PDF URL → `200`, pipeline awaited before response; PDF URL → `202` returned before
the mocked pipeline resolves, then pipeline invocation asserted (`TestClient` runs background
tasks synchronously as a context manager).

## Consequences

- PDF callers can no longer treat the `/ingest` response as "content is now live" — the note/DB
  row appears sometime after the `202`. Automations that read back a just-ingested PDF need to
  change.
- Background failures are only visible via the DB/vault, not at the `/ingest` call site.
- Sets the pattern for any future slow content type: a cheap loader-type check + `BackgroundTasks`,
  rather than introducing a message queue.
