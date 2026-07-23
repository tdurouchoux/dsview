# Async `/ingest` for PDF content, synchronous for everything else

## Context

`dsview/api.py`'s `/ingest` endpoint is called synchronously by external automations, and its
Ingress (`kubernetes/dsview_deployment.yaml`) sets no `proxy-read-timeout` override — unlike the
MCP ingress, which explicitly sets `3600`. nginx-ingress's default read timeout is 60s, so any
request that runs longer gets killed at the proxy regardless of what `dsview` itself does. That's
the real constraint behind the "60 second" budget explored earlier.

Investigation into PDF extraction quality (comparing `pypdf`, `docling`, and `xberg`/`kreuzberg`'s
layout detection) showed that getting real document *structure* out of a PDF (headings, isolated
figures/formulas — not just raw text) costs roughly 1.1-1.5s per page on this CPU-only deployment
(`dsview-api` is `replicas: 1`, `limits.cpu: 4000m` per the deployment yaml), a floor that more
threads/cores don't move — confirmed by raising `num_threads` and batch size and seeing wall time
barely change while CPU-seconds consumed roughly doubled. A 37-page paper alone took 45-57s. That
makes "quality analysis of long PDF documents" structurally incompatible with a hard 60s
synchronous deadline: a 60+ page survey would blow the timeout on PDF processing alone, before any
LLM extraction call even runs, and there's currently no message queue available to defer this kind
of work generically.

The decision: keep every other content type (web pages, YouTube, arXiv-via-alphaXiv) exactly as
synchronous as today, since those are fast and callers currently expect a response only once
ingestion is fully done. Only PDF ingestion — the one case with a real, unavoidable latency floor —
should return immediately and finish in the background. This sidesteps the timeout entirely for
PDFs without needing to bound or degrade PDF extraction quality, and keeps the synchronous contract
for every other caller unchanged.

## Design

### 1. Detecting "is this a PDF" cheaply, before loading anything

`get_content_loader(link)` (`dsview/extraction/content_loader.py`) already returns `PdfUrlLoader`
for direct PDF links and `ArxivContentLoader` for arXiv links — and `ArxivContentLoader` subclasses
`PdfUrlLoader` specifically (this was the reuse reason it was built that way in the prior arXiv
work on this branch). So:

```python
isinstance(get_content_loader(content.link), PdfUrlLoader)
```

is a single, correct, side-effect-free check (construction does no I/O — only `.load()` does) that
covers both "always a PDF" and "usually alphaXiv, but PDF if that misses." Treating *all* arXiv
links as the async path even when they'll actually resolve via the fast alphaXiv-overview path in
under a second is a deliberate simplification: we can't know in advance whether alphaXiv has an
overview without making the request, and erring toward "async" is always safe, never wrong. No
changes needed to `content_loader.py` itself — this reuses the exact class hierarchy already in
place.

### 2. Restructuring `/ingest` in `dsview/api.py`

The current `@api_sync_vault` decorator (`dsview/obsidian/sync_vault.py:88`) enforces a fixed
`pull_changes()` → `await function(...)` → `upload_changes(...)` sequence, which can't
conditionally defer the "await" and "push" parts based on request content. So `/ingest` drops that
decorator and inlines the same two helper calls directly, branching on the PDF check:

```python
from fastapi import BackgroundTasks
from starlette.responses import JSONResponse

from dsview.extraction.content_loader import PdfUrlLoader, get_content_loader
from dsview.obsidian.sync_vault import config as vault_config, pull_changes, upload_changes


async def _ingest_and_sync(content: InputContent, session: Session):
    await ingest_pipeline.async_ingest_content(content, session)

    if vault_config.github_vault.repository is not None:
        upload_changes("Adding content")


@app.post("/ingest")
async def ingest(
    content: InputContent, session: SessionDep, background_tasks: BackgroundTasks
):
    check_db_connection(session)

    content = InputContent.model_validate(content)
    if content.source == "None":
        content.source = None

    if vault_config.github_vault.repository is not None:
        pull_changes()

    if isinstance(get_content_loader(content.link), PdfUrlLoader):
        background_tasks.add_task(_ingest_and_sync, content, session)
        return JSONResponse(status_code=202, content={"status": "processing"})

    await ingest_pipeline.async_ingest_content(content, session)
    if vault_config.github_vault.repository is not None:
        upload_changes("Adding content")
```

Key points:
- **Non-PDF path is byte-for-byte the same behavior as today**: pull → await pipeline → push →
  implicit `200` with no body. Callers relying on synchronous completion for web/YouTube/arXiv-
  overview content see no change.
- **PDF path**: pull stays synchronous (fast; needed before the background task starts writing
  notes into the vault checkout), then the pipeline call *and* the vault push both move into the
  background task, and the endpoint returns `202 {"status": "processing"}` immediately. The push
  has to move too, not just the pipeline call — the pipeline is what writes the Obsidian notes
  (`IngestPipeline._write`), so pushing before it's done would push nothing new or race the write.
- **DB session lifetime is safe.** `SessionDep` (`dsview/api.py:35-40`) is a `yield`-based FastAPI
  dependency. FastAPI has a specific, documented guarantee for exactly this pattern: cleanup code
  after `yield` in a dependency runs *after* `BackgroundTasks` finish, not after the response is
  sent — so the same `session` object stays valid and open for the whole background task, no
  separate session needs to be opened.
- **No new failure handling needed.** `IngestPipeline.async_ingest_content`
  (`dsview/ingest_source.py`) already wraps its body in `try/except Exception` →
  `save_failed_ingestion` + `session.rollback()`/`commit()`, so a background-task failure is
  already captured to the DB rather than raising unhandled inside the background task.
- `check_db_connection(session)` and `pull_changes()` stay synchronous in both branches — both are
  fast and unrelated to the PDF-processing cost that motivated this change.
- No changes to `IngestPipeline`, `ingest_source.py`, or the CLI (`dsview ingest`) — this is purely
  an HTTP-API-layer concern tied to the nginx timeout; the CLI has no such deadline and stays fully
  synchronous as-is.

### 3. Testing

There's no existing `tests/test_api.py`. A hermetic test is possible via FastAPI's `TestClient` +
`app.dependency_overrides[get_session]`, with `ingest_pipeline.async_ingest_content`,
`pull_changes`, and `upload_changes` monkeypatched so no real DB/git/network calls happen. One
wrinkle worth flagging up front: importing `dsview.api` runs `init_vault()` at module import time,
which is a no-op if `config.vault_path` already exists, or just an empty `mkdir()` if
`github_vault.repository` is unset — no git clone happens unless `VAULT_REPOSITORY` is actually
configured in the test environment. Two cases to cover:
- A non-PDF link (e.g. a plain web URL) → `200`, and the mocked pipeline was awaited before the
  response was returned (i.e. called synchronously).
- A PDF link (e.g. `https://arxiv.org/pdf/...` or any `.pdf` URL) → `202` with
  `{"status": "processing"}`, returned before the mocked pipeline call resolves; then assert the
  mocked pipeline was actually invoked (`TestClient`, when used as a context manager, runs
  Starlette's background tasks synchronously after the request completes, so this is
  straightforward to assert on).

## Files touched

- `dsview/api.py` — restructure `/ingest` as described above; new imports
  (`BackgroundTasks`, `JSONResponse`, `PdfUrlLoader`, `get_content_loader`, and `pull_changes`/
  `upload_changes`/`config` from `sync_vault`). `/relevance` is untouched and keeps using
  `@api_sync_vault` unchanged.
- `tests/test_api.py` — new file, the two cases above.

No changes to `dsview/extraction/content_loader.py`, `dsview/ingest_source.py`,
`dsview/obsidian/sync_vault.py`, or the Kubernetes manifests (the Ingress timeout is exactly what
this change works around, not something that needs adjusting).

## Verification

1. `uvx ruff format . && uvx ruff check .` on touched files.
2. `uv run pytest tests/test_api.py -v` — new hermetic tests, no live network/DB/LLM calls.
3. Manual check: run the API locally (`uv run fastapi run dsview/api.py` or equivalent) against a
   real Postgres, POST a plain web URL to `/ingest` and confirm it blocks until done (`200`); POST
   a PDF/arXiv URL and confirm it returns `202` almost immediately while the note still shows up in
   the vault/DB a bit later once the background task finishes.
