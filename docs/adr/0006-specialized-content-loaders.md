---
status: "accepted"
date: "2026-08-03"
decision-makers: "tdurouchoux"
---

# Specialized content loaders for arXiv, GitHub, and YouTube links

## Context and Problem Statement

ADR-0003 established that a blanket, generic improvement to web-page loading (Jina Reader markdown
extraction for every URL) measurably hurt topic extraction and wasn't worth adopting — the existing
`requests` + `BeautifulSoup.get_text()` `UrlLoader` remains the default for arbitrary web pages.
That verdict doesn't generalize to three specific, extremely common link types in this project's
actual usage, where the generic scrape isn't merely "less clean" but structurally wrong for the
content:

* **YouTube** pages are a JS-driven single-page app; a plain GET returns page chrome and, at best,
  some metadata embedded in a large inline JSON blob — never the video's actual substance (what's
  said). There is no way to make `BeautifulSoup` output useful here; the real content lives in the
  caption track, not the HTML.
* **arXiv abstract pages** contain only the abstract — the substantive content is the paper body,
  reachable only via the PDF (expensive: docling conversion + token-budget-aware chunking, see the
  existing `PdfUrlLoader`) or a cleaner third-party markdown rendering.
* **GitHub repository pages** wrap the README in a large amount of navigation/UI chrome; the actual
  documentation is available directly and far more cheaply via GitHub's own REST API, with no
  rendering noise to strip.

These three link types are frequent enough in this project's real usage (a personal DS
knowledge/reading-list tool) to justify a bespoke integration per platform, unlike ADR-0003's
rejected blanket swap across every web page.

## Considered Options

* Add a specialized `ContentLoader` subclass per platform — YouTube, arXiv, GitHub (chosen)
* Keep the generic scrape (`UrlLoader`) for these link types too

## Decision Outcome

Chosen option: "Add one `ContentLoader` subclass per platform — `YoutubeContentLoader`,
`ArxivContentLoader`, `GithubContentLoader`", because the generic scrape is structurally wrong or
near-empty for all three platforms (see Context above), unlike ADR-0003's rejected blanket swap
which only made otherwise-usable content marginally cleaner.

Each subclass uses a data source purpose-built for that platform instead of scraping the rendered
page. Dispatch happens in `get_content_loader()` (`dsview/extraction/content_loader.py`), purely on
`link.host`/`link.path` — no I/O — before falling through to the `.pdf`-suffix check and finally
the generic `UrlLoader`. This is the same low-cost extension point already used for the
PDF-vs-generic split.

Two different patterns emerged for handling a specialized source that might not be available for a
given link:

* **Specialized source with fallback** (`ArxivContentLoader`, `GithubContentLoader`): a private
  helper (`_load_alphaxiv_overview` / `_load_readme`) tries the clean source, does a minimal
  quality check (alphaXiv: reject sub-200-character/garbage responses; GitHub: reject blank or
  undecodable API responses), sets `self.content` and returns `True` on success, or logs and
  returns `False` on failure — distinguishing an expected 404 (the source doesn't exist for this
  link, logged at info) from a genuine error (logged as a warning). `_load_content` tries the
  helper first and falls back to the generic path on `False`: full PDF extraction for arXiv
  (`PdfUrlLoader`, hence `ArxivContentLoader(PdfUrlLoader)`), or the plain webpage scrape for GitHub
  (`UrlLoader`, hence `GithubContentLoader(UrlLoader)`). A recognized link of these types therefore
  never hard-fails — worst case, it degrades to what any other link of that base type would get.
* **Specialized source with no fallback** (`YoutubeContentLoader`): if a transcript genuinely isn't
  obtainable (`TranscriptsDisabled`, `VideoUnavailable`, or no transcript listed at all),
  `_load_content` raises `YoutubeTranscriptUnavailable` rather than falling back to scraping the
  page. There's no meaningful fallback here — the rendered YouTube page contains no substitute
  content worth extracting, so degrading to `UrlLoader` would produce a mostly-empty result rather
  than a legitimately-worse-but-usable one. The caller (`IngestPipeline.async_ingest_content`)
  already catches and records any content-loading exception via `save_failed_ingestion` (see
  ADR-0004's context), so this surfaces as a normal failed ingestion, not a special case.

`GithubContentLoader` specifically subclasses `UrlLoader` (not a new base) for two reasons: it
reuses the fallback's BeautifulSoup scrape, and — independently —
`ContentExtractor.run_extraction` only runs `LinksExtractor` when
`isinstance(content_loader, UrlLoader)` (`content_extraction.py`), so this is also what makes the
README's own links eligible for extraction at all. Its `content_links` are the README's absolute
(non-relative) markdown links, extracted via `markdown_it`'s token stream rather than regex over
the raw text. No further curation happens in the loader — `LinksExtractor` is itself an LLM task
that judges relevance from the full candidate list, so the loader only needs to supply a reasonable
pool, not a perfectly filtered one.

### Confirmation

Tests in `tests/test_content_loader.py` largely hit live network/APIs rather than mocking — a
deliberate exception to this project's default hermetic-testing rule (see `CLAUDE.md`), since
dispatch and real content shape are exactly what's being verified. Monkeypatching is used only to
force a specific fallback branch deterministically (e.g.
`monkeypatch.setattr(GithubContentLoader, "_load_readme", lambda self: False)`).

### Consequences

* Good, because adding another slow- or awkward-to-scrape platform in the future only needs: a new
  `ContentLoader` subclass, a `get_content_loader` dispatch branch keyed on host/path (no I/O), and
  a choice between the two patterns above based on whether the base type's fallback (PDF, generic
  scrape) is actually usable for that platform.
* Neutral, because arXiv/GitHub links are resilient to their specialized source being temporarily
  or permanently unavailable (rate-limited, deleted README, alphaXiv down), while YouTube links are
  not — a transcript-less video is a real, expected `failed` ingestion, not a bug to fix.
* Bad, because `content_extraction.py`'s `isinstance(content_loader, UrlLoader)` check for link
  extraction is a live coupling to keep in mind: any future specialized loader that should feed
  `LinksExtractor` needs to subclass `UrlLoader` for that reason alone, not just for its fallback
  behavior.
* Neutral, because unlike ADR-0003's rejected generic swap, no eval-harness comparison was run for
  these three loaders — the case for each is architectural (the generic scrape is unusable or
  near-empty for that link shape) rather than a measured topic-extraction quality delta, so there's
  no equivalent cost/benefit table here.
