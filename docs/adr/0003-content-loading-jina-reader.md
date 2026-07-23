# ADR 0003: Content loading — Jina Reader for generic web pages

## Status

Rejected

## Context

`dsview/extraction/content_loader.py` loads generic web pages with `requests` +
`BeautifulSoup.get_text()`: flat text that still carries nav bars, footers and cookie banners,
with no structure. The hypothesis was that a markdown extraction service
([Jina AI Reader](https://jina.ai/reader/)) would give the extraction pipeline cleaner, better
structured input, and that this would show up as better topic extraction — the most crucial task
in the project.

Content loading is not currently a parameter of the evaluation: `evaluate()` scores the `content`
persisted in `labels.labelledcontent`, captured once at label-ingest time. Testing it therefore
needed a harness that re-fetches the labelled URLs, built as
`notebooks/jina_content_loader_topics.py`. PDF rows kept the existing `PdfUrlLoader` path so only
the HTML loader changed.

The labels themselves are loader-independent: the Streamlit form never displays loaded content,
so the labeller works from the live website. Ground truth is not biased toward either loader.

## Options considered

All figures are the labelled `eval` split (25 rows), `mistral-small-2603` + the Exp 2 prompt
(ADR 0002). Repeat runs at temperature 0 differ by ±0.011–0.022 precision and up to ±10 correct
topics, so treat smaller differences as noise.

| Loader | Prompt | Temp | Precision | Recall | AP | Correct topics |
|---|---|---|---|---|---|---|
| **BeautifulSoup (production)** | Exp 2 | 0 | **0.716 / 0.704** | 0.665 / 0.637 | 0.662 / 0.641 | 132 / 122 |
| Jina Reader | Exp 2 | unpinned | 0.662 / 0.664 | 0.627 / 0.622 | 0.609 / 0.598 | 122 / 121 |
| Jina Reader | selective (`eval_configs/topics_jina_selective.yaml`) | 0 | 0.620 | 0.519 | 0.549 | 102 |

Key findings:

- **Jina costs ~0.04–0.05 precision and buys nothing.** Both loaders find the same number of
  correct topics (122), but Jina's richer markdown draws 16 more predictions (7.16 → 7.80 per row)
  with zero additional hits — a mechanical precision loss. The gap clears the noise floor.
- **Per-row results churn rather than improve**: 8 rows found fewer correct topics, 7 found more,
  10 were unchanged. There is no subset of content where Jina clearly wins.
- **The over-enumeration is not a fixable prompt artifact.** The Exp 2 prompt asks the model to
  capture *all* named entities up to a cap of 10, so richer input pushes output toward the cap. A
  variant that keeps the exhaustive entity scan but adds a selection pass filtering candidates
  against the whole content did reduce output (7.80 → 6.6 topics/row) — and made everything worse
  (precision 0.620, 102 correct). The filter removed proportionally more correct topics than
  incorrect ones.
- The extra topics Jina elicits are real named entities (`loguru`, `MinIO`, `Prometheus`,
  `Grok 4`), not hallucinations. The labeller saw the same live page, used 7 of 10 available slots
  on average, and chose not to include them — they are genuinely marginal.

## Decision

Keep the existing `BeautifulSoup` loader. Do not adopt Jina Reader.

The measured effect on topic extraction is negative, and the change would add an external API
dependency, a third API key, rate limits and per-page latency to every ingestion.

## Consequences

- Content loading stays out of the evaluation as a parameter. The harness notebook is kept as the
  way to re-test it if a future loader is proposed.
- `eval_configs/topics_jina_selective*.{yaml,txt}` are kept as the record of the selectivity
  experiment. The result is a useful negative: on this task, telling the model to be more selective
  loses more good topics than bad ones.
- Untested: whether markdown structure helps the *other* extraction tasks (links, description),
  where document structure plausibly matters more than it does for topic naming.
