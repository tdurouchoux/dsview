"""Compute the digest week, fetch its content/topics, and build the digest email.

Orchestrates `dsview.db.query` + `DigestSummaryGenerator` into the sequences
`dsview.notification.digest.render_digest` expects, and returns the `(subject,
html)` pair. Sending the result is the caller's decision, not this module's.
"""

import logging
import random
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

import logfire
from sqlmodel import Session

from dsview.config import notification_config
from dsview.db.query import (
    get_content_by_date_range,
    get_content_linked_topics,
    get_failed_ingestions,
    get_filtered_content,
    get_resurfaced_content,
    get_topic_min_upload_dates,
)
from dsview.db.schemas import ExtractionTopic, InputContent
from dsview.db.schemas.extraction_schema import ExtractionResult
from dsview.notification.digest import (
    ContentRef,
    Item,
    Stats,
    Topic,
    digest_subject,
    render_digest,
    vault_content_url,
    vault_topic_url,
)
from dsview.notification.digest_summary import DigestSummaryGenerator

logger = logging.getLogger(__name__)


def _digest_item(
    content: InputContent,
    extraction: ExtractionResult,
    summary_generator: DigestSummaryGenerator,
    why: str | None = None,
) -> Item:
    digest_summary = summary_generator.predict({"content": extraction.summary})

    return Item(
        content_type=extraction.content_type,
        title=extraction.title,
        source_url=str(content.link),
        upload_date=content.upload_date,
        note_url=vault_content_url(extraction.title, extraction.content_type),
        meta=extraction.tags[0].name if extraction.tags else None,
        gist=digest_summary.one_liner,
        bullets=digest_summary.bullets,
        why=why,
    )


def _with_extractions(
    content_list: list[InputContent], session: Session
) -> list[ExtractionResult]:
    """Completed extractions for `content_list`, skipping content not yet extracted."""
    extractions = (
        session.get(ExtractionResult, content.id) for content in content_list
    )
    return [extraction for extraction in extractions if extraction is not None]


def _extraction_of(content: InputContent, session: Session) -> ExtractionResult:
    """The extraction for `content`, which callers must only pass content already
    known to be extracted - either filtered through `_with_extractions`, or
    fetched via `get_resurfaced_content`, whose join guarantees an extraction."""
    extraction = session.get(ExtractionResult, content.id)
    assert extraction is not None, f"content {content.id} has no extraction"
    return extraction


def _select_must_read(
    candidates: list[InputContent], limit: int, pool_multiplier: int
) -> list[InputContent]:
    """Top `limit * pool_multiplier` by (read_priority, date), then a random
    sample of `limit` from that pool - keeps the digest from always leading
    with the same handful of high-priority items.

    `candidates` is meant to be unread content across the whole vault, not just
    this week's additions: must-reads resurface a backlog, they aren't tied to
    what just got ingested.
    """
    pool = sorted(
        candidates, key=lambda c: (c.read_priority, c.upload_date), reverse=True
    )[: limit * pool_multiplier]
    return random.sample(pool, k=min(limit, len(pool)))


def _new_topic_ids(
    topic_ids: Iterable[int],
    topic_min_dates: dict[int, date],
    week_start: date,
    week_end: date,
) -> list[int]:
    """Topic ids whose earliest connected content falls in `[week_start, week_end]`,
    most recently created first."""
    return sorted(
        (
            topic_id
            for topic_id in topic_ids
            if week_start <= topic_min_dates[topic_id] <= week_end
        ),
        key=lambda topic_id: topic_min_dates[topic_id],
        reverse=True,
    )


def _fetch_new_content(
    week_start: date, week_end: date, session: Session
) -> list[InputContent]:
    """All content ingested this week, whether or not its extraction succeeded -
    content with a failed extraction still counts as new, it's just excluded
    from the rendered "New contents" section (see `_build_content_refs`)."""
    return get_content_by_date_range(week_start, week_end, session)


def _build_content_refs(
    new_input_content: list[InputContent], session: Session
) -> list[ContentRef]:
    """`ContentRef` list for the "New contents" section - content whose
    extraction failed has no title or note to link to, so it's skipped."""
    return [
        ContentRef(title=e.title, note_url=vault_content_url(e.title, e.content_type))
        for e in _with_extractions(new_input_content, session)
    ]


def _select_must_read_content(session: Session) -> list[InputContent]:
    """Selected must-reads, drawn from unread content across the whole vault -
    not just this week's additions, since must-reads resurface a backlog."""
    candidates = get_filtered_content(session, already_read=False, limit=None)
    extracted_candidates = [
        c for c in candidates if session.get(ExtractionResult, c.id) is not None
    ]
    return _select_must_read(
        extracted_candidates,
        notification_config.must_read_limit,
        notification_config.must_read_pool_multiplier,
    )


def _fetch_week_topics(
    new_input_content: list[InputContent], session: Session
) -> dict[int, ExtractionTopic]:
    """Topics linked to any of this week's content, keyed by id - content with
    a failed extraction simply contributes none."""
    return {
        topic.id: topic
        for content in new_input_content
        for topic in get_content_linked_topics(content.id, session)
    }


def _fetch_new_topics(
    new_input_content: list[InputContent],
    week_start: date,
    week_end: date,
    session: Session,
) -> dict[int, ExtractionTopic]:
    """The topics linked to this week's content that are brand new this week,
    keyed by id in most-recently-created-first order.

    A topic's "creation date" is the earliest upload date among all its
    connected contents - it's new this week only if that date falls in
    `[week_start, week_end]`, i.e. it wasn't already tied to older content.
    """
    week_topics_by_id = _fetch_week_topics(new_input_content, session)
    topic_min_dates = get_topic_min_upload_dates(list(week_topics_by_id), session)
    new_topic_ids = _new_topic_ids(
        week_topics_by_id, topic_min_dates, week_start, week_end
    )
    return {topic_id: week_topics_by_id[topic_id] for topic_id in new_topic_ids}


def _build_topic_refs(new_topics_by_id: dict[int, ExtractionTopic]) -> list[Topic]:
    """Render-ready `Topic` list for the "New topics" section, capped for display."""
    return [
        Topic(
            name=topic.name,
            type=topic.type,
            url=vault_topic_url(topic.name, topic.type),
        )
        for topic in list(new_topics_by_id.values())[
            : notification_config.new_topics_limit
        ]
    ]


def _fetch_resurfaced_content(
    new_input_content: list[InputContent], week_start: date, session: Session
) -> list[tuple[InputContent, set[str]]]:
    """Older content sharing a topic with this week's content, paired with the
    specific topic names each one shares - "worth revisiting" candidates plus
    the context needed to explain why."""
    week_topics_by_id = _fetch_week_topics(new_input_content, session)
    topic_names = {topic.name for topic in week_topics_by_id.values()}

    resurfaced_content = get_resurfaced_content(
        list(week_topics_by_id),
        week_start,
        session,
        limit=notification_config.resurfaced_limit,
    )
    return [
        (
            content,
            {t.name for t in get_content_linked_topics(content.id, session)}
            & topic_names,
        )
        for content in resurfaced_content
    ]


def _build_resurfaced_items(
    resurfaced_content: list[tuple[InputContent, set[str]]],
    summary_generator: DigestSummaryGenerator,
    session: Session,
) -> list[Item]:
    items = []
    for content, shared_topics in resurfaced_content:
        extraction = _extraction_of(content, session)
        why = f"Relevance {content.relevance}/5"
        if shared_topics:
            why += f" · ties back to {next(iter(shared_topics))}"
        items.append(_digest_item(content, extraction, summary_generator, why))

    return items


def _build_stats(
    new_input_content: list[InputContent],
    week_start: date,
    week_end: date,
    new_topic_count: int,
    session: Session,
) -> Stats:
    window_length = week_end - week_start
    previous_week_end = week_start - timedelta(days=1)
    previous_week_start = previous_week_end - window_length
    previous_input_content = get_content_by_date_range(
        previous_week_start, previous_week_end, session
    )

    return Stats(
        contents=len(new_input_content),
        contents_delta=len(new_input_content) - len(previous_input_content),
        new_topics=new_topic_count,
        sources=len({c.source for c in new_input_content if c.source}),
        failed=len(
            get_failed_ingestions(session, start_date=week_start, end_date=week_end)
        ),
    )


def build_digest(session: Session) -> tuple[str, str]:
    """Fetch last week's Monday-Sunday content and return `(subject, html)`."""
    today = datetime.now(UTC).date()
    this_monday = today - timedelta(days=today.weekday())
    week_start = this_monday - timedelta(days=7)  # last week's Monday
    week_end = this_monday - timedelta(days=1)  # last week's Sunday

    summary_generator = DigestSummaryGenerator()

    with logfire.span("Fetching digest content"):
        new_input_content = _fetch_new_content(week_start, week_end, session)
        new_content_refs = _build_content_refs(new_input_content, session)

        new_topics_by_id = _fetch_new_topics(
            new_input_content, week_start, week_end, session
        )
        new_topics = _build_topic_refs(new_topics_by_id)
        stats = _build_stats(
            new_input_content, week_start, week_end, len(new_topics_by_id), session
        )

    with logfire.span("Building digest items"):
        must_read_content = _select_must_read_content(session)
        resurfaced_content = _fetch_resurfaced_content(
            new_input_content, week_start, session
        )

        must_read_items = [
            _digest_item(c, _extraction_of(c, session), summary_generator)
            for c in must_read_content
        ]
        resurfaced_items = _build_resurfaced_items(
            resurfaced_content, summary_generator, session
        )

    logfire.info(
        f"Weekly digest: {len(new_input_content)} new contents "
        f"({len(new_content_refs)} extracted), "
        f"{len(must_read_items)} must-read, {len(resurfaced_items)} resurfaced"
    )

    with logfire.span("Rendering digest email"):
        html = render_digest(
            new_content_refs,
            must_read_items,
            resurfaced_items,
            week_start,
            week_end,
            stats=stats,
            new_topics=new_topics,
        )

    return digest_subject(week_start), html
