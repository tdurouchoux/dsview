from datetime import date

from dsview.db.schemas import ExtractionResult, InputContent
from dsview.notification.digest_pipeline import (
    _build_content_refs,
    _new_topic_ids,
    _select_must_read,
)

MUST_READ_LIMIT = 5
MUST_READ_POOL_MULTIPLIER = 5


def _content(link: str, read_priority: int, upload_date: date) -> InputContent:
    return InputContent(link=link, upload_date=upload_date, read_priority=read_priority)


def test_select_must_read_only_draws_from_top_priority_pool():
    pool_size = MUST_READ_LIMIT * MUST_READ_POOL_MULTIPLIER
    high_priority = [
        _content(f"https://example.com/high-{i}", 5, date(2026, 8, 24))
        for i in range(pool_size)
    ]
    low_priority = [
        _content(f"https://example.com/low-{i}", 0, date(2026, 8, 24))
        for i in range(10)
    ]

    selected = _select_must_read(
        high_priority + low_priority, MUST_READ_LIMIT, MUST_READ_POOL_MULTIPLIER
    )

    assert len(selected) == MUST_READ_LIMIT
    assert all(c.read_priority == 5 for c in selected)


def test_select_must_read_breaks_ties_with_date_before_sampling():
    pool_size = MUST_READ_LIMIT * MUST_READ_POOL_MULTIPLIER
    recent = [
        _content(f"https://example.com/recent-{i}", 3, date(2026, 8, 24))
        for i in range(pool_size)
    ]
    older = [
        _content(f"https://example.com/older-{i}", 3, date(2026, 1, 1))
        for i in range(10)
    ]

    selected = _select_must_read(
        recent + older, MUST_READ_LIMIT, MUST_READ_POOL_MULTIPLIER
    )

    assert all(c.upload_date == date(2026, 8, 24) for c in selected)


def test_select_must_read_returns_all_candidates_when_fewer_than_limit():
    candidates = [_content("https://example.com/a", 1, date(2026, 8, 24))]

    assert (
        _select_must_read(candidates, MUST_READ_LIMIT, MUST_READ_POOL_MULTIPLIER)
        == candidates
    )


def test_new_topic_ids_excludes_topics_created_outside_the_week():
    topic_min_dates = {
        1: date(2026, 8, 25),  # inside the week
        2: date(2026, 1, 1),  # long-established topic
        3: date(2026, 9, 5),  # after the week (shouldn't happen, but be safe)
    }

    result = _new_topic_ids(
        topic_min_dates.keys(), topic_min_dates, date(2026, 8, 24), date(2026, 8, 30)
    )

    assert result == [1]


def test_new_topic_ids_orders_most_recently_created_first():
    topic_min_dates = {1: date(2026, 8, 24), 2: date(2026, 8, 27), 3: date(2026, 8, 30)}

    result = _new_topic_ids(
        topic_min_dates.keys(), topic_min_dates, date(2026, 8, 24), date(2026, 8, 30)
    )

    assert result == [3, 2, 1]


def test_build_content_refs_skips_content_with_failed_extraction(session):
    extracted = InputContent(
        link="https://example.com/ok", upload_date=date(2026, 8, 26)
    )
    failed = InputContent(
        link="https://example.com/failed", upload_date=date(2026, 8, 27)
    )
    session.add(extracted)
    session.add(failed)
    session.commit()
    session.add(
        ExtractionResult(
            content_id=extracted.id,
            title="ok",
            content_type="Article",
            summary="summary",
            embedding=None,
        )
    )
    session.commit()

    refs = _build_content_refs([extracted, failed], session)

    assert [ref.title for ref in refs] == ["ok"]
