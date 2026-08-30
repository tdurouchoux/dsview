from datetime import UTC, date, datetime

from sqlmodel import Session

from dsview.db.query import (
    get_content_by_date_range,
    get_failed_ingestion,
    get_failed_ingestions,
    get_filtered_content,
    get_resurfaced_content,
    get_topic_min_upload_dates,
)
from dsview.db.schemas import (
    ContentTopicRelation,
    ExtractionResult,
    ExtractionTopic,
    FailedIngestion,
    InputContent,
)


def _saved_content(
    session: Session,
    link: str = "https://example.com/a",
    upload_date: date | None = None,
    relevance: int = 0,
    already_read: bool = False,
) -> InputContent:
    content = InputContent(
        link=link,
        upload_date=upload_date or datetime.now(UTC).date(),
        relevance=relevance,
        already_read=already_read,
    )
    session.add(content)
    session.commit()
    return content


def _saved_extraction(
    content_id: int, topic: ExtractionTopic, session: Session
) -> None:
    session.add(
        ExtractionResult(
            content_id=content_id,
            title=f"content {content_id}",
            content_type="Article",
            summary="summary",
            embedding=None,
        )
    )
    session.add(ContentTopicRelation(content_id=content_id, topic_id=topic.id))
    session.commit()


def _link_topic(content_id: int, topic: ExtractionTopic, session: Session) -> None:
    session.add(ContentTopicRelation(content_id=content_id, topic_id=topic.id))
    session.commit()


def test_get_failed_ingestion_returns_none_when_no_failure_recorded(session):
    content = _saved_content(session)

    assert get_failed_ingestion(content.id, session) is None


def test_get_failed_ingestion_returns_matching_failure(session):
    content = _saved_content(session)
    session.add(
        FailedIngestion(
            content_id=content.id,
            original_link="https://example.com/a",
            error_type="ValueError",
            error_message="boom",
        )
    )
    session.commit()

    failure = get_failed_ingestion(content.id, session)

    assert failure.error_type == "ValueError"
    assert failure.error_message == "boom"


def test_get_failed_ingestions_filters_by_date_range(session):
    in_range = _saved_content(
        session, "https://example.com/in-range", date(2026, 8, 20)
    )
    session.add(
        FailedIngestion(
            content_id=in_range.id,
            original_link="https://example.com/in-range",
            error_type="ValueError",
            error_message="boom",
        )
    )
    before_range = _saved_content(
        session, "https://example.com/before", date(2026, 8, 1)
    )
    session.add(
        FailedIngestion(
            content_id=before_range.id,
            original_link="https://example.com/before",
            error_type="ValueError",
            error_message="boom",
        )
    )
    session.commit()

    results = get_failed_ingestions(
        session, start_date=date(2026, 8, 15), end_date=date(2026, 8, 22)
    )

    assert [content.id for content in results] == [in_range.id]


def test_get_content_by_date_range_excludes_content_outside_the_window(session):
    in_range = _saved_content(
        session, "https://example.com/in-range", date(2026, 8, 20)
    )
    _saved_content(session, "https://example.com/before", date(2026, 8, 1))
    _saved_content(session, "https://example.com/after", date(2026, 9, 1))

    results = get_content_by_date_range(date(2026, 8, 15), date(2026, 8, 22), session)

    assert [content.id for content in results] == [in_range.id]


def test_get_resurfaced_content_finds_older_content_sharing_a_topic(session):
    topic = ExtractionTopic(
        type="Library", name="pandas", description="...", embedding=None
    )
    session.add(topic)
    session.commit()

    old_content = _saved_content(session, "https://example.com/old", date(2026, 1, 1))
    _saved_extraction(old_content.id, topic, session)

    unrelated_old_content = _saved_content(
        session, "https://example.com/unrelated", date(2026, 1, 2)
    )

    results = get_resurfaced_content([topic.id], date(2026, 8, 1), session)

    assert [content.id for content in results] == [old_content.id]
    assert unrelated_old_content.id not in [content.id for content in results]


def test_get_resurfaced_content_excludes_content_after_the_cutoff(session):
    topic = ExtractionTopic(
        type="Library", name="pandas", description="...", embedding=None
    )
    session.add(topic)
    session.commit()

    recent_content = _saved_content(
        session, "https://example.com/recent", date(2026, 8, 20)
    )
    _saved_extraction(recent_content.id, topic, session)

    results = get_resurfaced_content([topic.id], date(2026, 8, 15), session)

    assert results == []


def test_get_resurfaced_content_returns_empty_list_with_no_topic_ids(session):
    assert get_resurfaced_content([], date(2026, 8, 15), session) == []


def test_get_resurfaced_content_ranks_by_relevance_not_recency(session):
    topic = ExtractionTopic(
        type="Library", name="pandas", description="...", embedding=None
    )
    session.add(topic)
    session.commit()

    most_relevant = _saved_content(
        session, "https://example.com/most-relevant", date(2026, 1, 1), relevance=5
    )
    _saved_extraction(most_relevant.id, topic, session)

    most_recent_but_less_relevant = _saved_content(
        session, "https://example.com/most-recent", date(2026, 7, 1), relevance=1
    )
    _saved_extraction(most_recent_but_less_relevant.id, topic, session)

    results = get_resurfaced_content([topic.id], date(2026, 8, 1), session, limit=1)

    assert [content.id for content in results] == [most_relevant.id]


def test_get_resurfaced_content_ranks_by_paths_times_relevance(session):
    topic_a = ExtractionTopic(
        type="Library", name="pandas", description="...", embedding=None
    )
    topic_b = ExtractionTopic(
        type="Library", name="numpy", description="...", embedding=None
    )
    session.add(topic_a)
    session.add(topic_b)
    session.commit()

    # relevance 3, 1 shared topic -> score 3
    single_path_high_relevance = _saved_content(
        session, "https://example.com/single-path", date(2026, 1, 1), relevance=3
    )
    _saved_extraction(single_path_high_relevance.id, topic_a, session)

    # relevance 2, 2 shared topics -> score 4, should outrank the above
    double_path_lower_relevance = _saved_content(
        session, "https://example.com/double-path", date(2026, 1, 2), relevance=2
    )
    _saved_extraction(double_path_lower_relevance.id, topic_a, session)
    _link_topic(double_path_lower_relevance.id, topic_b, session)

    results = get_resurfaced_content(
        [topic_a.id, topic_b.id], date(2026, 8, 1), session
    )

    assert [content.id for content in results] == [
        double_path_lower_relevance.id,
        single_path_high_relevance.id,
    ]


def test_get_topic_min_upload_dates_returns_earliest_connected_content_date(session):
    new_topic = ExtractionTopic(
        type="Library", name="polars", description="...", embedding=None
    )
    established_topic = ExtractionTopic(
        type="Library", name="pandas", description="...", embedding=None
    )
    session.add(new_topic)
    session.add(established_topic)
    session.commit()

    only_content = _saved_content(
        session, "https://example.com/only", date(2026, 8, 24)
    )
    _saved_extraction(only_content.id, new_topic, session)

    older_content = _saved_content(
        session, "https://example.com/older", date(2026, 1, 1)
    )
    _saved_extraction(older_content.id, established_topic, session)
    newer_content = _saved_content(
        session, "https://example.com/newer", date(2026, 8, 24)
    )
    _saved_extraction(newer_content.id, established_topic, session)

    min_dates = get_topic_min_upload_dates(
        [new_topic.id, established_topic.id], session
    )

    assert min_dates == {
        new_topic.id: date(2026, 8, 24),
        established_topic.id: date(2026, 1, 1),
    }


def test_get_topic_min_upload_dates_returns_empty_dict_with_no_topic_ids(session):
    assert get_topic_min_upload_dates([], session) == {}


def test_get_filtered_content_already_read_filters_correctly(session):
    unread = _saved_content(session, "https://example.com/unread", already_read=False)
    _saved_content(session, "https://example.com/read", already_read=True)

    results = get_filtered_content(session, already_read=False, limit=None)

    assert [content.id for content in results] == [unread.id]


def test_get_filtered_content_limit_none_returns_all_matches(session):
    for i in range(25):
        _saved_content(session, f"https://example.com/{i}")

    results = get_filtered_content(session, limit=None)

    assert len(results) == 25
