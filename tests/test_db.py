from datetime import date

from sqlmodel import Session

from dsview.db.query import get_failed_ingestion
from dsview.db.schemas import FailedIngestion, InputContent


def _saved_content(session: Session) -> InputContent:
    content = InputContent(link="https://example.com/a", upload_date=date.today())
    session.add(content)
    session.commit()
    return content


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
