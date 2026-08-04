from datetime import date

import pytest
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from dsview.db.query import get_failed_ingestion
from dsview.db.schemas import FailedIngestion, InputContent


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def attach_content_schema(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("ATTACH DATABASE ':memory:' AS content")
        cursor.close()

    SQLModel.metadata.create_all(
        engine, tables=[InputContent.__table__, FailedIngestion.__table__]
    )

    with Session(engine) as session:
        yield session


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
