from datetime import date
from typing import Literal

from sqlalchemy import func
from sqlmodel import Session, select

from ..schemas import (
    ContentTopicRelation,
    FailedIngestion,
    InputContent,
)


def get_content(
    link: str,
    session: Session,
) -> InputContent:
    return session.exec(select(InputContent).where(InputContent.link == link)).first()


def get_content_by_id(
    content_id: int,
    session: Session,
) -> InputContent | None:
    return session.get(InputContent, content_id)


def get_content_list(
    session: Session,
    start_id: int = 1,
    end_id: int | None = None,
) -> list[InputContent]:
    statement = select(InputContent).where(InputContent.id >= start_id)

    if end_id is not None:
        statement = statement.where(InputContent.id <= end_id)

    statement = statement.order_by(InputContent.id.asc())

    return session.exec(statement).all()


def get_failed_ingestions(
    session: Session,
    ignore_errors: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[InputContent]:
    failed_content_stmt = select(FailedIngestion.content_id)

    if ignore_errors is not None:
        failed_content_stmt = failed_content_stmt.where(
            FailedIngestion.error_type.not_in(ignore_errors)
        )

    content_stmt = select(InputContent).where(InputContent.id.in_(failed_content_stmt))

    if start_date is not None:
        content_stmt = content_stmt.where(InputContent.upload_date >= start_date)

    if end_date is not None:
        content_stmt = content_stmt.where(InputContent.upload_date <= end_date)

    return session.exec(content_stmt).all()


def get_failed_ingestion(
    content_id: int,
    session: Session,
) -> FailedIngestion | None:
    return session.exec(
        select(FailedIngestion).where(FailedIngestion.content_id == content_id)
    ).first()


def get_filtered_content(
    session: Session,
    already_read: bool | None = None,
    read_priority: int | None = None,
    relevance: int | None = None,
    source: str | None = None,
    date_ordering: Literal["asc", "desc"] | None = "desc",
    limit: int | None = 20,
) -> list[InputContent]:

    stmt = select(InputContent)

    if already_read is not None:
        stmt = stmt.where(InputContent.already_read == already_read)

    if read_priority is not None:
        stmt = stmt.where(InputContent.read_priority == read_priority)

    if relevance is not None:
        stmt = stmt.where(InputContent.relevance == relevance)

    if source is not None:
        stmt = stmt.where(InputContent.source == source)

    if date_ordering == "asc":
        stmt = stmt.order_by(InputContent.upload_date.asc())
    else:
        stmt = stmt.order_by(InputContent.upload_date.desc())

    if limit is not None:
        stmt = stmt.limit(limit)

    return session.exec(stmt).all()


def get_content_by_date_range(
    start_date: date,
    end_date: date,
    session: Session,
) -> list[InputContent]:
    statement = (
        select(InputContent)
        .where(InputContent.upload_date >= start_date)
        .where(InputContent.upload_date <= end_date)
        .order_by(InputContent.upload_date.asc())
    )

    return session.exec(statement).all()


def get_resurfaced_content(
    topic_ids: list[int],
    before_date: date,
    session: Session,
    limit: int = 3,
) -> list[InputContent]:
    """Older content (before `before_date`) sharing one of `topic_ids`.

    Surfaces past reads the week's new content ties back into, for the digest's
    "worth revisiting" section - the top `limit` by
    `(number of shared topics) * relevance`, so a content tied back into several
    of this week's topics outranks one that only shares a single, equally
    relevant topic.
    """
    if not topic_ids:
        return []

    path_count = func.count(ContentTopicRelation.topic_id)
    score = path_count * InputContent.relevance

    statement = (
        select(InputContent)
        .join(
            ContentTopicRelation,
            ContentTopicRelation.content_id == InputContent.id,
        )
        .where(ContentTopicRelation.topic_id.in_(topic_ids))
        .where(InputContent.upload_date < before_date)
        .group_by(InputContent.id)
        .order_by(score.desc())
        .limit(limit)
    )

    return session.exec(statement).all()


# class ContentIndex(DuckDBIndex)
