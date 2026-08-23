from typing import Literal

from sqlmodel import Session, select

from ..schemas import FailedIngestion, InputContent


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
) -> list[InputContent]:
    failed_content_stmt = select(FailedIngestion.content_id)

    if ignore_errors is not None:
        failed_content_stmt = failed_content_stmt.where(
            FailedIngestion.error_type.not_in(ignore_errors)
        )

    content_list = session.exec(
        select(InputContent).where(InputContent.id.in_(failed_content_stmt))
    ).all()

    return content_list


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
    limit: int = 20,
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

    stmt = stmt.limit(limit)

    return session.exec(stmt).all()


# class ContentIndex(DuckDBIndex)
