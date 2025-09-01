from sqlmodel import Session, select

from ..schemas import FailedIngestion, InputContent


def get_content(
    link: str,
    session: Session,
) -> InputContent:
    return session.exec(select(InputContent).where(InputContent.link == link)).first()


def get_content_list(
    session: Session,
    start_id: int = 1,
    end_id: int = None,
) -> list[InputContent]:
    statement = select(InputContent).where(InputContent.id >= start_id)

    if end_id is not None:
        statement = statement.where(InputContent.id <= end_id)

    statement = statement.order_by(InputContent.id.asc())

    return session.exec(statement).all()


def get_failed_ingestions(
    session: Session,
    ignore_errors: list[str] = None,
) -> list[InputContent]:
    failed_content_stmt = select(FailedIngestion.content_id)

    if ignore_errors is not None:
        failed_content_stmt = failed_content_stmt.where(FailedIngestion.error_type.not_in(ignore_errors))

    content_list = session.exec(
        select(InputContent).where(
            InputContent.id.in_(failed_content_stmt)
        )
    ).all()

    return content_list


# class ContentIndex(DuckDBIndex)
