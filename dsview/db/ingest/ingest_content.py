from sqlmodel import Session

from ..query import get_content
from ..schemas import FailedIngestion, InputContent
from .update import update_instance


class ContentAlreadyExists(Exception):
    def __init__(self, link: str):
        super().__init__(f"Content with link {link} already exists")


def save_content(
    content: InputContent,
    session: Session,
) -> InputContent:
    existing_content = get_content(content.link, session)

    if existing_content is not None:
        raise ContentAlreadyExists(content.link)

    session.add(content)

    return content


def save_failed_ingestion(
    content: InputContent,
    original_link: str,
    error: Exception,
    session: Session,
):
    failed_ingestion = FailedIngestion(
        content_id=content.id,
        original_link=original_link,
        error_type=error.__class__.__name__,
        error_message=str(error),
    )
    session.add(failed_ingestion)


def update_content(
    session: Session,
    content_id: int | None = None,
    content_link: str | None = None,
    already_read: bool | None = None,
    read_priority: int | None = None,
    relevance: int | None = None,
) -> InputContent:
    if all((already_read is None, read_priority is None, relevance is None)):
        raise ValueError("No value to update")

    update_attributes = {
        "already_read": already_read,
        "read_priority": read_priority,
        "relevance": relevance,
    }

    return update_instance(
        session,
        InputContent,
        row_id=content_id,
        filter_attributes={"link": content_link},
        **{
            attr: value
            for attr, value in update_attributes.items()
            if value is not None
        },
    )
