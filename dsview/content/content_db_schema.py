from datetime import date
from pathlib import Path
from typing import Dict

from pydantic import HttpUrl
from sqlalchemy.types import String, TypeDecorator
from sqlmodel import Field, Session, SQLModel, select

from dsview.config import load_extraction_config

# TODO Move Medium redirect to WebLoader

config = load_extraction_config()


class LinkType(TypeDecorator):
    impl = String(2083)

    def process_bind_param(self, value, dialect) -> str:
        return str(value)

    def process_result_value(self, value, dialect) -> HttpUrl:
        if Path(value).exists():
            return Path(value)
        return HttpUrl(value)

    def process_literal_param(self, value, dialect) -> str:
        return str(value)


class InputContent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)  # Add an ID as primary key
    link: HttpUrl | Path = Field(unique=True, sa_type=LinkType)
    upload_date: date
    already_read: bool = Field(default=False)
    read_priority: int = Field(default=0, ge=0, le=5)
    relevance: int = Field(default=0, ge=0, le=5)
    source: str | None = Field(default=None)

    def get_str_dict(self) -> Dict:
        instance_dict = self.model_dump()
        instance_dict["link"] = str(self.link)
        instance_dict["upload_date"] = self.upload_date.isoformat()

        if self.link is None:
            del instance_dict["source"]
        if "id" in instance_dict:
            del instance_dict["id"]

        return instance_dict


def update_content(
    session: Session,
    content_id: int = None,
    content_link: str = None,
    already_read: bool = None,
    read_priority: int = None,
    relevance: int = None,
):
    if content_id is None and content_link is None:
        raise ValueError(
            "At least one of id or link should be provided to perform an update."
        )

    if all((already_read is None, read_priority is None, relevance is None)):
        raise ValueError("No value to update")

    statement = select(InputContent)

    if content_id is not None:
        statement = statement.where(InputContent.id == content_id)
    else:
        statement = statement.where(InputContent.link == content_link)

    result = session.exec(statement)
    input_content = result.one()

    if already_read is not None:
        input_content.already_read = already_read

    if read_priority is not None:
        input_content.read_priority = read_priority

    if relevance is not None:
        input_content.relevance = relevance

    session.add(input_content)
    session.commit()
    session.refresh(input_content)
