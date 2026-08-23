from datetime import date
from pathlib import Path

from pydantic import HttpUrl, field_serializer
from sqlalchemy.types import String, TypeDecorator
from sqlmodel import Field, SQLModel

INPUT_CONTENT_SCHEMA = "content"


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
    __tablename__ = "inputcontent"
    __table_args__ = {"schema": INPUT_CONTENT_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)  # Add an ID as primary key
    link: HttpUrl = Field(unique=True, sa_type=LinkType)
    upload_date: date
    already_read: bool = Field(default=False)
    read_priority: int = Field(default=0, ge=0, le=5)
    relevance: int = Field(default=0, ge=0, le=5)
    source: str | None = Field(default=None)

    @field_serializer("link")
    def url_to_string(self, value: HttpUrl) -> str:
        return str(value)

    def get_str_dict(self) -> dict:
        instance_dict = self.model_dump()
        instance_dict["link"] = str(self.link)
        instance_dict["upload_date"] = self.upload_date.isoformat()

        if self.link is None:
            del instance_dict["source"]
        if "id" in instance_dict:
            del instance_dict["id"]

        return instance_dict


class FailedIngestion(SQLModel, table=True):
    __tablename__ = "failedingestion"
    __table_args__ = {"schema": INPUT_CONTENT_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(
        unique=True, foreign_key=f"{INPUT_CONTENT_SCHEMA}.inputcontent.id"
    )
    original_link: str
    error_type: str
    error_message: str
