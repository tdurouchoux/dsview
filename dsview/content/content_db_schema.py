from datetime import date
from pathlib import Path
from typing import Dict

from sqlmodel import SQLModel, Field
from sqlalchemy.types import String, TypeDecorator
from pydantic import HttpUrl

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
    read_priority: int = Field(default=1, ge=0, le=5)
    source: str | None = Field(default=None)

    def get_str_dict(self) -> Dict:
        instance_dict = self.model_dump()
        instance_dict["link"] = str(self.link)
        instance_dict["upload_date"] = self.upload_date.isoformat()

        if self.link is None:
            del instance_dict["source"]
        del instance_dict["id"]

        return instance_dict
