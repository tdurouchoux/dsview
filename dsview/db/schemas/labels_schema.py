from pathlib import Path

from pydantic import HttpUrl
from sqlmodel import Field, SQLModel

# from dsview.obsidian.sync_vault import label_sync_vault
from .input_content_schema import LinkType


class LabelledContent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    link: HttpUrl | Path = Field(unique=True, sa_type=LinkType)
    content: str
    test: bool = Field(default=False)


class TitleLabels(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key="labelledcontent.id")
    title: str


class ContentTypeLabels(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key="labelledcontent.id")
    content_type: str


class TagLabels(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key="labelledcontent.id")
    tag: str


class TopicsLabels(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    type: str
    rank: int = Field(ge=1, lt=10)
    content_id: int = Field(foreign_key="labelledcontent.id")


class LinksLabels(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    hyperlink: str
    rank: int = Field(ge=1, lt=10)
    content_id: int = Field(foreign_key="labelledcontent.id")


class ERLabels(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    er_comparison_id: int = Field(foreign_key="ercomparison.id")
    merge: bool
