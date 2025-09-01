from pathlib import Path

from pydantic import HttpUrl
from sqlmodel import Field, SQLModel

# from dsview.obsidian.sync_vault import label_sync_vault
from .content_schema import LinkType

LABELS_SCHEMA = "labels"

class LabelledContent(SQLModel, table=True):
    __tablename__ = "labelledcontent"
    __table_args__ = {"schema": LABELS_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    link: HttpUrl | Path = Field(unique=True, sa_type=LinkType)
    content: str
    test: bool = Field(default=False)


class TitleLabels(SQLModel, table=True):
    __tablename__ = "titlelabels"
    __table_args__ = {"schema": LABELS_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key=f"{LABELS_SCHEMA}.labelledcontent.id")
    title: str


class ContentTypeLabels(SQLModel, table=True):
    __tablename__ = "contenttypelabels"
    __table_args__ = {"schema": LABELS_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key=f"{LABELS_SCHEMA}.labelledcontent.id")
    content_type: str


class TagLabels(SQLModel, table=True):
    __tablename__ = "taglabels"
    __table_args__ = {"schema": LABELS_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key=f"{LABELS_SCHEMA}.labelledcontent.id")
    tag: str


class TopicsLabels(SQLModel, table=True):
    __tablename__ = "topicslabels"
    __table_args__ = {"schema": LABELS_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    name: str
    type: str
    rank: int = Field(ge=1, lt=10)
    content_id: int = Field(foreign_key=f"{LABELS_SCHEMA}.labelledcontent.id")


class LinksLabels(SQLModel, table=True):
    __tablename__ = "linkslabels"
    __table_args__ = {"schema": LABELS_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    hyperlink: str
    rank: int = Field(ge=1, lt=10)
    content_id: int = Field(foreign_key=f"{LABELS_SCHEMA}.labelledcontent.id")


class ERLabels(SQLModel, table=True):
    __tablename__ = "erlabels"
    __table_args__ = {"schema": LABELS_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    name_1: str = Field(nullable=False)
    type_1: str = Field(nullable=False)
    description_1: str = Field(nullable=False)
    name_2: str = Field(nullable=False)
    type_2: str = Field(nullable=False)
    description_2: str = Field(nullable=False)
    merge: bool
