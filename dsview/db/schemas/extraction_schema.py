import importlib.metadata
import logging

from sqlmodel import Field, SQLModel

logger = logging.getLogger(__name__)

__version__ = importlib.metadata.version("dsview")


class ExtractionResult(SQLModel, table=True):
    content_id: int = Field(foreign_key="inputcontent.id", primary_key=True)
    title: str
    content_type: str
    summary: str
    # embedding: str


class ExtractionTag(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key="extractionresult.content_id")
    name: str


class ExtractionLink(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key="extractionresult.content_id")
    name: str
    url: str
    description: str


class ExtractionTopic(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    type: str
    name: str
    description: str
    embedding: str


class ContentTopicRelation(SQLModel, table=True):
    content_id: int = Field(foreign_key="extractionresult.content_id", primary_key=True)
    topic_id: int = Field(foreign_key="extractiontopic.id", primary_key=True)


class ERComparison(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name_1: str = Field(nullable=False)
    type_1: str = Field(nullable=False)
    description_1: str = Field(nullable=False)
    name_2: str = Field(nullable=False)
    type_2: str = Field(nullable=False)
    description_2: str = Field(nullable=False)


class ERDecision(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    comparison_id: int = Field(foreign_key="ercomparison.id")
    code_version: str = Field(default=__version__)
    decision_date: str
    merge_topic: bool = Field(nullable=False)
    merge_name: str | None = Field(default=None, nullable=True)
    merge_type: str | None = Field(default=None, nullable=True)
    merge_description: str | None = Field(default=None, nullable=True)
