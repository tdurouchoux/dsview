import importlib.metadata
import logging
from datetime import datetime

from sqlmodel import ARRAY, Column, Field, Float, SQLModel, Relationship

logger = logging.getLogger(__name__)

__version__ = importlib.metadata.version("dsview")

EXTRACTION_SCHEMA = "extraction"

# TODO Change types as enum ?


class ContentTopicRelation(SQLModel, table=True):
    __tablename__ = "contenttopicrelation"
    __table_args__ = {"schema": EXTRACTION_SCHEMA}

    content_id: int = Field(
        foreign_key=f"{EXTRACTION_SCHEMA}.extractionresult.content_id",
        primary_key=True,
        ondelete="CASCADE",
    )
    topic_id: int = Field(
        foreign_key=f"{EXTRACTION_SCHEMA}.extractiontopic.id",
        primary_key=True,
        ondelete="CASCADE",
    )


class ExtractionTag(SQLModel, table=True):
    __tablename__ = "extractiontag"
    __table_args__ = {"schema": EXTRACTION_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(
        foreign_key=f"{EXTRACTION_SCHEMA}.extractionresult.content_id"
    )
    name: str

    extraction: "ExtractionResult" = Relationship(back_populates="tags")


class ExtractionLink(SQLModel, table=True):
    __tablename__ = "extractionlink"
    __table_args__ = {"schema": EXTRACTION_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(
        foreign_key=f"{EXTRACTION_SCHEMA}.extractionresult.content_id"
    )
    name: str
    url: str
    description: str

    extraction: "ExtractionResult" = Relationship(back_populates="links")


class ExtractionTopic(SQLModel, table=True):
    __tablename__ = "extractiontopic"
    __table_args__ = {"schema": EXTRACTION_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    type: str
    name: str
    description: str
    embedding: list[float] = Field(sa_column=Column(ARRAY(Float)))

    extractions: list["ExtractionResult"] = Relationship(
        back_populates="topics", link_model=ContentTopicRelation
    )


class ExtractionResult(SQLModel, table=True):
    __tablename__ = "extractionresult"
    __table_args__ = {"schema": EXTRACTION_SCHEMA}

    content_id: int = Field(foreign_key="content.inputcontent.id", primary_key=True)
    title: str
    content_type: str
    summary: str
    embedding: list[float] = Field(sa_column=Column(ARRAY(Float)))
    extraction_time: datetime = Field(default_factory=datetime.now)

    tags: list[ExtractionTag] = Relationship(back_populates="extraction")
    links: list[ExtractionLink] = Relationship(back_populates="extraction")
    topics: list[ExtractionTopic] = Relationship(
        back_populates="extractions", link_model=ContentTopicRelation
    )


class ERComparison(SQLModel, table=True):
    __tablename__ = "ercomparison"
    __table_args__ = {"schema": EXTRACTION_SCHEMA}

    id: int | None = Field(default=None, primary_key=True)
    name_1: str = Field(nullable=False)
    type_1: str = Field(nullable=False)
    description_1: str = Field(nullable=False)
    name_2: str = Field(nullable=False)
    type_2: str = Field(nullable=False)
    description_2: str = Field(nullable=False)
    fts_score: float = Field(nullable=True)
    vss_distance: float = Field(nullable=True)
    decision_date: str = Field(nullable=False)
    merge_topic: bool = Field(nullable=False)
    merge_name: str | None = Field(default=None, nullable=True)
    merge_type: str | None = Field(default=None, nullable=True)
    merge_description: str | None = Field(default=None, nullable=True)
