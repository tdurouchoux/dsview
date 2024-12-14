import importlib.metadata
import logging

from sqlmodel import SQLModel, Field, Session, select

logger = logging.getLogger(__name__)

__version__ = importlib.metadata.version("dsview")


class InvalidTag(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key="inputcontent.id")
    name: str


class InvalidTopic(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(foreign_key="inputcontent.id")
    type: str
    name: str
    description: str


class ERComparison(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name_1: str = Field(nullable=False)
    type_1: str = Field(nullable=False)
    description_1: str = Field(nullable=False)
    name_2: str = Field(nullable=False)
    type_2: str = Field(nullable=False)
    description_2: str = Field(nullable=False)


def find_or_add_comparison(topic_comparison: dict, session: Session) -> ERComparison:
    # Check if the comparison already exists in the database
    existing_comparison = session.exec(
        select(ERComparison).where(
            *[
                getattr(ERComparison, key) == value
                for key, value in topic_comparison.items()
            ]
        )
    ).first()

    if existing_comparison is not None:
        logger.warning(
            "The exact same comparison has already been done, check decisions in db."
        )
        return existing_comparison
    else:
        # Add the comparison to the database
        er_comparison = ERComparison(**topic_comparison)
        session.add(er_comparison)
        session.commit()
        return er_comparison


class ERDecision(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    comparison_id: int = Field(foreign_key="ercomparison.id")
    code_version: str = Field(default=__version__)
    decision_date: str
    merge_topic: bool = Field(nullable=False)
    merge_name: str | None = Field(default=None, nullable=True)
    merge_type: str | None = Field(default=None, nullable=True)
    merge_description: str | None = Field(default=None, nullable=True)
