from pathlib import Path

import pandas as pd
from pydantic import HttpUrl
from sqlmodel import SQLModel, Field, create_engine, Session, select

from dsview.content.content_db_schema import LinkType


class LabelledContent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    link: HttpUrl | Path = Field(unique=True, sa_type=LinkType)
    content: str
    test: bool = Field(default=False)


# To be used also during data validation
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


def get_engine(db_url: str):
    engine = create_engine(db_url)
    SQLModel.metadata.create_all(engine)

    return engine


def check_link_labelled(engine, link: str) -> bool:
    with Session(engine) as session:
        statement = select(LabelledContent).where(LabelledContent.link == link)
        result = session.exec(statement).first()
        return result is not None


def save_labels(
    engine,
    link: str,
    content: str,
    title: str,
    content_type: str,
    tags: list[str],
    topic_ranking: pd.DataFrame,
    links_ranking: pd.DataFrame,
):
    with Session(engine) as session:
        labelled_content = LabelledContent(link=link, content=content)
        session.add(labelled_content)
        session.commit()

        title_label = TitleLabels(content_id=labelled_content.id, title=title)
        content_type_label = ContentTypeLabels(
            content_id=labelled_content.id, content_type=content_type
        )
        tag_labels = [
            TagLabels(content_id=labelled_content.id, tag=tag) for tag in tags
        ]
        topic_labels = [
            TopicsLabels(
                content_id=labelled_content.id,
                name=row["name"],
                type=row["type"],
                rank=row["rank"],
            )
            for _, row in topic_ranking.dropna(how="any").iterrows()
        ]
        links_labels = [
            LinksLabels(
                hyperlink=row["hyperlink"],
                rank=row["rank"],
                content_id=labelled_content.id,
            )
            for _, row in links_ranking.dropna(how="any").iterrows()
        ]

        session.add_all(
            [
                title_label,
                content_type_label,
                *tag_labels,
                *topic_labels,
                *links_labels,
            ]
        )
        session.commit()


def clear_content_labelling(engine):
    SQLModel.metadata.drop_all(
        engine,
        tables=[
            LabelledContent.__table__,
            TitleLabels.__table__,
            ContentTypeLabels.__table__,
            TagLabels.__table__,
            TopicsLabels.__table__,
            LinksLabels.__table__,
        ],
    )


def clear_er_labelling(engine):
    SQLModel.metadata.drop_all(engine, tables=[ERLabels.__table__])
