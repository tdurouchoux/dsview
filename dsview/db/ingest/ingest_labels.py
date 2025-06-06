import pandas as pd
from sqlmodel import Session

from dsview.config import get_sqlite_engine

from ..schemas import (
    ContentTypeLabels,
    ERLabels,
    LabelledContent,
    LinksLabels,
    TagLabels,
    TitleLabels,
    TopicsLabels,
)
from .update import update_instance

engine = get_sqlite_engine()


# @label_sync_vault("content")
def save_labels(
    link: str,
    content: str,
    title: str,
    content_type: str,
    tags: list[str],
    topic_ranking: pd.DataFrame,
    links_ranking: pd.DataFrame,
    session: Session,
):
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


# TODO Remove (or not) Session from arguments


# @label_sync_vault("er")
def save_er_label(session: Session, er_comparison_id: int, merge: bool):
    er_label = ERLabels(
        er_comparison_id=er_comparison_id,
        merge=merge,
    )

    session.add(er_label)
    session.commit()


def update_er_label(session: Session, label_id: int, merge: bool):
    update_instance(session, ERLabels, row_id=label_id, merge=merge)
