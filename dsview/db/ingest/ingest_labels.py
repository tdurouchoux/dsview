import pandas as pd
from sqlmodel import Session

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
    tag_labels = [TagLabels(content_id=labelled_content.id, tag=tag) for tag in tags]
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


def save_er_label(
    session: Session,
    name_1: str,
    type_1: str,
    description_1: str,
    name_2: str,
    type_2: str,
    description_2: str,
    merge: bool,
):
    er_label = ERLabels(
        name_1=name_1,
        type_1=type_1,
        description_1=description_1,
        name_2=name_2,
        type_2=type_2,
        description_2=description_2,
        merge=merge,
    )

    session.add(er_label)
    session.commit()


def update_er_label(session: Session, label_id: int, merge: bool):
    update_instance(session, ERLabels, row_id=label_id, merge=merge)
