import pandas as pd
import streamlit as st

from dsview.db.ingest import save_labels
from dsview.extraction.content_loader import WebContentLoader
from dsview.extraction.models.description_generation import (
    ContentDescription,
    ContentType,
    TagsType,
)
from dsview.extraction.models.links_extraction import RelevantLink
from dsview.extraction.models.topics_extraction import DataScienceTopic, TopicType


def topics_labelling(topics: list[DataScienceTopic]) -> pd.DataFrame:
    st.subheader("Content topics")

    df_topics = pd.DataFrame(
        [
            {"rank": i + 1, "name": topic.name, "type": topic.type}
            for i, topic in enumerate(topics)
        ]
        + [{"rank": i + 1, "name": None, "type": None} for i in range(len(topics), 10)]
    )

    topic_rank_column = st.column_config.NumberColumn(
        "Rank", width="small", min_value=1, max_value=10, step=1, disabled=True
    )

    topic_name_column = st.column_config.TextColumn(
        "Topic name",
        width="medium",
    )

    topic_type_column = st.column_config.SelectboxColumn(
        "Topic type",
        width="medium",
        options=TopicType,
    )

    topic_ranking = st.data_editor(
        df_topics,
        column_config={
            "rank": topic_rank_column,
            "name": topic_name_column,
            "type": topic_type_column,
        },
        num_rows="fixed",
        hide_index=True,
    )

    return topic_ranking


def links_labelling(
    content_links: list[RelevantLink], all_links: list[str]
) -> pd.DataFrame:
    st.subheader("Content links")

    hyperlink_rank_column = st.column_config.NumberColumn(
        "Rank", width="small", min_value=1, max_value=10, step=1, disabled=True
    )

    hyperlink_column = st.column_config.SelectboxColumn(
        "Hyperlink",
        width="large",
        options=all_links,
    )

    df_links = pd.DataFrame(
        [{"rank": i + 1, "hyperlink": link.url} for i, link in enumerate(content_links)]
        + [{"rank": i + 1, "hyperlink": None} for i in range(len(content_links), 10)]
    )

    links_ranking = st.data_editor(
        df_links,
        column_config={
            "rank": hyperlink_rank_column,
            "hyperlink": hyperlink_column,
        },
        num_rows="fixed",
        hide_index=True,
    )

    return links_ranking


def generate_labelling_form(
    content_loader: WebContentLoader,
    content_description: ContentDescription,
    topics: list[DataScienceTopic],
    content_links: str,
    all_links: list[str],
):
    with st.form("content labellization"):
        st.subheader("Content description")

        title = st.text_input("Title", value=content_description.title)

        content_type = st.multiselect(
            "Type",
            ContentType,
            default=content_description.content_type,
            max_selections=1,
        )[0]

        tags = st.multiselect(
            "Tags",
            TagsType,
            default=[tag.name for tag in content_description.tags],
            max_selections=3,
        )

        topics_ranking = topics_labelling(topics)

        links_ranking = links_labelling(content_links, all_links)

        submit = st.form_submit_button(
            "Confirm labels",
            type="primary",
        )

        if submit:
            save_labels(
                content_loader.link,
                content_loader.content,
                title,
                content_type,
                tags,
                topics_ranking,
                links_ranking,
            )

            st.session_state["labelling"] = False
            st.rerun()
