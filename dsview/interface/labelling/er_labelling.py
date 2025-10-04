import streamlit as st
from sqlmodel import Session

from dsview.db import engine
from dsview.db.ingest import save_er_label
from dsview.db.query import get_random_missing_er_label

st.set_page_config(page_title="ER comparison labelling", page_icon="small_icon.png")

# TODO package create engine and db
# TODO seelect comparisons based on decisionsh


def main():
    st.title("ER comparison labelling")

    topic_display_template = (
        "## Topic {}\n\n **Name** : {}\n\n **Type** : {}\n\n **Description** : \n\n {}"
    )

    with Session(engine) as session:
        comparison = get_random_missing_er_label(session)

        if comparison is None:
            st.info("No more comparisons to label")
            return

        col1, col2 = st.columns(2)
        col1.markdown(
            topic_display_template.format(
                1, comparison.name_1, comparison.type_1, comparison.description_1
            )
        )
        col2.markdown(
            topic_display_template.format(
                2, comparison.name_2, comparison.type_2, comparison.description_2
            )
        )

        st.subheader("Are these topics the same ?")
        col1, col2 = st.columns(2)
        merge = col1.button("Yes", type="primary", use_container_width=True)
        not_merge = col2.button("No", type="primary", use_container_width=True)

        if merge or not_merge:
            save_er_label(
                session,
                comparison.name_1,
                comparison.type_1,
                comparison.description_1,
                comparison.name_2,
                comparison.type_2,
                comparison.description_2,
                merge,
            )
            st.rerun()


main()
