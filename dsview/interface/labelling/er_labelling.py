import streamlit as st
from sqlmodel import Session

from dsview.db import engine
from dsview.db.ingest import save_er_label
from dsview.db.query import count_er_labels, get_random_missing_er_label

st.set_page_config(page_title="ER comparison labelling", page_icon="small_icon.png")

# TODO package create engine and db
# TODO seelect comparisons based on decisionsh


def main():
    if not "comparison" in st.session_state:
        st.session_state["comparison"] = None

    st.title("ER comparison labelling")

    topic_display_template = (
        "## Topic {}\n\n **Name** : {}\n\n **Type** : {}\n\n **Description** : \n\n {}"
    )

    with Session(engine) as session:

        count_total_labeled, count_total_comparisons = count_er_labels(session)

        header_col_1, header_col_2 = st.columns([7, 3])

        header_col_2.metric(
            "Labelled couples",
            f"{count_total_labeled:_} / {count_total_comparisons:_}",
            border=True,
        )

        if st.session_state.comparison is None:
            st.session_state.comparison = get_random_missing_er_label(session)
            header_col_1.success("New comparison loaded")

        if st.session_state.comparison is None:
            header_col_1.info("No more comparisons to label")
            return


        col1, col2 = st.columns(2)
        col1.markdown(
            topic_display_template.format(
                1, st.session_state.comparison.name_1, st.session_state.comparison.type_1, st.session_state.comparison.description_1
            )
        )
        col2.markdown(
            topic_display_template.format(
                2, st.session_state.comparison.name_2, st.session_state.comparison.type_2, st.session_state.comparison.description_2
            )
        )

        st.subheader("Are these topics the same ?")
        col1, col2 = st.columns(2)
        merge = col1.button("Yes", type="primary", use_container_width=True)
        not_merge = col2.button("No", type="primary", use_container_width=True)

        skip = st.button("Skip", type="secondary", use_container_width=True)

        if skip or merge or not_merge:
            if not skip:
                save_er_label(
                    session,
                    st.session_state.comparison.name_1,
                    st.session_state.comparison.type_1,
                    st.session_state.comparison.description_1,
                    st.session_state.comparison.name_2,
                    st.session_state.comparison.type_2,
                    st.session_state.comparison.description_2,
                    merge,
                )

            st.session_state.comparison = None
            st.rerun()


main()
