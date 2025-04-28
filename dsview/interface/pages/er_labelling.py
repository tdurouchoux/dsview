import streamlit as st

from sqlmodel import Session, select

from dsview.config import get_sqlite_url
from dsview.extraction.extraction_db_schema import ERComparison
from dsview.evaluation.labels_schema import ERLabels, get_engine, save_er_label

from dsview.interface.interface_utils import get_sidebar

sqlite_url = get_sqlite_url()

st.set_page_config(page_title="ER comparison labelling", page_icon="small_icon.png")

# TODO package create engine and db


def get_missing_label_comparison(session: Session) -> ERComparison:
    er_labels = session.exec(select(ERLabels)).all()

    if len(er_labels) == 0:
        max_comparison_id = 1
    else:
        max_comparison_id = max(er_label.er_comparison_id for er_label in er_labels) + 1

    comparison = session.exec(
        select(ERComparison).where(ERComparison.id == max_comparison_id)
    ).first()

    return comparison


def main():
    get_sidebar()

    engine = get_engine(sqlite_url)

    st.title("ER comparison labelling")

    topic_display_template = (
        "## Topic {}\n\n **Name** : {}\n\n **Type** : {}\n\n **Description** : \n\n {}"
    )

    with Session(engine) as session:
        comparison = get_missing_label_comparison(session)

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
            save_er_label(session, comparison.id, merge)
            st.rerun()


main()
