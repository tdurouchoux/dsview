from sqlalchemy.sql.expression import label
import streamlit as st
from sqlmodel import Session, text

from dsview.config import get_sqlite_engine
from dsview.db.ingest import save_er_label
from dsview.db.schemas import ERComparison
from dsview.obsidian.sync_vault import label_sync_vault

st.set_page_config(page_title="ER comparison labelling", page_icon="small_icon.png")

# TODO package create engine and db
# TODO seelect comparisons based on decisionsh


def get_missing_label_comparison(session: Session) -> ERComparison:
    result = session.exec(
        text("""
            SELECT * FROM ercomparison
            WHERE (name_1, name_2) NOT IN (
                SELECT name_1, name_2 FROM erlabels
            ) ORDER BY RANDOM()""")
    ).first()

    return ERComparison(**result._asdict())


@label_sync_vault("content", debug_mode=False)
def save_and_sync_er_label(*args):
    save_er_label(*args)


def main():
    engine = get_sqlite_engine()
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
