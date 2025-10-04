from typing import Literal, Type

import numpy as np
import pandas as pd
from sqlmodel import Session, select, text
from sqlmodel.main import SQLModel

from ..schemas import ERComparison, ERLabels, LabelledContent

DEFAULT_SPLIT_RATIOS = {
    "eval": 0.7,
    "test": 0.3,
}
DEFAULT_RANDOM_STATE = 42


def assign_rows(
    df: pd.DataFrame,
    split_ratios: dict[str, float],
    random_state: int,
) -> pd.DataFrame:
    df["row_type"] = np.random.RandomState(random_state).choice(
        list(split_ratios.keys()),
        size=df.shape[0],
        p=list(split_ratios.values()),
    )

    return df


def get_labeled_content(
    set_type: Literal["eval", "test"],
    session: Session,
    split_ratios: dict[str, float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    df_labeled = pd.read_sql_table(LabelledContent.__tablename__, session.bind)

    df_labeled = assign_rows(df_labeled, split_ratios, random_state)

    return df_labeled[df_labeled["row_type"] == set_type].drop(columns=["row_type"])


def get_labels_data(
    labels_tables: list[Type[SQLModel]],
    set_type: Literal["eval", "test"],
    session: Session,
) -> pd.DataFrame:
    df_labeled = get_labeled_content(set_type, session)

    for labels_table in labels_tables:
        df_labels = pd.read_sql_table(labels_table.__tablename__, session.bind)

        df_labeled = df_labeled.merge(
            df_labels.drop(columns=["id"]),
            left_on="id",
            right_on="content_id",
            how="left",
        )

    return df_labeled


def check_link_labelled(link: str, session: Session) -> bool:
    statement = select(LabelledContent).where(LabelledContent.link == link)
    result = session.exec(statement).first()
    return result is not None


def get_er_labels(
    split_ratios: dict[str, float],
    random_state: int,
    session: Session,
) -> pd.DataFrame:
    query = """
        SELECT
           	erlabels.id,
            erlabels.merge,
            ercomparison.name_1,
            ercomparison.description_1,
            ercomparison.type_1,
            ercomparison.name_2,
            ercomparison.description_2,
            ercomparison.type_2
        FROM erlabels
        JOIN ercomparison
        ON erlabels.er_comparison_id = ercomparison.id
    """

    # Use the connection with pandas read_sql
    df = pd.read_sql(
        query,
        session.bind,
        index_col="id",
    )

    df = assign_rows(df, split_ratios, random_state)

    return df


def get_random_missing_er_label(session: Session) -> ERComparison | None:
    result = session.exec(
        text(f"""
            SELECT * FROM {ERComparison.__table__.schema}.{ERComparison.__table__.name}
            WHERE (name_1, name_2) NOT IN (
                SELECT name_1, name_2 FROM {ERLabels.__table__.schema}.{ERLabels.__table__.name}
            ) ORDER BY RANDOM()""")
    ).first()

    if result is None:
        return None

    return ERComparison(**result._asdict())
