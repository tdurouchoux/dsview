from typing import Literal, Type
import hashlib
import random

import pandas as pd
from sqlalchemy.sql import schema
from sqlmodel import Session, select, text, func
from sqlmodel.main import SQLModel

from ..schemas import LABELS_SCHEMA, ERComparison, ERLabels, LabelledContent

DEFAULT_SPLIT_RATIOS = {
    "eval": 0.7,
    "test": 0.3,
}
DEFAULT_RANDOM_STATE = 42


def assign_rows(
    df: pd.DataFrame,
    split_ratios: dict[str, float],
    random_state: int,
    keys: pd.Series,
) -> pd.DataFrame:
    """Deterministically assign each row to a split based on a hash of its key.

    Unlike positional random sampling, a row's split membership depends only
    on its own key, so it stays stable as new labelled rows are appended —
    positional sampling reshuffled every existing row's eval/test membership
    whenever the labelled set grew, silently invalidating past MLflow runs.
    """

    def _bucket(key) -> str:
        digest = hashlib.sha256(f"{random_state}:{key}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64

        cumulative = 0.0
        for set_type, ratio in split_ratios.items():
            cumulative += ratio
            if fraction < cumulative:
                return set_type

        return next(reversed(split_ratios))

    df["row_type"] = [_bucket(key) for key in keys]

    return df


def get_labeled_content(
    set_type: Literal["eval", "test"],
    session: Session,
    split_ratios: dict[str, float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    df_labeled = pd.read_sql_table(
        LabelledContent.__tablename__, session.bind, schema=LABELS_SCHEMA
    )

    df_labeled = assign_rows(
        df_labeled, split_ratios, random_state, keys=df_labeled["id"]
    )

    return df_labeled[df_labeled["row_type"] == set_type].drop(columns=["row_type"])


def get_labels_data(
    labels_tables: list[Type[SQLModel]],
    set_type: Literal["eval", "test"],
    session: Session,
) -> pd.DataFrame:
    df_labeled = get_labeled_content(set_type, session)

    for labels_table in labels_tables:
        df_labels = pd.read_sql_table(
            labels_table.__tablename__, session.bind, schema=LABELS_SCHEMA
        )

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
    query = f"""
        SELECT
           	erlabels.id,
            erlabels.merge,
            ercomparison.name_1,
            ercomparison.description_1,
            ercomparison.type_1,
            ercomparison.name_2,
            ercomparison.description_2,
            ercomparison.type_2
        FROM {ERLabels.__table__.schema}.{ERLabels.__table__.name}
        JOIN {ERComparison.__table__.schema}.{ERComparison.__table__.name}
        ON erlabels.name_1 = ercomparison.name_1 AND erlabels.name_2 = ercomparison.name_2
    """

    # Use the connection with pandas read_sql
    df = pd.read_sql(
        query,
        session.bind,
        index_col="id",
    )

    df = assign_rows(df, split_ratios, random_state, keys=df.index)

    return df


def count_er_labels(session: Session) -> tuple[int, int]:

    count_total_labeled = session.scalar(select(func.count(ERLabels.id)))

    count_total_comparisons = session.scalar(select(func.count(ERComparison.id)))

    return count_total_labeled, count_total_comparisons


def get_random_missing_er_label(
    session: Session,
    class_weight: float | None = 0.4,
    bias_expr: str | None = "1 - vss_distance",
    bias_strength: float = 0.5,
) -> ERComparison | None:

    query = f"""
        SELECT * FROM {ERComparison.__table__.schema}.{ERComparison.__table__.name}
        WHERE (name_1, name_2) NOT IN (
            SELECT name_1, name_2 FROM {ERLabels.__table__.schema}.{ERLabels.__table__.name}
        )
    """

    if class_weight is not None:
        class_selection = random.random() < class_weight
        query += f" AND merge_topic = {'true' if class_selection else 'false'}"

    if bias_expr is not None:
        query += f"""
            ORDER BY POWER(random(), 1.0 / GREATEST(POWER({bias_expr}, {bias_strength}), 0.0001)) DESC;
        """
    else:
        query += " ORDER BY RANDOM()"

    result = session.exec(text(query)).first()

    if result is None:
        if class_weight is not None:
            return get_random_missing_er_label(
                session,
                class_weight=None,
                bias_expr=bias_expr,
                bias_strength=bias_strength,
            )

        return None

    return ERComparison(**result._asdict())
