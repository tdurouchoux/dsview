from typing import Literal, Type

import numpy as np
import pandas as pd
from sqlmodel import Session, create_engine, select
from sqlmodel.main import SQLModel

from dsview.config import get_sqlite_url
from dsview.evaluation.labels_schema import LabelledContent

SPLIT_RATIOS = {
    "eval": 0.7,
    "test": 0.3,
}
RANDOM_STATE = 42

engine = create_engine(get_sqlite_url())


def statement_to_df(statement) -> pd.DataFrame:
    with Session(engine) as session:
        results = session.exec(statement).all()
        data = [row.dict() for row in results]
    return pd.DataFrame.from_records(data)


def get_labeled_content(set_type: Literal["eval", "test"]) -> pd.DataFrame:
    statement = select(LabelledContent)
    df_labeled = statement_to_df(statement)

    df_labeled["row_type"] = np.random.RandomState(RANDOM_STATE).choice(
        list(SPLIT_RATIOS.keys()),
        size=df_labeled.shape[0],
        p=list(SPLIT_RATIOS.values()),
    )

    return df_labeled[df_labeled["row_type"] == set_type].drop(columns=["row_type"])


def get_labels_data(
    labels_tables: list[Type[SQLModel]], set_type: Literal["eval", "test"]
) -> pd.DataFrame:
    df_labeled = get_labeled_content(set_type)

    for labels_table in labels_tables:
        statement = select(labels_table)
        df_labels = statement_to_df(statement)

        df_labeled = df_labeled.merge(
            df_labels.drop(columns=["id"]),
            left_on="id",
            right_on="content_id",
            how="left",
        )

    return df_labeled
