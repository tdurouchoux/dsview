import mlflow
import numpy as np
import pandas as pd
from rich.progress import track
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sqlmodel import Session, create_engine

from dsview.config import ModelConfig, get_sqlite_url
from dsview.models.llm_models import ERClassifier

RANDOM_STATE = 42
SPLIT_RATIOS = {
    "example": 0.4,
    "eval": 0.5,
    "test": 0.1,
}

engine = create_engine(get_sqlite_url())


def get_er_data() -> pd.DataFrame:
    with Session(engine) as session:
        conn = session.connection()

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
            conn,
            index_col="id",
        )

    df["row_type"] = np.random.RandomState(RANDOM_STATE).choice(
        list(SPLIT_RATIOS.keys()),
        size=df.shape[0],
        p=list(SPLIT_RATIOS.values()),
    )

    return df


def evaluate(
    model_config: ModelConfig = None,
    system_prompt: str = None,
    user_prompt: str = None,
    set_type: str = "eval",
    remove_descr=False,
):
    er_data = get_er_data()
    run_data = er_data[er_data["row_type"] == set_type]

    er_classifier = ERClassifier(
        model_config=model_config, system_prompt=system_prompt, user_prompt=user_prompt
    )

    if remove_descr:
        run_data["description_1"] = ""
        run_data["description_2"] = ""

    if mlflow.active_run() is not None:
        er_classifier.log_params()

    merge_pred = []

    for comparison in track(run_data.itertuples(index=False), total=run_data.shape[0]):
        topic_comparison = {
            "name_1": comparison.name_1,
            "type_1": comparison.type_1,
            "description_1": comparison.description_1,
            "name_2": comparison.name_2,
            "type_2": comparison.type_2,
            "description_2": comparison.description_2,
        }

        merge_pred.append(er_classifier.predict(topic_comparison).merge_topic)

    run_data["merge_pred"] = merge_pred

    metrics = {
        "accuracy": accuracy_score(run_data["merge"], merge_pred),
        "precision": precision_score(run_data["merge"], merge_pred),
        "recall": recall_score(run_data["merge"], merge_pred),
        "f1": f1_score(run_data["merge"], merge_pred),
    }

    print({key: f"{value:.2f}" for key, value in metrics.items()})

    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics)

    return run_data
