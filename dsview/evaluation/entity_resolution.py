import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sqlmodel import Session

from dsview.config import ModelConfig
from dsview.db import engine
from dsview.db.query import get_er_labels
from dsview.extraction.models import ERClassifier
from dsview.extraction.models.topics_extraction import DataScienceTopic

RANDOM_STATE = 42
SPLIT_RATIOS = {
    "example": 0.1,
    "eval": 0.7,
    "test": 0.2,
}


def evaluate(
    model_config: ModelConfig = None,
    system_prompt: str = None,
    user_prompt: str = None,
    set_type: str = "eval",
    remove_descr=False,
):
    with Session(engine) as session:
        er_data = get_er_labels(SPLIT_RATIOS, RANDOM_STATE, session)
    run_data = er_data[er_data["row_type"] == set_type].copy()

    er_classifier = ERClassifier(
        model_config=model_config, system_prompt=system_prompt, user_prompt=user_prompt
    )

    if remove_descr:
        run_data["description_1"] = ""
        run_data["description_2"] = ""

    if mlflow.active_run() is not None:
        er_classifier.log_params()

    topic_pairs = [
        (
            DataScienceTopic(
                name=comparison.name_1,
                type=comparison.type_1,
                description=comparison.description_1,
            ),
            DataScienceTopic(
                name=comparison.name_2,
                type=comparison.type_2,
                description=comparison.description_2,
            ),
        )
        for comparison in run_data.itertuples(index=False)
    ]

    merge_pred = [
        result.merge_topic for result in er_classifier.predict_batch(topic_pairs)
    ]

    run_data["merge_pred"] = merge_pred

    metrics = {
        "accuracy": accuracy_score(run_data["merge"], merge_pred),
        "precision": precision_score(run_data["merge"], merge_pred),
        "recall": recall_score(run_data["merge"], merge_pred),
        "f1": f1_score(run_data["merge"], merge_pred),
        "f0.75": fbeta_score(run_data["merge"], merge_pred, beta=0.75),
    }

    print({key: f"{value:.2f}" for key, value in metrics.items()})

    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics)

    return run_data
