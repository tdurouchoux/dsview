from typing import Literal

import mlflow
import pandas as pd
from pydantic import BaseModel
from sqlmodel import Session

from dsview.config import ModelConfig
from dsview.db import engine
from dsview.db.query import get_labels_data
from dsview.db.schemas import TopicsLabels
from dsview.extraction.models.topics_extraction import (
    TopicsExtractor,
    TopicType,
)
from dsview.model_utils import LLMModel

MAX_N_TOPICS = 10


class SimpleERResult(BaseModel):
    analysis: bool
    merge_topic: bool


class SimpleERModel(LLMModel):
    DEFAULT_SYSTEM_PROMPT_FILE = "system_eval_simple_er.txt"
    DEFAULT_USER_PROMPT_FILE = "user_eval_simple_er.txt"
    DEFAULT_SYSTEM_PROMPT_FORMAT = [",".join(TopicType)]
    DEFAULT_STRUCTURED_OUTPUT_CLASS = SimpleERResult


def get_topics_extraction_data(set_type: Literal["eval", "test"]) -> pd.DataFrame:
    with Session(engine) as session:
        df_labels = get_labels_data([TopicsLabels], set_type, session)

    df_labels = (
        df_labels.dropna(subset=["name", "type"], how="any")
        .sort_values("rank", ascending=True)
        .groupby("id")
        .agg({"content": "first", "name": list, "type": list})
    )

    return df_labels


def find_er_matches(simple_er: SimpleERModel, df: pd.DataFrame) -> dict[tuple, str]:
    """Match predicted topics to labelled topics with the LLM judge.

    Returns a mapping of (row id, predicted topic position) to the matched
    labelled topic name. Predicted topics whose name is an exact labelled
    name are matched without a judge call.
    """
    er_inputs = []
    er_keys = []

    for row_id, row in df.iterrows():
        for topic_position, topic in enumerate(row["pred_topics"]):
            if topic.name in row["name"]:
                continue

            for labelled_name, labelled_type in zip(row["name"], row["type"]):
                er_inputs.append(
                    {
                        "name_1": labelled_name,
                        "type_1": labelled_type,
                        "name_2": topic.name,
                        "type_2": topic.type.value,
                    }
                )
                er_keys.append((row_id, topic_position, labelled_name))

    er_results = simple_er.predict_batch(er_inputs)

    er_matches = {}
    for (row_id, topic_position, labelled_name), result in zip(er_keys, er_results):
        key = (row_id, topic_position)
        # Labelled topics are rank-ordered: keep the first match.
        if result.merge_topic and key not in er_matches:
            er_matches[key] = labelled_name

    return er_matches


def score_row(row_id, row: pd.Series, er_matches: dict[tuple, str]) -> pd.Series:
    pred_topics_match = []
    precision_at_i = []
    count_type_correct = 0

    for topic_position, topic in enumerate(row["pred_topics"]):
        if topic.name in row["name"]:
            topic_match = topic.name
        else:
            topic_match = er_matches.get((row_id, topic_position))

        pred_topics_match.append(topic_match)

        if topic_match is not None:
            precision_at_i.append(
                1 - (pred_topics_match.count(None) / (topic_position + 1))
            )

            # Check if type is also correct
            topic_match_index = row["name"].index(topic_match)
            if topic.type.value == row["type"][topic_match_index]:
                count_type_correct += 1

    count_correct_topic = len(row["pred_topics"]) - pred_topics_match.count(None)

    if len(row["pred_topics"]) == 0:
        # Predicting nothing is always a failure here (every labelled row has
        # topics to find), so an empty prediction must score 0 — not 1. Scoring
        # it 1 rewarded abstention and silently inflated precision/AP on rows
        # where the model returned nothing (e.g. a blocked/empty source page).
        precision = 0
        recall = 0
        average_precision = 0
    else:
        precision = count_correct_topic / len(row["pred_topics"])
        recall = len({name for name in pred_topics_match if name is not None}) / min(
            len(row["name"]), MAX_N_TOPICS
        )

        average_precision = sum(precision_at_i) / len(row["pred_topics"])

    return pd.Series(
        [
            count_correct_topic,
            precision,
            recall,
            average_precision,
            count_type_correct,
        ]
    )


def evaluate(
    model_config: ModelConfig | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    set_type: Literal["eval", "test"] = "eval",
):
    topics_extractor = TopicsExtractor(
        model_config=model_config, system_prompt=system_prompt, user_prompt=user_prompt
    )
    simple_er = SimpleERModel()

    if mlflow.active_run():
        topics_extractor.log_params()

    df = get_topics_extraction_data(set_type)

    extraction_results = topics_extractor.predict_batch(
        [{"content": content} for content in df["content"]]
    )
    df["pred_topics"] = [result.topics for result in extraction_results]

    er_matches = find_er_matches(simple_er, df)

    df[
        [
            "count_correct_topic",
            "precision",
            "recall",
            "average_precision",
            "count_type_correct",
        ]
    ] = df.apply(lambda row: score_row(row.name, row, er_matches), axis=1)

    metrics = {
        f"mean_{metric}": df[metric].mean()
        for metric in ("precision", "recall", "average_precision")
    }
    metrics["count_correct_topic"] = df["count_correct_topic"].sum()
    metrics["type_accuracy"] = (
        df["count_type_correct"].sum() / df["count_correct_topic"].sum()
    )

    print({key: f"{value:.2f}" for key, value in metrics.items()})

    if mlflow.active_run():
        mlflow.log_metrics(metrics)

    return df
