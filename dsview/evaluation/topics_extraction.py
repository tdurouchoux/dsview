import asyncio
from typing import Literal

import mlflow
import nest_asyncio
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from dsview.config import ModelConfig, load_extraction_config
from dsview.db.query import get_labels_data
from dsview.db.schemas import TopicsLabels
from dsview.extraction.models.topics_extraction import (
    DataScienceTopic,
    TopicsExtractor,
    TopicType,
)
from dsview.model_utils import LLMModel

tqdm.pandas()
nest_asyncio.apply()

extraction_config = load_extraction_config()

MAX_N_TOPICS = 10


class SimpleERResult(BaseModel):
    merge_topic: bool


class SimpleERModel(LLMModel):
    DEFAULT_SYSTEM_PROMPT_FILE = "system_eval_simple_er.txt"
    DEFAULT_USER_PROMPT_FILE = "user_eval_simple_er.txt"
    DEFAULT_SYSTEM_PROMPT_FORMAT = [",".join(TopicType)]
    DEFAULT_STRUCTURED_OUTPUT_CLASS = SimpleERResult


def get_topics_extraction_data(set_type: Literal["eval", "test"]) -> pd.DataFrame:
    df_labels = get_labels_data([TopicsLabels], set_type)

    df_labels = (
        df_labels.dropna(subset=["name", "type"], how="any")
        .sort_values("rank", ascending=True)
        .groupby("id")
        .agg({"content": "first", "name": list, "type": list})
    )

    return df_labels


async def is_topic_close(
    simple_er: SimpleERModel,
    relevant_topic_name: str,
    relevant_topic_type: str,
    topic_name: str,
    topic_type: str,
) -> bool:
    result = await simple_er.async_predict(
        {
            "name_1": relevant_topic_name,
            "type_1": relevant_topic_type,
            "name_2": topic_name,
            "type_2": topic_type,
        }
    )

    return result.merge_topic


def find_close_topic(
    simple_er: SimpleERModel,
    topic: DataScienceTopic,
    relevant_topic_name_list: list[str],
    relevant_topic_type_list: list[str],
) -> str:
    tasks = []

    for relevant_topic_name, relevant_topic_type in zip(
        relevant_topic_name_list, relevant_topic_type_list
    ):
        tasks.append(
            is_topic_close(
                simple_er,
                relevant_topic_name,
                relevant_topic_type,
                topic.name,
                topic.type.value,
            )
        )

    results = asyncio.run(asyncio.gather(*tasks))

    if True not in results:
        return None
    return relevant_topic_name_list[results.index(True)]


def eval_row(
    topics_extractor: TopicsExtractor, simple_er: SimpleERModel, row: pd.Series
) -> pd.Series:
    pred_topics = topics_extractor.predict({"content": row["content"]}).topics

    pred_topics_match = []
    precision_at_i = []
    count_type_correct = 0

    print(f"Number of predicted topics : {len(pred_topics)}")
    print(f"Number of er comparison to make : {len(pred_topics) * len(row['name'])}")

    for i, topic in enumerate(pred_topics):
        if topic.name in row["name"]:
            topic_match = topic.name
        else:
            topic_match = find_close_topic(simple_er, topic, row["name"][:1], row["type"][:1])

        pred_topics_match.append(topic_match)

        if topic_match is not None:
            precision_at_i.append(1 - (pred_topics_match.count(None) / (i + 1)))

            # Check if type is also correct
            topic_match_index = row["name"].index(topic_match)
            if topic.type.value == row["type"][topic_match_index]:
                count_type_correct += 1

    count_correct_topic = len(pred_topics) - pred_topics_match.count(None)

    if len(pred_topics) == 0:
        precision = 1
        recall = 0
        average_precision = 1
    else:
        precision = count_correct_topic / len(pred_topics)
        recall = len({name for name in pred_topics_match if name is not None}) / min(
            len(row["name"]), MAX_N_TOPICS
        )

        average_precision = sum(precision_at_i) / len(pred_topics)
        # Compute type precision

    return pd.Series(
        [
            pred_topics,
            count_correct_topic,
            precision,
            recall,
            average_precision,
            count_type_correct,
        ]
    )


# Beware invalid topic type during evaluation


def evaluate(
    model_config: ModelConfig = None,
    system_prompt: str = None,
    user_prompt: str = None,
    set_type: Literal["eval", "test"] = "eval",
):
    topics_extractor = TopicsExtractor(
        model_config=model_config, system_prompt=system_prompt, user_prompt=user_prompt
    )
    simple_er = SimpleERModel()

    if mlflow.active_run():
        topics_extractor.log_params()

    df = get_topics_extraction_data(set_type)

    df[
        [
            "pred_topics",
            "count_correct_topic",
            "precision",
            "recall",
            "average_precision",
            "count_type_correct",
        ]
    ] = df.progress_apply(
        lambda row: eval_row(topics_extractor, simple_er, row), axis=1
    )

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
