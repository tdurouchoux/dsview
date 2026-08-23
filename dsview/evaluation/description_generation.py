from typing import Literal

import mlflow
import pandas as pd
from rouge import Rouge
from sqlmodel import Session
from tqdm import tqdm

from dsview.config import ModelConfig
from dsview.db import engine
from dsview.db.query import get_labels_data
from dsview.db.schemas import ContentTypeLabels, TagLabels, TitleLabels
from dsview.extraction.models import DescriptionGenerator

from .semantic_score import semantic_score

tqdm.pandas()


def get_description_generation_data(set_type: Literal["eval", "test"]) -> pd.DataFrame:
    with Session(engine) as session:
        df_labels = get_labels_data(
            [ContentTypeLabels, TagLabels, TitleLabels], set_type, session
        )

    df_labels = df_labels.groupby("id").agg(
        {"content": "first", "title": "first", "content_type": "first", "tag": list}
    )

    return df_labels


def score_row(row: pd.Series) -> pd.Series:
    pred_content_description = row["pred_content_description"]

    rouge = Rouge()
    title_rouge_f1 = rouge.get_scores(pred_content_description.title, row["title"])[0][
        "rouge-1"
    ]["f"]
    title_semantic_f1 = semantic_score(pred_content_description.title, row["title"])[
        "f1"
    ]

    correct_content_type = pred_content_description.content_type == row["content_type"]

    pred_tags = {tag.name.value for tag in pred_content_description.tags}
    actual_tags = set(row["tag"])
    if len(pred_tags) == 0:
        tag_precision = 0
        tag_recall = 0
    else:
        tags_intersection = pred_tags.intersection(actual_tags)

        tag_precision = len(tags_intersection) / len(pred_tags)
        tag_recall = len(tags_intersection) / len(actual_tags)

    return pd.Series(
        [
            pred_content_description.title,
            pred_content_description.content_type,
            pred_content_description.tags,
            title_rouge_f1,
            title_semantic_f1,
            correct_content_type,
            tag_precision,
            tag_recall,
        ]
    )


def evaluate(
    model_config: ModelConfig | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    set_type: Literal["eval", "test"] = "eval",
):
    description_generator = DescriptionGenerator(
        model_config=model_config, system_prompt=system_prompt, user_prompt=user_prompt
    )

    if mlflow.active_run():
        description_generator.log_params()

    df_eval = get_description_generation_data(set_type)

    df_eval["pred_content_description"] = description_generator.predict_batch(
        [{"content": content} for content in df_eval["content"]]
    )

    df_eval[
        [
            "pred_title",
            "pred_content_type",
            "pred_tags",
            "title_rouge_f1",
            "title_semantic_f1",
            "correct_content_type",
            "tag_precision",
            "tag_recall",
        ]
    ] = df_eval.progress_apply(score_row, axis=1)

    metrics = {
        f"mean_{col}": df_eval[col].mean()
        for col in [
            "title_rouge_f1",
            "title_semantic_f1",
            "tag_precision",
            "tag_recall",
        ]
    }

    metrics["content_type_accuracy"] = (
        df_eval["correct_content_type"].sum() / df_eval.shape[0]
    )

    print({key: f"{value:.2f}" for key, value in metrics.items()})

    if mlflow.active_run():
        mlflow.log_metrics(metrics)

    return df_eval
