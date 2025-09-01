from typing import Literal

import mlflow
import pandas as pd
from pydantic import HttpUrl
from sqlmodel import Session
from tqdm import tqdm


from dsview.config import ModelConfig
from dsview.db import engine
from dsview.db.query import get_labels_data
from dsview.db.schemas import LinksLabels
from dsview.extraction.content_loader import UrlLoader
from dsview.extraction.models import LinksExtractor

MAX_N_EXTRACTED_LINKS = 5

tqdm.pandas()


def get_links_extraction_data(set_type: Literal["test", "eval"]) -> pd.DataFrame:

    with Session(engine) as session:
        df_labels = get_labels_data([LinksLabels], set_type, session)

    df_labels = (
        df_labels.dropna(subset=["hyperlink"])
        .sort_values("rank", ascending=True)
        .groupby("id")
        .agg({"link": "first", "hyperlink": list})
    )

    return df_labels


def compute_average_precision(
    target_ranking: list[str], pred_ranking: list[str]
) -> float:
    precision_at_i = []
    for i in range(len(pred_ranking)):
        if pred_ranking[i] in target_ranking:
            precision_at_i.append(
                len(set(pred_ranking[: i + 1]).intersection(set(target_ranking)))
                / (i + 1)
            )
        else:
            precision_at_i.append(0)

    return sum(precision_at_i) / len(pred_ranking)


def eval_row(links_extractor: LinksExtractor, row: pd.Series) -> pd.Series:
    content_loader = UrlLoader(HttpUrl(row["link"]), 64_000)
    content_loader.load()

    extracted_links = [
        link.url for link in links_extractor.predict(content_loader).links
    ]
    n_extracted_links = len(extracted_links)
    relevant_links = row["hyperlink"]

    links_intersection = set(extracted_links).intersection(set(relevant_links))

    if n_extracted_links == 0:
        precision = 1
        recall = 0
        average_precision = 1
    else:
        precision = len(links_intersection) / n_extracted_links
        # recall = len(links_intersection) / min(len(relevant_links), MAX_N_EXTRACTED_LINKS)
        recall = len(links_intersection) / len(relevant_links)
        average_precision = compute_average_precision(relevant_links, extracted_links)

    return pd.Series([extracted_links, precision, recall, average_precision])


def evaluate(
    model_config: ModelConfig = None,
    system_prompt: str = None,
    user_prompt: str = None,
    set_type: Literal["eval", "test"] = "eval",
):
    links_extractor = LinksExtractor(
        model_config, system_prompt=system_prompt, user_prompt=user_prompt
    )

    if mlflow.active_run():
        links_extractor.log_params()

    df = get_links_extraction_data(set_type)

    # TODO remove token_limit from model_config
    df[["pred_hyperlink", "precision", "recall", "average_precision"]] = (
        df.progress_apply(lambda row: eval_row(links_extractor, row), axis=1)
    )

    metrics = {
        "mean_precision": df["precision"].mean(),
        "mean_recall": df["recall"].mean(),
        "mean_average_precision": df["average_precision"].mean(),
    }

    print({key: f"{value:.2f}" for key, value in metrics.items()})

    if mlflow.active_run():
        mlflow.log_metrics(metrics)

    return df
