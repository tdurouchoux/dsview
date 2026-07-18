"""CLI entry points for the evaluation modules.

Each command runs one task's evaluate() inside an MLflow run. An optional
YAML file overrides the pipeline defaults, e.g.:

    model_config:            # built as a full ModelConfig
      chat_model: mistral-medium-2508
      provider: MISTRAL
      token_limit: 100000
    system_prompt_file: experiments/system_topics_v2.txt
    user_prompt_file: experiments/user_topics_v2.txt

All keys are optional; anything omitted falls back to the pipeline default
(LLMModel handles the coalescing). Without a YAML file, the default pipeline
configuration is evaluated.
"""

import json
import logging
import tempfile
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import mlflow
import pandas as pd
import typer
from pydantic import BaseModel

from dsview.config import ModelConfig, load_evaluation_config

logger = logging.getLogger(__name__)


def _to_parquet_safe(value):
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, list):
        return json.dumps(
            [
                item.model_dump() if isinstance(item, BaseModel) else item
                for item in value
            ]
        )
    return value


def _holds_pydantic_model(value) -> bool:
    if isinstance(value, BaseModel):
        return True
    if isinstance(value, list):
        return any(isinstance(item, BaseModel) for item in value)
    return False


def _prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize columns holding pydantic model outputs (e.g. pred_topics) so
    to_parquet doesn't choke on Arrow's inability to infer their type."""
    df = df.copy()
    for col in df.columns:
        if df[col].map(_holds_pydantic_model).any():
            df[col] = df[col].map(_to_parquet_safe)
    return df

evaluate_app = typer.Typer(
    help="Run model evaluations on the labelled datasets (results go to MLflow)"
)

CONFIG_FILE_ARGUMENT = typer.Argument(
    None,
    help="YAML file overriding model config and/or prompt files "
    "(default: pipeline configuration)",
)


class SetType(str, Enum):
    EVAL = "eval"
    TEST = "test"


SET_TYPE_OPTION = typer.Option(
    SetType.EVAL,
    help="Labelled split to evaluate on; keep 'test' for final measurements",
)


def load_eval_overrides(
    config_file: Path,
) -> tuple[Optional[ModelConfig], Optional[str], Optional[str]]:
    eval_config = load_evaluation_config(config_file)

    system_prompt = (
        eval_config.system_prompt_file.read_text()
        if eval_config.system_prompt_file is not None
        else None
    )
    user_prompt = (
        eval_config.user_prompt_file.read_text()
        if eval_config.user_prompt_file is not None
        else None
    )

    return eval_config.model_config, system_prompt, user_prompt


def run_evaluation(
    evaluate_fn: Callable,
    experiment: str,
    config_file: Optional[Path],
    set_type: SetType,
    run_name: Optional[str] = None,
    **evaluate_kwargs,
):
    model_config = system_prompt = user_prompt = None

    if config_file is not None:
        model_config, system_prompt, user_prompt = load_eval_overrides(config_file)

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("set_type", set_type.value)
        if config_file is not None:
            mlflow.log_param("eval_config_file", str(config_file))

        result_df = evaluate_fn(
            model_config=model_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            set_type=set_type.value,
            **evaluate_kwargs,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "eval_results.parquet"
            _prepare_for_parquet(result_df).to_parquet(result_path)
            mlflow.log_artifact(str(result_path))


@evaluate_app.command(help="Evaluate topic extraction (fans out an LLM judge)")
def topics(
    config_file: Optional[Path] = CONFIG_FILE_ARGUMENT,
    set_type: SetType = SET_TYPE_OPTION,
    experiment: str = typer.Option("Topics extraction", help="MLflow experiment name"),
    run_name: Optional[str] = typer.Option(None, help="MLflow run name"),
):
    from dsview.evaluation import topics_extraction

    run_evaluation(
        topics_extraction.evaluate, experiment, config_file, set_type, run_name
    )


@evaluate_app.command(help="Evaluate links extraction (fetches the labelled URLs)")
def links(
    config_file: Optional[Path] = CONFIG_FILE_ARGUMENT,
    set_type: SetType = SET_TYPE_OPTION,
    experiment: str = typer.Option("Links extraction", help="MLflow experiment name"),
    run_name: Optional[str] = typer.Option(None, help="MLflow run name"),
):
    from dsview.evaluation import links_extraction

    run_evaluation(
        links_extraction.evaluate, experiment, config_file, set_type, run_name
    )


@evaluate_app.command(help="Evaluate content description generation")
def description(
    config_file: Optional[Path] = CONFIG_FILE_ARGUMENT,
    set_type: SetType = SET_TYPE_OPTION,
    experiment: str = typer.Option(
        "Content description", help="MLflow experiment name"
    ),
    run_name: Optional[str] = typer.Option(None, help="MLflow run name"),
):
    from dsview.evaluation import description_generation

    run_evaluation(
        description_generation.evaluate, experiment, config_file, set_type, run_name
    )


@evaluate_app.command(help="Evaluate entity resolution classification")
def er(
    config_file: Optional[Path] = CONFIG_FILE_ARGUMENT,
    set_type: SetType = SET_TYPE_OPTION,
    experiment: str = typer.Option("Entity resolution", help="MLflow experiment name"),
    run_name: Optional[str] = typer.Option(None, help="MLflow run name"),
    remove_descr: bool = typer.Option(
        False, help="Blank out topic descriptions before classification"
    ),
    sync: bool = typer.Option(
        False, help="Use synchronous API calls instead of the provider batch API"
    ),
):
    import os

    from dsview.evaluation import entity_resolution
    from dsview.model_utils.model_provider import DISABLE_NATIVE_BATCH_ENV_VAR

    if sync:
        os.environ[DISABLE_NATIVE_BATCH_ENV_VAR] = "1"

    run_evaluation(
        entity_resolution.evaluate,
        experiment,
        config_file,
        set_type,
        run_name,
        remove_descr=remove_descr,
    )
