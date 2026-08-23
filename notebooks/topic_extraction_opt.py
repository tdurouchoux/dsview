import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import Literal

    import marimo as mo
    import mlflow
    import pandas as pd
    from tqdm import tqdm

    from dsview.config import (
        LLMProvider,
        ModelConfig,
        load_extraction_config,
    )
    from dsview.db.query import get_er_labels
    from dsview.evaluation.topics_extraction import (
        SimpleERModel,
        evaluate,
        get_topics_extraction_data,
    )

    return (
        LLMProvider,
        Literal,
        ModelConfig,
        SimpleERModel,
        evaluate,
        get_er_labels,
        get_topics_extraction_data,
        load_extraction_config,
        mlflow,
        mo,
        pd,
        tqdm,
    )


@app.cell
def _():
    import os

    os.environ["PROMPT_DIR"] = "./prompts/"


@app.cell
def _(tqdm):
    tqdm.pandas()


@app.cell
def _(mo):
    mo.md(r"""## Simple ER eval""")


@app.cell
def _(mlflow):
    simple_er_experiment = mlflow.set_experiment("Simple ER")
    return (simple_er_experiment,)


@app.cell
def _(
    SimpleERModel,
    accuracy_score,
    f1_score,
    get_er_labels,
    mlflow,
    pd,
    precision_score,
    recall_score,
):
    from typing import Literal

    SPLIT_RATIOS = {
        "example": 0.4,
        "eval": 0.5,
        "test": 0.1,
    }
    RANDOM_STATE = 42

    def evaluate_simple_er(
        simple_er: SimpleERModel, set_type: Literal["eval", "test"] = "eval"
    ):

        def get_simple_er_pred(eval_row: pd.Series) -> bool:
            result = simple_er.predict(
                {
                    "name_1": eval_row["name_1"],
                    "type_1": eval_row["type_1"],
                    "name_2": eval_row["name_2"],
                    "type_2": eval_row["type_2"],
                }
            )

            return result.merge_topic

        er_labels = get_er_labels(SPLIT_RATIOS, RANDOM_STATE)
        eval_data = er_labels[er_labels["row_type"] == set_type]

        simple_er.log_params()

        eval_data["merge_pred"] = eval_data.progress_apply(
            get_simple_er_pred,
            axis=1,
        )

        metrics = {
            "accuracy": accuracy_score(eval_data["merge"], eval_data["merge_pred"]),
            "precision": precision_score(eval_data["merge"], eval_data["merge_pred"]),
            "recall": recall_score(eval_data["merge"], eval_data["merge_pred"]),
            "f1": f1_score(eval_data["merge"], eval_data["merge_pred"]),
        }

        mlflow.log_metrics(metrics)

        print(metrics)

        return eval_data

    return (evaluate_simple_er,)


@app.cell
def _(SimpleERModel):
    simple_er = SimpleERModel()
    return (simple_er,)


@app.cell
def _(evaluate_simple_er, simple_er):
    evaluate_simple_er(simple_er, "test")


@app.cell
def _(evaluate_simple_er, mlflow, simple_er, simple_er_experiment):
    with mlflow.start_run(
        run_name="default config",
        experiment_id=simple_er_experiment.experiment_id,
    ):
        evaluate_simple_er(simple_er)


@app.cell
def _(mo):
    mo.md(r"""## Compare models""")


@app.cell
def _(load_extraction_config, mlflow):
    experiment = mlflow.set_experiment("Topics extraction mistral")

    extraction_config = load_extraction_config()
    return experiment, extraction_config


@app.cell
def _(get_topics_extraction_data):
    eval_data = get_topics_extraction_data("eval")
    return (eval_data,)


@app.cell
def _(eval_data):
    eval_data.loc[14]


@app.cell
def _(eval_data):
    eval_data["len_content"] = eval_data["content"].apply(len)


@app.cell
def _(eval_data):
    eval_data.sort_values("len_content", ascending=False)


@app.cell
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    mistral_medium_config = ModelConfig(
        chat_model="mistral-medium-3-5",
        provider=LLMProvider.MISTRAL,
        token_limit=100_000,
    )

    with mlflow.start_run(
        run_name="Mistral medium 3.5",
        experiment_id=experiment.experiment_id,
    ):
        eval_results = evaluate(model_config=mistral_medium_config)
    return (eval_results,)


@app.cell
def _(eval_results):
    eval_results["pred_name"] = eval_results["pred_topics"].apply(
        lambda topics: [topic.name for topic in topics]
    )
    eval_results[["name", "pred_name", "count_correct_topic"]]


@app.cell
def _(extraction_config):
    ", ".join(extraction_config.tags.values())


@app.cell
def _(extraction_config):
    prompt_version = "2.1.3"

    user_prompt = f"""
    Your task is to analyze text content related to Data Science and extract the most crucial and relevant topics.

    You will be provided with a text about Data Science or a related subject. Your goal is to identify and extract the main technical topics that are introduced, explained, or mentioned in the text. These topics should be highly relevant and potentially useful for a Data Scientist in their work.

    After your analysis, provide your final list of topics (at least one but at most 10). Each topic should be formatted as follows:
    - type: Type of the described topic in the source.
    - name: Name of the described topic in the source.
    - description: General and detailed description of the topic. This should not be a description of how the source discusses the topic, but rather a comprehensive explanation of the topic itself, using both the provided information and any relevant prior knowledge.

    Also, you should exclude any topic from this list (they are considered as themes and not topics) : {", ".join(extraction_config.tags.values())}

    Here are general examples or relevant topics:
    <examples>
    - Example 1 :
        type : Concept
        name : AutoML
    - Example 2 :
        type : Library
        name : duckDB
    - Example 3 :
        type : Platform
        name : AWS
    - Example 4 :
        type : Dataset
        name : IMDB Dataset
    - Example 5 :
        type : Model
        name : TimeGPT-1
    - Example 6 :
        type : Concept
        name : Knowledge Graphs
    - Example 7 :
        type : Concept
        name : Reinforcment Learning from Human Feedback (RLHF)
    </examples>

    Here is the source text for your analysis:
    <source_text>
    {{content}}
    </source_text>

    Please proceed with your analysis and topic extraction based on the provided source text.

    """
    return prompt_version, user_prompt


@app.cell
def _(
    LLMProvider,
    ModelConfig,
    evaluate,
    experiment,
    mlflow,
    prompt_version,
    user_prompt,
):
    magistral_small_config = ModelConfig(
        chat_model="magistral-small-latest",
        provider=LLMProvider.MISTRAL,
        token_limit=100_000,
    )

    with mlflow.start_run(
        run_name=f"magistral small prompt {prompt_version}",
        experiment_id=experiment.experiment_id,
    ):
        eval_results_magistral_small = evaluate(
            user_prompt=user_prompt,
            model_config=magistral_small_config,
        )
    return (eval_results_magistral_small,)


@app.cell
def _(eval_results_magistral_small, mo):
    magistral_results = mo.ui.table(eval_results_magistral_small)
    magistral_results
    return (magistral_results,)


@app.cell
def _(eval_results_magistral_small):
    eval_results_magistral_small["pred_name"] = eval_results_magistral_small[
        "pred_topics"
    ].apply(lambda topics: [topic.name for topic in topics])
    eval_results_magistral_small[["name", "pred_name"]]


@app.cell
def _(magistral_results):
    magistral_results.value


@app.cell
def _(eval_results_magistral_small):
    eval_results_magistral_small["pred_topics"].apply(len).sum()


@app.cell
def _(
    LLMProvider,
    ModelConfig,
    evaluate,
    experiment,
    mlflow,
    prompt_version,
    user_prompt,
):
    mistral_small_config = ModelConfig(
        chat_model="mistral-small-latest",
        provider=LLMProvider.MISTRAL,
        token_limit=100_000,
    )

    with mlflow.start_run(
        run_name=f"mistral small prompt {prompt_version}",
        experiment_id=experiment.experiment_id,
    ):
        eval_results_mistral_small = evaluate(
            user_prompt=user_prompt,
            model_config=mistral_small_config,
        )
    return (eval_results_mistral_small,)


@app.cell
def _(eval_results_mistral_small):
    eval_results_mistral_small["pred_name"] = eval_results_mistral_small[
        "pred_topics"
    ].apply(lambda topics: [topic.name for topic in topics])
    eval_results_mistral_small


@app.cell
def _(eval_results_mistral_small):
    eval_results_mistral_small["pred_topics"].apply(len).sum()


@app.cell
def _(eval_results):
    eval_results


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
