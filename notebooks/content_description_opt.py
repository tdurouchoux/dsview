import marimo

__generated_with = "0.15.2"
app = marimo.App(
    width="columns",
    layout_file="layouts/content_description_opt.grid.json",
)


@app.cell(column=0)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import mlflow

    from dsview.evaluation.description_generation import (
        evaluate,
        get_description_generation_data,
    )
    from dsview.config import (
        ModelConfig,
        LLMProvider,
        load_extraction_config,
    )
    from dsview.db import engine
    from dsview.model_utils.providers.ollama import OllamaProvider

    return LLMProvider, ModelConfig, evaluate, load_extraction_config, mlflow


@app.cell
def _():
    from dsview.extraction.models.description_generation import TagsType

    return (TagsType,)


@app.cell
def _(TagsType):
    ", ".join(TagsType)
    return


@app.cell
def _(mo):
    mo.md(r"""## Compare models""")
    return


@app.cell
def _(load_extraction_config, mlflow):
    experiment = mlflow.set_experiment("Content description mistral")

    extraction_config = load_extraction_config()
    return (experiment,)


@app.cell
def _():
    import os

    os.environ["PROMPT_DIR"] = "./prompts/"
    return


@app.cell
def _(evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="Mistral small 4", experiment_id=experiment.experiment_id
    ):
        eval_results = evaluate()
    return (eval_results,)


@app.cell
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    ollama_config = ModelConfig(
        chat_model="gemma3:4b",
        embedding_model="mxbai-embed-large",
        provider=LLMProvider.OLLAMA,
        token_limit=128_000,
    )

    with mlflow.start_run(
        run_name="new config gemma3", experiment_id=experiment.experiment_id
    ):
        _ = evaluate(
            model_config=ollama_config,
        )
    return


@app.cell
def _(eval_results):
    eval_results
    return


@app.cell
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    gpt_4_1_mini_config = ModelConfig(
        chat_model="gpt-4.1-mini",
        provider=LLMProvider.OPENAI,
        token_limit=128_000,
    )

    with mlflow.start_run(
        run_name="GTP 4.1 mini", experiment_id=experiment.experiment_id
    ):
        _ = evaluate(
            model_config=gpt_4_1_mini_config,
        )
    return


@app.cell
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    gpt_4_1_nano_config = ModelConfig(
        chat_model="gpt-4.1-nano",
        provider=LLMProvider.OPENAI,
        token_limit=128_000,
    )

    with mlflow.start_run(
        run_name="GTP 4.1 nano", experiment_id=experiment.experiment_id
    ):
        _ = evaluate(
            model_config=gpt_4_1_nano_config,
        )
    return


@app.cell
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    mistral_small_config = ModelConfig(
        chat_model="mistral-small-latest",
        provider=LLMProvider.MISTRAL,
        token_limit=64_000,
    )

    with mlflow.start_run(
        run_name="Mistral small", experiment_id=experiment.experiment_id
    ):
        eval_results_mistral_small = evaluate(
            model_config=mistral_small_config,
        )
    return eval_results_mistral_small, mistral_small_config


@app.cell
def _(eval_results_mistral_small):
    eval_results_mistral_small
    return


@app.cell
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    mistral_medium_config = ModelConfig(
        chat_model="mistral-medium-latest",
        provider=LLMProvider.MISTRAL,
        token_limit=64_000,
    )

    with mlflow.start_run(
        run_name="Mistral medium", experiment_id=experiment.experiment_id
    ):
        eval_results_mistral_medium = evaluate(
            model_config=mistral_medium_config,
        )
    return (eval_results_mistral_medium,)


@app.cell
def _(eval_results_mistral_medium):
    eval_results_mistral_medium
    return


@app.cell
def _(eval_results):
    eval_results["title"].apply(len).max()
    return


@app.cell
def _(ebv):
    ebv
    return


@app.cell
def _(eval_results_mistral_small):
    eval_results_mistral_small["pred_title"].apply(len).max()
    return


@app.cell
def _(eval_results):
    eval_results["pred_tags"].apply(len).plot.hist()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - mistral medium ~ gpt 4.1 mini
    - Mistral small ~ gpt 4o mini
    """
    )
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""## Prompt tuning mistral small""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Objectifs :  

    - Améliorer l'extraction de titre
        - Réduire la taille des titres
        - Améliorer le Rouge f1
    - Augmenter la précision des tags
    """
    )
    return


@app.cell
def _():
    run_name = "Mistral small prompt v1.5.4"

    system_prompt = """
    You are an expert extraction algorithm specialized in Data Science-related subjects. Your task is to analyze a given piece of content and extract key information about it.

    Please follow these steps to analyze the content and provide the required information:

    1. Title Extraction/Generation:
       - If the content has an existing title, extract it. Only take the main title, subtitles are not relevant. If the content originates from a github repository, only take the repository name.
       - Only if no title exists, generate one that accurately represents the content.
       - Ensure the title does not exceed 80 characters, trim if necessary

    2. Content Type Determination:
       - Identify the type of content from the following options: Course, Repository, Documentation, Blog post, Product main page, Scientific article
       - Choose the most appropriate type based on the content's structure and purpose.

    3. Tag Extraction:
       - Identify the main Data Science topics (focus on the primary themes, not related subjects)
       - Reduce the number of tags as much as possible to ensure precision. You can choose up to 3 tags, but most of the time one tag is enough

    Remember:
    - The title must not exceed 80 characters
    - Be as accurate as possible in determining the content type.
    - Only include the most relevant and prominent Data Science topics as tags.

    """
    return run_name, system_prompt


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


@app.cell
def _(
    evaluate,
    experiment,
    mistral_small_config,
    mlflow,
    run_name,
    system_prompt,
):
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        eval_results_mistral_small_prompt = evaluate(
            model_config=mistral_small_config, system_prompt=system_prompt
        )
    return (eval_results_mistral_small_prompt,)


@app.cell
def _(mo):
    mo.md(r"""## Tags prediction""")
    return


@app.cell
def _(eval_results):
    eval_results["pred_tags"].apply(len).plot.hist()
    return


@app.cell
def _(eval_results_mistral_small_prompt):
    eval_results_mistral_small_prompt["pred_tags"].apply(len).plot.hist()
    return


@app.cell
def _(mo):
    mo.md(r"""## Title predictions""")
    return


@app.cell
def _(eval_results_mistral_small):
    eval_results_mistral_small[["title", "pred_title"]]
    return


@app.cell
def _(eval_results_mistral_small):
    eval_results_mistral_small["pred_title"].apply(len).hist()
    return


@app.cell
def _(eval_results_mistral_small_prompt):
    eval_results_mistral_small_prompt[["title", "pred_title"]]
    return


@app.cell
def _(eval_results_mistral_small_prompt):
    eval_results_mistral_small_prompt["pred_title"].apply(len).hist()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
