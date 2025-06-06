import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _(load_extraction_config, mlflow):
    experiment = mlflow.set_experiment("Topics extraction mistral")

    extraction_config = load_extraction_config()
    return (experiment,)


@app.cell
def _():
    import marimo as mo
    import mlflow

    from dsview.evaluation.topics_extraction import (
        evaluate,
        get_topics_extraction_data,
    )
    from dsview.config import (
        ModelConfig,
        LLMProvider,
        load_extraction_config,
    )
    return (
        LLMProvider,
        ModelConfig,
        evaluate,
        load_extraction_config,
        mlflow,
        mo,
    )


@app.cell
def _(mo):
    mo.md(r"""## Compare models""")
    return


@app.cell
def _(evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="default config",
        experiment_id=experiment.experiment_id,
    ):
        eval_results = evaluate()
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
    return


@app.cell(disabled=True)
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
    return


if __name__ == "__main__":
    app.run()
