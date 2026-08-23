import marimo

__generated_with = "0.13.11"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import mlflow

    from dsview.config import (
        LLMProvider,
        ModelConfig,
    )
    from dsview.evaluation.links_extraction import (
        evaluate,
    )

    return LLMProvider, ModelConfig, evaluate, mlflow, mo


@app.cell
def _(mlflow):
    experiment = mlflow.set_experiment("Links extraction mistral")
    return (experiment,)


@app.cell
def _(mo):
    mo.md(r"""## Compare models""")


@app.cell
def _(evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="Default config", experiment_id=experiment.experiment_id
    ):
        eval_results = evaluate()
    return (eval_results,)


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
        evaluate(
            model_config=mistral_small_config,
        )
    return (mistral_small_config,)


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
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    mistral_large_config = ModelConfig(
        chat_model="mistral-large-latest",
        provider=LLMProvider.MISTRAL,
        token_limit=64_000,
    )

    with mlflow.start_run(
        run_name="Mistral large", experiment_id=experiment.experiment_id
    ):
        evaluate(
            model_config=mistral_large_config,
        )


@app.cell
def _(eval_results):
    eval_results


@app.cell
def _(eval_results):
    eval_results["pred_hyperlink"].apply(len).hist()


@app.cell
def _(eval_results_mistral_medium):
    eval_results_mistral_medium


@app.cell
def _(eval_results_mistral_medium):
    eval_results_mistral_medium["pred_hyperlink"].apply(len).hist()


@app.cell(column=1)
def _():
    ## Prompt tuning
    return


@app.cell
def _():
    run_name = "Mistral small prompt 1.3"

    system_prompt = """
    You are an advanced AI assistant integrated into a knowledge management system
    designed to help Data Scientists track and understand new trends and tools in
    their field. You are a part of multiple agents responsible for information
    extraction in an user provided content. You should be as accurate as possible,
    keeping in mind that the user is looking for relevant technical insights.
    """

    user_prompt = """
    Your task is to extract the most relevant links within a website content. The content will be related to Data Science, and I am looking for the links that will provide the most value for my knowledge management system.

    Here is the content url :
    <content_url>
    {url}
    </content_url>

    Here is the content : 
    <content>
    {content}
    </content>

    Here is the links within this source (to select from) : 
    <links>
    {content_links}
    </links>

    Instructions: 
    1. Read and analyze the provided text (within <content>) carefully.
    2. For each link (provided in <links>), describe the related content and identify what they are pointing to 
    3. Exclude any internal links or self referencing link (compared to <content_url>)
    4. Exclude any social networks links
    5. Select the most relevant links, at most 5 but you should aim for the lowest amount possible

    Provide a list of links with additionnal information, following this format : 
    <output>
    - name: Name describing the content of the first link.
      url : Url to the first link
      description : Short description of the first link content.
    - name: Name describing the content of the second link.
      url : Url to the second link
      description : Short description of the first second content.
    </output>

    Please select the most relevant links regarding the provided content

    """
    return run_name, system_prompt, user_prompt


@app.cell
def _():
    return


@app.cell(column=2)
def _(
    evaluate,
    experiment,
    mistral_small_config,
    mlflow,
    run_name,
    system_prompt,
    user_prompt,
):
    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        eval_results_prompt = evaluate(
            model_config=mistral_small_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    return (eval_results_prompt,)


@app.cell
def _(eval_results_prompt):
    eval_results_prompt["pred_hyperlink"].apply(len).hist()


@app.cell
def _(eval_results_prompt):
    eval_results_prompt


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
