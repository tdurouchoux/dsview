

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mlflow

    from dsview.evaluation.links_extraction import (
        evaluate,
        get_links_extraction_data,
    )
    from dsview.config import (
        ModelConfig,
        LLMProvider,
        get_sqlite_url,
        load_extraction_config,
    )
    return LLMProvider, ModelConfig, evaluate, mlflow


@app.cell
def _(mlflow):
    experiment = mlflow.set_experiment("Links extraction")
    return (experiment,)


@app.cell(disabled=True)
def _(evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="Default config", experiment_id=experiment.experiment_id
    ):
        evaluate()
    return


@app.cell
def _():
    anthropic_user_prompt = """
    Your task is to carefully analyze the source content and the provided links, then select up to 5 links that are the most technically relevant to the source. Follow these guidelines:
    1. Ignore any social media links (e.g., Twitter, Facebook, LinkedIn) and internal website links.
    2. Do not include duplicate links in your selection.
    3. If no relevant links are found, return an empty list.

    Before providing your final output, you should evaluate each provided links.

    Instructions
    1. Briefly summarize the main topic(s) of the source content.
    2. List all the links that you consider technically relevant, explaining why each one is important.
    3. Evaluate each link's relevance on a scale of 1-5, with 5 being the most relevant.
    4. If you have more than 5 relevant links, explain your criteria for selecting the top 5.

    After your evaluation, provide your final selection of up to 5 links in the following format:

    - name: Name describing the content of the link
      url: URL of the link
      description: Short description of the link's content

    Here is the content of the Data Science source:

    <source_content>
    {content}
    </source_content>

    And here is the list of links found within this source:

    <source_links>
    {content_links}
    </source_links>

    Please extract the most relevant links.

    """


    system_prompt = """You are an integral part of a knowledge management system specializing in Data Science resources. Your primary function is to analyze Data Science sources and extract the most relevant technical links. These sources may cover a wide range of topics including mathematics, Python, Machine Learning, Large Language Models, Data engineering, and related subjects."""
    return anthropic_user_prompt, system_prompt


@app.cell
def _(anthropic_user_prompt, evaluate, experiment, mlflow, system_prompt):
    with mlflow.start_run(
        run_name="Anthropic user prompt 1.4",
        experiment_id=experiment.experiment_id,
    ):
        eval_results = evaluate(
            system_prompt=system_prompt,
            user_prompt=anthropic_user_prompt,
        )
    return (eval_results,)


@app.cell
def _(eval_results):
    eval_results["pred_hyperlink"].apply(len)
    return


@app.cell
def _(eval_results):
    eval_results["n_predicted_links"] = eval_results["pred_hyperlink"].apply(len)
    eval_results
    return


@app.cell
def _():
    anthropic_user_prompt_no_limit = """Your goal is to analyze Data Science sources and extract the most relevant technical links. These sources may cover a wide range of topics including mathematics, Python, Machine Learning, Large Language Models, Data engineering, and related subjects.

    Your task is to carefully analyze the source content and the provided links, then select links that are the most technically relevant to the source. Follow these guidelines:

    Instructions : 
    1. Briefly summarize the main topic(s) of the source content.
    2. List all the links that you consider technically relevant, explaining why each one is important.
    3. Evaluate each link's relevance on a scale of 1-5, with 5 being the most relevant.
    4. Exclude any social media links (e.g., Twitter, Facebook, LinkedIn) and internal website links
    5. Select the most relevant links, without duplicates. If no relevant links are found, return an empty list.

    Remenber quality is better than quantity, only return the most relevant ones.

    Here is the expected output format for each selected links : 
    - name: Name describing the content of the link
      url: URL of the link
      description: Short description of the link's content
  
    Here is the content of the Data Science source from url {url}:

    <source_content>
    {content}
    </source_content>

    And here is the list of links found within this source:

    <source_links>
    {content_links}
    </source_links>

    Please extract the relevant links
    """
    return (anthropic_user_prompt_no_limit,)


@app.cell
def _(anthropic_user_prompt_no_limit, evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="Anthropic system prompt 2.1 no limit",
        experiment_id=experiment.experiment_id,
    ):
        eval_results_no_limit = evaluate(
            system_prompt="",
            user_prompt=anthropic_user_prompt_no_limit,
        )
    return (eval_results_no_limit,)


@app.cell
def _(eval_results_no_limit):
    eval_results_no_limit
    return


@app.cell(disabled=True)
def _(
    LLMProvider,
    ModelConfig,
    anthropic_user_prompt,
    evaluate,
    experiment,
    mlflow,
):
    mistral_config = ModelConfig(
        chat_model="mistral-small-latest",
        provider=LLMProvider.MISTRAL,
        token_limit=32_000,
    )

    with mlflow.start_run(
        run_name="Mistral anthropic user prompt",
        experiment_id=experiment.experiment_id,
    ):
        evaluate(
            model_config=mistral_config,
            system_prompt="",
            user_prompt=anthropic_user_prompt,
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
