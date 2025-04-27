

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


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
        get_sqlite_url,
        load_extraction_config,
    )
    return ModelConfig, evaluate, load_extraction_config, mlflow, mo


@app.cell
def _(load_extraction_config, mlflow):
    experiment = mlflow.set_experiment("Topics extraction")

    extraction_config = load_extraction_config()
    return experiment, extraction_config


@app.cell
def _(evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="New Default config", experiment_id=experiment.experiment_id
    ):
        eval_results = evaluate()
    return


@app.cell(disabled=True)
def _(ModelConfig, evaluate, experiment, mlflow):
    gpt_4_1_config = ModelConfig(
        chat_model="gpt-4.1-nano",
        token_limit=128_000,
    )

    with mlflow.start_run(
        run_name="4.1 default config", experiment_id=experiment.experiment_id
    ):
        print(evaluate(model_config=gpt_4_1_config))
    return


@app.cell
def _():
    system_prompt = """
    You are an advanced AI assistant integrated into a knowledge management system designed to help Data Scientists track and understand new trends and tools in their field. You are a part of multiple agents responsible for information extraction in an user provided content. You should be as accurate as possible, keeping in mind that the user is looking for relevant technical insights.
    """
    return (system_prompt,)


@app.cell
def _(extraction_config):
    anthropic_user_prompt = """
    Your task is to analyze text content related to Data Science and extract the most crucial and relevant topics.

    You will be provided with a text about Data Science or a related subject. Your goal is to identify and extract the main technical topics that are introduced, explained, or mentioned in the text. These topics should be highly relevant and potentially useful for a Data Scientist in their work.

    Here is the source text for your analysis:
    <source_text>
    {content}
    </source_text>

    Instructions:
    1. Read and analyze the provided text carefully.
    2. Identify the main technical topics, focusing on:
       - Concepts or techniques introduced or explained
       - Specific practical tools or platform mentioned
       - Names of libraries or products (especially if the text is from a repository main page)
    3. Ensure that the extracted topics are as generic as possible, unless they refer to a specific tool or product.
    4. Ignore too broad topics that are included in : Deep Learning, Cloud Computing, Statistics, Mathematics, MlOps, Data Engineering, Feature Engineering, Data Analysis, Data Visualization, DevOps, Python, Large Language Model, AI agent, Natural Language Processing, Time Series, Graph, Computer Vision, Supervised Learning, Unsupervised Learning, Semi-supervised Learning, Dimensionality reduction, Active Learning, AI regulation, Data Quality, Model evaluation, Development tool, Data teams management
    5. Ensure there are no duplicate topics in your list.
    6. Select the most relevant topics

    After your analysis, provide your final list of topics. Each topic should be formatted as follows:
    - type: Type of the described topic in the source.
    - name: Name of the described topic in the source.
    - description: General and detailed description of the topic. This should not be a description of how the source discusses the topic, but rather a comprehensive explanation of the topic itself, using both the provided information and any relevant prior knowledge.

    Remember, it's better to return fewer topics if there aren't enough highly relevant and crucial topics in the text. Quality and relevance are more important than quantity.

    Please proceed with your analysis and topic extraction based on the provided source text.

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

    Remember, it's better to return fewer topics if there aren't enough highly relevant and crucial topics in the text. Quality and relevance are more important than quantity.

    Please proceed with your analysis and topic extraction based on the provided source text.
    """

    anthropic_user_prompt = anthropic_user_prompt.format(
        topic_categories=", ".join(extraction_config.tags.values()),
        content="{content}",
    )
    return (anthropic_user_prompt,)


@app.cell(disabled=True)
def _(anthropic_user_prompt, evaluate, experiment, mlflow, system_prompt):
    with mlflow.start_run(
        run_name="anthropic user prompt 1.3e 10 + system prompt",
        experiment_id=experiment.experiment_id,
    ):
        eval_results_2 = evaluate(
            system_prompt=system_prompt,
            user_prompt=anthropic_user_prompt,
        )
    return (eval_results_2,)


@app.cell
def _(eval_results_2):
    eval_results_2
    return


@app.cell
def _(eval_results_2):
    eval_results_2["pred_topics"].apply(len)
    return


@app.cell
def _(eval_results_2):
    eval_results_2["pred_topics"].apply(len).sum() / 28
    return


@app.cell
def _(mo):
    mo.md(r"""Check metric validity > when total number of matched topics increase, metrics decreased""")
    return


@app.cell
def _(mo):
    mo.md(r"""> No improvments yet""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
