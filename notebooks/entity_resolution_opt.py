import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from dotenv import load_dotenv
    import marimo as mo
    import mlflow

    from dsview.evaluation.entity_resolution import evaluate
    from dsview.db import engine
    from dsview.config import LLMProvider, ModelConfig, load_extraction_config

    return (
        LLMProvider,
        ModelConfig,
        engine,
        evaluate,
        load_dotenv,
        load_extraction_config,
        mlflow,
        mo,
    )


@app.cell
def _(load_dotenv):
    load_dotenv()
    return


@app.cell
def _(load_extraction_config, mlflow):
    experiment = mlflow.set_experiment("Entity Resolution mistral")

    extraction_config = load_extraction_config()
    return experiment, extraction_config


@app.cell
def _(engine, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM labels.erlabels
        """,
        engine=engine
    )
    return


@app.cell
def _(engine, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM extraction.ercomparison
        """,
        engine=engine
    )
    return


@app.cell
def _(extraction_config):
    extraction_config.topic_categories.values()
    return


@app.cell
def _(eval_result):
    eval_result
    return


@app.cell
def _(evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="Default config",
        experiment_id=experiment.experiment_id
    ):
        eval_results = evaluate()
    return (eval_results,)


@app.cell
def _(eval_results):
    eval_results
    return


@app.cell
def _(eval_results):
    eval_results[eval_results["merge"]!=eval_results["merge_pred"]]
    return


@app.cell
def _():
    return


@app.cell
def _():
    prompt_version = "2.0"

    system_prompt = """
    You are an advanced AI assistant integrated into a knowledge management system
    designed to help Data Scientists track and understand new trends and tools in
    their field. You are a part of multiple agents responsible for information
    extraction in an user provided content. You should be as accurate as possible,
    keeping in mind that the user is looking for relevant technical insights.
    """

    user_prompt = """
    You are an integral part of a knowledge management system designed to assist Data Scientists in their technical review process. Your specific task is to deduplicate Data Science topics by determining whether two given topics should be merged. This process aims to maintain a precise and non-redundant database of Data Science concepts.

    Here are the available topic types:
    <topic_tags>
    Library, Tool, Model, Platform, Concept, Dataset
    </topic_tags>

    You will be presented with two Data Science topics, each defined by a name, type (from the list above), and description. Your goal is to decide whether these topics should be merged based on the following rules:

    1. Topics should represent the same generic theme or tool to qualify for merging.
    2. Topics do not need to match exactly, spelling variations or slight rephrasing should be merged regardless of other differences.
    3. If one topic encompasses or significantly intersects with the other, they should be merged.
    4. If the two topics share the same root (e.g., the same library) but describe distinct concepts within this subject, they should NOT be merged.

    When analyzing the topics, complete your analysis inside <topic_comparison> tags using the following structure:

    <topic_comparison>
    1. Name Comparison:
       [Check if the names are close]

    2. Key Concepts :
       [List key concepts from each description]

    3. Concept Overlap:
       [Identify overlapping concepts between the two topics]

    4. Significant Differences:
       [Note any significant differences between the topics]

    5. Similarity Assessment:
       [Evaluate how similar the topics are, considering their theme, tool, or concept]

    6. Confidence Level:
       [Assess your confidence in the similarity of the topics on a scale of 1-10]

    7. Merging Decision:
        [Based on the above analysis, decide whether to merge the topics or not]
    </topic_comparison>

    The output should take the following format :
    <output>
        merge_topic (bool): Wether or not the two provided topics should be merged.
        topic : Result of the merge between the two topics, only provided if topics should be merged (None if merge_topic is False)
            name : Merge topic name
            type : Merged topic type from topic_tags
            description : Merged topic description
    </output>


    Here are some examples :
    <examples>
    Example 1 :
    	- Topic 1 :
    		- name : Evaluation metrics
    		- type :  Concept
    		- description : Methods and criteria used to assess the performance and quality of generated content, particularly in the context of language models.
    	- Topic 2 :
    		- name : Evaluation Strategies
    		- type : Concept
    		- description : Methods and techniques for assessing the performance and output quality of language models.
    	- Result :
            - merge_topic : False

    Example 2 :
    	- Topic 1 :
    		- name : Data Manipulation
    		- type :  Concept
    		- description : The process of adjusting, organizing, and transforming data to make it more suitable for analysis.
    	- Topic 2 :
    		- name : Data Transformation
    		- type : Concept
    		- description : The process of converting data from one format or structure into another, often a key step in data pipelines.
        - Result :
            - merge_topic : True
            - topic :
                - name : Data Transformation
                - type : Concept
                - description : The process of transforming and organizing data to make it more suitable for analysis or consumption, often a key step in data pipelines.

    Example 3 :
    	- Topic 1 :
    		- name : vLLM
    		- type :  Product
    		- description : A serving framework for large language models that includes caching mechanisms and optimizations for improved inference throughput.
    	- Topic 2 :
    		- name : vLLM
    		- type : Library
    		- description : A fast and easy-to-use library for LLM inference and serving, featuring state-of-the-art serving throughput and efficient memory management.
    	- Result :
            - merge_topic : True
            - topic :
                - name : vLLM
                - type : Library
                - description : An easy-to-use serving framework for large language models, featuring state-of-the-art performances and capabilities such
                as caching mechanisms and efficient memory management.

    Example 4 :
    	- Topic 1 :
    		- name : Machine Learning Systems Design
    		- type :  Concept
    		- description : The process of defining the software architecture, infrastructure, algorithms, and data for a machine learning system to satisfy specified requirements.
    	- Topic 2 :
    		- name : Machine Learning Pipeline
    		- type : Concept
    		- description : The end-to-end process of building and deploying machine learning models.
    	- Result :
            - merge_topic : False

    Example 5 :
    	- Topic 1 :
    		- name : TensorFlow.js
    		- type :  Library
    		- description : A JavaScript library for machine learning that can run in the browser and on Node.js.
    	- Topic 2 :
    		- name : TensorFlow
    		- type : Library
    		- description : An open-source library for machine learning and deep learning applications, including a framework for implementing federated learning and federated analytics.
    	- Result :
            - merge_topic : False

    </examples>

    Here is the first Data Science topic : 
        - name : {name_1}
        - type : {type_1}
        - description : {description_1}

    Here is the second Data Science topic :
        - name : {name_2}
        - type : {type_2}
        - description : {description_2}

    Please proceed with your analysis and decision.

    """
    return prompt_version, system_prompt, user_prompt


@app.cell
def _(
    LLMProvider,
    ModelConfig,
    evaluate,
    experiment,
    mlflow,
    prompt_version,
    system_prompt,
    user_prompt,
):
    mistral_small_config = ModelConfig(
        chat_model="mistral-small-latest",
        provider=LLMProvider.MISTRAL,
        token_limit=64_000,
    )

    with mlflow.start_run(
            run_name=f"Mistral small prompt {prompt_version}", experiment_id=experiment.experiment_id
        ):
        eval_results_mistral_small = evaluate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_config=mistral_small_config,
        )
    return (mistral_small_config,)


@app.cell
def _(evaluate, experiment, mistral_small_config, mlflow):
    with mlflow.start_run(
            run_name=f"Mistral small default", experiment_id=experiment.experiment_id
        ):
        eval_results_mistral_small_default = evaluate(
            # system_prompt=system_prompt,
            # user_prompt=user_prompt,
            model_config=mistral_small_config,
        )
    return (eval_results_mistral_small_default,)


@app.cell
def _(eval_results, eval_results_mistral_small_default):
    eval_results[(eval_results_mistral_small_default["merge_pred"] != eval_results["merge_pred"])]
    return


@app.cell
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    anthropic_config = ModelConfig(
        chat_model='claude-sonnet-4-20250514',
        provider=LLMProvider.ANTHROPIC,
        model_specific_config={"max_tokens": 1_024},
    )

    with mlflow.start_run(
            run_name="Anthropic", experiment_id=experiment.experiment_id
        ):
        evaluate(
            model_config=anthropic_config,
        )
    return


app._unparsable_cell(
    r"""
    sprompt_version = \"2.0\"


    system_prompt = \"\"\"
    \"\"\"

    user_prompt = f\"\"'
    \"\"\"
    """,
    name="_"
)


@app.cell
def _(evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="Default config",
        experiment_id=experiment.experiment_id
    ):
        evaluate()
    return


if __name__ == "__main__":
    app.run()
