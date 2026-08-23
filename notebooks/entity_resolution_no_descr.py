import marimo

__generated_with = "0.13.2"
app = marimo.App()


@app.cell
def _():
    from dotenv import load_dotenv
    load_dotenv()
    return


@app.cell
def _():
    from dsview.config import load_extraction_config
    config = load_extraction_config()
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## No descr Er model
        """
    )
    return


@app.cell
def _():
    from openai import OpenAI
    from pydantic import BaseModel
    return BaseModel, OpenAI


@app.cell
def _(BaseModel):
    class SimpleErResult(BaseModel):
        merge_topic: bool
    return (SimpleErResult,)


@app.cell
def _(config):
    system_prompt = f"""
    You are an integral part of a knowledge management system designed to assist Data Scientists in their technical review process. Your specific task is to deduplicate Data Science topics by determining whether two given topics should be merged. This process aims to maintain a precise and non-redundant database of Data Science concepts.

    Here are the available topic types:
    <topic_tags>
    {','.join(config.topic_categories)}
    </topic_tags>

    You will be presented with two Data Science topics, each defined by a name and a type (from the list above). Your goal is to decide whether these topics should be merged based on the following rules:

    1. Topics should represent the same generic theme or tool to qualify for merging.
    2. Topics do not need to match exactly, spelling variations or slight rephrasing should be merged regardless of other differences.
    3. If one topic encompasses or intersects with the other, they should be merged.

    When analyzing the topics, complete your analysis inside <topic_comparison> tags using the following structure:

    <topic_comparison>
    1. Name Comparison:
       [Check if the names are close]

    2. Define topics
       [Define each topics]

    3. Key Concepts :
       [List key concepts from each description]

    4. Concept Overlap:
       [Identify overlapping concepts between the two topics]

    5. Significant Differences:
       [Note any significant differences between the topics]

    6. Similarity Assessment:
       [Evaluate how similar the topics are, considering their theme, tool, or concept]

    7. Confidence Level:
       [Assess your confidence in the similarity of the topics on a scale of 1-10]

    8. Merging Decision:
        [Based on the above analysis, decide whether to merge the topics or not]
    </topic_comparison>

    Here are some examples :
    <examples>
    Example 1 :
    	- Topic 1 :
    		- name : Evaluation metrics
    		- type :  Concept
    	- Topic 2 :
    		- name : Evaluation Strategies
    		- type : Concept
    	- Result :
            - merge_topic : False

    Example 2 :
    	- Topic 1 :
    		- name : Data Manipulation
    		- type :  Concept
    	- Topic 2 :
    		- name : Data Transformation
    		- type : Concept
        - Result :
            - merge_topic : True

    Example 3 :
    	- Topic 1 :
    		- name : vLLM
    		- type :  Product
    	- Topic 2 :
    		- name : vLLM
    		- type : Library
    	- Result :
            - merge_topic : True

    Example 4 :
    	- Topic 1 :
    		- name : Machine Learning Systems Design
    		- type :  Concept
    	- Topic 2 :
    		- name : Machine Learning Pipeline
    		- type : Concept
    	- Result :
            - merge_topic : False

    Example 5 :
    	- Topic 1 :
    		- name : TensorFlow.js
    		- type :  Library
    	- Topic 2 :
    		- name : TensorFlow
    		- type : Library
    	- Result :
            - merge_topic : False

    </examples>
    """
    return (system_prompt,)


@app.cell
def _():
    user_prompt = """
    Here is the first Data Science topic :
        - name : {name_1}
        - type : {type_1}

    Here is the second Data Science topic :
        - name : {name_2}
        - type : {type_2}

    Please proceed with your analysis and decision.
    """
    return (user_prompt,)


@app.cell
def _(OpenAI):
    client = OpenAI()
    return (client,)


@app.cell
def _(SimpleErResult, client, system_prompt, user_prompt):
    def predict_merge(name_1, type_1, name_2, type_2):
        prompt = user_prompt.format(name_1=name_1, type_1=type_1, name_2=name_2, type_2=type_2)
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
            ],
            response_format=SimpleErResult,
        )
        return response.choices[0].message.parsed
    return (predict_merge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Test prediction
        """
    )
    return


@app.cell
def _():
    from dsview.evaluation.entity_resolution import get_er_data
    return (get_er_data,)


@app.cell
def _(get_er_data):
    er_data = get_er_data()
    er_data
    eval_data = er_data.loc[er_data["row_type"]=="eval"]
    return (eval_data,)


@app.cell
def _(eval_data):
    eval_data
    return


@app.cell
def _(eval_data, predict_merge):
    predict_merge(
        name_1=eval_data.iloc[0]["name_1"],
        type_1=eval_data.iloc[0]["type_1"],
        name_2=eval_data.iloc[0]["name_2"],
        type_2=eval_data.iloc[0]["type_2"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Eval
        """
    )
    return


@app.cell
def _():
    import mlflow
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    experiment = mlflow.set_experiment("Topic extraction ER evaluator")
    return (
        accuracy_score,
        experiment,
        f1_score,
        mlflow,
        precision_score,
        recall_score,
    )


@app.cell
def _(
    accuracy_score,
    eval_data,
    experiment,
    f1_score,
    mlflow,
    precision_score,
    predict_merge,
    recall_score,
    system_prompt,
    user_prompt,
):
    with mlflow.start_run(run_name="default_adj_2_er_pipeline", experiment_id = experiment.experiment_id):

        mlflow.log_param("system_prompt", system_prompt)
        mlflow.log_param("user_prompt", user_prompt)

        eval_data["merge_pred"] = eval_data.apply(lambda row: predict_merge(
            name_1=row["name_1"],
            type_1=row["type_1"],
            name_2=row["name_2"],
            type_2=row["type_2"],
        ).merge_topic, axis=1)

        metrics = {
            "accuracy": accuracy_score(eval_data["merge"], eval_data["merge_pred"]),
            "precision": precision_score(eval_data["merge"], eval_data["merge_pred"]),
            "recall": recall_score(eval_data["merge"], eval_data["merge_pred"]),
            "f1": f1_score(eval_data["merge"], eval_data["merge_pred"]),
        }

        mlflow.log_metrics(metrics)
    return (metrics,)


@app.cell
def _(eval_data):
    eval_data[eval_data["merge"]!=eval_data["merge_pred"]]
    return


@app.cell
def _(metrics):
    metrics
    return


@app.cell
def _(metrics):
    metrics
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
