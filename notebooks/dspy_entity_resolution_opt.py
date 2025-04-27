import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import mlflow

    from dsview.evaluation.entity_resolution import evaluate, get_er_data
    return evaluate, get_er_data, mlflow


@app.cell
def _(mo):
    mo.md(r"""## Current model""")
    return


@app.cell
def _(mlflow):
    experiment = mlflow.set_experiment("Entity resolution")
    return (experiment,)


@app.cell(disabled=True)
def _(evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="refresh benchmark", experiment_id=experiment.experiment_id
    ):
        evaluate()
    return


@app.cell
def _():
    from dsview.config import ModelConfig, LLMProvider
    return LLMProvider, ModelConfig


@app.cell
def _(LLMProvider, ModelConfig):
    ollama_model_config = ModelConfig(
        chat_model="gemma3:1b",
        embedding_model="all-minilm:latest",
        provider=LLMProvider.OLLAMA,
        token_limit=32_000,
    )
    return (ollama_model_config,)


@app.cell
def _(evaluate, ollama_model_config):
    evaluate(model_config=ollama_model_config)
    return


@app.cell
def _(mo):
    mo.md("""## Dspy implementation""")
    return


@app.cell
def _():
    import dspy
    return (dspy,)


@app.cell
def _(dspy):
    class EntityResolution(dspy.Signature):
        """Decide if two Data Science topics are close or not."""

        name_1: str = dspy.InputField()
        type_1: str = dspy.InputField()
        description_1: str = dspy.InputField()
        name_2: str = dspy.InputField()
        type_2: str = dspy.InputField()
        description_2: str = dspy.InputField()
        merge_topic: bool = dspy.OutputField()
    return (EntityResolution,)


@app.cell
def _(dspy):
    def get_example_set(dataset):
        example_set = []

        for row in dataset.itertuples():
            example_set.append(
                dspy.Example(
                    name_1=row.name_1,
                    type_1=row.type_1,
                    description_1=row.description_1,
                    name_2=row.name_2,
                    type_2=row.type_2,
                    description_2=row.description_2,
                    merge_topic=bool(row.merge),
                ).with_inputs(
                    "name_1",
                    "type_1",
                    "description_1",
                    "name_2",
                    "type_2",
                    "description_2",
                )
            )

        return example_set
    return (get_example_set,)


@app.cell
def _(get_er_data, get_example_set):
    er_data = get_er_data()
    train_set = get_example_set(er_data[er_data["row_type"] == "example"])
    eval_set = get_example_set(er_data[er_data["row_type"] == "eval"])
    return er_data, eval_set, train_set


@app.cell
def _(EntityResolution, dspy):
    er_classifier = dspy.Predict(EntityResolution)
    return (er_classifier,)


@app.cell
def _(dspy, er_classifier, train_set):
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        test_result = er_classifier(**train_set[0].inputs())

    test_result
    return (test_result,)


@app.cell
def equality_metric():
    def equality_metric(example, pred, trace=None):
        return example.merge_topic == pred.merge_topic
    return (equality_metric,)


@app.cell(disabled=True)
def _(dspy, equality_metric, er_classifier, train_set):
    from dspy.teleprompt import BootstrapFewShotWithRandomSearch

    optimizer_1 = BootstrapFewShotWithRandomSearch(
        metric=equality_metric,
        num_threads=3,
        max_errors=20,
    )

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        er_classifier_opt_1 = optimizer_1.compile(
            er_classifier, trainset=train_set
        )
    return BootstrapFewShotWithRandomSearch, er_classifier_opt_1, optimizer_1


@app.cell
def _(equality_metric, mlflow):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from tqdm import tqdm


    def evaluate_dspy(classifier, eval_set):
        merge_example = []
        merge_pred = []

        for example in tqdm(eval_set):
            pred = classifier(**example.inputs())
            merge_example.append(example.merge_topic)
            merge_pred.append(equality_metric(example, pred))

        metrics = {
            "accuracy": accuracy_score(merge_example, merge_pred),
            "precision": precision_score(merge_example, merge_pred),
            "recall": recall_score(merge_example, merge_pred),
            "f1": f1_score(merge_example, merge_pred),
        }

        print(metrics)

        mlflow.log_metrics(metrics)
    return (
        accuracy_score,
        evaluate_dspy,
        f1_score,
        precision_score,
        recall_score,
        tqdm,
    )


@app.cell
def _(dspy, er_classifier_opt_1, eval_set, evaluate_dspy, mlflow):
    with mlflow.start_run(run_name="dspy boostrap random search"):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
            evaluate_dspy(er_classifier_opt_1, eval_set)
    return


@app.cell
def _(er_classifier_opt_1):
    er_classifier_opt_1.save("er_classifier_opt_1.json")
    return


@app.cell(disabled=True)
def _(dspy, equality_metric, er_classifier, train_set):
    from dspy.teleprompt import MIPROv2

    optimizer_2 = MIPROv2(
        metric=equality_metric,
        prompt_model=dspy.LM("openai/gpt-4o"),
        num_threads=3,
        auto="medium",
    )

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        er_classifier_opt_2 = optimizer_2.compile(
            er_classifier, trainset=train_set
        )
    return MIPROv2, er_classifier_opt_2, optimizer_2


@app.cell
def _(dspy, er_classifier_opt_2, eval_set, evaluate_dspy, mlflow):
    with mlflow.start_run(run_name="dspy miprov2 medium"):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
            evaluate_dspy(er_classifier_opt_2, eval_set)
    return


@app.cell
def _(er_classifier_opt_2):
    er_classifier_opt_2.save("er_classifier_miprov2_medium.json")
    return


@app.cell
def _(EntityResolution, dspy):
    er_classifier_load = dspy.Predict(EntityResolution)
    er_classifier_load.load("er_classifier_miprov2_medium.json")
    return (er_classifier_load,)


@app.cell
def _(dspy, er_classifier_load, eval_set, evaluate_dspy, mlflow):
    with mlflow.start_run(run_name="dspy miprov2 medium"):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
            evaluate_dspy(er_classifier_load, eval_set)
    return


if __name__ == "__main__":
    app.run()
