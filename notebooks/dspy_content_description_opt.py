import marimo

__generated_with = "0.11.20"
app = marimo.App(
    width="medium",
    layout_file="layouts/content_description_opt.grid.json",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from dsview.config import LLMProvider, ModelConfig, get_sqlite_url
    from dsview.model_utils.providers.ollama import OllamaProvider

    return (
        LLMProvider,
        ModelConfig,
        OllamaProvider,
        get_sqlite_url,
    )


@app.cell
def _():
    from typing import Literal

    import dspy

    from dsview.config import load_extraction_config

    return Literal, dspy, load_extraction_config


@app.cell
def _(mo):
    mo.md(r"""### Model definition""")


@app.cell
def _(load_extraction_config):
    extraction_config = load_extraction_config()
    return (extraction_config,)


@app.cell
def _(Literal, dspy, extraction_config):
    class ContentDescription(dspy.Signature):
        content: str = dspy.InputField()
        title: str = dspy.OutputField()
        content_type: Literal[*extraction_config.content_types.values()] = (
            dspy.OutputField()
        )
        tags: list[Literal[*extraction_config.tags.values()]] = dspy.OutputField()

    return (ContentDescription,)


@app.cell
def _(eval_data):
    test_content = eval_data.loc[0, "content"]
    return (test_content,)


@app.cell
def _(ContentDescription, dspy):
    content_descriptor = dspy.Predict(
        ContentDescription,
    )
    return (content_descriptor,)


@app.cell
def _(content_descriptor, dspy, test_content):
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        content_description = content_descriptor(content=test_content)
    return (content_description,)


@app.cell
def _(content_description):
    content_description


@app.cell
def _(eval_data):
    eval_data.drop(columns=["content"]).loc[0]


@app.cell
def _(mo):
    mo.md(r"""### Generate data""")


@app.cell
def _(get_description_generation_data):
    eval_data = get_description_generation_data("eval").reset_index(drop=True)
    return (eval_data,)


@app.cell
def _(dspy, eval_data):
    datasets = []

    for row in eval_data.itertuples():
        datasets.append(
            dspy.Example(
                content=row.content,
                title=row.title,
                content_type=row.content_type,
                tags=row.tag,
            ).with_inputs("content")
        )
    return datasets, row


@app.cell
def _(datasets):
    len(datasets)


@app.cell
def _(mo):
    mo.md(r"""### Metric definition""")


@app.cell
def _(mo):
    mo.md(
        """
        Compute aggregate f1 score:
        - semantic score for title (could be replaced with a dspy module judge)
        - equality for content type
        - F1 for tags
        """
    )


@app.cell
def _():
    from dsview.evaluation.semantic_score import semantic_score

    return (semantic_score,)


app._unparsable_cell(
    r"""
    def aggregate_f1_score(example, pred, trace=None):
        title_semantic_f1 = semantic_score(example.title, pred.title)[\"f1\"]

        correct_content_type =  .content_type == pred.content_type

        actual_tags = set(example.tags)
        pred_tags = set(pred.tags)

        if len(pred_tags) == 0:
            tag_precision = 0
            tag_recall = 0
        else:
            tags_intersection = pred_tags.intersection(actual_tags)

            tag_precision = len(tags_intersection) / len(pred_tags)
            tag_recall = len(tags_intersection) / len(actual_tags)

        if tag_precision == 0 and tag_precision == 0:
            tag_f1 = 0
        else:
            tag_f1 = (
                2 * (tag_precision * tag_recall) / (tag_precision + tag_recall)
            )

        return (title_semantic_f1 + correct_content_type + tag_f1) / 3
    """,
    name="_",
)


@app.cell
def _():
    return


@app.cell
def _(aggregate_f1_score, content_descriptor, datasets, dspy):
    test_example = datasets[1]

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        test_result = content_descriptor(**test_example.inputs())
        test_score = aggregate_f1_score(test_example, test_result)

    test_score
    return test_example, test_result, test_score


@app.cell
def _(test_example):
    test_example.labels()


@app.cell
def _(test_result):
    test_result


@app.cell
def _(mo):
    mo.md(r"""### Test optimization""")


@app.cell
def _(datasets):
    import random
    from math import ceil

    train_size = ceil(len(datasets) * 0.3)
    random.shuffle(datasets)
    train_set = datasets[:train_size]
    eval_set = datasets[train_size:]
    return ceil, eval_set, random, train_set, train_size


@app.cell
def _(aggregate_f1_score, dspy):
    from tqdm import tqdm

    def evaluate_set(dataset, predictor, batch_size: int = 5):
        scores = []

        for example in tqdm(dataset):
            with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
                test_result = predictor(**example.inputs())
                scores.append(aggregate_f1_score(example, test_result))

        return sum(scores) / len(scores)

    return evaluate_set, tqdm


@app.cell
def _(content_descriptor, evaluate_set, train_set):
    evaluate_set(train_set, content_descriptor)


@app.cell
def _(content_descriptor, eval_set, evaluate_set):
    evaluate_set(eval_set, content_descriptor)


@app.cell(disabled=True)
def _(aggregate_f1_score, content_descriptor, dspy, train_set):
    optimizer = dspy.BootstrapFewShot(metric=aggregate_f1_score)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        content_descriptor_opt = optimizer.compile(
            content_descriptor.deepcopy(), trainset=train_set
        )
    return content_descriptor_opt, optimizer


@app.cell
def _(content_descriptor_opt, evaluate_set, train_set):
    evaluate_set(train_set, content_descriptor_opt)


@app.cell
def _(content_descriptor_opt, eval_set, evaluate_set):
    evaluate_set(eval_set, content_descriptor_opt)


@app.cell(disabled=True)
def _(aggregate_f1_score, content_descriptor, dspy, train_set):
    from dspy.teleprompt import MIPROv2

    miprov2_optimizer = MIPROv2(
        metric=aggregate_f1_score,
        auto="light",
        max_bootstrapped_demos=0,  # ZERO FEW-SHOT EXAMPLES
        max_labeled_demos=0,  # ZERO FEW-SHOT EXAMPLES)
    )

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        content_descriptor_miprov2_opt = miprov2_optimizer.compile(
            content_descriptor.deepcopy(),
            trainset=train_set,
        )
    return MIPROv2, content_descriptor_miprov2_opt, miprov2_optimizer


@app.cell
def _(content_descriptor_miprov2_opt, evaluate_set, train_set):
    evaluate_set(train_set, content_descriptor_miprov2_opt)


@app.cell
def _(content_descriptor_miprov2_opt, eval_set, evaluate_set):
    evaluate_set(eval_set, content_descriptor_miprov2_opt)


@app.cell
def _(content_descriptor_miprov2_opt):
    content_descriptor_miprov2_opt.save("content_description_miprov2_opt.json")


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
