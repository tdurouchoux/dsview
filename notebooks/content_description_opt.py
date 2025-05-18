

import marimo

__generated_with = "0.13.0"
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
    import mlflow

    from dsview.evaluation.description_generation import (
        evaluate,
        get_description_generation_data,
    )
    from dsview.config import (
        ModelConfig,
        LLMProvider,
        get_sqlite_url,
        load_extraction_config,
    )
    from dsview.model_utils.providers import OllamaProvider
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
def _(load_extraction_config, mlflow):
    experiment = mlflow.set_experiment("Content description")

    extraction_config = load_extraction_config()
    return experiment, extraction_config


@app.cell(disabled=True)
def _(evaluate, experiment, mlflow):
    with mlflow.start_run(
        run_name="Default config", experiment_id=experiment.experiment_id
    ):
        eval_results = evaluate()
    return


@app.cell
def _(mo):
    mo.md(r"""## Model testing""")
    return


@app.cell(disabled=True)
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    gpt4o_config = ModelConfig(
        chat_model="gpt-4o",
        embedding_model="text-embedding-3-small",
        provider=LLMProvider.OPENAI,
        token_limit=128_000,
    )

    with mlflow.start_run(
        run_name="Default config GPT4o", experiment_id=experiment.experiment_id
    ):
        evaluate(model_config=gpt4o_config)
    return


@app.cell(disabled=True)
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    mistral_small_config = ModelConfig(
        chat_model="mistral-small-latest",
        provider=LLMProvider.MISTRAL,
        token_limit=32_000,
    )

    with mlflow.start_run(
        run_name="Default config mistral small",
        experiment_id=experiment.experiment_id,
    ):
        evaluate(model_config=mistral_small_config)
    return


@app.cell(disabled=True)
def _(LLMProvider, ModelConfig, evaluate, experiment, mlflow):
    anthropic_config = ModelConfig(
        chat_model="claude-3-5-haiku-20241022",
        provider=LLMProvider.ANTHROPIC,
        token_limit=200_000,
        model_specific_config={"max_tokens": 8_192},
    )

    with mlflow.start_run(
        run_name="Default config anthropic haiku",
        experiment_id=experiment.experiment_id,
    ):
        evaluate(model_config=anthropic_config)
    return (anthropic_config,)


@app.cell
def _():
    ## System prompt opt
    return


@app.cell
def _():
    system_prompt = """
    You are an expert extraction algorithm specialized in Data Science-related subjects. Your task is to analyze a given piece of content and extract key information about it.

    Please follow these steps to analyze the content and provide the required information:

    1. Title Extraction/Generation:
       - If the content has an existing title, extract it, you are not allowed any rephrasing.
       - Only if no title exists, generate one that accurately represents the content.
       - Ensure the title does not exceed 80 characters.

    2. Content Type Determination:
       - Identify the type of content from the following options: Course, Repository, Documentation, Blog post, Product main page, Scientific article
       - Choose the most appropriate type based on the content's structure and purpose.

    3. Tag Extraction:
       - Identify the main Data Science topics (focus on the primary themes, not related subjects)
       - Reduce the number of tags as much as possible to ensure precision (aim for 1-3 tags maximum).

    Remember:
    - Be as accurate as possible in determining the content type.
    - Only include the most relevant and prominent Data Science topics as tags.


    Here as some examples (only extracts):
    <examples>
    Example 1 :

    - content :
        Building effective agents
        Published Dec 19, 2024

        We've worked with dozens of teams building LLM agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

        Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

        In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.
    - Output :

        - title : Building effective agents
        - content_type : Blog post
        - tags : Large Language Model

    Example 2 :
    - Content :
        DuckDB is a fast
        in-process|
        database system
        Query and transform your data anywhere
        using DuckDB's feature-rich SQL dialect

        Installation Documentation

        Live demo
        DuckDB at a glance
        Simple
        DuckDB is easy to install and deploy. It has zero external dependencies and runs in-process in its host application or as a single binary.

        Read more
        Portable
        DuckDB runs on Linux, macOS, Windows, Android, iOS and all popular hardware architectures. It has idiomatic client APIs for major programming languages.

        Read more
        Feature-rich
        DuckDB offers a rich SQL dialect. It can read and write file formats such as CSV, Parquet, and JSON, to and from the local file system and remote endpoints such as S3 buckets.

        Read more
        Fast
        DuckDB runs analytical queries at blazing speed thanks to its columnar engine, which supports parallel execution and can process larger-than-memory workloads.

        Read more
        Extensible
        DuckDB is extensible by third-party features such as new data types, functions, file formats and new SQL syntax. User contributions are available as community extensions.

        Read more
        Free
        DuckDB and its core extensions are open-source under the permissive MIT License. The intellectual property of the project is held by the DuckDB Foundation.

        Read more
    - Output :
        - title : DuckDB is a fast in-process database system
        - content_type :  Product main page
        - tags : Data Engineering

    Example 3 :
    - Content :
        Owner avatar
        timely-dataflow
        Public
        TimelyDataflow/timely-dataflow
        Go to file
        t
        Name
        frankmcsherry
        frankmcsherry
        Linear connectivity (#651)
        d0ea86f
         ·
        5 days ago
        .github
        Add support for release-plz (#548)
        5 months ago
        bytes
        chore: release (#634)
        last month
        communication
        chore: release (#643)
        3 weeks ago
        container
        chore: release (#643)
        3 weeks ago
        logging
        chore: release (#643)
        3 weeks ago
        mdbook
        Apply various Clippy recommendations (#603)
        4 months ago
        timely
        Linear connectivity (#651)
        5 days ago
        .gitignore
        initial kafka check-in
        8 years ago
        CHANGELOG.md
        chore: release (#643)
        3 weeks ago
        CONTRIBUTING.md
        cleanup: remove trailing whitespace
        7 years ago
        COPYRIGHT
        Update COPYRIGHT to include myself
        8 years ago
        Cargo.toml
        Update columnar to 0.3, make workspace dependency (#639)
        last month
        LICENSE
        Initial commit
        11 years ago
        README.md
        Rust updates, better doc testing (#598)
        4 months ago
        release-plz.toml
        Add support for release-plz (#548)
        5 months ago
        Repository files navigation
        README
        MIT license
        Timely Dataflow
        Timely dataflow is a low-latency cyclic dataflow computational model, introduced in the paper Naiad: a timely dataflow system. This project is an extended and more modular implementation of timely dataflow in Rust.

        This project is something akin to a distributed data-parallel compute engine, which scales the same program up from a single thread on your laptop to distributed execution across a cluster of computers. The main goals are expressive power and high performance. It is probably strictly more expressive and faster than whatever you are currently using, assuming you aren't yet using timely dataflow.

        Be sure to read the documentation for timely dataflow. It is a work in progress, but mostly improving. There is more long-form text in mdbook format with examples tested against the current builds. There is also a series of blog posts (part 1, part 2, part 3) introducing timely dataflow in a different way, though be warned that the examples there may need tweaks to build against the current code.

    - output:
        - title : Timely Dataflow
        - content_type :  Repository
        - tags : Data Engineering
    </examples>
    """


    user_prompt = """
    Here is the content that I want you to describe :
    <content>
    {content}
    </content>

    Please procede with your analysis, and provide an accurate description.
    """
    return system_prompt, user_prompt


@app.cell
def _(evaluate, experiment, mlflow, system_prompt, user_prompt):
    with mlflow.start_run(
        run_name="Anthropic user prompt 2.2e.1",
        experiment_id=experiment.experiment_id,
    ):
        evaluate(system_prompt=system_prompt, user_prompt=user_prompt)
    return


@app.cell
def _(extraction_config):
    prompt_version = "2.2e"

    anthropic_system_prompt = """
    You are an expert extraction algorithm specialized in Data Science-related subjects. Your task is to analyze a given piece of content and extract key information about it.

    Please follow these steps to analyze the content and provide the required information:

    1. Title Extraction/Generation:
       - If the content has an existing title, extract it, you are not allowed any rephrasing.
       - Only if no title exists, generate one that accurately represents the content.
       - Ensure the title does not exceed 80 characters.

    2. Content Type Determination:
       - Identify the type of content from the following options: Course, Repository, Documentation, Blog post, Product main page, Scientific article
       - Choose the most appropriate type based on the content's structure and purpose.

    3. Tag Extraction:
       - Identify the main Data Science topics (focus on the primary themes, not related subjects)
       - Reduce the number of tags as much as possible to ensure precision (aim for 1-3 tags maximum).

    Remember:
    - Be as accurate as possible in determining the content type.
    - Only include the most relevant and prominent Data Science topics as tags.

    """


    formatted_system_prompt = anthropic_system_prompt.format(
        content_types=", ".join(extraction_config.content_types.values()),
        tags=", ".join(extraction_config.tags.values()),
    )
    return formatted_system_prompt, prompt_version


@app.cell
def _(anthropic_config, evaluate, experiment, formatted_system_prompt, mlflow):
    with mlflow.start_run(
        run_name="Default config anthropic haiku",
        experiment_id=experiment.experiment_id,
    ):
        evaluate(model_config=anthropic_config, system_prompt=formatted_system_prompt)
    return


@app.cell(disabled=True)
def _(
    LLMProvider,
    ModelConfig,
    evaluate,
    experiment,
    formatted_system_prompt,
    mlflow,
):
    anthropic_3_7_config = ModelConfig(
        chat_model="claude-3-7-sonnet-20250219",
        provider=LLMProvider.ANTHROPIC,
        token_limit=200_000,
        model_specific_config={"max_tokens": 8_192},
    )

    with mlflow.start_run(
        run_name="Claude 3.7 prompt v2",
        experiment_id=experiment.experiment_id,
    ):
        evaluate(model_config=anthropic_3_7_config, system_prompt=formatted_system_prompt)

    return


@app.cell(disabled=True)
def _(evaluate, experiment, formatted_system_prompt, mlflow, prompt_version):
    with mlflow.start_run(
        run_name=f"Anthropic system prompt v{prompt_version}",
        experiment_id=experiment.experiment_id,
    ):
        results = evaluate(system_prompt=formatted_system_prompt)
    return (results,)


@app.cell(disabled=True)
def _(evaluate, formatted_system_prompt):
    evaluate(system_prompt=formatted_system_prompt, set_type="test")
    return


@app.cell
def _(results):
    results.loc[:, "pred_tags"] = results["pred_tags"].apply(
        lambda l: [e.name.value for e in l]
    )
    return


@app.cell
def _(results):
    results[["pred_tags", "tag"]]
    return


@app.cell
def _(results):
    results[["pred_title", "title"]]
    return


@app.cell
def _(results):
    results["content"].apply(lambda x: len(x.split(" ")))
    return


@app.cell
def _(results):
    results.loc[6, "content"]
    return


@app.cell
def _():
    return


@app.cell
def _(extraction_config):
    ", ".join(extraction_config.content_types.values())
    return


@app.cell
def _(extraction_config):
    ", ".join(extraction_config.tags.values())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
