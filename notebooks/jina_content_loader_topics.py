import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    from dotenv import load_dotenv

    os.environ["PROMPT_DIR"] = "./prompts/"
    load_dotenv()

    # Native batch is left enabled (cheaper, and the point is to match the baseline
    # runs' code path). If Mistral's batch API starts 404-ing, uncomment the block
    # below: the fallback path calls asyncio.run(), which raises inside marimo's
    # already-running event loop, and nest_asyncio patches that. Notebook-only
    # workaround; production code under dsview/ is intentionally left untouched.
    #
    # import nest_asyncio
    #
    # os.environ["DSVIEW_DISABLE_NATIVE_BATCH"] = "Disable"
    # nest_asyncio.apply()
    return (os,)


@app.cell
def _():
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import marimo as mo
    import mlflow
    import pandas as pd
    import requests
    from pydantic import HttpUrl
    from sqlmodel import Session
    from tenacity import retry, stop_after_attempt, wait_random_exponential
    from tqdm import tqdm

    from dsview.db import engine
    from dsview.db.query import get_labels_data
    from dsview.db.schemas import TopicsLabels

    # Aliased: marimo treats leading-underscore names as cell-local, so the original
    # name could not be handed to the eval cell below.
    from dsview.evaluation.cli import _prepare_for_parquet as prepare_for_parquet
    from dsview.evaluation.topics_extraction import (
        SimpleERModel,
        find_er_matches,
        score_row,
    )
    from dsview.extraction.content_loader import PdfUrlLoader, get_content_loader
    from dsview.extraction.models.topics_extraction import TopicsExtractor

    return (
        HttpUrl,
        Path,
        PdfUrlLoader,
        Session,
        SimpleERModel,
        ThreadPoolExecutor,
        TopicsExtractor,
        TopicsLabels,
        engine,
        find_er_matches,
        get_content_loader,
        get_labels_data,
        mlflow,
        mo,
        pd,
        prepare_for_parquet,
        requests,
        retry,
        score_row,
        stop_after_attempt,
        tempfile,
        tqdm,
        wait_random_exponential,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Jina Reader as content loader — Topics extraction

    `dsview/extraction/content_loader.py` loads generic web pages with `requests` +
    `BeautifulSoup.get_text()`: raw text carrying nav bars, footers and cookie banners,
    with no structure. **Hypothesis:** a markdown extraction service
    ([Jina AI Reader](https://jina.ai/reader/)) gives `TopicsExtractor` cleaner and
    structurally richer input, and that shows up as better topic precision/recall.

    Content loading is not currently a parameter of the evaluation — `evaluate()` scores
    the `content` text persisted in `labels.labelledcontent`, written once at label-ingest
    time. This notebook re-fetches the labelled URLs through Jina and runs the *existing*
    topics eval on that content, logging one run to the **"Topics extraction"** experiment
    so it sits next to the past default-config runs.

    **Scope**
    - Only rows that `get_content_loader` routes to `UrlLoader` (generic web pages) get
      Jina content. PDF rows keep their stored content untouched, so they contribute
      exactly what they contributed to the baseline run and the delta is attributable to
      the HTML path.
    - One Jina config (markdown, images dropped). The request options are notebook
      variables — variants can be tried later without re-deriving the harness.

    **Caveat, stated up front:** the baseline is stored content captured at labelling
    time, so part of any delta may be page drift rather than the loader. The diagnostics
    section below (free, no LLM calls) is where that gets sanity-checked *before* spending
    anything.

    > This notebook is a test harness. If the result is positive, changing
    > `content_loader.py` is a separate, explicitly requested task — nothing here edits
    > production code.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ## Load the labelled eval split
    """)


@app.cell
def _(Session, TopicsLabels, engine, get_labels_data):
    # Same aggregation as evaluation.topics_extraction.get_topics_extraction_data(),
    # with "link" kept — that function drops it, and the fetch needs it. Everything
    # downstream (find_er_matches, score_row) sees the identical frame shape.
    def get_topics_extraction_data_with_link(set_type: str):
        with Session(engine) as session:
            df_labels = get_labels_data([TopicsLabels], set_type, session)

        return (
            df_labels.dropna(subset=["name", "type"], how="any")
            .sort_values("rank", ascending=True)
            .groupby("id")
            .agg({"link": "first", "content": "first", "name": list, "type": list})
        )

    return (get_topics_extraction_data_with_link,)


@app.cell
def _(get_topics_extraction_data_with_link):
    df = get_topics_extraction_data_with_link("eval")
    return (df,)


@app.cell
def _(HttpUrl, PdfUrlLoader, df, get_content_loader):
    # Reuse production routing so "which rows are PDFs" cannot drift from what the
    # pipeline actually does.
    df["is_pdf"] = [
        isinstance(get_content_loader(HttpUrl(link)), PdfUrlLoader)
        for link in df["link"]
    ]

    # Medium links were rewritten to readmedium.com at ingest time
    # (IngestPipeline._clean_content_url), and readmedium sits behind Cloudflare:
    # Jina gets a 403 block page back. Jina reads Medium directly, so point the
    # fetch at the original article instead. _clean_content_url drops the original
    # host (medium.com / towardsdatascience.com / netflixtechblog.com all collapse
    # to readmedium.com), so it cannot be recovered automatically — this is a manual
    # rewrite, correct for the readmedium rows currently in the split. Re-check it
    # if the labelled set grows.
    READMEDIUM_ORIGIN_HOST = "towardsdatascience.com"

    df["fetch_link"] = df["link"].str.replace(
        "readmedium.com", READMEDIUM_ORIGIN_HOST, regex=False
    )
    df.loc[8, "fetch_link"] = (
        "https://web.archive.org/web/20251216015402/https://neptune.ai/blog/arima-vs-prophet-vs-lstm"
    )


@app.cell
def _(df, mo):
    mo.md(f"""
    Eval split: **{len(df)}** labelled rows — {(~df["is_pdf"]).sum()} web pages (Jina
    arm), {df["is_pdf"].sum()} PDFs (stored content, unchanged).
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ## Fetch through Jina Reader

    `GET https://r.jina.ai/<url>`. Retries use the same shape as the provider calls in
    `dsview/model_utils/model_provider.py` so 429s and transient 5xx are absorbed. One
    failure never aborts the sweep: the status is recorded per row and dealt with below.
    """)


@app.cell
def _(os):
    JINA_ENDPOINT = "https://r.jina.ai/"
    JINA_OPTIONS = {
        "X-Return-Format": "markdown",
        "X-Retain-Images": "none",
    }
    JINA_TIMEOUT = 120
    JINA_MAX_WORKERS = 5
    # Jina can answer 200 with a placeholder/paywall/JS-shell body; anything this short
    # is treated as a failed fetch rather than as content.
    JINA_MIN_CONTENT_CHARS = 200
    # Jina reports upstream failures inside a 200 response body.
    JINA_FAILURE_MARKERS = (
        "Warning: Target URL returned error",
        "Attention Required! | Cloudflare",
    )
    JINA_MARKER_SCAN_CHARS = 1000

    JINA_HEADERS = {
        "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
        **JINA_OPTIONS,
    }
    return (
        JINA_ENDPOINT,
        JINA_FAILURE_MARKERS,
        JINA_HEADERS,
        JINA_MARKER_SCAN_CHARS,
        JINA_MAX_WORKERS,
        JINA_MIN_CONTENT_CHARS,
        JINA_OPTIONS,
        JINA_TIMEOUT,
    )


@app.cell
def _(
    JINA_ENDPOINT,
    JINA_FAILURE_MARKERS,
    JINA_HEADERS,
    JINA_MARKER_SCAN_CHARS,
    JINA_MIN_CONTENT_CHARS,
    JINA_TIMEOUT,
    requests,
    retry,
    stop_after_attempt,
    wait_random_exponential,
):
    @retry(
        wait=wait_random_exponential(multiplier=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _fetch_jina(url: str) -> str:
        response = requests.get(
            JINA_ENDPOINT + str(url),
            headers=JINA_HEADERS,
            timeout=JINA_TIMEOUT,
        )
        response.raise_for_status()
        return response.text

    def fetch_jina(url: str) -> tuple[str | None, str]:
        """Return (content, status). Never raises — status carries the failure."""
        try:
            content = _fetch_jina(url)
        except Exception as error:  # noqa: BLE001 - a sweep must survive any failure
            return None, f"error: {type(error).__name__}: {error}"

        # Jina answers 200 and puts the upstream failure in the body, so a Cloudflare
        # block page or paywall stub looks like a successful fetch. Caught one this
        # way (readmedium, 1128 chars of "Attention Required! | Cloudflare"), which
        # is well over the length floor below.
        for marker in JINA_FAILURE_MARKERS:
            if marker in content[:JINA_MARKER_SCAN_CHARS]:
                return content, f"blocked: {marker!r} in body"

        if len(content) < JINA_MIN_CONTENT_CHARS:
            return content, f"suspicious: only {len(content)} chars"

        return content, "ok"

    return (fetch_jina,)


@app.cell
def _(mo):
    fetch_button = mo.ui.run_button(label="Fetch web pages through Jina")
    fetch_button
    return (fetch_button,)


@app.cell
def _(
    JINA_MAX_WORKERS,
    ThreadPoolExecutor,
    df,
    fetch_button,
    fetch_jina,
    mo,
    tqdm,
):
    mo.stop(not fetch_button.value, mo.md("*Press the button above to fetch.*"))

    # fetch_link, not link: readmedium rows are fetched from the original Medium host.
    web_links = df.loc[~df["is_pdf"], "fetch_link"]

    with ThreadPoolExecutor(max_workers=JINA_MAX_WORKERS) as executor:
        jina_fetches = list(
            tqdm(
                executor.map(fetch_jina, web_links),
                total=len(web_links),
                desc="Fetching through Jina",
            )
        )

    jina_content = dict(zip(web_links.index, (content for content, _ in jina_fetches)))
    jina_status = dict(zip(web_links.index, (status for _, status in jina_fetches)))
    return jina_content, jina_status


@app.cell
def _(df, jina_content, jina_status):
    df["jina_content"] = df.index.map(jina_content)
    df["jina_status"] = df.index.map(jina_status).fillna("skipped: pdf")

    # PDFs keep the stored content, and so do rows Jina could not serve — that keeps the
    # row set identical to the baseline run instead of scoring a hole. content_source
    # makes both cases visible in the artifact.
    df["content_source"] = [
        "jina" if status == "ok" else "stored" for status in df["jina_status"]
    ]
    df["eval_content"] = [
        jina if source == "jina" else stored
        for source, jina, stored in zip(
            df["content_source"], df["jina_content"], df["content"]
        )
    ]


@app.cell
def _(mo):
    mo.md(r"""
    ## Diagnostics — free, run these before spending anything

    A broken Jina config (paywall page, JS shell, error body returned with 200) is much
    cheaper to catch here than after a full eval + judge fan-out.
    """)


@app.cell
def _(df):
    df["jina_status"].value_counts()


@app.cell
def _(df):
    df["stored_chars"] = df["content"].str.len()
    df["jina_chars"] = df["jina_content"].str.len()
    df["char_ratio"] = df["jina_chars"] / df["stored_chars"]

    length_stats = df.loc[
        df["content_source"] == "jina", ["stored_chars", "jina_chars", "char_ratio"]
    ].describe()
    length_stats


@app.cell
def _(df, pd):
    # Extremes are where the interesting failures live: a ratio near 0 means Jina
    # returned nothing useful, a ratio far above 1 means it kept boilerplate the old
    # loader stripped (or the stored content was truncated).
    _ranked = df.loc[
        df["content_source"] == "jina",
        ["link", "stored_chars", "jina_chars", "char_ratio"],
    ].sort_values("char_ratio")

    pd.concat([_ranked.head(5), _ranked.tail(5)])


@app.cell
def _(df, mo):
    row_picker = mo.ui.dropdown(
        options={str(link): row_id for row_id, link in df["link"].items()},
        label="Inspect a row",
    )
    row_picker
    return (row_picker,)


@app.cell
def _(df, mo, row_picker):
    mo.stop(row_picker.value is None, mo.md("*Pick a row to compare side by side.*"))

    _row = df.loc[row_picker.value]

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(f"### Stored ({_row['stored_chars']} chars)"),
                    mo.plain_text(_row["content"][:5000]),
                ]
            ),
            mo.vstack(
                [
                    mo.md(f"### Jina — {_row['jina_status']}"),
                    mo.plain_text((_row["jina_content"] or "")[:5000]),
                ]
            ),
        ],
        widths="equal",
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Pick the baseline run to compare against

    The comparison is against a past default-config run of the same experiment. Choose it
    explicitly and note its `run_id` — it gets logged as a param on the new run, so the
    pairing stays recoverable later.
    """)


@app.cell
def _(mlflow):
    # No set_tracking_uri: the backend is whatever the environment resolves to.
    experiment = mlflow.set_experiment("Topics extraction")
    return (experiment,)


@app.cell
def _(experiment, mlflow):
    past_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    past_runs[
        [
            column
            for column in (
                "run_id",
                "tags.mlflow.runName",
                "start_time",
                "params.set_type",
                "params.content_loader",
                "metrics.mean_precision",
                "metrics.mean_recall",
                "metrics.mean_average_precision",
                "metrics.type_accuracy",
            )
            if column in past_runs.columns
        ]
    ]


@app.cell
def _(mo):
    baseline_run_id = mo.ui.text(
        label="Baseline run_id",
        placeholder="paste a run_id from the table above",
        full_width=True,
    )
    baseline_run_id
    return (baseline_run_id,)


@app.cell
def _():
    from dsview.config import LLMProvider, ModelConfig

    return LLMProvider, ModelConfig


@app.cell
def _(mo):
    # Reuse the CLI's override loader so this arm and a `dsview evaluate topics
    # <config>` arm are built from the exact same YAML — no drift between them.
    from dsview.evaluation.cli import load_eval_overrides

    eval_config_file = mo.ui.dropdown(
        options={
            "exp2 default prompt (pipeline)": None,
            "jina selective prompt": "eval_configs/topics_jina_selective.yaml",
        },
        value="jina selective prompt",
        label="Eval config",
    )
    eval_config_file
    return eval_config_file, load_eval_overrides


@app.cell
def _(
    LLMProvider,
    ModelConfig,
    Path,
    eval_config_file,
    load_eval_overrides,
    mo,
):
    # Both arms pin temperature 0. The earlier Jina runs left it unpinned while the
    # exp2 baseline pinned it, which put sampling noise inside the comparison.
    if eval_config_file.value is None:
        eval_model_config, eval_system_prompt, eval_user_prompt = (
            ModelConfig(
                chat_model="mistral-small-2603",
                embedding_model="mistral-embed",
                provider=LLMProvider.MISTRAL,
                token_limit=100_000,
                model_specific_config={"temperature": 0},
            ),
            None,
            None,
        )
        eval_variant = "exp2_default_prompt"
    else:
        eval_model_config, eval_system_prompt, eval_user_prompt = load_eval_overrides(
            Path(eval_config_file.value)
        )
        eval_variant = Path(eval_config_file.value).stem

    run_name = f"{eval_variant}_jina_reader_temp0_eval"
    mo.md(f"Run name: `{run_name}`")
    return eval_model_config, eval_system_prompt, eval_user_prompt, run_name


@app.cell
def _(mo):
    mo.md(r"""
    ## Run the evaluation

    Mirrors `dsview/evaluation/topics_extraction.evaluate()` with `eval_content` swapped
    in for the stored content. `TopicsExtractor()` is built with **no overrides** so the
    model and prompts match the baseline exactly — the loader is the only thing that
    changed. Temperature is deliberately not pinned here: the point is to match the past
    default runs, not to minimise noise against a fresh arm.

    **This costs real API money** (one extraction per row, plus the LLM-judge fan-out over
    every predicted × labelled topic pair).
    """)


@app.cell
def _(mo):
    run_eval_button = mo.ui.run_button(label="Run the eval (costs money)")
    run_eval_button
    return (run_eval_button,)


@app.cell
def _(
    JINA_OPTIONS,
    Path,
    SimpleERModel,
    TopicsExtractor,
    baseline_run_id,
    df,
    eval_config_file,
    eval_model_config,
    eval_system_prompt,
    eval_user_prompt,
    experiment,
    find_er_matches,
    mlflow,
    mo,
    prepare_for_parquet,
    run_eval_button,
    run_name,
    score_row,
    tempfile,
):
    mo.stop(not run_eval_button.value, mo.md("*Press the button above to run.*"))

    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        topics_extractor = TopicsExtractor(
            model_config=eval_model_config,
            system_prompt=eval_system_prompt,
            user_prompt=eval_user_prompt,
        )
        simple_er = SimpleERModel()

        topics_extractor.log_params()
        mlflow.log_params(
            {
                "set_type": "eval",
                "content_loader": "jina_reader",
                "jina_options": JINA_OPTIONS,
                "eval_config_file": eval_config_file.value,
                "baseline_run_id": baseline_run_id.value or None,
                "n_jina_content": int((df["content_source"] == "jina").sum()),
                "n_stored_content": int((df["content_source"] == "stored").sum()),
            }
        )

        extraction_results = topics_extractor.predict_batch(
            [{"content": content} for content in df["eval_content"]]
        )
        df["pred_topics"] = [result.topics for result in extraction_results]

        er_matches = find_er_matches(simple_er, df)

        df[
            [
                "count_correct_topic",
                "precision",
                "recall",
                "average_precision",
                "count_type_correct",
            ]
        ] = df.apply(lambda row: score_row(row.name, row, er_matches), axis=1)

        metrics = {
            f"mean_{metric}": df[metric].mean()
            for metric in ("precision", "recall", "average_precision")
        }
        metrics["count_correct_topic"] = df["count_correct_topic"].sum()
        metrics["type_accuracy"] = (
            df["count_type_correct"].sum() / df["count_correct_topic"].sum()
        )
        metrics["mean_n_pred_topics"] = df["pred_topics"].map(len).mean()

        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "eval_results.parquet"
            prepare_for_parquet(df).to_parquet(result_path)
            mlflow.log_artifact(str(result_path))

        jina_run_id = mlflow.active_run().info.run_id

    metrics
    return (jina_run_id,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Analyse the result

    Reading a logged artifact costs nothing — never re-run an eval just to inspect it.
    """)


@app.cell
def _(baseline_run_id, jina_run_id, mlflow, mo, pd):
    mo.stop(not baseline_run_id.value, mo.md("*Set a baseline run_id to compare.*"))

    _baseline_metrics = mlflow.get_run(baseline_run_id.value).data.metrics
    _jina_metrics = mlflow.get_run(jina_run_id).data.metrics

    comparison = pd.DataFrame(
        {"baseline": _baseline_metrics, "jina": _jina_metrics}
    ).dropna()
    comparison["delta"] = comparison["jina"] - comparison["baseline"]
    comparison


@app.cell
def _(baseline_run_id, mlflow, mo, pd):
    # json is local to this cell: adding it to the shared import cell would make
    # that cell stale and cascade a re-run through the whole notebook.
    import json

    mo.stop(not baseline_run_id.value, mo.md("*Set a baseline run_id to compare.*"))

    # Downloading a logged artifact is free — never re-run an eval to inspect it.
    # pred_topics was serialised by _prepare_for_parquet, so it comes back as a JSON
    # string, not as DataScienceTopic objects.
    _baseline_path = mlflow.artifacts.download_artifacts(
        run_id=baseline_run_id.value, artifact_path="eval_results.parquet"
    )
    baseline_df = pd.read_parquet(_baseline_path)
    baseline_df["pred_topics"] = baseline_df["pred_topics"].map(json.loads)

    baseline_df.head()
    return (baseline_df,)


@app.cell
def _(baseline_df, baseline_run_id, df, mo):
    mo.stop(not baseline_run_id.value, mo.md("*Set a baseline run_id to compare.*"))

    # Per-row join: which rows actually moved, and does that correlate with how much the
    # content changed?
    per_row = df[
        [
            "link",
            "content_source",
            "char_ratio",
            "precision",
            "recall",
            "average_precision",
        ]
    ].join(
        baseline_df[["precision", "recall", "average_precision"]],
        rsuffix="_baseline",
    )
    for _metric in ("precision", "recall", "average_precision"):
        per_row[f"{_metric}_delta"] = per_row[_metric] - per_row[f"{_metric}_baseline"]

    per_row.sort_values("average_precision_delta")


@app.cell
def _(mo):
    mo.md(r"""
    ### Predicted topics, side by side

    The metric deltas say *how much* a row moved; this says *what changed*. Pick a row
    with the selector above — labelled topics on the left, then what each arm predicted,
    in rank order. Topics matching a labelled name exactly are marked ✅;
    everything else
    went to the LLM judge, so an unmarked topic is not necessarily wrong.
    """)


@app.cell
def _(df):
    df["pred_topics"].apply(len).sum()


@app.cell
def _(baseline_df):
    baseline_df["pred_topics"].apply(len).sum()


@app.cell
def _(baseline_df, baseline_run_id, df, mo, row_picker):
    mo.stop(
        "pred_topics" not in df.columns or not baseline_run_id.value,
        mo.md("*Needs a finished Jina run and a baseline run_id.*"),
    )
    mo.stop(row_picker.value is None, mo.md("*Pick a row above.*"))

    _row_id = row_picker.value
    _labelled = list(zip(df.loc[_row_id, "name"], df.loc[_row_id, "type"]))
    _labelled_names = set(df.loc[_row_id, "name"])

    def _format_topics(topics, as_dict: bool) -> str:
        if not len(topics):
            return "*(nothing predicted)*"

        lines = []
        for rank, topic in enumerate(topics, start=1):
            name = topic["name"] if as_dict else topic.name
            type_ = topic["type"] if as_dict else topic.type.value
            hit = " ✅" if name in _labelled_names else ""
            lines.append(f"{rank}. **{name}** — `{type_}`{hit}")

        return "\n".join(lines)

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("#### Labelled"),
                    mo.md(
                        "\n".join(
                            f"{rank}. **{name}** — `{type_}`"
                            for rank, (name, type_) in enumerate(_labelled, start=1)
                        )
                    ),
                ]
            ),
            mo.vstack(
                [
                    mo.md("#### Baseline (stored content)"),
                    mo.md(
                        _format_topics(baseline_df.loc[_row_id, "pred_topics"], True)
                    ),
                ]
            ),
            mo.vstack(
                [
                    mo.md(f"#### Jina ({df.loc[_row_id, 'content_source']})"),
                    mo.md(_format_topics(df.loc[_row_id, "pred_topics"], False)),
                ]
            ),
        ],
        widths="equal",
        gap=2,
    )


if __name__ == "__main__":
    app.run()
