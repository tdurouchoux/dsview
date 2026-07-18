import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import nest_asyncio

    os.environ["PROMPT_DIR"] = "./prompts/"
    os.environ["DSVIEW_DISABLE_NATIVE_BATCH"] = "Disable"

    # Native batch is disabled above, so batch calls fall back to asyncio.run(),
    # which raises inside marimo's already-running event loop. nest_asyncio patches
    # asyncio to allow that reentrant run. Notebook-only workaround; production code
    # under dsview/ is intentionally left untouched.
    nest_asyncio.apply()
    return


@app.cell
def _():
    from types import SimpleNamespace

    import marimo as mo
    import mlflow
    import pandas as pd
    from tqdm import tqdm

    from dsview.evaluation.topics_extraction import (
        MAX_N_TOPICS,
        SimpleERModel,
        find_er_matches,
        get_topics_extraction_data,
        score_row,
    )
    from dsview.extraction.models.topics_extraction import TopicType

    tqdm.pandas()
    return (
        MAX_N_TOPICS,
        SimpleERModel,
        SimpleNamespace,
        TopicType,
        find_er_matches,
        get_topics_extraction_data,
        mlflow,
        mo,
        pd,
        score_row,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Topic extraction baselines

    Compares the LLM-based `TopicsExtractor` (see `notebooks/topic_extraction_opt.py`,
    baseline logged as `mistral_small_2603_baseline_eval_sync`) against single-document
    unsupervised keyword extraction methods: YAKE and KeyBERT.

    These operate one document at a time (no shared corpus-level topics assumed), matching
    how DSView actually processes content, unlike corpus-level methods (LDA/NMF/BERTopic)
    which assume documents share a recurring set of latent topics — not a good fit for a
    heterogeneous personal reading list.

    `yake` and `keybert` are ephemeral dependencies for this notebook only (not added to
    `pyproject.toml`). Run with:

    ```
    uv run --with yake --with keybert marimo edit notebooks/topic_modeling_baselines.py
    ```
    """)
    return


@app.cell
def _(get_topics_extraction_data):
    eval_data = get_topics_extraction_data("eval")
    return (eval_data,)


@app.cell
def _(mlflow):
    experiment = mlflow.set_experiment("Topics extraction")
    return (experiment,)


@app.cell
def _(TopicType):
    # These methods have no type taxonomy (Concept/Library/...), so type-match
    # scoring is never meaningful for them here — placeholder keeps score_row()
    # from erroring on `.type.value`; type_accuracy is reported as N/A, not 0%.
    PLACEHOLDER_TYPE = next(iter(TopicType))
    return (PLACEHOLDER_TYPE,)


@app.cell
def _(MAX_N_TOPICS, PLACEHOLDER_TYPE, SimpleNamespace):
    def to_pred_topics(names: list[str]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name=name, type=PLACEHOLDER_TYPE)
            for name in names[:MAX_N_TOPICS]
        ]

    return (to_pred_topics,)


@app.cell
def _(SimpleERModel, find_er_matches, mlflow, pd, score_row):
    def score_predictions(df: pd.DataFrame, method: str, params: dict) -> pd.DataFrame:
        simple_er = SimpleERModel()

        df = df.copy()
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

        print(method, {key: f"{value:.2f}" for key, value in metrics.items()})

        mlflow.log_param("method", method)
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metrics(metrics)

        return df

    return (score_predictions,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Method 1 — YAKE
    """)
    return


@app.cell
def _():
    import yake

    return (yake,)


@app.cell
def _(MAX_N_TOPICS, eval_data, to_pred_topics, tqdm, yake):
    YAKE_NGRAM_MAX = 3
    # YAKE's built-in near-duplicate suppression (lower = stricter). The 0.9 default
    # is permissive and lets overlapping n-grams of one phrase through; those score
    # as separate correct predictions and inflate precision.
    YAKE_DEDUP_LIM = 0.7

    yake_extractor = yake.KeywordExtractor(
        n=YAKE_NGRAM_MAX, top=MAX_N_TOPICS, dedup_lim=YAKE_DEDUP_LIM
    )


    def extract_yake(content: str):
        keywords = yake_extractor.extract_keywords(content)
        return to_pred_topics([name for name, _score in keywords])


    yake_data = eval_data.copy()
    yake_data["pred_topics"] = [
        extract_yake(content) for content in tqdm(yake_data["content"])
    ]
    return YAKE_DEDUP_LIM, YAKE_NGRAM_MAX, yake_data


@app.cell
def _(
    MAX_N_TOPICS,
    YAKE_DEDUP_LIM,
    YAKE_NGRAM_MAX,
    experiment,
    mlflow,
    score_predictions,
    yake_data,
):
    with mlflow.start_run(
        run_name=f"YAKE_dedup{YAKE_DEDUP_LIM}_eval", experiment_id=experiment.experiment_id
    ):
        yake_results = score_predictions(
            yake_data,
            method="YAKE",
            params={
                "ngram_max": YAKE_NGRAM_MAX,
                "top_n": MAX_N_TOPICS,
                "dedup_lim": YAKE_DEDUP_LIM,
            },
        )
    return (yake_results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Method 2 — KeyBERT
    """)
    return


@app.cell
def _():
    from keybert import KeyBERT

    keybert_model = KeyBERT()
    return (keybert_model,)


@app.cell
def _(MAX_N_TOPICS, eval_data, keybert_model, to_pred_topics, tqdm):
    KEYBERT_NGRAM_MAX = 3
    # KeyBERT ranks candidates by similarity to the whole-document embedding, so on a
    # DuckDB post every "duckdb X" phrase scores ~0.65 while distinct entities that are
    # mentioned but not central (MotherDuck, AWS Athena) rank low. Without MMR that
    # yields ~10 paraphrases of one theme, which score_row counts as separate correct
    # predictions (measured 3.3x precision inflation).
    # MMR suppresses that redundancy. Calibrated on eval docs: 0.2/0.3 still return
    # near-duplicates, 0.7 collapses relevance into scrape noise; 0.5 is the point
    # where distinct labelled entities (dbt-core, open-source model) start surfacing.
    KEYBERT_DIVERSITY = 0.5


    def extract_keybert(content: str):
        keywords = keybert_model.extract_keywords(
            content,
            keyphrase_ngram_range=(1, KEYBERT_NGRAM_MAX),
            top_n=MAX_N_TOPICS,
            stop_words="english",
            use_mmr=True,
            diversity=KEYBERT_DIVERSITY,
        )
        return to_pred_topics([name for name, _score in keywords])


    keybert_data = eval_data.copy()
    keybert_data["pred_topics"] = [
        extract_keybert(content) for content in tqdm(keybert_data["content"])
    ]
    return KEYBERT_DIVERSITY, KEYBERT_NGRAM_MAX, keybert_data


@app.cell
def _(
    KEYBERT_DIVERSITY,
    KEYBERT_NGRAM_MAX,
    MAX_N_TOPICS,
    experiment,
    keybert_data,
    mlflow,
    score_predictions,
):
    with mlflow.start_run(
        run_name=f"KeyBERT_mmr{KEYBERT_DIVERSITY}_eval",
        experiment_id=experiment.experiment_id,
    ):
        keybert_results = score_predictions(
            keybert_data,
            method="KeyBERT",
            params={
                "ngram_max": KEYBERT_NGRAM_MAX,
                "top_n": MAX_N_TOPICS,
                "use_mmr": True,
                "diversity": KEYBERT_DIVERSITY,
            },
        )
    return (keybert_results,)


@app.cell
def _(keybert_results):
    keybert_results
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Method 3 — GLiNER (zero-shot NER)

    Unlike YAKE/KeyBERT, GLiNER takes the entity labels as input, so it is given the
    `TopicType` taxonomy directly and returns `(span, type)` pairs — the same output
    shape as `TopicsExtractor`. This makes `type_accuracy` a real metric here rather
    than N/A, and reflects that ~57% of labelled topics (Model/Library/Tool/Platform/
    Dataset) are named entities rather than corpus-level themes.

    GLiNER's context window is ~384 tokens but the median eval document is ~3.9k
    tokens, so content is chunked and entities are aggregated across chunks.
    """)
    return


@app.cell
def _():
    from gliner import GLiNER

    GLINER_MODEL_NAME = "urchade/gliner_medium-v2.1"

    gliner_model = GLiNER.from_pretrained(GLINER_MODEL_NAME)
    return GLINER_MODEL_NAME, gliner_model


@app.cell
def _(MAX_N_TOPICS, SimpleNamespace, TopicType, eval_data, gliner_model, tqdm):
    # GLiNER is given the taxonomy itself as its label set.
    GLINER_LABELS = [topic_type.value for topic_type in TopicType]
    GLINER_THRESHOLD = 0.5
    # ~300 tokens, kept under GLiNER's ~384-token window.
    GLINER_CHUNK_CHARS = 1200


    def chunk_text(text: str, size: int) -> list[str]:
        """Split text into <=size-char chunks on word boundaries."""
        chunks, current, length = [], [], 0
        for word in text.split():
            if length + len(word) + 1 > size and current:
                chunks.append(" ".join(current))
                current, length = [], 0
            current.append(word)
            length += len(word) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks


    def to_typed_pred_topics(ranked: list[tuple[float, str, str]]) -> list[SimpleNamespace]:
        """Build pred_topics carrying GLiNER's predicted type (no placeholder needed)."""
        return [
            SimpleNamespace(name=name, type=TopicType(label))
            for _score, name, label in ranked[:MAX_N_TOPICS]
        ]


    def extract_gliner(content: str) -> list[SimpleNamespace]:
        # Dedupe spans across chunks case-insensitively, keeping the best-scoring hit.
        best: dict[str, tuple[float, str, str]] = {}
        for _chunk in chunk_text(content, GLINER_CHUNK_CHARS):
            for _ent in gliner_model.predict_entities(
                _chunk, GLINER_LABELS, threshold=GLINER_THRESHOLD
            ):
                _name = _ent["text"].strip()
                _key = _name.lower()
                if not _key:
                    continue
                if _key not in best or _ent["score"] > best[_key][0]:
                    best[_key] = (_ent["score"], _name, _ent["label"])

        # score_row scores by rank, so order by confidence before truncating.
        _ranked = sorted(best.values(), key=lambda item: item[0], reverse=True)
        return to_typed_pred_topics(_ranked)


    gliner_data = eval_data.copy()
    gliner_data["pred_topics"] = [
        extract_gliner(content) for content in tqdm(gliner_data["content"])
    ]
    return GLINER_CHUNK_CHARS, GLINER_THRESHOLD, gliner_data


@app.cell
def _(
    GLINER_CHUNK_CHARS,
    GLINER_MODEL_NAME,
    GLINER_THRESHOLD,
    MAX_N_TOPICS,
    experiment,
    gliner_data,
    mlflow,
    score_predictions,
):
    with mlflow.start_run(
        run_name="GLiNER_baseline_eval", experiment_id=experiment.experiment_id
    ):
        gliner_results = score_predictions(
            gliner_data,
            method="GLiNER",
            params={
                "gliner_model": GLINER_MODEL_NAME,
                "threshold": GLINER_THRESHOLD,
                "chunk_chars": GLINER_CHUNK_CHARS,
                "top_n": MAX_N_TOPICS,
            },
        )

        # GLiNER predicts a real type per span, so unlike YAKE/KeyBERT this is
        # meaningful; score_predictions omits it since it is N/A for those methods.
        _correct = gliner_results["count_correct_topic"].sum()
        gliner_type_accuracy = (
            gliner_results["count_type_correct"].sum() / _correct if _correct else 0.0
        )
        mlflow.log_metric("type_accuracy", gliner_type_accuracy)
        print(f"GLiNER type_accuracy: {gliner_type_accuracy:.2f}")
    return (gliner_results,)


@app.cell
def _(gliner_data):
    gliner_data
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparison
    """)
    return


@app.cell
def _(gliner_results, keybert_results, pd, yake_results):
    comparison = pd.DataFrame(
        {
            method: {
                f"mean_{metric}": results[metric].mean()
                for metric in ("precision", "recall", "average_precision")
            }
            for method, results in {
                "YAKE": yake_results,
                "KeyBERT": keybert_results,
                "GLiNER": gliner_results,
            }.items()
        }
    )
    comparison
    return


if __name__ == "__main__":
    app.run()
