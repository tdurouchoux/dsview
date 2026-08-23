import marimo

__generated_with = "0.14.16"
app = marimo.App(
    width="medium",
    layout_file="layouts/search_vault.grid.json",
    css_file="layouts/marimo_style.css",
)


@app.cell
def _():
    import marimo as mo

    from dsview.db.query import ExtractionIndex, TopicsIndex
    from dsview.interface.dashboard.marimo_sidebar import get_sidebar
    from dsview.obsidian.obsidian_utils import get_content_url_link

    return ExtractionIndex, TopicsIndex, get_content_url_link, get_sidebar, mo


@app.cell
def _(get_sidebar):
    get_sidebar()


@app.cell
def _(mo):
    mo.center(mo.md(r"""# Search vault"""))


@app.cell
def _(mo):
    query = mo.ui.text(label="Query some content :")
    mo.center(query)
    return (query,)


@app.cell
def _(mo):
    n_topics = mo.ui.slider(
        start=5, stop=20, value=10, label="Number of topics", show_value=True
    )
    return (n_topics,)


@app.cell
def _(TopicsIndex, mo, n_topics, query):
    mo.stop(query.value == "")

    with TopicsIndex() as topics_index:
        topics = topics_index.query_with_rff(query.value, n_topics.value)

    display_topics = mo.ui.table(
        topics.drop(columns=["rff_score"]),
        selection="single",
        show_data_types=False,
    )

    mo.vstack(
        (
            mo.md("## Related topics"),
            mo.center(n_topics),
            display_topics
            if len(topics) > 0
            else mo.callout("No topic found.", kind="warn"),
        )
    )
    return (display_topics,)


@app.cell
def _(display_topics, get_content_url_link, mo):
    mo.stop(len(display_topics.value) == 0)

    topic_link = mo.center(
        mo.md(
            f"[Go to obsidian note]({get_content_url_link(display_topics.value.iloc[0]['name'])})"
        )
    )
    topic_link


@app.cell
def _(mo):
    n_content = mo.ui.slider(
        start=5, stop=20, value=10, label="Number of content", show_value=True
    )

    return (n_content,)


@app.cell
def _(ExtractionIndex, mo, n_content, query):
    mo.stop(query.value == "")

    with ExtractionIndex() as extraction_index:
        extraction = extraction_index.query_with_rff(query.value, n_content.value)

    display_extraction = mo.ui.table(
        extraction.drop(columns=["rff_score", "extraction_time"]),
        selection="single",
        show_data_types=False,
    )

    mo.vstack(
        (
            mo.md("## Related contents"),
            mo.center(n_content),
            (
                display_extraction
                if len(extraction) > 0
                else mo.callout("No content found.", kind="warn")
            ),
        )
    )
    return (display_extraction,)


@app.cell
def _(display_extraction, get_content_url_link, mo):
    mo.stop(len(display_extraction.value) == 0)

    content_link = mo.center(
        mo.md(
            f"[Go to obsidian note]({get_content_url_link(display_extraction.value.iloc[0]['title'])})"
        )
    )
    content_link


if __name__ == "__main__":
    app.run()
