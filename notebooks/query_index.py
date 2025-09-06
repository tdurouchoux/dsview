import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    Interactions : 
    - query field expressed as natural language
    - return close topics
    - return close content ?
    - some kind of display ?
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    from dsview.db.query import TopicsIndex
    from dsview.obsidian.obsidian_utils import get_content_url_link
    return TopicsIndex, get_content_url_link, mo


@app.cell
def _(TopicsIndex):
    index = TopicsIndex()
    return (index,)


@app.cell
def _(index):
    index.build()
    return


@app.cell
def _(mo):
    query = mo.ui.text(
        placeholder="Search for any topic",
        full_width=True,
        label="Query",
        debounce=True,
    )
    query
    return (query,)


@app.cell
def _(get_content_url_link, index, mo, query):
    mo.stop(query.value == "")

    fts_query_result, vss_query_result = index.query(query.value)

    fts_query_result["link"] = fts_query_result["name"].apply(
        lambda link: mo.md(f"[note]({get_content_url_link(link)})")
    )
    vss_query_result["link"] = vss_query_result["name"].apply(
        lambda link: mo.md(f"[note]({get_content_url_link(link)})")
    )
    return fts_query_result, vss_query_result


@app.cell
def _(fts_query_result, mo):
    mo.ui.table(
        fts_query_result[["name", "link", "type", "description", "score"]],
        selection=None,
        freeze_columns_left=["name"],
        label="FTS results",
    )
    return


@app.cell
def _(mo, vss_query_result):
    mo.ui.table(
        vss_query_result[["name", "link", "type", "description", "distance"]],
        selection=None,
        freeze_columns_left=["name"],
        label="VSS results",
    )
    return


if __name__ == "__main__":
    app.run()
