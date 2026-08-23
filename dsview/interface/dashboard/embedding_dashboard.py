import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", css_file="layouts/marimo_style.css")


@app.cell
def _(mo):
    mo.center(mo.md("# Embedding dashboard"))


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd
    import umap

    from dsview.db import engine
    from dsview.interface.dashboard.marimo_sidebar import get_sidebar
    from dsview.obsidian.obsidian_utils import get_content_url_link

    return alt, engine, get_content_url_link, get_sidebar, mo, pd, umap


@app.cell
def _(get_sidebar):
    get_sidebar()


@app.cell
def _(alt, mo, pd, umap):
    @mo.cache
    def reduce_and_display_embeddings(
        data: pd.DataFrame,
        embedding_col: str,
        n_neighbors: int,
        tooltip_cols: list[str],
        color_col: str | None = None,
    ):
        embeddings = data[embedding_col].to_list()

        reducer = umap.UMAP(n_neighbors=n_neighbors)
        projection = reducer.fit_transform(embeddings)

        data["reduced_embedding_x"] = projection[:, 0]
        data["reduced_embedding_y"] = projection[:, 1]

        chart = (
            alt.Chart(data)
            .mark_point()
            .encode(
                x="reduced_embedding_x:Q",
                y="reduced_embedding_y:Q",
                color=color_col,
                tooltip=tooltip_cols,
            )
        )

        return mo.ui.altair_chart(chart)

    return (reduce_and_display_embeddings,)


@app.cell
def _(mo):
    mo.md(r"""## Content embeddings""")


@app.cell
def _(engine, extraction, extractionresult, mo):
    extracted_content = mo.sql(
        """
        SELECT
            title,
            content_type,
            embedding
        FROM
            extraction.extractionresult
        """,
        output=False,
        engine=engine,
    )
    return (extracted_content,)


@app.cell
def _(mo):
    n_neighbors_content = mo.ui.slider(
        start=1,
        stop=20,
        value=5,
        label="Neighbors for UMAP projection :",
        show_value=True,
    )
    mo.center(n_neighbors_content)
    return (n_neighbors_content,)


@app.cell
def _(extracted_content, n_neighbors_content, reduce_and_display_embeddings):
    umap_content_mo_chart = reduce_and_display_embeddings(
        extracted_content,
        "embedding",
        n_neighbors=n_neighbors_content.value,
        tooltip_cols=["title"],
        color_col="content_type",
    )
    umap_content_mo_chart
    return (umap_content_mo_chart,)


@app.cell
def _(get_content_url_link, mo, umap_content_mo_chart):
    mo.stop(len(umap_content_mo_chart.value) == 0)
    selected_content = umap_content_mo_chart.value.to_dict("records")[0]

    mo.center(
        mo.md(
            f"[Go to obsidian note]({get_content_url_link(selected_content['title'])})"
        )
    )


@app.cell
def _(mo):
    mo.md(r"""## Topics embeddings""")


@app.cell
def _(engine, extraction, extractiontopic, mo):
    topics = mo.sql(
        """
        SELECT
            type,
            name,
            description,
            embedding
        FROM
            extraction.extractiontopic
        """,
        output=False,
        engine=engine,
    )
    return (topics,)


@app.cell
def _(mo):
    n_neighbors_topics = mo.ui.slider(
        start=1,
        stop=30,
        value=5,
        label="Neighbors for UMAP projection :",
        show_value=True,
    )
    mo.center(n_neighbors_topics)
    return (n_neighbors_topics,)


@app.cell
def _(n_neighbors_topics, reduce_and_display_embeddings, topics):
    umap_topic_mo_chart = reduce_and_display_embeddings(
        topics,
        "embedding",
        n_neighbors=n_neighbors_topics.value,
        tooltip_cols=["name"],
        color_col="type",
    )
    umap_topic_mo_chart
    return (umap_topic_mo_chart,)


@app.cell
def _(mo, umap_topic_mo_chart):
    mo.stop(len(umap_topic_mo_chart.value) == 0)

    if len(umap_topic_mo_chart.value) == 1:
        selected_topic = umap_topic_mo_chart.value.to_dict("records")[0]

        display_markdown = f"""
            **Name** : {selected_topic["name"]}

            **Type** : {selected_topic["type"]}

            **Description** : {selected_topic["description"]}
        """
    else:
        display_markdown = "**Topics titles :**\n\n"

        display_markdown += "- " + "\n- ".join(umap_topic_mo_chart.value["name"].values)

    mo.md(display_markdown)


if __name__ == "__main__":
    app.run()
