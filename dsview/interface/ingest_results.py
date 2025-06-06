import marimo

__generated_with = "0.13.11"
app = marimo.App(
    width="medium",
    layout_file="layouts/ingest_results.grid.json",
)


@app.cell
def _(mo):
    mo.center(mo.md(r"""# Extraction results"""))
    return


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import umap


    from dsview.db.schemas import engine
    return alt, engine, mo, np, umap


@app.cell
def _(mo):
    mo.md(r"""## Content type""")
    return


@app.cell
def _(engine, extractionresult, mo):
    count_content_type = mo.sql(
        f"""
        SELECT
            content_type,
            count(content_id) as count
        FROM
            extractionresult
        GROUP by
            content_type
        """,
        output=False,
        engine=engine
    )
    return (count_content_type,)


@app.cell
def _(alt, count_content_type):
    alt.Chart(count_content_type, title="Number of content per type").mark_arc().encode(
        theta="count",
        color="content_type",
        tooltip=["content_type", "count"]
    )
    return


@app.cell
def _():
    # TODO Add content embeddings
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## Tags
    """
    )
    return


@app.cell
def _(engine, extractiontag, mo):
    content_tags = mo.sql(
        f"""
        SELECT
            content_id,
            name
        FROM
            extractiontag
        """,
        output=False,
        engine=engine
    )
    return (content_tags,)


@app.cell
def _(alt, content_tags):
    count_content_tags = (
        content_tags.value_counts("name")
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    alt.Chart(count_content_tags, title="Most common tags").mark_bar().encode(
        x="count:Q",
        y=alt.Y("name:N", sort="-x"),
    )
    return


@app.cell
def _(alt, content_tags):
    alt.Chart(content_tags, title="Number of tags per content").transform_window(
        count="count()",
        groupby=["content_id"]
    ).mark_arc().encode(
        theta="count(count):Q",
        color=alt.Theta("count:N", title="Number"),
        tooltip=["count:N", "count(count):Q"]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## Topics
    """
    )
    return


@app.cell
def _(engine, extractiontopic, mo):
    topics = mo.sql(
        f"""
        SELECT
            type,
            name,
            description,
            embedding
        FROM
            extractiontopic
        """,
        output=False,
        engine=engine
    )
    return (topics,)


@app.cell
def _(alt, topics):
    alt.Chart(topics[["type"]], title="Topics per type").mark_arc().encode(
        theta=alt.Theta("count(type):Q", title="count per type"),
        color="type:N",
        tooltip=["type:N", "count(type):Q"]
    )
    return


@app.cell
def _(alt, mo, np, topics, umap):
    embeddings = np.array(
        list(
            topics["embedding"]
            .apply(lambda x: [float(e) for e in x.split(",")])
            .values
        )
    )

    reducer = umap.UMAP(n_neighbors=10)

    reduced_embeddings = reducer.fit_transform(embeddings)

    topics["reduced_embedding_x"] = reduced_embeddings[:, 0]
    topics["reduced_embedding_y"] = reduced_embeddings[:, 1]

    umap_chart = (
        alt.Chart(
            topics[
                [
                    "name",
                    "type",
                    "description",
                    "reduced_embedding_x",
                    "reduced_embedding_y",
                ]
            ],
            title="Topics embeddings",
        )
        .mark_point()
        .encode(
            x="reduced_embedding_x:Q",
            y="reduced_embedding_y:Q",
            color="type:N",
            tooltip=["name"],
        )
    )
    umap_mo_chart = mo.ui.altair_chart(umap_chart)
    umap_mo_chart
    return (umap_mo_chart,)


@app.cell
def _(umap_mo_chart):
    len(umap_mo_chart.value)
    return


@app.cell
def _(mo, umap_mo_chart):
    mo.stop(len(umap_mo_chart.value)==0)

    if len(umap_mo_chart.value)==1: 
        selected_topic = umap_mo_chart.value.to_dict("records")[0]
    
        display_markdown = f"""
            **Name** : {selected_topic["name"]}
    
            **Type** : {selected_topic["type"]}
    
            **Description** : {selected_topic["description"]}
        """
    else: 
        display_markdown = "**Topics titles :**\n\n"
        
        display_markdown += "- " + "\n- ".join(umap_mo_chart.value["name"].values) 

    mo.md(display_markdown)
    return


@app.cell
def _():
    ## Graph analysis ? 
    return


if __name__ == "__main__":
    app.run()
