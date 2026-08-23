import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx

    from dsview.db import engine

    return engine, mo, nx, plt


@app.cell
def _(contenttopicrelation, engine, mo):
    relations = mo.sql(
        """
        SELECT 
         *
        FROM contenttopicrelation
        """,
        engine=engine,
    )
    return (relations,)


@app.cell
def _(relations):
    relations_prep = relations.astype(str)
    relations_prep["content_id"] = "content_" + relations_prep["content_id"]
    relations_prep["topic_id"] = "topic_" + relations_prep["topic_id"]

    # relations["content_id"] = relations["id"]
    return (relations_prep,)


@app.cell
def _(relations_prep):
    relations_prep


@app.cell
def _(nx, relations_prep):
    topic_graph = nx.from_pandas_edgelist(
        relations_prep,
        source="content_id",
        target="topic_id",
    )
    return (topic_graph,)


@app.cell
def _(nx, topic_graph):
    comp = nx.connected_components(topic_graph)
    return (comp,)


@app.cell
def _(comp):
    [c for c in comp]


@app.cell
def _(nx, plt, topic_graph):
    nx.draw(topic_graph)
    plt.show()


@app.cell
def _(nx, topic_graph):
    pr = nx.pagerank(topic_graph)
    return (pr,)


@app.cell
def _(pr):
    pr


@app.cell
def _(pr):
    dict(sorted(pr.items(), key=lambda item: item[1]))


@app.cell
def _(engine, extractiontopic, mo):
    _df = mo.sql(
        """
        SELECT
            *
        FROM extractiontopic
        WHERE id=3
        """,
        engine=engine,
    )


@app.cell
def _():
    # Have some metrics on connected components
    # Display top topics according to pagerank
    # Display most central topic / content (with some centrality metrics)
    # Detect some kind of communities
    # density ? k-cores ?

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
