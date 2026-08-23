import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from dsview.db import get_engine

    return get_engine, mo


@app.cell
def _(get_engine):
    engine = get_engine()
    return (engine,)


@app.cell(hide_code=True)
def _(engine, mo):
    extraction_results = mo.sql(
        """
        SELECT * FROM extraction.extractionresult
        """,
        engine=engine,
    )
    return (extraction_results,)


@app.cell
def _(extraction_results):
    extraction_results["title"].nunique()


@app.cell
def _(extraction_results):
    extraction_results["title_lower"] = extraction_results["title"].str.lower()


@app.cell
def _(extraction_results):
    extraction_results[
        extraction_results.duplicated(subset=["title_lower"], keep=False)
    ]


@app.cell
def _(extraction_results):
    extraction_results[extraction_results.duplicated(subset=["title"], keep=False)]


@app.cell(hide_code=True)
def _(engine, mo):
    _df = mo.sql(
        """
        SELECT * FROM content.inputcontent WHERE id IN (643, 744, 745)
        """,
        engine=engine,
    )


@app.cell(hide_code=True)
def _(engine, mo):
    extraction_topics = mo.sql(
        """
        SELECT * FROM extraction.extractiontopic
        """,
        engine=engine,
    )
    return (extraction_topics,)


@app.cell
def _(extraction_topics):
    extraction_topics["name"].nunique()


@app.cell
def _(extraction_topics):
    extraction_topics["name_lower"] = extraction_topics["name"].str.lower()


@app.cell
def _(extraction_topics):
    extraction_topics[extraction_topics.duplicated(subset=["name_lower"], keep=False)]


@app.cell
def _(extraction_topics):
    extraction_topics[extraction_topics["name_lower"].str.startswith("in-context")]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    update date > Max date linked contents
    """)


@app.cell
def _():
    708 > 740


@app.cell(hide_code=True)
def _(engine, mo):
    _df = mo.sql(
        """
        SELECT
            *
        FROM
            extraction.contenttopicrelation t1
        JOIN
            content.inputcontent t2
        on t1.content_id=t2.id
        WHERE
            topic_id = 1198
        LIMIT
            100
        """,
        engine=engine,
    )


@app.cell(hide_code=True)
def _(engine, mo):
    _df = mo.sql(
        """
        SELECT * FROM content.failedingestion
        """,
        engine=engine,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
