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
        f"""
        SELECT * FROM extraction.extractionresult
        """,
        engine=engine
    )
    return (extraction_results,)


@app.cell
def _(extraction_results):
    extraction_results["title"].nunique()
    return


@app.cell
def _(extraction_results):
    extraction_results["title_lower"] = extraction_results["title"].str.lower()
    return


@app.cell
def _(extraction_results):
    extraction_results[extraction_results.duplicated(subset=["title_lower"], keep=False)]
    return


@app.cell
def _(extraction_results):
    extraction_results[extraction_results.duplicated(subset=["title"], keep=False)]
    return


@app.cell(hide_code=True)
def _(engine, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM content.inputcontent WHERE id IN (643, 744, 745)
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _(engine, mo):
    extraction_topics = mo.sql(
        f"""
        SELECT * FROM extraction.extractiontopic
        """,
        engine=engine
    )
    return (extraction_topics,)


@app.cell
def _(extraction_topics):
    extraction_topics["name"].nunique()
    return


@app.cell
def _(extraction_topics):
    extraction_topics["name_lower"] = extraction_topics["name"].str.lower()
    return


@app.cell
def _(extraction_topics):
    extraction_topics[extraction_topics.duplicated(subset=["name_lower"], keep=False)]
    return


@app.cell
def _(extraction_topics):
    extraction_topics[extraction_topics["name_lower"].str.startswith("in-context")]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    update date > Max date linked contents
    """)
    return


@app.cell
def _():
    708 > 740
    return


@app.cell(hide_code=True)
def _(engine, mo):
    _df = mo.sql(
        f"""
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
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _(engine, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM content.failedingestion
        """,
        engine=engine
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
