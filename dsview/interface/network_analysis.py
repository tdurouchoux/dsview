import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from dsview.db import engine
    return engine, mo


@app.cell
def _(contenttopicrelation, engine, mo):
    relations = mo.sql(
        f"""
        SELECT 
         *
        FROM contenttopicrelation
        """,
        engine=engine
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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
