import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from sqlmodel import SQLModel

    # Get table names from SQLModel metadata
    from dsview.db import engine
    from dsview.interface.dashboard.marimo_sidebar import get_sidebar
    return SQLModel, engine, get_sidebar, mo


@app.cell
def _(get_sidebar):
    get_sidebar()
    return


@app.cell
def _(mo):
    mo.center(mo.md(r"""# Database explorer"""))
    return


@app.cell
def _(SQLModel):
    table_names = list(SQLModel.metadata.tables.keys())

    return (table_names,)


@app.cell
def _(mo, table_names):
    select_table = mo.ui.dropdown(table_names, label="Select table to display :")
    mo.center(select_table)
    return (select_table,)


@app.cell
def _(engine, mo, select_table):
    mo.stop(select_table.value is None)

    _df = mo.sql(
        f"""
        SELECT * FROM {select_table.value}
        """,
        engine=engine
    )
    return


if __name__ == "__main__":
    app.run()
