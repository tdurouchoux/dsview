import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium", layout_file="layouts/db_management.grid.json")


@app.cell
def _(mo):
    mo.center(mo.md(r"""# Content upload dashboard"""))
    return


@app.cell
def _():
    from datetime import date

    import altair as alt
    import dayplot as dp
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from pydantic import HttpUrl


    from dsview.db.schemas import engine
    return HttpUrl, alt, date, dp, engine, mo, pd, plt


@app.cell
def _(engine, inputcontent, mo):
    min_date = mo.sql(
        f"""
        SELECT 
            min(upload_date) as min_date
        FROM
            inputcontent
        """,
        engine=engine
    )
    return (min_date,)


@app.cell
def _(date, min_date, mo):
    min_year = int(min_date.loc[0, "min_date"][:4])
    max_year = date.today().year

    select_year = mo.md("## Uploads for year {year}").batch(
        year=mo.ui.dropdown(
            options=[year for year in range(min_year, max_year + 1)],
            value=max_year,
        )
    )
    select_year
    return (select_year,)


@app.cell
def _(select_year):
    selected_year = select_year.value["year"]
    return (selected_year,)


@app.cell
def _(engine, inputcontent, mo, selected_year):
    content_upload = mo.sql(
        f"""
        SELECT
            upload_date,
            link,
            source
        FROM 
            inputcontent
        WHERE upload_date > \"{str(selected_year)}-01-01\" and upload_date < \"{str(selected_year + 1)}-01-01\"
        """,
        engine=engine,
    )
    return (content_upload,)


@app.cell
def _(content_upload, dp, plt, selected_year):
    content_upload_dates = content_upload.value_counts("upload_date").reset_index()

    fig, ax = plt.subplots(figsize=(15, 6))
    dp.calendar(
        dates=content_upload_dates["upload_date"],
        values=content_upload_dates["count"],
        start_date=f"{selected_year}-01-01",
        end_date=f"{selected_year}-12-31",
        boxstyle="circle",
        ax=ax,
    )
    return


@app.cell
def _(alt, content_upload):
    time_chart = (
        alt.Chart(content_upload, title="Uploads per source")
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(upload_date):T").title("month"),
            y="count(source)",
            color="source",
            tooltip=["source", "yearmonth(upload_date)"],
        )
    )

    time_chart
    return


@app.cell
def _(HttpUrl, content_upload, mo):
    content_upload["host"] = content_upload["link"].apply(
        lambda l: HttpUrl(l).host
    )

    mo.ui.table(
        content_upload.value_counts("host").to_frame().head(10),
        selection=None,
        label="Most frequent content link host",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
        ## Failed ingestion
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""---""")
    return


@app.cell
def _(engine, failedingestion, inputcontent, mo):
    failed_ingestion = mo.sql(
        f"""
        SELECT
            link,
            upload_date,
            error_type,
            error_message
        FROM
            failedingestion
            JOIN inputcontent ON failedingestion.content_id = inputcontent.id
        ORDER BY
            upload_date DESC
        """,
        engine=engine
    )
    return (failed_ingestion,)


@app.cell
def _(engine, extractionresult, mo):
    count_success = mo.sql(
        f"""
        SELECT
            count(*) as count
        FROM
            extractionresult
        """,
        engine=engine
    )
    return (count_success,)


@app.cell
def _(alt, count_success, failed_ingestion, pd):
    count_input = pd.DataFrame(
        [
            {
                "status": "failed",
                "count": len(failed_ingestion),
            },
            {
                "status": "success",
                "count": count_success["count"].values[0],
            },
        ]
    )

    alt.Chart(count_input, title="Number of failed ingestion").mark_arc().encode(
        theta="count:Q",
        color="status:N",
        tooltip="status:N",
    )
    return


@app.cell
def _(alt, failed_ingestion):
    alt.Chart(
        failed_ingestion, title="Failed ingestion per error type"
    ).mark_bar().encode(
        alt.X("count(error_type):Q").title("count"),
        alt.Y("error_type:N"),
        tooltip=["error_type"],
    )
    return


@app.cell
def _(failed_ingestion, mo):
    mo.ui.table(
        failed_ingestion,
        selection=None,
        page_size=5,
    )
    return


if __name__ == "__main__":
    app.run()
