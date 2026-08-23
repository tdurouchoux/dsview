import marimo

__generated_with = "0.14.16"
app = marimo.App(
    width="medium",
    layout_file="layouts/upload_dashboard.grid.json",
    css_file="layouts/marimo_style.css",
)


@app.cell
def _():
    from datetime import date

    import altair as alt
    import dayplot as dp
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from pydantic import HttpUrl

    from dsview.db import engine
    from dsview.interface.dashboard.marimo_sidebar import get_sidebar

    return HttpUrl, alt, date, dp, engine, get_sidebar, mo, pd, plt


@app.cell
def _(get_sidebar):
    get_sidebar()


@app.cell
def _(mo):
    mo.center(mo.md(r"""# Upload dashboard"""))



@app.cell
def _(content, engine, inputcontent, mo):
    min_date = mo.sql(
        """
        SELECT
            min(upload_date) as min_date
        FROM
            content.inputcontent
        """,
        output=False,
        engine=engine,
    )
    return (min_date,)


@app.cell
def _(date, min_date, mo):
    min_year = min_date.loc[0, "min_date"].year
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
def _(content, engine, inputcontent, mo, selected_year):
    content_upload = mo.sql(
        f"""
        SELECT
            upload_date,
            link,
            source
        FROM
            content.inputcontent
        WHERE upload_date > \'{selected_year!s}-01-01\' and upload_date < \'{selected_year + 1!s}-01-01\'


        """,
        engine=engine,
        output=False,
    )
    return (content_upload,)


@app.cell
def _(content_upload, dp, plt, selected_year):
    content_upload_dates = content_upload.value_counts("upload_date").reset_index()

    _, ax = plt.subplots(figsize=(15, 6))
    dp.calendar(
        dates=content_upload_dates["upload_date"],
        values=content_upload_dates["count"],
        start_date=f"{selected_year}-01-01",
        end_date=f"{selected_year}-12-31",
        boxstyle="circle",
        ax=ax,
    )


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


@app.cell
def _(HttpUrl, content_upload, mo):
    content_upload["host"] = content_upload["link"].apply(lambda l: HttpUrl(l).host)

    mo.ui.table(
        content_upload.value_counts("host").to_frame().head(10),
        selection=None,
        label="Most frequent content link host",
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## Failed ingestion
    """
    )


@app.cell
def _(content, engine, failedingestion, inputcontent, mo):
    failed_ingestion = mo.sql(
        """
        SELECT
            link,
            upload_date,
            error_type,
            error_message
        FROM
            content.failedingestion
            JOIN content.inputcontent ON failedingestion.content_id = inputcontent.id
        ORDER BY
            upload_date DESC
        """,
        output=False,
        engine=engine,
    )
    return (failed_ingestion,)


@app.cell
def _(engine, extraction, extractionresult, mo):
    count_success = mo.sql(
        """
        SELECT
            count(*) as count
        FROM
            extraction.extractionresult
        """,
        output=False,
        engine=engine,
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


@app.cell
def _(alt, failed_ingestion):
    alt.Chart(
        failed_ingestion, title="Failed ingestion per error type"
    ).mark_bar().encode(
        alt.X("count(error_type):Q").title("count"),
        alt.Y("error_type:N"),
        tooltip=["error_type"],
    )


@app.cell
def _(failed_ingestion, mo):
    mo.ui.table(
        failed_ingestion,
        selection=None,
        page_size=5,
    )


if __name__ == "__main__":
    app.run()
