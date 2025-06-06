import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium", layout_file="layouts/dashboard.grid.json")


@app.cell
def _(mo):
    mo.center(mo.md(r"""# Content dashboard"""))
    return


@app.cell
def _():
    import altair as alt
    import marimo as mo
    from sqlmodel import Session

    from dsview.db.schemas import engine
    from dsview.db.ingest import update_content
    from dsview.obsidian.obsidian_utils import get_content_url_link
    return Session, alt, engine, get_content_url_link, mo, update_content


@app.cell
def _(mo):
    mo.md(r"""## Sources""")
    return


@app.cell
def _(mo):
    get_refresh, set_refresh = mo.state(0)
    return get_refresh, set_refresh


@app.cell
def _(engine, extractionresult, get_refresh, inputcontent, mo):
    get_refresh()

    sources = mo.sql(
        f"""
        SELECT
            inputcontent.id,
            title,
            content_type,
            upload_date,
            already_read,
            read_priority,
            relevance,
            source,
            link
        FROM
            extractionresult
            JOIN inputcontent ON inputcontent.id = extractionresult.content_id
        ORDER BY
            inputcontent.upload_date DESC
        """,
        output=False,
        engine=engine,
    )
    return (sources,)


@app.cell
def _(mo):
    mo.md(r"""## Display and filter""")
    return


@app.function
def coalesce_df(*args):
    if len(args) == 0:
        return None
    elif len(args[0]) == 0:
        return coalesce_df(*args[1:])
    return args[0]


@app.cell
def _(alt, mo, sources):
    read_sources_chart = mo.ui.altair_chart(
        alt.Chart(sources, title="Number of read content")
        .mark_arc()
        .encode(
            theta=alt.Theta("count(already_read):Q").title("count"),
            color="already_read:N",
        )
    )
    read_sources_chart
    return (read_sources_chart,)


@app.cell
def _(read_sources_chart, sources):
    filtered_read_sources = coalesce_df(read_sources_chart.value, sources)
    return (filtered_read_sources,)


@app.cell
def _(alt, filtered_read_sources, mo):
    read_priority_chart = mo.ui.altair_chart(
        alt.Chart(filtered_read_sources, title="Read priority accross contents")
        .mark_bar()
        .encode(
            x=alt.X("read_priority:N").axis(labelAngle=0),
            y=alt.Y("count(read_priority):Q").title("count"),
            color="read_priority:N",
        ),
    )

    read_priority_chart
    return (read_priority_chart,)


@app.cell
def _(filtered_read_sources, read_priority_chart):
    filtered_priority_sources = coalesce_df(
        read_priority_chart.value, filtered_read_sources
    )
    return (filtered_priority_sources,)


@app.cell
def _(alt, filtered_priority_sources, mo):
    relevance_chart = mo.ui.altair_chart(
        alt.Chart(filtered_priority_sources, title="Relevance accross contents")
        .mark_bar()
        .encode(
            x=alt.X("relevance:N").axis(labelAngle=0),
            y=alt.Y("count(relevance):Q").title("count"),
            color="relevance:N",
        ),
    )

    relevance_chart
    return (relevance_chart,)


@app.cell
def _(filtered_priority_sources, relevance_chart):
    filtered_sources = coalesce_df(
        relevance_chart.value, filtered_priority_sources
    )
    return (filtered_sources,)


@app.cell
def _(mo):
    mo.md(r"""## Display sources""")
    return


@app.cell
def _(filtered_sources, mo):
    sources_table = mo.ui.table(
        filtered_sources,
        selection="single",
        show_column_summaries=False,
        label="Ingested content",
        page_size=10,
    )
    sources_table
    return (sources_table,)


@app.cell
def _(mo, sources_table):
    mo.stop(len(sources_table.value) == 0)

    selected_content = sources_table.value.to_dict("records")[0]
    return (selected_content,)


@app.cell
def fetch_note_summary(engine, extractionresult, mo, selected_content):
    note_summary = mo.sql(
        f"""
        Select
            summary
        FROM 
            extractionresult
        WHERE content_id = {selected_content["id"]}
        """,
        output=False,
        engine=engine,
    )
    return (note_summary,)


@app.cell
def note_link(get_content_url_link, mo, selected_content):
    mo.center(
        mo.md(
            f"[Go to obsidian note]({get_content_url_link(selected_content['title'])})"
        )
    )
    return


@app.cell
def note_summary(mo, note_summary):
    mo.accordion({"Note summary": note_summary.loc[0, "summary"]})
    return


@app.cell
def _(mo):
    mo.md(r"""## Update form""")
    return


@app.cell
def _(mo, selected_content):
    already_read = mo.ui.checkbox(
        label="Content read", value=selected_content["already_read"] == 1
    )
    already_read
    return (already_read,)


@app.cell
def _(already_read, mo, selected_content):
    read_priority = mo.ui.slider(
        start=0,
        stop=5,
        value=(selected_content["read_priority"] if not already_read.value else 0),
        show_value=True,
        label="Read priority : ",
        disabled=already_read.value,
    )
    read_priority
    return (read_priority,)


@app.cell
def _(already_read, mo, selected_content):
    content_relevance = mo.ui.slider(
        start=0,
        stop=5,
        value=(selected_content["relevance"] if already_read.value else 0),
        show_value=True,
        label="Content relevance : ",
        disabled=not already_read.value,
    )
    content_relevance
    return (content_relevance,)


@app.cell
def _(mo, sources_table):
    mo.stop(len(sources_table.value) == 0)

    submit_update = mo.ui.run_button(
        label="Update content",
        kind="success",
        full_width=True,
    )
    submit_update
    return (submit_update,)


@app.cell
def _(
    Session,
    already_read,
    content_relevance,
    engine,
    mo,
    read_priority,
    selected_content,
    set_refresh,
    submit_update,
    update_content,
):
    mo.stop(not submit_update.value)

    with Session(engine) as session:
        update_content(
            session,
            content_id=selected_content["id"],
            already_read=already_read.value,
            read_priority=read_priority.value,
            relevance=content_relevance.value,
        )

    set_refresh(0)
    return


if __name__ == "__main__":
    app.run()
