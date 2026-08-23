import marimo

__generated_with = "0.14.16"
app = marimo.App(
    width="medium",
    layout_file="layouts/extraction_dashboard.grid.json",
    css_file="layouts/marimo_style.css",
)


@app.cell
def _(get_sidebar):
    get_sidebar()


@app.cell
def _(mo):
    mo.center(mo.md(r"""# Extraction dashboard"""))


@app.cell
def _():
    from urllib.parse import urlunparse

    import altair as alt
    import marimo as mo
    from pydantic import HttpUrl

    from dsview.db import engine
    from dsview.interface.dashboard.marimo_sidebar import get_sidebar

    return HttpUrl, alt, engine, get_sidebar, mo, urlunparse


@app.cell
def _(mo):
    mo.md(r"""## Content""")


@app.cell
def _(engine, extraction, extractionresult, mo):
    count_content_type = mo.sql(
        """
        SELECT
            content_type,
            count(content_id) as nb_content
        FROM
            extraction.extractionresult
        GROUP BY
            content_type
        """,
        output=False,
        engine=engine,
    )
    return (count_content_type,)


@app.cell
def _(alt, count_content_type):
    content_type_chart = (
        alt.Chart(count_content_type.reset_index(), title="Content per type")
        .mark_arc()
        .encode(
            theta="nb_content:Q",
            color="content_type:N",
            tooltip=["content_type", "nb_content"],
        )
    )
    return (content_type_chart,)


@app.cell
def _(engine, extraction, extractiontag, mo):
    count_tags_name = mo.sql(
        """
        SELECT
            name,
            count(id) as nb_content
        FROM
            extraction.extractiontag
        GROUP BY
            name
        ORDER BY nb_content DESC
        """,
        output=False,
        engine=engine,
    )
    return (count_tags_name,)


@app.cell
def _(alt, count_tags_name):
    top_tags_chart = (
        alt.Chart(count_tags_name.head(10), title="Most common tags")
        .mark_bar()
        .encode(
            x="nb_content:Q",
            y=alt.Y("name:N", sort="-x"),
        )
    )
    return (top_tags_chart,)


@app.cell
def _(engine, extraction, extractiontag, mo):
    count_content_tag = mo.sql(
        """
        SELECT
            count(id) as nb_tags
        FROM
            extraction.extractiontag
        GROUP BY
            content_id
        """,
        output=False,
        engine=engine,
    )
    return (count_content_tag,)


@app.cell
def _(alt, count_content_tag):
    content_tags_chart = (
        alt.Chart(count_content_tag, title="Number of tags per content")
        .mark_bar()
        .encode(
            y="count(nb_tags):Q",
            x="nb_tags:O",
            color=alt.Theta("nb_tags:N", title="Number"),
            tooltip=["nb_tags:Q"],
        )
    )
    return (content_tags_chart,)


@app.cell
def _(engine, extraction, extractionresult, mo):
    summary_size = mo.sql(
        """
        SELECT
            LENGTH(summary) as len_summary
        FROM
            extraction.extractionresult
        """,
        output=False,
        engine=engine,
    )
    return (summary_size,)


@app.cell
def _(alt, summary_size):
    summary_size_chart = (
        alt.Chart(summary_size, title="Summary size distribution")
        .mark_bar()
        .encode(
            alt.X("len_summary:Q", bin=True, title="Summary size"),
            alt.Y("count()", title="Number of content"),
        )
    )
    return (summary_size_chart,)


@app.cell
def _(
    content_tags_chart,
    content_type_chart,
    mo,
    summary_size_chart,
    top_tags_chart,
):
    content_layout = mo.vstack(
        (
            mo.hstack(
                (content_type_chart, content_tags_chart, summary_size_chart), gap=0
            ),
            top_tags_chart,
            # mo.hstack((), justify="space-around")
        )
    )
    return (content_layout,)


@app.cell
def _(mo):
    mo.md(r"""## Topics""")


@app.cell
def _(engine, extraction, extractiontopic, mo):
    topics = mo.sql(
        """
        SELECT
            type,
            name
        FROM
            extraction.extractiontopic
        """,
        output=False,
        engine=engine,
    )
    return (topics,)


@app.cell
def _(alt, mo, topics):
    topic_type_chart = mo.ui.altair_chart(
        alt.Chart(topics, title="Topics per type")
        .mark_arc()
        .encode(
            theta=alt.Theta("count(type):Q", title="count per type"),
            color="type:N",
            tooltip=["type:N", "count(type):Q"],
        )
    )
    return (topic_type_chart,)


@app.cell
def _(topic_type_chart):
    selected_topic = topic_type_chart.value.drop(columns="type")
    return (selected_topic,)


@app.cell
def _(contenttopicrelation, engine, extraction, mo):
    topic_per_content = mo.sql(
        """
        SELECT
            count(topic_id) as nb_link
        FROM
            extraction.contenttopicrelation
        GROUP BY
            content_id
        """,
        output=False,
        engine=engine,
    )
    return (topic_per_content,)


@app.cell
def _(alt, topic_per_content):
    topic_per_content_chart = (
        alt.Chart(topic_per_content, title="Number of topic per content")
        .mark_bar()
        .encode(
            alt.X("nb_link:O", title="Number of topic"),
            alt.Y("count(nb_link):Q", title="Number of content"),
        )
    )
    return (topic_per_content_chart,)


@app.cell
def _(mo, selected_topic, topic_per_content_chart, topic_type_chart):
    topic_layout = mo.vstack(
        (
            mo.hstack((topic_type_chart, selected_topic), justify="center"),
            topic_per_content_chart,
        )
    )
    return (topic_layout,)


@app.cell
def _(mo):
    mo.md(r"""## ER comparisons""")


@app.cell
def _(engine, ercomparison, extraction, mo):
    er_decision_stat = mo.sql(
        """
        SELECT
            sum(CASE WHEN merge_topic THEN 1 ELSE 0 END) as count_pos,
            count(id) as nb_comparison
        FROM
            extraction.ercomparison
        """,
        output=False,
        engine=engine,
    )
    return (er_decision_stat,)


@app.cell
def _(engine, extraction, extractionresult, mo):
    count_content = mo.sql(
        """
        SELECT
            COUNT(DISTINCT content_id) as nb_content
        FROM
            extraction.extractionresult
        """,
        output=False,
        engine=engine,
    )
    return (count_content,)


@app.cell
def _(count_content, er_decision_stat, mo):
    nb_comp_stat = mo.stat(
        label="Total number of comparisons",
        value=er_decision_stat.loc[0, "nb_comparison"],
        bordered=True,
    )

    avg_comp_per_content = (
        er_decision_stat.loc[0, "nb_comparison"] / count_content.loc[0, "nb_content"]
    )

    avg_comp_stat = mo.stat(
        label="Average number of ER comparison per content",
        value=f"{avg_comp_per_content:.1f}",
        bordered=True,
    )

    pos_rate = (
        er_decision_stat.loc[0, "count_pos"] / er_decision_stat.loc[0, "nb_comparison"]
    )

    pos_rate_stat = mo.stat(
        label="Percentage of positive comparison",
        value=f"{pos_rate * 100:.1f} %",
        bordered=True,
    )
    return avg_comp_stat, nb_comp_stat, pos_rate_stat


@app.cell
def _(engine, ercomparison, extraction, mo):
    er_topic_distance = mo.sql(
        """
        SELECT
            name_1,
            name_2,
            fts_score,
            vss_distance,
            merge_topic
        FROM
            extraction.ercomparison
        """,
        output=False,
        engine=engine,
    )
    return (er_topic_distance,)


@app.cell
def _(alt, er_topic_distance):
    candidate_distance_chart = (
        alt.Chart(er_topic_distance, title="ER comparison distance")
        .mark_point()
        .encode(
            alt.X("fts_score:Q", title="Full text distance"),
            alt.Y("vss_distance:Q", title="Embeddding cosine score"),
            color="merge_topic:N",
            tooltip=["name_1", "name_2"],
        )
    )
    return (candidate_distance_chart,)


@app.cell
def _():
    # Most common names from comparisons (name_2)
    return


@app.cell
def _(engine, ercomparison, extraction, mo):
    count_er_name = mo.sql(
        """
        SELECT
            name_2,
            COUNT(id) as nb_comparison
        FROM extraction.ercomparison
        GROUP BY name_2
        ORDER BY nb_comparison DESC
        """,
        output=False,
        engine=engine,
    )
    return (count_er_name,)


@app.cell
def _(alt, count_er_name):
    candidate_topic_chart = (
        alt.Chart(count_er_name.iloc[:10], title="Most common candidate topic")
        .mark_bar()
        .encode(
            alt.X("nb_comparison:Q", title="Number of comparisons"),
            alt.Y("name_2:N", title="Candidate topic name", sort="-x"),
        )
    )
    return (candidate_topic_chart,)


@app.cell
def _(
    avg_comp_stat,
    candidate_distance_chart,
    candidate_topic_chart,
    mo,
    nb_comp_stat,
    pos_rate_stat,
):
    er_layout = mo.hstack(
        (
            mo.vstack((nb_comp_stat, avg_comp_stat, pos_rate_stat), justify="center"),
            mo.vstack((candidate_distance_chart, candidate_topic_chart)),
        ),
        widths=[0.2, 0.8],
        align="center",
    )
    return (er_layout,)


@app.cell
def _(mo):
    mo.md(r"""## Content link""")


@app.cell
def _(engine, extraction, extractionlink, mo):
    link_per_content = mo.sql(
        """
        SELECT
            count(id) as nb_link
        FROM
            extraction.extractionlink
        GROUP BY
            content_id
        """,
        output=False,
        engine=engine,
    )
    return (link_per_content,)


@app.cell
def _(alt, link_per_content):
    link_per_content_chart = (
        alt.Chart(link_per_content, title="Number of link per content")
        .mark_bar()
        .encode(
            alt.X("nb_link:O", title="Number of links"),
            alt.Y("count(nb_link):Q", title="Number of content"),
        )
    )
    return (link_per_content_chart,)


@app.cell
def _(engine, extraction, extractionlink, mo):
    links_url = mo.sql(
        """
        SELECT
            url
        FROM
            extraction.extractionlink
        """,
        engine=engine,
    )
    return (links_url,)


@app.cell
def _(HttpUrl, links_url, urlunparse):
    links_url["url"] = links_url["url"].apply(lambda url: HttpUrl(url))
    links_url["host"] = links_url["url"].apply(lambda url: url.host)
    links_url["clean_url"] = links_url["url"].apply(
        lambda url: HttpUrl(urlunparse((url.scheme, url.host, url.path, "", "", "")))
    )


@app.cell
def _(alt, links_url):
    most_common_host_chart = (
        alt.Chart(links_url.value_counts("host").reset_index().head(10))
        .mark_bar()
        .encode(
            alt.X("count:Q", title="Number of reference"),
            alt.Y("host:N", title="Url host", sort="-x"),
        )
    )
    return (most_common_host_chart,)


@app.cell
def _(content, engine, inputcontent, mo):
    input_content_link = mo.sql(
        """
        SELECT
            link
        FROM content.inputcontent
        """,
        output=False,
        engine=engine,
    )
    return (input_content_link,)


@app.cell
def _(HttpUrl, input_content_link):
    input_content_link["link"] = input_content_link["link"].apply(
        lambda link: HttpUrl(link)
    )


@app.cell
def _(input_content_link, link_per_content, links_url, mo):
    nb_link_stat = mo.stat(
        value=link_per_content.nb_link.sum(),
        label="Total number of links",
        bordered=True,
    )

    ratio_existing_url = (
        links_url.clean_url.isin(input_content_link.link).sum()
        / link_per_content.nb_link.sum()
    )

    existing_url_stat = mo.stat(
        value=f"{ratio_existing_url * 100:.2f} %",
        label="Percentage of existing link in input",
        bordered=True,
    )
    return existing_url_stat, nb_link_stat


@app.cell
def _(
    existing_url_stat,
    link_per_content_chart,
    mo,
    most_common_host_chart,
    nb_link_stat,
):
    content_links_layout = mo.hstack(
        (
            mo.vstack((nb_link_stat, existing_url_stat)),
            mo.vstack((link_per_content_chart, most_common_host_chart)),
        ),
        widths=[0.2, 0.8],
        align="center",
    )
    return (content_links_layout,)


@app.cell
def _(content_links_layout):
    content_links_layout


@app.cell
def _(mo):
    mo.md(r"""## Full layout""")


@app.cell
def _(content_layout, content_links_layout, er_layout, mo, topic_layout):
    mo.ui.tabs(
        {
            "Content extraction": content_layout,
            "Content links": content_links_layout,
            "Topics detection": topic_layout,
            "ER comparisons": er_layout,
        }
    )



@app.cell
def _():
    ## Graph analysis ?
    return


if __name__ == "__main__":
    app.run()
