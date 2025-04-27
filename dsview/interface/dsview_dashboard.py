from typing import Literal

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlmodel import Session, create_engine
from streamlit.commands.page_config import Layout


from dsview.config import get_sqlite_url
from dsview.content.content_db_schema import update_content
from dsview.obsidian.obsidian_utils import get_content_url_link
from dsview.interface.interface_utils import get_sidebar

load_dotenv()

engine = create_engine(get_sqlite_url())

st.set_option("client.showSidebarNavigation", False)
st.set_page_config(page_title="Content dashboard", page_icon="📊", layout="wide")

# TODO Add option to change relevance


def extract_content() -> pd.DataFrame:
    extract_stmt = """
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
        FROM extractionresult
        JOIN inputcontent
        ON inputcontent.id = extractionresult.content_id
        ORDER BY inputcontent.upload_date DESC
    """

    with Session(engine) as session:
        df = pd.read_sql(extract_stmt, session.bind)

    df = df.astype({"already_read": bool})
    df["note_link"] = df["title"].apply(get_content_url_link)

    return df


def display_distrib_and_filter(
    df_content: pd.DataFrame,
    group_col: str,
    chart_type: Literal["pie", "bar"] = "pie",
    count_column: str = "title",
    title: str = "",
    column=None,
) -> pd.DataFrame:
    selection = alt.selection_point(name=f"count_{group_col}", toggle=False)

    raw_chart = alt.Chart(df_content, title=title)

    if chart_type == "pie":
        chart = raw_chart.mark_arc().encode(
            theta=alt.Theta(f"count({group_col}):Q").title("count"),
            color=alt.condition(selection, f"{group_col}:N", alt.value("lightgray")),
        )

    elif chart_type == "bar":
        chart = raw_chart.mark_bar().encode(
            x=alt.X(f"{group_col}:N").axis(labelAngle=0),
            y=alt.Y(f"count({group_col}):Q").title("count"),
            color=alt.condition(selection, f"{group_col}:N", alt.value("lightgray")),
        )
    else:
        raise ValueError(
            "Only pie and bar chart are supported by display_distrib_and_filter function."
        )

    chart = chart.add_params(selection)

    if column is None:
        event = st.altair_chart(chart, on_select="rerun")
    else:
        event = getattr(column, "altair_chart")(chart, on_select="rerun")

    param_selection = event["selection"][f"count_{group_col}"]
    if len(param_selection) > 0:
        df_content_filtered = df_content[
            df_content[group_col] == param_selection[0][group_col]
        ]

        return df_content_filtered
    return df_content


def extract_failed_ingestion() -> pd.DataFrame:
    extract_stmt = """
        SELECT
            link,
            upload_date,
            error_type,
            error_message
        FROM failedingestion
        JOIN inputcontent
        ON failedingestion.content_id=inputcontent.id
        ORDER BY upload_date DESC
    """
    with Session(engine) as session:
        df = pd.read_sql(extract_stmt, session.bind)

    return df


def display_contents(df_content: pd.DataFrame) -> list[int]:
    column_config = {
        "id": None,
        "title": st.column_config.TextColumn(
            "Title",
            # pinned=True,
        ),
        "content_type": None,
        "upload_date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
        "already_read": st.column_config.CheckboxColumn("Read ?", width="small"),
        "read_priority": st.column_config.ProgressColumn(
            "Read priority",
            format="%d/5",
            min_value=0,
            max_value=5,
            width="small",
        ),
        "relevance": st.column_config.ProgressColumn(
            "Relevance",
            format="%d/5",
            min_value=0,
            max_value=5,
            width="small",
        ),
        "source": st.column_config.TextColumn("Source", width="small"),
        "link": st.column_config.LinkColumn("Link", width="large"),
        "note_link": st.column_config.LinkColumn(
            "Note", display_text="note", width="small"
        ),
    }

    event = st.dataframe(
        df_content,
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="multi-row",
    )

    return event.selection["rows"]


def display_failed_ingestion(df_failed: pd.DataFrame):
    with st.expander("Failed ingestion"):
        column_config = {
            "link": st.column_config.LinkColumn("Link", width="large"),
            "upload_date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
            "error_type": st.column_config.TextColumn("Error type", width="small"),
            "error_message": st.column_config.TextColumn("Message", width="large"),
        }
        st.dataframe(
            df_failed,
            hide_index=True,
            use_container_width=True,
            column_config=column_config,
        )


# TODO add change relevance, read_priority and already_read


def update_form(selected_ids: list[int]):
    with st.container(border=True):
        col1, col2, col3 = st.columns([0.2, 0.4, 0.4], vertical_alignment="bottom")

        already_read = col1.toggle("already_read")

        if already_read:
            read_priority = col2.number_input("read_priority", value=0, disabled=True)
        else:
            read_priority = col2.number_input("read_priority", min_value=0, max_value=5)
        relevance = col3.number_input("relevance", min_value=0, max_value=5)

        update = st.button("Update contents", type="primary", use_container_width=True)

        if not update:
            return
        if len(selected_ids) == 0:
            st.warning("No content were selected for update")
            return
        with Session(engine) as session:
            for id in selected_ids:
                update_content(
                    session,
                    content_id=id,
                    already_read=already_read,
                    read_priority=read_priority,
                    relevance=relevance,
                )
        st.rerun()


def main():
    get_sidebar()
    st.title("Vault DB Viewer")

    df_content = extract_content()
    df_failed = extract_failed_ingestion()

    col1, col2, col3 = st.columns(3)

    df_content_filtered = display_distrib_and_filter(
        df_content,
        "already_read",
        chart_type="pie",
        title="Number of content already read",
        column=col1,
    )

    df_content_filtered = display_distrib_and_filter(
        df_content_filtered,
        "read_priority",
        chart_type="bar",
        title="Number of content per read priority",
        column=col2,
    )

    df_content_filtered = display_distrib_and_filter(
        df_content_filtered,
        "relevance",
        chart_type="bar",
        title="Number of content per relevance",
        column=col3,
    )

    df_content_filtered = df_content_filtered.reset_index(drop=True)

    selected_rows = display_contents(df_content_filtered)
    selected_ids = list(df_content_filtered.loc[selected_rows, "id"])

    update_form(selected_ids)

    display_failed_ingestion(df_failed)


if __name__ == "__main__":
    main()
