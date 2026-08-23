import asyncio

import streamlit as st
from pydantic import HttpUrl
from sqlmodel import Session

from dsview.db import engine
from dsview.db.query import check_link_labelled
from dsview.extraction.content_extraction import ContentExtractor
from dsview.extraction.content_loader import get_content_loader
from dsview.interface.labelling.content_labelling_form import generate_labelling_form

st.set_page_config(page_title="Content labelling", page_icon="small_icon.png")
# how to feed new url ? > list input or random

# Delta storage ?
# Multiple run of extraction ?


def content_extraction(link: str):
    content_loader = get_content_loader(HttpUrl(link))
    content_loader.load()

    content_extractor = ContentExtractor()

    # TODO Clean this
    _, content_description, topics, content_links = asyncio.run(
        content_extractor.run_extraction(
            content_loader,
        )
    )

    return (
        content_loader,
        content_description,
        topics,
        content_links,
        content_loader.content_links,
    )


# TODO would be better if session was a cached resource


def main():
    if "labelling" not in st.session_state:
        st.session_state["labelling"] = False

    # st.title("Content labelling")
    with Session(engine) as session:
        col1, col2 = st.columns([2, 1], vertical_alignment="bottom")

        link = col1.text_input("Content url")
        confirm_url = col2.button("Confirm")

        if confirm_url:
            already_exist = check_link_labelled(link, session)
            if already_exist:
                st.error("This url has already been labelled.")
                return

            st.session_state["labelling"] = True
            with st.spinner("Extracting content..."):
                st.session_state.extraction_results = content_extraction(link)

        if st.session_state["labelling"]:
            generate_labelling_form(session, *st.session_state.extraction_results)


main()
