import streamlit as st

from langchain_openai import ChatOpenAI
from pydantic import HttpUrl

from dsview.content.content_loader import get_content_loader
from dsview.extraction.content_extraction import ContentExtractor
from dsview.labelling.form import generate_labelling_form
from dsview.labelling.schema import get_engine, check_link_labelled
from dsview.config import load_db_config, load_model_config

# Load config
db_config = load_db_config()
model_config = load_model_config()

# how to feed new url ? > list input or random

# Delta storage ?
# Multiple run of extraction ?


def content_extraction(link: str):
    content_loader = get_content_loader(HttpUrl(link), model_config.token_limit)

    llm = ChatOpenAI(temperature=0, model_name=model_config.name)
    content_extractor = ContentExtractor(llm)

    _, content_description, topics, content_links = content_extractor.extract_content(
        content_loader
    )

    content_loader.get_hyperlink()
    all_links = content_loader.content_links

    return content_loader, content_description, topics, content_links, all_links


def main():
    if "engine" not in st.session_state:
        st.session_state["engine"] = get_engine(db_config.sqlite_url)

    if "labelling" not in st.session_state:
        st.session_state["labelling"] = False

    st.title("Content labelling")

    col1, col2 = st.columns([2, 1], vertical_alignment="bottom")

    link = col1.text_input("Content url")
    confirm_url = col2.button("Confirm")

    if confirm_url:
        
        already_exist = check_link_labelled(st.session_state.engine, link)
        if already_exist:
            st.error("This url has already been labelled.")
            return
        
        st.session_state["labelling"] = True
        with st.spinner("Extracting content..."):
            st.session_state.extraction_results = content_extraction(link)

    if st.session_state["labelling"]:
        generate_labelling_form(*st.session_state.extraction_results)


if __name__ == "__main__":
    main()
