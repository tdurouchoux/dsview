import streamlit as st


def get_sidebar():
    with st.sidebar:
        st.page_link("dsview_dashboard.py", label="Content dashboard", icon="📊")
        st.page_link("pages/content_labelling.py", label="Content labelling", icon="📚")
        st.page_link(
            "pages/er_labelling.py", label="ER comparison labelling", icon="⛓️‍💥"
        )
