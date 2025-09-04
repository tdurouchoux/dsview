import streamlit as st


# --server.baseUrlPath=/labelling --server.port 8000
def main():
    content_labelling_page = st.Page(
        "labelling/content_labelling.py", title="Content labelling"
    )
    er_labelling_page = st.Page("labelling/er_labelling.py", title="ER labelling")

    pg = st.navigation(
        [
            content_labelling_page,
            er_labelling_page,
        ]
    )

    pg.run()


if __name__ == "__main__":
    main()
