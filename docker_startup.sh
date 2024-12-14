#!/bin/bash
/app/.venv/bin/python dsview/obsidian/setup.py

/app/.venv/bin/fastapi run dsview/api.py & \
/app/.venv/bin/streamlit run dsview/labelling/interface.py