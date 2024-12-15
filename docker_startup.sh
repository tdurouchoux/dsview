#!/bin/bash
/app/.venv/bin/python dsview/obsidian/setup.py

/app/.venv/bin/fastapi run dsview/api.py --host 0.0.0.0 & \
/app/.venv/bin/streamlit run dsview/labelling/interface.py --server.address 0.0.0.0