#!/bin/bash
/app/.venv/bin/python dsview/obsidian/setup.py

/app/.venv/bin/fastapi run dsview/api.py --host 0.0.0.0 & \
/app/.venv/bin/streamlit run dsview/interface/labelling_interface.py --server.address 0.0.0.0 & \
/app/.venv/bin/python dsview/interface/run_dashboard.py &
